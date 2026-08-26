# Real-Time GraphSAGE Privilege Escalation Detection

## Overview

This project detects AWS privilege-escalation attacks from CloudTrail
logs using a heterogeneous Graph Neural Network (GraphSAGE), Neo4j, and
incremental streaming inference. Trained on procedurally-generated
synthetic CloudTrail sessions, validated against real attack data collected
with [Stratus Red Team](https://stratus-red-team.cloud/) across 4
independent AWS accounts.

## Results

All figures below are on **the same 238 held-out real test sessions**.

| | Precision | Recall | F1 |
|---|---|---|---|
| GraphSAGE, session-level, real held-out test data | 0.874 | 0.830 | **0.851** [95% CI: 0.794, 0.900] |
| Rule-based baseline (GuardDuty-style, 11 rules) | 0.878 | 0.650 | 0.747 [95% CI: 0.667, 0.811] |

**Paired bootstrap on the difference: +0.104 F1, 95% CI [+0.040, +0.171], p = 0.0008.**

The GNN clears the rule-based baseline on real, previously-unseen attack
sessions — verified with a dev-set-only selected threshold (0.65), checked once
on held-out test data, threshold-stability checked, and — because both systems
score the *same* sessions — compared with a **paired** bootstrap rather than by
eyeballing two separate confidence intervals.
Getting here required diagnosing and fixing a real synthetic-to-real
generalization gap (two structural bugs, one ruled-out hypothesis, and the
actual fix — a rank-normalization feature transform). Full evidence trail,
caveats, and what's still open: **`docs/PROJECT_STATUS_REPORT.md`**. Full runnable
commands with expected output at each step: **`docs/DEMO_GUIDE.md`**.

Two honest caveats up front:

1. **The win is a *session-level* effect** — the model correctly flags at least
   one edge per attack session while staying quiet on benign ones. It is not
   accurate per-action classification.
2. **Edge-level ranking on real data is inverted** (test AUC ≈ 0.26). Pooled
   across all 25,984 in-schema edges, real attack edges score *lower* than real
   benign ones, yet the per-session maximum separates the two classes well
   (session AUC 0.921). This is a real, unexplained phenomenon and an open
   question — do not describe this system as an edge-level detector.

Both were verified against controls the result survives: a permutation test
that preserves session sizes while destroying the edge→session association
(observed session AUC beats all 200 permutations), and within-length-strata
AUCs of 0.998 / 0.970 / 0.872 / 0.759 where session length alone is
uninformative. See `docs/PROJECT_STATUS_REPORT.md` §6.17 for the full picture.

## Architecture

The path every reported result actually comes from is batch, not streaming
(see the note under Streaming Inference below):

``` text
Raw CloudTrail (CSV / JSON)
    │
    ▼
feature_engine9.py
    │
    ├──→ structural.csv → build_graph.py → Neo4j → data_loader.py → GraphSAGE → evaluate_session_level.py
    │                                                                            (session-level F1, reported results)
    └──→ temporal.csv → LSTMTransformerV5
```

`ensemble.py` is a separate, third consumer of the same two feature CSVs: it
combines a pure-topology GNN score (deliberately not the trained GraphSAGE
checkpoint's edge probability — see its module docstring) with the LSTM's
per-event probability into one `risk_score` per event. See the Ensemble
section below.

## Why GraphSAGE?

-   Inductive learning for unseen AWS entities
-   Efficient neighborhood sampling by prioritising edges with higher probability as attack in the sample
-   Streaming inference without retraining *(design goal — see the status note under Streaming Inference)*
-   Scales to continuously growing graphs
-   Works naturally with heterogeneous IAM graphs

## Streaming Inference

> ### ⚠️ NOT CURRENTLY OPERATIONAL
>
> The streaming path below is **the design, not the current state.** It is broken in two independent ways and must not be claimed in a paper or demo until both are fixed:
>
> 1. **`infer.py` feature-schema desync.** Its `_edge_features()` builds a numeric vector that no longer matches what the trained checkpoint's `edge_scaler` expects, so step 6 raises rather than running. (It fails closed, not silently — but it does not run.)
> 2. **The incremental updater does not reproduce the batch graph.** Three `TestBatchIncrementalEquivalence` assertions fail (see `docs/PROJECT_STATUS_REPORT.md` §6.9). Two of them are on `hop_count` and `distance_to_sensitive_resource` — both live model inputs — so step 3 would feed the model different values than the batch pipeline it was trained against.
>
> The batch evaluation path (`graph_construction/build_graph.py` → `graph_construction/data_loader.py` → `graph_construction/evaluate_session_level.py`) is unaffected, and every reported result comes from it. Fix the two defects above, or scope streaming out of the write-up, before making any real-time claim.

Design (once fixed): watch `incoming/` → feature-engineer each new event →
incrementally update Neo4j → extract the affected k-hop neighbourhood →
run GraphSAGE → trigger blast radius on a malicious verdict → write a JSON
alert into `alerts/`. See the command at the bottom of this file for how
it's invoked once operational.

## Repository

-   `feature_engine9.py` --- feature engineering: raw CloudTrail (CSV/JSON) -> structural.csv (GNN) + temporal.csv (LSTM). Also the fast-lane alert on defense-evasion actions.
-   `ensemble.py` --- combined GNN structural + LSTM temporal risk score, one 0-10 `risk_score` per EVENT
-   `leakage_guard.py` --- one shared definition of "held out"; audits any file for train/test contamination across the graph and sequence tracks
-   `datasets/privilege-escalation/generate_synthetic_data.py` --- synthetic training data generator
-   `datasets/privilege-escalation/evaluate_baselines.py` --- rule-based baselines
-   `graph_construction/train.py` --- training (GraphSAGE and GAT)
-   `graph_construction/infer.py` --- streaming inference + checkpoint wrapping (see note below)
-   `graph_construction/model_graphsage.py` / `graph_construction/model_gat.py` --- models
-   `graph_construction/data_loader.py` --- graph loading, feature normalization
-   `graph_construction/privilege_features.py` --- node/edge identity, relation classification
-   `graph_construction/neo4j_graph_builder.py` --- batch graph construction
-   `graph_construction/incremental_updater.py` --- streaming graph updates
-   `graph_construction/build_graph.py` --- CLI wrapper to load a structural CSV into Neo4j
-   `graph_construction/evaluate_on_real.py` --- edge-level real-data evaluation
-   `graph_construction/evaluate_session_level.py` --- session-level real-data evaluation (comparable to the rule baseline)
-   `graph_construction/blast_radius.py` --- downstream reachability/impact analysis (not yet exercised)
-   `graph_construction/explainability.py` --- prediction explanations (not yet exercised)
-   `tests/run_tests.py` --- one command to run the project's test suites
-   `docs/PROJECT_STATUS_REPORT.md` --- full evaluation history, evidence, and publication roadmap
-   `docs/DEMO_GUIDE.md` --- runnable demo script with expected output at each step

## Setup

``` bash
pip install -r requirements.txt
```
Neo4j must be running (`bolt://localhost:7687` by default) for graph
construction, training, and evaluation. Full environment setup in
`docs/DEMO_GUIDE.md`.

## Feature engineering

Raw CloudTrail in, `structural.csv` (GNN) + `temporal.csv` (LSTM) out.
Fires a `[FAST-LANE ALERT]` immediately on any defense-evasion action
(`StopLogging`, `DeleteTrail`, `UpdateDetector`, `DeleteFlowLogs`), ahead of
full feature computation:

``` bash
# One-shot batch, default input (datasets/privilege-escalation/synthetic_cloudtrail.csv)
python feature_engine9.py

# Any other CSV/JSON input -- always freeze the vocab on evaluation data
python feature_engine9.py --input datasets/privilege-escalation/real_dataset_dev.csv --freeze-vocab

# Watch a folder and process each new log file as it lands (Ctrl+C to stop)
python feature_engine9.py --watch incoming
# ...and simulate arrivals by chunking --input into it while watching
python feature_engine9.py --watch incoming --simulate
```

`action_risk_prior` / `principal_type_prior_risk` are label-derived
features: only the training input (`synthetic_cloudtrail.csv`) may fit
them (`--fit-priors`); every other input is frozen automatically, and
`--fit-priors` on a non-training input is refused outright.

## Training

``` bash
python graph_construction/train.py \
    --model sage \
    --epochs 100 \
    --save_dir ./checkpoints
```

## Wrap checkpoint

``` bash
python graph_construction/infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt --wrapped-output checkpoints/best_GraphSAGE_wrapped.pt
```

## Evaluate against real data

``` bash
python graph_construction/evaluate_on_real.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage
python graph_construction/evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage --raw-csv datasets/privilege-escalation/real_dataset_test.csv --threshold 0.35
```

Full setup (Neo4j, environment variables, expected output) in `docs/DEMO_GUIDE.md`.

## Ensemble: combined GNN + LSTM risk score

Fuses the GNN's structural score (pure graph topology, not the trained
classifier's edge probability -- see `ensemble.py`'s module docstring for
why) with the LSTM's per-event probability into one 0-10 `risk_score` per
EVENT, in arrival order, so an attack chain shows up as a run of rising
scores:

This file is the single entry point -- it calls `feature_engine9.py`
internally, so you never invoke that script by hand:

``` bash
# One-shot: score one file
python ensemble.py --input datasets/privilege-escalation/real_dataset_dev.csv --out risk_scores_dev.csv --freeze-vocab

# Batch/continuous: watch a folder and re-score the full accumulated
# dataset each time a new log file lands (Ctrl+C to stop)
python ensemble.py --watch incoming --out risk_scores.csv --freeze-vocab
# ...and simulate arrivals by chunking --input into it while watching
python ensemble.py --watch incoming --simulate --input datasets/privilege-escalation/real_dataset_dev.csv --freeze-vocab
```

`--weight-gnn`/`--weight-lstm` (default 0.5/0.5, must sum to 1.0), `--source
csv|neo4j` (build the graph from the CSV directly, or read it back from a
live Neo4j already loaded via `build_graph.py`), `--show-table` (also print
the scored table to the terminal, one-shot mode only). The LSTM step is the
slow one on CPU -- budget a couple of minutes per full rescore.
`--freeze-vocab` on non-training input, same as `feature_engine9.py`; in
`--watch` mode the label-derived priors are always frozen, never fit.

## Run tests

``` bash
python tests/run_tests.py            # fast suites only (~2s, no external deps)
python tests/run_tests.py --all      # adds the slow batch/streaming equivalence suite (~10 min)
```

## Audit for train/test leakage

Checks any file for rows that belong to the held-out dev/test split --
before you let anything train on it:

``` bash
python leakage_guard.py temporal-analysis/data/lstm/train_temporal.csv
```

## Run live streaming inference

``` bash
python graph_construction/infer.py   --checkpoint checkpoints/best_GraphSAGE_wrapped.pt   --watch incoming   --alert-dir alerts   --threshold 0.5   --seed-from-neo4j
```
Insert json logs into the incoming directory to get a real-time prediction
of the action performed — once the two defects flagged above are fixed
(see `docs/PROJECT_STATUS_REPORT.md` §6.16 for the current feature-schema
desync). Use the batch evaluation commands above for anything that needs
to be trusted right now.

