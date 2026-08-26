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
caveats, and what's still open: **`PROJECT_STATUS_REPORT.md`**. Full runnable
commands with expected output at each step: **`DEMO_GUIDE.md`**.

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
uninformative. See `PROJECT_STATUS_REPORT.md` §6.17 for the full picture.

## Architecture

``` text
CloudTrail
    │
    ▼
Feature Engineering
    │
    ▼
Incremental Graph Update
(Neo4j + In-Memory Graph)
    │
    ▼
K-Hop Neighbourhood Extraction
    │
    ▼
PyTorch Geometric HeteroData
    │
    ▼
GraphSAGE
    │
    ├── Benign
    └── Malicious
            │
            ▼
     Blast Radius Analysis
            │
            ▼
        JSON Alert
```

## Why GraphSAGE?

-   Inductive learning for unseen AWS entities
-   Efficient neighborhood sampling by prioritising edges with higher probability as attack in the sample
-   Streaming inference without retraining *(design goal — see the status note under Streaming Inference)*
-   Scales to continuously growing graphs
-   Works naturally with heterogeneous IAM graphs

## Training Pipeline
 Build Neo4j graph.
 Convert to PyTorch Geometric HeteroData.
 Fit scalers and label encoders.
 Train GraphSAGE.
 Save checkpoint.

## Streaming Inference

> ### ⚠️ NOT CURRENTLY OPERATIONAL
>
> The streaming path below is **the design, not the current state.** It is broken in two independent ways and must not be claimed in a paper or demo until both are fixed:
>
> 1. **`infer.py` feature-schema desync.** Its `_edge_features()` builds a numeric vector that no longer matches what the trained checkpoint's `edge_scaler` expects, so step 6 raises rather than running. (It fails closed, not silently — but it does not run.)
> 2. **The incremental updater does not reproduce the batch graph.** Three `TestBatchIncrementalEquivalence` assertions fail (see `PROJECT_STATUS_REPORT.md` §6.9). Two of them are on `hop_count` and `distance_to_sensitive_resource` — both live model inputs — so step 3 would feed the model different values than the batch pipeline it was trained against.
>
> The batch evaluation path (`build_graph.py` → `data_loader.py` → `evaluate_session_level.py`) is unaffected, and every reported result comes from it. Fix the two defects above, or scope streaming out of the write-up, before making any real-time claim.

1.  Watch incoming directory.
2.  Feature engineer new event.
3.  Incrementally update Neo4j.
4.  Extract affected k-hop neighborhood.
5.  Build HeteroData.
6.  Apply training scalers.
7.  Run GraphSAGE.
8.  Trigger blast radius if malicious.
9.  Save JSON alert.

## Tests

```powershell
python run_tests.py            # fast suites, ~2s, no Neo4j/Docker/checkpoint needed
python run_tests.py --all      # adds the batch/streaming equivalence suite (~10 min)
```

| Suite | Covers |
|---|---|
| `test_data_loader.py` | rank normalization, node/edge feature construction, `<UNK>` fallback, one-hot encoding, scaler discipline, global edge ordering |
| `test_evaluation_integrity.py` | edge→session join, graph/CSV provenance guard, session max-aggregation, paired baseline comparison |
| `test_models.py` | logit/label alignment contract for GraphSAGE and GAT, untrained-triple handling, feature widths |
| `test_incremental_updater.py` | batch vs. streaming graph equivalence (slow; **3 known failures**, see `PROJECT_STATUS_REPORT.md` §6.9) |

The first three suites exist because an end-to-end audit found eight defects,
and none of them were caught by the existing tests — every one lived in a
module with no coverage. Each test names the specific defect it pins, so the
failure message tells you what regressed rather than just that something did.
The known failures in the fourth suite are real and documented; don't "fix"
them by deleting the assertions.

## Repository

-   train.py --- training (GraphSAGE and GAT)
-   infer.py --- streaming inference + checkpoint wrapping (see note below)
-   model_graphsage.py / model_gat.py --- models
-   data_loader.py --- graph loading, feature normalization
-   privilege_features.py --- node/edge identity, relation classification
-   graph_construction/neo4j_graph_builder.py --- batch graph construction
-   incremental_updater.py --- streaming graph updates
-   feature_engine9.py --- feature engineering (raw CloudTrail -> structural/temporal CSVs)
-   datasets/privilege-escalation/generate_synthetic_data.py --- synthetic training data generator
-   build_graph.py --- CLI wrapper to load a structural CSV into Neo4j
-   evaluate_on_real.py --- edge-level real-data evaluation
-   evaluate_session_level.py --- session-level real-data evaluation (comparable to the rule baseline)
-   datasets/privilege-escalation/evaluate_baselines.py --- rule-based baselines
-   blast_radius.py --- downstream reachability/impact analysis (not yet exercised)
-   explainability.py --- prediction explanations (not yet exercised)
-   PROJECT_STATUS_REPORT.md --- full evaluation history, evidence, and publication roadmap
-   DEMO_GUIDE.md --- runnable demo script with expected output at each step

## Training

``` bash
python train.py \
    --model sage \
    --epochs 100 \
    --save_dir ./checkpoints
```

## Wrap checkpoint

``` bash
python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt --wrapped-output checkpoints/best_GraphSAGE_wrapped.pt
```

## Evaluate against real data

``` bash
python evaluate_on_real.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage
python evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage --raw-csv datasets/privilege-escalation/real_dataset_test.csv --threshold 0.35
```

Full setup (Neo4j, environment variables, expected output) in `DEMO_GUIDE.md`.

## Run live streaming inference

``` bash
python infer.py   --checkpoint checkpoints/best_GraphSAGE_wrapped.pt   --watch incoming   --alert-dir alerts   --threshold 0.5   --seed-from-neo4j
```
Insert json logs into incoming directory to get real time prediction of the action performed.

**Known issue**: `infer.py`'s live single-event feature builder constructs
edge features independently of `data_loader.py` and has not yet been
updated for the rank-normalized feature schema behind the current best
checkpoint (see `PROJECT_STATUS_REPORT.md` section 6.16) — it will run
without erroring but produce incorrect scores until this is fixed. Use the
batch evaluation commands above for anything that needs to be trusted right
now.

## Outputs

Alerts are written into the alerts directory as JSON.

## Design Principles

-   Incremental graph updates
-   No retraining
-   Inductive GNN
-   Explainable blast radius
-   Consistent preprocessing between training and inference
