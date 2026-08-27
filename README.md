# Real-Time GraphSAGE Privilege Escalation Detection

Detects AWS privilege-escalation attacks from CloudTrail logs using a
heterogeneous Graph Neural Network (GraphSAGE), Neo4j, and an LSTM
sequence model. Trained on synthetic CloudTrail sessions, validated
against real attack data collected with
[Stratus Red Team](https://stratus-red-team.cloud/) across 4 independent
AWS accounts.

## Results

Session-level, on 238 held-out real test sessions:

| | Precision | Recall | F1 |
|---|---|---|---|
| GraphSAGE | 0.874 | 0.830 | **0.851** [95% CI: 0.794, 0.900] |
| Rule-based baseline (GuardDuty-style) | 0.878 | 0.650 | 0.747 [95% CI: 0.667, 0.811] |

Paired bootstrap on the difference: +0.104 F1, 95% CI [+0.040, +0.171], p = 0.0008.

Two caveats: the win is a *session-level* effect, not accurate per-action
classification; and edge-level ranking on real data is inverted (AUC ≈
0.26) even though session-level aggregation works (AUC 0.921) -- a real,
unexplained phenomenon. Full evidence trail: `docs/PROJECT_STATUS_REPORT.md`.
Full runnable walkthrough: `docs/DEMO_GUIDE.md`.

## Architecture

Every reported result comes from the batch path (streaming is not
operational -- see below):

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

`ensemble.py` is a third consumer of the same two feature CSVs: it
combines a pure-topology GNN score (not the trained checkpoint's edge
probability -- see its module docstring for why) with the LSTM's
per-event probability into one `risk_score` per event.

## Streaming inference: not operational

`graph_construction/infer.py`'s live path is broken (feature-schema
desync, and the incremental updater doesn't reproduce the batch graph --
details in `docs/PROJECT_STATUS_REPORT.md` §6.9/§6.16). Don't claim
real-time inference works until both are fixed. Use the batch commands
below, or `ensemble.py --watch` (works today, see below).

## Repository

-   `feature_engine9.py` -- raw CloudTrail -> structural.csv (GNN) + temporal.csv (LSTM), plus fast-lane alerts
-   `ensemble.py` -- combined GNN + LSTM risk score, one 0-10 `risk_score` per event
-   `leakage_guard.py` -- audits any file for train/test contamination
-   `datasets/privilege-escalation/` -- synthetic data generator, rule baselines, raw/derived datasets
-   `graph_construction/` -- models, training, Neo4j graph construction, evaluation, streaming inference
-   `tests/run_tests.py` -- test suites
-   `docs/PROJECT_STATUS_REPORT.md` -- full evaluation history and evidence
-   `docs/DEMO_GUIDE.md` -- runnable demo with expected output

## Setup

``` bash
pip install -r requirements.txt
```

## Run

``` bash
python ensemble.py
```

Runs the full pipeline end-to-end (feature engineering → GNN + LSTM →
ensemble) on `datasets/privilege-escalation/synthetic_cloudtrail.csv`,
printing a `[FAST-LANE ALERT]` immediately on any defense-evasion action
and writing `risk_scores.csv` with one 0-10 `risk_score` per event.
No Neo4j required. For any other input, watch mode, training, evaluation,
tests, or the leakage audit, see `docs/DEMO_GUIDE.md` and each script's
own `--help`.
