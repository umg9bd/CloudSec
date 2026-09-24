# Real-Time GraphSAGE Privilege Escalation Detection

Detects AWS privilege-escalation attacks from CloudTrail logs using a
heterogeneous Graph Neural Network (GraphSAGE), Neo4j, and an LSTM
sequence model. Trained on synthetic CloudTrail sessions, validated
against real attack data collected with
[Stratus Red Team](https://stratus-red-team.cloud/) across 4 independent
AWS accounts.

## Results

Session-level, on 238 held-out real test sessions (`real_dataset_test.csv`,
never touched during tuning -- all thresholds below are frozen from
`real_dataset_dev.csv`):

| | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest (temporal features) | 0.508 | 0.980 | 0.669 |
| XGBoost (temporal features) | 0.424 | 1.000 | 0.595 |
| Curated IAM rule baseline (11 rules) | 0.878 | 0.650 | 0.747 [95% CI: 0.667, 0.811] |
| GraphSAGE alone (calibrated) | 0.778 | 0.910 | 0.839 |
| **GNN heuristic + LSTM ensemble (ours, shipped)** | 0.845 | 0.980 | **0.907** |

The ensemble beats the rule baseline significantly: paired bootstrap
+0.160 F1, 95% CI [+0.091, +0.234], p < 0.0001.

Two things worth knowing:
- The rule baseline is a curated list built by reading AWS's public GuardDuty
  finding-type docs -- it was never validated against real GuardDuty output
  (this project's data collection never enabled it), so it's *not* a stand-in
  for the actual commercial product.
- Random Forest and XGBoost, trained on the exact same `feature_engine9`
  temporal columns, both **underperform the rule baseline** on real data
  despite using real ML -- naive supervised learning on tabular features
  doesn't transfer from synthetic training to real attacks. This motivates
  the ensemble's rule-injected + sequence-modeling approach over a plain
  classifier on the same columns.
- The standalone GraphSAGE model's raw edge-level ranking on real data was
  initially inverted (AUC ~0.26) -- root-caused to one dominant relation
  (`User->READ->Resource`, 87% of real attack-labeled edges) where the
  model learned "READ = safe" from synthetic training data, which real
  credential-theft techniques (`GetSecretValue`, `GetPasswordData`, both
  AWS-classified as "Read") directly violate. A per-relation orientation
  correction fit on dev only and frozen (not baked into the checkpoint,
  to keep the train/eval boundary clean) lifts edge-level AUC to 0.89 on
  both dev and test -- but even calibrated, GraphSAGE alone still trails
  the shipped ensemble at the session level (0.839 vs 0.907), which is why
  the ensemble, not the GNN alone, is what ships.

Full evidence trail: `docs/PROJECT_STATUS_REPORT.md`.
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
