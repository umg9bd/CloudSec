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
| Logistic regression (bag of actions) | 0.706 | 0.960 | 0.814 [95% CI: 0.756, 0.864] |
| Random Forest (temporal features) | 0.838 | 0.830 | 0.834 [95% CI: 0.777, 0.886] |
| XGBoost (temporal features) | 0.823 | 0.930 | 0.873 [95% CI: 0.822, 0.917] |
| Curated IAM rule baseline (11 rules) | 0.878 | 0.650 | 0.747 [95% CI: 0.667, 0.811] |
| GraphSAGE alone (calibrated) | 0.778 | 0.910 | 0.839 |
| **Ensemble candidate A** -- `ensemble.py`, fixed 0.5/0.5 sum | 0.845 | 0.980 | **0.907** |
| **Ensemble candidate B** -- `ensemble1.py`, stacked meta-learner | 0.838 | 0.980 | **0.903** |

Two ensemble candidates are kept as peers until a final choice is made. They
share the same GNN-heuristic and LSTM per-event scorers, CLI, and output
columns, and differ only in how the two branches are combined. With the
pre-leak-fix LSTM checkpoint (see the first note below -- these figures do not
hold with a clean LSTM), each beat the rule baseline significantly (paired
bootstrap: A +0.160 F1, 95% CI [+0.091,
+0.234]; B +0.156, [+0.086, +0.231]; both p < 0.0001); the difference between
them is not significant (B - A = -0.004, 95% CI [-0.019, +0.009], p = 0.79).
Reproduce the head-to-head with `datasets/privilege-escalation/compare_ensembles.py`.

Worth knowing:
- **The ensemble rows above are not publishable as-is.** They use an LSTM
  checkpoint that predates the sequence track's leak fix: it trained on the
  real Invictus capture and selected its epoch on a real attack user. Retrained
  on the leakage-clean, synthetic-only data, the ensembles score **A 0.769 /
  B 0.722** on the same test sessions -- **not significantly better than the
  rule baseline (A - rules = +0.021, 95% CI [-0.060, +0.104])**, and
  **significantly worse than the classical ML baselines above** (XGBoost - A =
  +0.105, 95% CI [+0.054, +0.159], p < 0.0001). See
  `docs/PROJECT_STATUS_REPORT.md` §6.20-6.22; which system the paper reports
  is an open decision.
- The rule baseline is a curated list built by reading AWS's public GuardDuty
  finding-type docs -- it was never validated against real GuardDuty output
  (this project's data collection never enabled it), so it's *not* a stand-in
  for the actual commercial product.
- The classical ML baselines (`evaluate_ml_baselines.py`) train on exactly
  the synthetic table the LSTM trains on, and each gets its configuration
  chosen on dev, as the proposed system did. An earlier version trained on
  less data and on three features the synthetic generator hardcodes for
  attacks (MFA fields, request-parameter length). It scored RF 0.669 /
  XGBoost 0.595, and its conclusion that "ML on these features doesn't
  transfer" was wrong. On equal footing XGBoost reaches 0.873, the best
  real-test result in the project so far.
- The standalone GraphSAGE model's raw edge-level ranking on real data was
  initially inverted (AUC ~0.26) -- root-caused to one dominant relation
  (`User->READ->Resource`, 87% of real attack-labeled edges) where the
  model learned "READ = safe" from synthetic training data, which real
  credential-theft techniques (`GetSecretValue`, `GetPasswordData`, both
  AWS-classified as "Read") directly violate. A per-relation orientation
  correction fit on dev only and frozen (not baked into the checkpoint,
  to keep the train/eval boundary clean) lifts edge-level AUC to 0.89 on
  both dev and test -- but even calibrated, GraphSAGE alone still trails
  both ensemble candidates at the session level (0.839 vs 0.903-0.907),
  which is why an ensemble, not the GNN alone, is the final system.

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
