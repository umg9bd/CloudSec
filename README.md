# Real-Time Privilege Escalation Detection for AWS CloudTrail

Detects AWS privilege-escalation attacks in CloudTrail logs as they arrive.
Two models score every event in parallel: a heterogeneous graph transformer
(HGT) over the principal/resource graph, and an LSTM-Transformer over each
principal's recent activity. An ensemble combines the two scores into one
risk score per event. Both models are trained on synthetic CloudTrail and
validated on real attacks collected with
[Stratus Red Team](https://stratus-red-team.cloud/) across 4 independent AWS
accounts.

## Architecture

``` text
                        +-> structural row -> graph (rolling 24 h window) -> HGT  -> p_graph ----+
incoming/<file> -> feature_engine9                                                               +-> ensemble -> alert
 (CloudTrail JSON,      +-> temporal row   -> the principal's last hour   -> LSTM -> p_sequence -+   0.4 p_graph + 0.6 p_sequence
  JSONL, or CSV)                                                                                     alert at >= 5.92/10
```

`pipeline.py` runs this. It watches `incoming/` and scores each new file's
events. It writes alerts to `alerts/alert_<id>.json` (one per principal per
file) and every event's scores to `output/risk_scores.csv`. Processed files
are moved to `incoming/processed/`.

- **Same scores as batch.** Each event gets the score a batch run over the same
  events would give it:
  - The LSTM branch matches a single batch pass on real dev to 3e-7.
  - The graph is rebuilt with the batch code on every file.
  - The Neo4j-free graph builder (`graph_construction/offline_graph.py`) is
    tensor-identical to the Neo4j loader on real dev and on the synthetic
    training graph.
  - `tests/test_pipeline.py` guards all of this.
- **Frozen at inference time.** Live traffic never changes the model's inputs:
  feature_engine9's vocabulary and risk priors are frozen from training, and
  both models use their training-time scalers.
- **Settings.** The ensemble weight, alert threshold, checkpoints and graph
  window live in `pipeline_config.json`. Its weight and threshold were chosen
  on `real_dataset_dev.csv` only, by
  `datasets/privilege-escalation/evaluate_pipeline.py`.

## Run it

One command, with Docker Desktop running:

``` powershell
.
un.cmd          # Windows (PowerShell or cmd)
./run.sh           # macOS / Linux / Git Bash
```

It builds the image if needed and starts the pipeline. It streams
`real_dataset_test.csv` into `incoming/`, 200 events every 5 s, and prints
every event's HGT, LSTM and risk score, with fast-lane and per-principal
alerts inline. Ctrl+C stops it. Options pass through, e.g.
`.
un.cmd --feed-interval 2 --feed-limit 2000`, or a different dataset:
`.
un.cmd datasets/privilege-escalation/real_dataset_dev.csv`.

The pieces, run individually. PyTorch runs in Docker because Windows Smart
App Control blocks torch's unsigned DLLs on the development machine.

``` bash
docker build -t cloudsec .
docker run --rm -v "$PWD:/app" cloudsec python pipeline.py --watch incoming
# then drop CloudTrail files into incoming/, e.g.
cp samples/cloudtrail/synthetic_attack_chain.json incoming/
```

``` text
[FAST-LANE ALERT] 2026-07-30 09:00:45+00:00 session1 StopLogging: CloudTrail logging disabled
[ALERT] session1: 9 event(s), max risk 9.91/10 (top: SetDefaultPolicyVersion)
[PIPELINE] synthetic_attack_chain.json: 10 events scored, 9 above threshold (0.5s)
```

To stream a whole dataset through it the way CloudTrail delivers logs (a new
file of events every few seconds), with a live line per event:

``` bash
docker run --rm -it -v "$PWD:/app" cloudsec python pipeline.py --watch incoming --show-events
python feed_incoming.py --batch-size 200 --interval 5     # second terminal (or add --feed to the line above)
```

``` text
2023-07-10 11:42:18  benjamin                     GetRegionOptStatus             HGT  0.92  LSTM 0.16  risk  4.63/10
[FAST-LANE ALERT] 2023-07-10 11:59:02+00:00 bert-jan DeleteTrail: CloudTrail trail deleted
[ALERT] bert-jan: 61 event(s), max risk 9.98/10 (top: TagInstanceProfile)
[PIPELINE] real_dataset_test_batch0005.csv: 200 events scored, 62 above threshold (1.6s)
```

Other commands:

- Score files once: `python pipeline.py --files a.json b.json`.
- Start with no per-principal history: add `--reset-state`.
- Tests: `python tests/run_tests.py` (101 tests).
- Re-tune on dev (add `--test` for the single held-out test run):
  `python datasets/privilege-escalation/evaluate_pipeline.py`.

## Results

Session-level results on 238 held-out real test sessions
(`real_dataset_test.csv`). A session is flagged if any of its events alerts.
Every configuration and threshold was frozen on `real_dataset_dev.csv` before
test was used, and each system was run on test once.

| | Precision | Recall | F1 [95% CI] |
|---|---|---|---|
| **Real-time pipeline** (HGT + LSTM, `pipeline.py`) | 0.825 | 0.925 | **0.872** [0.823, 0.915] |
| Random Forest (temporal features) | 0.841 | 0.888 | 0.864 [0.812, 0.909] |
| XGBoost (temporal features) | 0.838 | 0.916 | 0.875 [0.825, 0.918] |
| Logistic regression (bag of actions) | 0.600 | 0.869 | 0.710 [0.644, 0.769] |
| Curated IAM rule baseline (11 rules) | 0.904 | 0.701 | 0.789 [0.720, 0.849] |
| GraphSAGE alone (batch, session-level) | 0.868 | 0.925 | **0.896** |

> **All rows are on one footing** — the same 821-session real test set (4 collectors, including 30 real Stratus privilege-escalation detonations), one consistent feature engine, each model's threshold/config frozen on dev and test scored once. GraphSAGE (batch) leads at 0.896 (paired +0.107 over rules, p=0.0002; session AUC 0.987, beats all 200 size-preserving permutations). The pipeline, XGBoost and Random Forest are statistically tied (pipeline vs XGBoost -0.003 p=0.92; vs RF +0.009 p=0.68); all beat the rule baseline except logistic regression.

The pipeline beats the rule baseline significantly (paired bootstrap +0.083
F1, 95% CI [+0.023, +0.148], p = 0.008). It is statistically tied with the
classical ML baselines: vs Random Forest +0.009 (p = 0.68), vs XGBoost -0.003
(p = 0.92), vs logistic regression +0.162 (p < 0.001).

On dev, the ensemble beats each of its branches:

| Real dev | F1 |
|---|---|
| HGT alone | 0.868 |
| LSTM alone | 0.894 |
| Ensemble | 0.908 |

`pipeline.py` is the only ensemble. The earlier `ensemble.py` /
`ensemble1.py` were removed: their graph side was a hand-written topology
rule, not a trained GNN, and with a leak-clean LSTM they scored 0.769 / 0.722
on test (§6.21; the code is in git history at commit `2fe5977`).

Worth knowing:
- **What the pipeline's numbers do and don't show.** The ML baselines run on
  the same feature_engine9 features and synthetic training data, and they tie
  the pipeline. So the paper cannot yet claim that HGT + LSTM beats standard
  ML. The LSTM still carries known problems (`docs/PROJECT_STATUS_REPORT.md`
  §6.21-6.23), fixing them is the next lever, and each fix must be tuned on
  dev before test is used again.
- Older versions of this README reported 0.907 / 0.903 for those removed
  ensembles. Those numbers came from an LSTM checkpoint that had trained on
  real Invictus data, so they are not valid.
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
  transfer" was wrong. The numbers in the table above come from
  re-running them on the data merged from `feat/credential-access-chains`
  (§6.23).
- The standalone GraphSAGE model's raw edge-level ranking on real data was
  initially inverted (AUC ~0.26) -- root-caused to one dominant relation
  (`User->READ->Resource`, 87% of real attack-labeled edges) where the
  model learned "READ = safe" from synthetic training data, which real
  credential-theft techniques (`GetSecretValue`, `GetPasswordData`, both
  AWS-classified as "Read") directly violate. A per-relation orientation
  correction fit on dev only and frozen (not baked into the checkpoint,
  to keep the train/eval boundary clean) lifts edge-level AUC to 0.89 on
  both dev and test. The pipeline uses HGT rather than GraphSAGE. Trained on
  the same corrected schema, HGT's per-event ranking on real dev is healthier
  (event AUC 0.713 vs GraphSAGE's 0.517), which is what matters for an
  ensemble that combines per-event scores.

Full evidence trail: `docs/PROJECT_STATUS_REPORT.md`.
Full runnable walkthrough: `docs/DEMO_GUIDE.md`.

## Repository

-   `pipeline.py` -- the real-time detector (watch `incoming/`, HGT + LSTM, ensemble, alerts); settings in `pipeline_config.json`
-   `Dockerfile` -- the runtime everything torch-based runs in
-   `feature_engine9.py` -- raw CloudTrail -> structural rows (graph) + temporal rows (LSTM), plus fast-lane alerts
-   `graph_construction/offline_graph.py`, `gnn_scorer.py`, `model_hgt.py` -- Neo4j-free graph building and HGT/GraphSAGE/GAT scoring
-   `samples/cloudtrail/` -- example CloudTrail files to drop into `incoming/`
-   `feed_incoming.py` -- replays a dataset into `incoming/` batch by batch (a CloudTrail delivery simulator)
-   `run.cmd`, `run.sh` -- the one-command demo (pipeline + feeder in Docker)
-   `leakage_guard.py` -- audits any file for train/test contamination
-   `datasets/privilege-escalation/` -- synthetic data generator, rule baselines, raw/derived datasets
-   `graph_construction/` -- models (HGT, GraphSAGE, GAT), training (`train.py --model hgt --offline-csv ...`), graph construction, evaluation. `infer.py`'s incremental Neo4j path is legacy and not used by the pipeline.
-   `tests/run_tests.py` -- test suites
-   `docs/PROJECT_STATUS_REPORT.md` -- full evaluation history and evidence
-   `docs/DEMO_GUIDE.md` -- runnable demo with expected output

## Setup without Docker

``` bash
pip install -r requirements.txt
```

The batch tools (`feature_engine9.py`, the rule and ML baselines) run
natively. Anything that loads a torch checkpoint needs a machine where
PyTorch can load, or the Docker image above.
