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
 (CloudTrail JSON,      +-> temporal row   -> the principal's last hour   -> LSTM -> p_sequence -+   w p_graph + (1-w) p_sequence
  JSONL, or CSV)                                                                                     w, threshold: pipeline_config.json
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

### Alert explanations

Every alert says why it was flagged. The top `explain_top_events` events of each
alert (default 3, in `pipeline_config.json`; `--no-explain` turns it off) get an
explanation, computed on the exact graph window and LSTM history they were
scored with (`ensemble_explain.py`). A one-line summary prints under the alert:

``` text
[ALERT] stratus-redteam: 133 event(s), max risk 8.43/10 (top: PutRolePolicy)
        why: risk 8.43/10 (alert at 5.14) | LSTM 49% (p=0.82): the action PutRolePolicy (+2.40);
             risk learned for this action (+1.05); earlier GetUser 0.08 min before (+0.19)
             | HGT 51% (p=0.86): how common this action is overall (49%); known privilege-escalation action (39%)
```

How to read it:
- **Model shares** (`LSTM 49%`, `HGT 51%`) are exact: the ensemble is
  `w p_graph + (1-w) p_sequence`, so each term's share of the risk is its share
  of the alert. A model under 25% of the risk is reported as "not a driver". When
  the graph model has no weights for an event's relation, the LSTM decides alone
  and the summary says so.
- **LSTM reasons** come from `temporal-analysis/lstm_explain.py`: feature
  contributions by Integrated Gradients (logit units; they add up to the score's
  change from an all-absent input), and "earlier X n min before", the effect of
  removing that earlier event from the principal's 10-minute window.
- **HGT reasons** are gradient x input on the flagged event's edge in the window
  graph, as shares of the total (a first-order attribution, not an exact
  decomposition). Other events that moved the score through the graph are named
  when they carry at least 5% of it.
- **Fast-lane** events (trail deletion and similar) are flagged by rule whatever
  their risk; the summary starts with the rule.

The alert JSON's `explanations` list has the full detail per event: each model's
probability, weight, contribution and share, the graph's feature shares and
related events (with source, target and action), and the LSTM's per-feature and
per-earlier-event effects. Every feature carries its raw name and a label.

## Run it

One command, with Docker Desktop running:

``` powershell
.\run.cmd          # Windows (PowerShell or cmd)
./run.sh           # macOS / Linux / Git Bash
```

It does everything:
- builds the image if needed (and rebuilds it when the Dockerfile changes);
- clears the previous run's scores, alerts and per-principal history;
- starts the pipeline and streams `real_dataset_test.csv` into `incoming/`,
  200 events every 5 s, printing every event's HGT, LSTM and risk score, with
  fast-lane and per-principal alerts inline;
- serves the dashboard (`cloudsec_dashboard.py`) on http://localhost:8501 and
  opens it in the browser. The page refreshes itself as new scores arrive.

Ctrl+C stops all of it. Options pass through, e.g.
`.\run.cmd --feed-interval 2 --feed-limit 2000`, or a different dataset:
`.\run.cmd datasets/privilege-escalation/real_dataset_dev.csv`.

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
| GraphSAGE alone (batch, session-level) | -- | -- | session **AUC 0.986** (F1 threshold-sensitive) |

> **All rows are on one footing** — the same 821-session real test set (4 collectors, including 30 real Stratus privilege-escalation detonations), one consistent feature engine, each model's threshold/config frozen on dev and test scored once. GraphSAGE (batch) has session **AUC 0.986** (confound-controlled: beats all 200 size-preserving permutations, p=0.005) -- the robust primary metric, STABLE across every dataset iteration (0.982-0.987). Its operating-point F1 is threshold-sensitive (0.79-0.89 depending on where the dev-selected threshold lands on a flat plateau); we report AUC as primary for that reason. On held-out UNSEEN attack families the model reaches F1 0.929 / AUC 0.997. The pipeline, XGBoost and Random Forest are statistically tied (pipeline vs XGBoost -0.003 p=0.92; vs RF +0.009 p=0.68); all beat the rule baseline except logistic regression.

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
-   `run.cmd`, `run.sh` -- the one-command demo (pipeline + feeder + dashboard in Docker)
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
