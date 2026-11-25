> **Note.** These are the detailed notes for the Stage 4 real-time detector. The results
> table and alert example below come from the 14 Oct 2026 configuration (single HGT
> checkpoint, w = 0.5). The served configuration's numbers (3-seed HGT, w = 0.8,
> threshold 0.1849) are in the top-level [README](../README.md).

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
  on `real_dataset_dev.csv` only. The alert threshold (4.20/10) is the median
  best-F1 threshold over 1,000 bootstrap resamples of dev sessions, from the
  LSTM's out-of-fold scores (its heads were fitted on dev); `tuned_on` in the
  config records how.

### Alert explanations

Every alert says why it was flagged. The top `explain_top_events` events of each
alert (default 3, in `pipeline_config.json`; `--no-explain` turns it off), and
every fast-lane event whatever its risk, get an explanation, computed on the exact graph window and LSTM history they were
scored with (`ensemble_explain.py`). A one-line summary prints under the alert:

``` text
[ALERT] stratus-redteam: 133 event(s), max risk 8.43/10 (top: PutRolePolicy)
        why: risk 8.43/10 (alert at 4.20) | LSTM 49% (p=0.82): the action PutRolePolicy (+2.40);
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

It starts the whole system (`cloudsec.py`) and opens the CloudSec app on
http://localhost:8503. The app has two pages, switched from its sidebar:

| Page | What it does |
|---|---|
| **Agent Gate** (`agent_console.py`) | Ask the agent for an AWS change and watch every stage decide on it |
| **Live Dashboard** (`cloudsec_dashboard.py`) | The real-time pipeline's event stream, alerts and their explanations |

Behind the app:
- the Stage 2 watcher (`prompt_hardening_realtime`) hardens each SAFE prompt dropped
  into `prompt_hardening_realtime/incoming/`;
- the real-time pipeline (`pipeline.py`) scores every executed (or dropped-in)
  CloudTrail file in `incoming/` with HGT + LSTM, and prints each event in this terminal.

A request on the Agent Gate page goes through:

1. **Stage 1, screening:** a prompt injection or malicious request is refused, with
   the reason. Nothing is forwarded.
2. **Stage 2, hardening:** a SAFE prompt is written into Stage 2's `incoming/`. The
   watcher returns the **prompt template**: your request, plus a locked security
   section with the missing requirements, any conflicts and open questions.
3. **Stage 3, prediction:** the exact AWS calls and CloudTrail events the request
   would produce.
4. **Stage 4, assessment:** HGT + LSTM score those predicted events against the
   agent's stored history. It is a dry run, so nothing is saved.
5. **Revise:** edit the request in the template and send it back. It goes through
   every stage again. The security section is regenerated every round, so it
   cannot be edited away.
6. **Execute:** once every stage approves, **Execute** writes the plan's CloudTrail
   into `incoming/`. The real-time pipeline scores it, adds it to the agent's
   history, and the dashboard shows it. Execution is simulated, so no AWS API is
   called. It runs exactly the approved plan: any change to the calls, targets,
   parameters or principal is refused.

Ctrl+C stops all of it. Options:
- `--keep-state`: keep the agent's history, scores and alerts from the last run.
  Otherwise each start is fresh.
- `--feed [DATASET]`: also replay a dataset (default `real_dataset_test.csv`) as
  background traffic, as the pipeline-only demo did.

`agent_orchestrator.py` runs the same gate without the UI:
`python agent_orchestrator.py --execute "Create a private S3 bucket named audit-logs-prod with versioning."`.
Each step is logged, without the prompt text, to `output/agent_audit.jsonl`.
Service logs are in `logs/`.

### Dashboard

`cloudsec_dashboard.py` (Streamlit) reads what the pipeline writes and refreshes every
5 s ("Live updates" pauses it):

- **Fast-lane alerts** get their own orange panel at the top, newest first. They are
  rule-based (trail deleted, logging stopped, flow logs deleted, ...) and raised whatever
  the models score.
- **Processed logs**: every scored event in arrival order, new ones appended at the
  bottom, in a scrollable table. Filter it to All / Alerts / Fast-lane. The Flag column
  marks `⚡ FAST-LANE` and `▲ ALERT` rows; Why says whether an explanation exists.
- **Why was this flagged?** Click a row to get the event's explanation:
  - each model's exact share of the risk;
  - the plain-English summary;
  - what moved the LSTM (feature contributions and the earlier events that mattered);
  - what moved the HGT (feature shares and related graph events);
  - the attack chain.

  Explained events are the top `explain_top_events` of each alert plus every fast-lane
  event. For any other flagged event, the page shows its alert's top explanation.
- **Email alerts** (sidebar): the admin enters the address alerts are emailed to.

### Email alerts

Every alert whose risk is at or above the threshold is emailed to the admin, with the
principal, the risk, the plain-English reason and the riskiest events (`notify.py`).

1. Copy `email.env.example` to `email.env` and fill in the Gmail account that sends the
   alerts and its app password (Google Account -> Security -> 2-Step Verification -> App
   passwords; not the normal password). `email.env` is git-ignored.
2. Start `run.cmd` / `./run.sh`, then enter the admin's address in the dashboard's sidebar
   and press **Send test email**.

Detection never depends on it: without `email.env` or an address, alerts are only shown on
the dashboard; a failed send is logged and dropped; emails go out on a background thread.
At most 60 emails an hour are sent (`CLOUDSEC_EMAIL_MAX_PER_HOUR`), since Gmail locks
accounts that send in bulk -- the demo feed alone raises over 100 alerts. Alerts raised only
by a fast-lane rule, below the threshold, stay on the dashboard.

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
- Tests: `python tests/run_tests.py` (390 tests).
- Re-tune on dev (add `--test` for the single held-out test run):
  `python datasets/privilege-escalation/evaluate_pipeline.py`.

## Results

The pipeline exactly as served (LSTM v6.3-ft, alert threshold 0.42), scored on
14 Oct 2026 on 821 held-out real test sessions, 107 of them attacks
(`real_dataset_test.csv`; 4 collectors, including 30 real Stratus
privilege-escalation detonations). A session is flagged if any of its events
alerts. Every threshold and configuration was frozen on `real_dataset_dev.csv`
first.

**Is a session an attack?** Session level, best F1 first:

| | Precision | Recall | F1 [95% CI] | Pipeline minus this, paired bootstrap |
|---|---|---|---|---|
| XGBoost (temporal features) | 0.865 | 0.897 | 0.881 [0.832, 0.923] | -0.013 (p = 0.56) |
| **Real-time pipeline** (HGT + LSTM, `pipeline.py`) | 0.848 | 0.888 | **0.868** [0.816, 0.912] | |
| Random Forest (temporal features) | 0.841 | 0.888 | 0.864 [0.812, 0.909] | +0.004 (p = 0.89) |
| LSTM alone | 0.814 | 0.897 | 0.853 [0.800, 0.900] | +0.014 (p = 0.47) |
| HGT alone | 0.769 | 0.841 | 0.804 [0.741, 0.857] | +0.064 (p = 0.009) |
| Curated IAM rule baseline (11 rules) | 0.904 | 0.701 | 0.789 [0.720, 0.849] | +0.078 (p = 0.027) |
| Logistic regression (bag of actions) | 0.600 | 0.869 | 0.710 [0.644, 0.769] | +0.158 (p < 0.001) |

The pipeline is statistically tied with XGBoost, Random Forest and its own LSTM,
and significantly better than the rule baseline, the HGT alone and logistic
regression. Its session ranking is the strongest of its branches (AUC 0.989,
AP 0.927).

**Which event is the attack?** Event level, average precision over all 31,837
test events (3,452 attacks, base rate 0.108), nothing tuned on test:

| | Event AP |
|---|---|
| LSTM alone | 0.792 |
| **Real-time pipeline** (HGT + LSTM) | **0.654** |
| XGBoost | 0.250 |
| Random Forest | 0.233 |
| HGT alone | 0.127 |

This is where the pipeline is clearly ahead: it points to the attack event
about 2.6x better than XGBoost, which is what an alert's explanation depends
on. The HGT pulls event AP down from the LSTM's 0.792, so the graph branch is
the next thing to improve.

How these numbers came about, and what they do not show:
- **The threshold was stale for two weeks.** The LSTM was swapped to v6.3-ft on
  29 Sep without re-tuning the v5-era threshold (0.5139). As served until
  14 Oct, the pipeline scored F1 0.802 (P 0.938, R 0.701), significantly behind
  XGBoost (-0.079, p = 0.018). 0.42 was then chosen on dev from the LSTM's
  out-of-fold scores (see `tuned_on` in `pipeline_config.json`).
- **This test set is no longer untouched.** It was scored on 28 Sep and twice on
  14 Oct (old and new threshold). Final numbers need a fresh Stratus collection.
- **Earlier README figures** (pipeline 0.872, 28 Sep, commit `5c6d085`) were for
  the previous LSTM (v5), which relied on label-leaking features
  (`temporal-analysis/README_v6_3.md`).
- **Dev**, each model at its own best dev threshold, LSTM out-of-fold: pipeline
  0.909, XGBoost 0.907, Random Forest 0.904, LSTM 0.842, HGT 0.803.
- The threshold-selection, out-of-fold and event-level scripts behind these
  numbers are not in the repository yet; the replay is
  `evaluate_pipeline.py`'s and the baselines are `evaluate_ml_baselines.py`'s.

`pipeline.py` is the only ensemble. The earlier `ensemble.py` /
`ensemble1.py` were removed: their graph side was a hand-written topology
rule, not a trained GNN, and with a leak-clean LSTM they scored 0.769 / 0.722
on test (§6.21; the code is in git history at commit `2fe5977`).

Worth knowing:
- **What the pipeline's numbers do and don't show.** At session level the
  classical ML baselines tie the pipeline, so the paper cannot claim that
  HGT + LSTM detects attack sessions better than standard ML. It can claim
  that the pipeline identifies the attack event far better (test event AP
  0.654 vs XGBoost 0.250). Every further fix must be tuned on dev and checked
  on fresh test data.
- **GraphSAGE alone (batch)** has session AUC 0.986 on test (beats all 200
  size-preserving permutations, p = 0.005), stable across every dataset
  iteration (0.982-0.987), and F1 0.929 / AUC 0.997 on held-out unseen attack
  families. Its operating-point F1 is threshold-sensitive (0.79-0.89), so AUC
  is its primary metric. The pipeline uses HGT, not GraphSAGE.
- Older versions of this README reported 0.907 / 0.903 for those removed
  ensembles. Those numbers came from an LSTM checkpoint that had trained on
  real Invictus data, so they are not valid.
- The rule baseline is a curated list built by reading AWS's public GuardDuty
  finding-type docs -- it was never validated against real GuardDuty output
  (this project's data collection never enabled it), so it's *not* a stand-in
  for the actual commercial product.
- The classical ML baselines (`evaluate_ml_baselines.py`) train on
  `train_temporal_aug.csv`, the synthetic table the previous LSTM (v5) trained
  on, and each gets its configuration chosen on dev, as the proposed system
  did. The served LSTM (v6.3) trains on `cloudtrail_temporal.csv`; retrained
  on that table, XGBoost and Random Forest score lower on dev, so the table
  above reports the stronger baselines. An earlier version trained on
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

