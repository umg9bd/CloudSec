# CloudSec: A Pre-Execution Security Framework for AI Agents in AWS

AI agents are given AWS credentials to carry out tasks. A single manipulated
prompt can make an agent escalate privileges, harvest credentials, disable
logging or move data out of the account, using its own legitimate credentials.
CloudTrail monitoring and rule services such as GuardDuty only see this after
the API calls have run.

CloudSec checks an agent's request **before** anything reaches AWS. It screens
the prompt, adds missing security controls, predicts the CloudTrail events the
request would produce, scores those events for privilege-escalation risk, and
runs the plan only if every stage approves it.

Capstone project (UE23CS441A), Dept. of Computer Science and Engineering,
PES University, Aug–Nov 2026. Team 49: Akshaya N., Nandan D., Udita Negi,
Vanshitha Soma. Guide: Dr. Gokul Kannan Sadasivam.

## How it works

| Stage | What it does | Code |
|---|---|---|
| 1. Prompt screening | Unicode normalisation, 10 YAML rules (OWASP LLM / MITRE ATLAS), a fine-tuned multilingual DistilBERT, and a fail-closed combiner. Returns SAFE, REVIEW_REQUIRED or MALICIOUS; only SAFE goes on. | `prompt_injection_gate/` |
| 2. Security hardening | Finds the AWS services in the request and appends missing controls from a signed catalogue (OWASP, Microsoft, Red Hat, AWS). Conflicts, missing information and unsupported services block the request. | `prompt_hardening_realtime/` |
| 3. Pre-flight simulation | Extracts the intent (deterministic fast path, with an LLM fallback for intent only), plans the AWS calls and emits one predicted CloudTrail event per call from a versioned action catalogue. | `log-generation/` |
| 4. Risk scoring | `feature_engine9` builds 44 temporal features and a principal–resource graph. A 3-seed Heterogeneous Graph Transformer and a BiLSTM–Transformer score each event; risk = 0.8·p_graph + 0.2·p_sequence, alert at ≥ 0.1849. Defence-evasion calls (e.g. `StopLogging`) raise a fast-lane alert whatever the score. | `feature_engine9.py`, `graph_construction/`, `temporal-analysis/`, `pipeline.py` |
| 5. Execution gate | Approves only when every stage passes and risk is below the threshold. The plan is fingerprinted (SHA-256) so exactly the approved plan runs. Alerts carry an explanation (gradient attributions) and appear on the dashboard. | `agent_orchestrator.py`, `ensemble_explain.py`, `cloudsec_dashboard.py` |

Every stage fails closed: an error, an ambiguity or a missing artifact gives a
non-permissive outcome, never an approval.

## Results

Risk detector, scored once with the served configuration on **821 held-out real
sessions** (107 attacks) collected with Stratus Red Team across four AWS
accounts. The models were trained only on synthetic CloudTrail; the ensemble
weight and threshold were chosen on the 546-session dev split.

| | Session-level result | vs CloudSec (paired bootstrap) |
|---|---|---|
| **CloudSec (3-seed HGT + LSTM)** | **F1 0.872** [0.821, 0.916], **ROC-AUC 0.991** | |
| Curated 11-rule IAM baseline | F1 0.789 | CloudSec better, p = 0.019 |
| Logistic regression (bag of actions) | lower | CloudSec better, p < 0.001 |
| Random Forest (same features) | within 0.005 F1 | tie, p = 0.82 |
| XGBoost (same features) | CloudSec +0.017 F1 | not significant |

What these numbers do and don't show:
- At session level CloudSec **ties** Random Forest and XGBoost. Its advantage is
  finding the attacking event: in the earlier single-checkpoint configuration,
  event-level average precision was 0.654 against 0.250 for XGBoost. The LSTM
  alone reached 0.792 there, so the graph branch is the next part to improve.
- The test split was used by earlier configurations during development. Final
  claims need a fresh Stratus Red Team collection.

Prompt-injection classifier (DistilBERT v3) on 465 held-out cloud-agent prompts:
ROC-AUC 0.89, F1 0.76, 5.8% of benign prompts flagged as malicious, about 32 ms
per prompt on CPU. Held-out public test splits: ROC-AUC 0.97 (deepset) and 0.99
(jackhhao). The gate is cautious: only 41% of benign cloud prompts pass as SAFE,
and most of the rest go to review.

Full evaluation history: [`docs/PROJECT_STATUS_REPORT.md`](docs/PROJECT_STATUS_REPORT.md).
Detector details (alerts, dashboard, email alerts):
[`docs/DETECTOR_README.md`](docs/DETECTOR_README.md).

## Run it

With Docker Desktop running:

```powershell
.\run.cmd          # Windows
./run.sh           # macOS / Linux / Git Bash
```

This starts the Stage 2 watcher, the real-time pipeline and the CloudSec app on
http://localhost:8503, which has two pages:

- **Agent Gate**: ask the agent for an AWS change and watch each stage decide.
- **Live Dashboard**: scored events, alerts and their explanations.

Options: `--keep-state` keeps the previous run's history; `--feed [DATASET]`
replays a dataset as background traffic. Ctrl+C stops everything. Service logs
are in `logs/`.

Without the UI:

```bash
python agent_orchestrator.py --execute "Create a private S3 bucket named audit-logs-prod with versioning."
```

Execution from the Agent Gate is simulated: the approved plan's CloudTrail is
written into `incoming/` and no AWS API is called. `stage5_execute.py` can run an
approved plan in a sandbox account; it is a dry run by default and only executes
an allow-list of reversible S3 actions.

The Stage 1 classifier weights (~540 MB) are a GitHub Release asset, not in git.
`cloudsec.py` fetches them on first start; see
[`prompt_injection_gate/README.md`](prompt_injection_gate/README.md).

Setup without Docker: `pip install -r requirements.txt`. The batch tools run
natively; anything that loads a PyTorch checkpoint needs a machine where PyTorch
loads, or the Docker image.

## Repository

| Path | Contents |
|---|---|
| `prompt_injection_gate/` | Stage 1: normalisation, rules catalogue, classifier, combiner, tests |
| `prompt_hardening_realtime/` | Stage 2: task analysis, control catalogue, hardener, watcher |
| `log-generation/` | Stage 3: intent extraction, planner, action catalogue, event enumerator |
| `feature_engine9.py` | Raw CloudTrail → temporal rows (LSTM) and structural rows (graph), fast-lane alerts |
| `graph_construction/` | HGT / GraphSAGE / GAT models, training, offline (Neo4j-free) graph builder, blast radius |
| `temporal-analysis/` | BiLSTM–Transformer sequence model, training and explanations |
| `pipeline.py`, `pipeline_config.json` | Real-time detector and its frozen, dev-tuned settings |
| `agent_orchestrator.py`, `agent_console.py`, `stage5_execute.py` | Stage 5: orchestration, Agent Gate page, sandbox execution |
| `cloudsec_app.py`, `cloudsec_dashboard.py`, `ensemble_explain.py`, `notify.py` | App, dashboard, alert explanations, email alerts |
| `datasets/` | Synthetic data generator, real-data builders, baselines (`evaluate_ml_baselines.py`), prompt dataset |
| `leakage_guard.py` | Checks that no test session appears in any training file |
| `tests/` | Pipeline and cross-component tests |
| `docs/` | Status report, demo guide, design diagrams |

## Data and privacy

- Real CloudTrail was collected only from the team's own sandbox AWS accounts
  by detonating Stratus Red Team techniques. No production account was touched.
- AWS access-key IDs are redacted in the datasets. Prompts are stored only as
  SHA-256 hashes, and no credentials are kept.
- The raw logs still contain the sandbox account IDs and the source IP
  addresses of the machines used. Remove or mask them before making this
  repository public.

## Credits

Stratus Red Team (Datadog), Invictus IR cloud incident-response data,
`policy_sentry` (Salesforce), and the deepset/prompt-injections,
jackhhao/jailbreak-classification and Lakera Gandalf datasets on Hugging Face.
Models build on DistilBERT, the Heterogeneous Graph Transformer and PyTorch
Geometric.
