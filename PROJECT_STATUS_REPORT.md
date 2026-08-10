# Cloud Privilege-Escalation Detection — Project Status Report

**Branch:** `stratus_dataset`
**Goal:** GNN + sequence-model ensemble for detecting AWS privilege-escalation attacks from CloudTrail logs, with explainability, trained on synthetic data and validated against real, team-collected Stratus Red Team data.

---

## 1. Data Collection (Phase 1)

Built a full pipeline (`datasets/privilege-escalation/stratus_collection/`) to generate a real, labeled attack dataset using [Stratus Red Team](https://stratus-red-team.cloud/), since the project's original real data (`aws_dataset/`, from the public invictus-ir repo) was a single 55-minute capture with only ~3 real attacker identities — not enough for a statistically meaningful held-out test set.

- **`run_detonations.py`** — runs 11 MITRE-mapped AWS attack techniques (persistence, privilege-escalation, credential-access, defense-evasion) via the Stratus CLI: `warmup → detonate → revert → cleanup`, logging every run to a per-collector manifest CSV. Always attempts cleanup even on failure (try/finally), to avoid leaving billable AWS resources behind.
- **`collect_real_logs.py`** — syncs CloudTrail JSON logs from S3, decompresses them, and cross-references each manifest run against the *specific* expected CloudTrail events for that technique (not just "a log file exists") to confirm delivery.
- **`stratus_techniques.py`** — the 11 techniques with verified expected event names, pulled directly from `stratus show <technique>`, not guessed.
- **`combine_manifests.py`** — merges every teammate's `manifest_<collector>.csv` and flags any row with no corresponding JSON data pushed yet.
- **`build_combined_real_dataset.py`** — merges the deduped invictus data with every collector's Stratus sessions into one real dataset with correct session boundaries (see bug fix below).

**Team collection**: 4 collectors (vansh, akshaya, udita, nandan) ran detonations across their own separate AWS free-tier accounts (avoiding any credential sharing), each account producing its own CloudTrail trail and S3 bucket. All raw JSON logs and manifests are committed under `stratus_collection/` and `stratus_own_runs/CloudTrail/`.

### Bug found and fixed: session-boundary collapse
The initial session grouping (by `username`) collapsed dozens of separate detonation runs into one giant session per collector, because Stratus runs all execute as the same configured IAM identity. Fixed by deriving `session_id` from the manifest's per-run time window (with a priority-matching fix for back-to-back runs whose buffered windows overlap), and by gap-segmenting ambient benign background activity (30-minute inactivity timeout, the same convention web analytics platforms use) instead of bucketing it by calendar day. This took real session counts from a broken 45 (18 benign-day-buckets skewing the class balance to 70% attack) to a correct, balanced 397 sessions (167 attack / 230 benign).

---

## 2. Real Dataset & Train/Test Discipline

- **`real_dataset_combined.csv`** — the full combined real dataset (invictus + all 4 collectors), 46,786 rows / 397 sessions.
- **`split_real_dev_test.py`** — one-time stratified split (by collector × attack/benign, seed=42) into:
  - **`real_dataset_dev.csv`** (159 sessions) — for threshold tuning only.
  - **`real_dataset_test.csv`** (238 sessions) — touched once, for final reported numbers.
- **`synthetic_cloudtrail.csv`** — the training set (procedurally generated, see below). **Never mixed with real data.**

`invictus_enriched.csv` is retired from direct use (folded into `real_dataset_combined.csv`) but remains as an active input to the build script.

---

## 3. Rule-Based Baselines (`evaluate_baselines.py`)

Re-ran the original notebook's three rule sets (Minimal SIEM, GuardDuty-style, Post-incident) against the corrected data, with **bootstrap 95% confidence intervals** added (the original real-data baseline was computed on just 18 sessions).

| Method | Precision | Recall | F1 (real data) |
|---|---|---|---|
| Minimal SIEM (3 rules) | 0.881 | 0.353 | 0.504 |
| **GuardDuty-style (11 rules)** | 0.889 | 0.623 | **0.732** [95% CI: 0.672, 0.790] |
| Post-incident (23 rules, unfair upper bound) | 0.916 | 0.910 | 0.913 |

**GuardDuty-style F1=0.732 is the number any model needs to beat.** The gap from synthetic (F1=0.889) to real (F1=0.732) is expected and traced to a real cause: GuardDuty-style's 11 rules cover 5 of the 11 collected techniques (IAM-focused); the 6 it misses are credential-access-flavored (`GetPasswordData`, `GetSecretValue`, `DescribeParameters`/`GetParameters`, etc.) — a rule set tuned for one attack category missing an adjacent one.

---

## 4. Synthetic Data Generator (`generate_synthetic_data.py`)

Extracted out of `explore.ipynb` into a standalone, re-runnable script (the notebook still exists but is now superseded — do not re-run it to regenerate data, it has the old unpatched logic).

Three distribution mismatches found (via direct CSV inspection, then via formal chi-square/KS testing) and fixed, each calibrated against the real combined dataset's actual rates, not guessed:

| Field | Before | After | Real target |
|---|---|---|---|
| `target_resource` null rate | 93.9% | 27.8% | 28.2% |
| `mfa_authenticated` null rate | 0% | 68.2% | 69.1% |
| `principal_type` (AWSService/unknown/Root share) | 0% | ~21% | ~21% |

The third fix added three new generator functions modeling real AWS background noise found directly in the collected data: Resource Explorer's periodic `AssumeRole` calls, CloudTrail's periodic `GetBucketAcl` checks, Secrets Manager's own lifecycle events, and Root-level billing/cost background activity.

**`validate_synthetic_vs_real.py`** — formal two-sample chi-square (categorical) / KS (continuous) tests, using effect size (Cramér's V / KS statistic) rather than raw p-values as the pass/fail criterion, since p-values are near-meaningless at this sample size (46K+ real rows). Current state: **5 of 8 tested fields pass** the 0.10 effect-size threshold. `event_source`/`event_name` remain intentionally unmatched (documented reasoning: forcing them to match would work against planned cross-technique generalization testing). `hour_of_day` is borderline (KS=0.11).

---

## 5. Repository Cleanup

Removed (all confirmed unused by any current script before deletion):
- `botsv3/cloudtrail_dataset.csv` — 100% null request/response params, no labels, flagged since early in the project and never resolved until now.
- `stratus_collection/manifest_combined.csv` — superseded by `build_combined_real_dataset.py` reading manifests directly.
- 8 stray `.DS_Store` files (now gitignored).
- Local `raw_s3/` cache (already gitignored, redundant with organized JSON).

---

## 6. GNN Pipeline Integration — the major push this session

### 6.1 Repository recovery
Discovered the working repo had moved to a nested `C:\CloudSec\CloudSec\` directory with its own intact `.git`, while the outer `C:\CloudSec\` had lost its `.git` entirely (git commands from there were silently resolving to an unrelated, unrelated pre-existing repo at the `C:\` drive root). No data was lost — `C:\CloudSec\CloudSec\` is confirmed to be the correct, complete working copy going forward.

### 6.2 Branch merges
Merged three teammate branches into `stratus_dataset`:
- **`feature/Gnn`** — the actual GNN implementation: `model_gat.py`, `model_graphsage.py`, `train.py`, `data_loader.py`, `explainability.py`, `blast_radius.py`, `infer.py`, `incremental_updater.py`.
- **`fe-final`** — `feature_engine9.py`, the production feature engineering script (supersedes `feature_engine8.py`), plus its outputs.
- **`feature/graph_construction`** — `graph_construction/neo4j_graph_builder.py` (v3): builds a heterogeneous privilege-propagation graph (`User`/`Role`/`UnresolvedPrincipal`/`Resource`/`Service`/`Policy` nodes; `ASSUMES`/`LIST`/`READ`/`WRITE`/`TAGGING`/`PERMISSIONS_MANAGEMENT`/`UNKNOWN_ACTION` relations) in Neo4j.

4 merge conflicts resolved (`.gitignore`, `.vscode/settings.json`, root `README.md`, `requirements.txt`) — all config/docs, no data or logic conflicts.

### 6.3 Code review findings on the merged GNN code
- **No label leakage** confirmed in the current graph builder — `is_known_attacker_identity` is tracked but explicitly excluded from the model's feature tensor (verified both in code and at runtime: "200 principals flagged... metadata only, NOT a model feature").
- `feature_engine9.py` had the same bug as the earlier-reviewed `feature_engine8.py`: hardcoded output paths regardless of `--input`, meaning evaluating real data would have silently mixed its features into the same file as synthetic training data. **Fixed**: output paths now derive from the input filename (default input keeps its original filenames for backward compatibility); added a `--freeze-vocab` flag so evaluating real data doesn't grow the event-name vocabulary past what a trained model's embedding table supports.
- `data_loader.py` always fit fresh `StandardScaler`/`LabelEncoder` instances on whatever graph it loaded — meaning naively evaluating a trained model against a real graph would have silently re-normalized features on the real data's own statistics instead of the training statistics. **Fixed**: `PrivilegePropagationGraphLoader` now accepts optional `fit_artifacts` and applies `.transform()` only (never re-fits) when provided, with graceful unseen-category fallback to `<UNK>`/first-known-class for both node and edge categorical features.

### 6.4 Environment & infrastructure
Installed the full ML stack into `myenv` (torch, torch_geometric, neo4j driver, scikit-learn, etc.), started Docker Desktop and the project's `neo4j-local` container, resolved several cross-module import path issues (`graph_construction/` and repo root both needed to be on `PYTHONPATH`), and fixed recurring Windows console encoding crashes (`PYTHONIOENCODING=utf-8`) that were killing otherwise-successful runs on their final cosmetic print statement.

### 6.5 Training results

**First smoke test (15 epochs, original 7-triple graph schema):**

| | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| GraphSAGE | 0.000 | 0.000 | 0.000 | 0.740 |
| GAT | 1.000 | 0.194 | 0.326 | 0.984 |

**After fixing the `policy_sentry` dependency gap** (see 6.6) and retraining for 50 epochs on the resulting 14-triple schema — dramatic improvement on synthetic-only evaluation:

| | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| GraphSAGE | 1.000 | 0.931 | 0.964 | 0.997 |
| GAT | 1.000 | 0.944 | 0.971 | 0.990 |

### 6.6 `policy_sentry` fix
Found that the graph builder's relation-type classifier (`resolve_relation_type`) was falling back to a 13-action partial table because the optional `policy_sentry` package wasn't installed — meaning the vast majority of real AWS actions (and most synthetic ones) were being dumped into a generic `UNKNOWN_ACTION` relation type instead of their correct AWS-documented category (`LIST`/`READ`/`WRITE`/`TAGGING`/`PERMISSIONS_MANAGEMENT`). Installing it (`pip install policy_sentry`) dropped `UNKNOWN_ACTION` from 85.5% to 2.6% of synthetic edges and from 83.5% to 3.1% of real edges.

### 6.7 Real-data evaluation infrastructure (`evaluate_on_real.py`, new)
Built to properly evaluate a trained checkpoint against real data:
1. `infer.py --wrap-checkpoint` bakes the training-fitted scalers/encoders into the checkpoint (must run *before* the training graph is wiped from Neo4j).
2. The real test graph is built fresh in Neo4j from `real_dataset_test_structural.csv`.
3. `evaluate_on_real.py` loads the wrapped checkpoint, reuses its fitted scalers (via the `data_loader.py` fix above) to load the real graph, and — critically — **filters evaluation down to only the edge-type triples the model was actually trained on**, since GraphSAGE's per-relation-type weight matrices only exist for triples seen during training. Reports exactly how many real edges get excluded this way, rather than silently crashing or silently scoring garbage.

### 6.8 The honest finding: real evaluation result

| Run | Precision | Recall | F1 | AUC | Real-graph edges excluded (out-of-schema) |
|---|---|---|---|---|---|
| GraphSAGE, pre-`policy_sentry` (7-triple schema) | 0.000 | 0.000 | 0.000 | 0.681 | 35.3% |
| GraphSAGE, post-`policy_sentry` (14-triple schema) | 0.037 | 0.010 | 0.015 | 0.602 | 35.6% |

**`policy_sentry` fixed the classification granularity correctly, but did not close the real gap.** The real graph's distinct triple count doubled right alongside synthetic's (24→48 vs. 7→14) — previously-lumped `UNKNOWN_ACTION` edges just spread across more *precisely labeled but still numerous* node-type combinations (`Resource→Resource`, `Role→Role`, `User→User`, `User→Policy`, `User→Service`) that the synthetic generator's session structure fundamentally never produces, regardless of how individual actions are classified.

**Root cause, now well-understood**: synthetic sessions are structurally simple — one principal acting on targets. There's no generator mechanism for a resource to act on another resource, or a role on another role — relationships that arise naturally in real CloudTrail data from resource-based policies and cross-references. This is a synthetic **session-topology** gap, not a feature-calibration or classification-accuracy problem, and is not fixable by another round of statistical patching.

---

## 7. Current Repository State

**Uncommitted, ready for review:**
- Modified: `data_loader.py` (fit_artifacts reuse fix), `feature_engine9.py` (path derivation + `--freeze-vocab`)
- New: `evaluate_on_real.py`, `checkpoints/` (trained model weights), real-test structural/temporal CSVs and their state files

**Not yet pushed to origin** — the branch merges and all fixes above are local only, pending review.

---

## 8. Next Steps, in priority order

1. **Redesign synthetic session topology** to produce the relationship diversity real data has — not more of the same event types, but new *kinds* of graph structure (resource-to-resource references, role-to-role chains) that don't exist in the current one-principal-per-session generator design. This is the actual blocker; nothing downstream is trustworthy until real-graph schema coverage substantially improves.
2. **Tune the classification threshold on `real_dataset_dev.csv`** (still never done) — independent of #1, can happen in parallel. The AUC values seen throughout (0.60–0.98 depending on run) suggest the default 0.5 threshold is not well-matched to this class imbalance.
3. **Re-run the full train → wrap → real-eval cycle** once #1 lands, to get the actual comparable number against the GuardDuty-style F1=0.732 baseline.
4. **Evaluate GAT against real data** — never done; it substantially outperformed GraphSAGE on synthetic-only evaluation both before and after the `policy_sentry` fix.
5. **Build the sequence/temporal branch and the ensemble fusion layer** — still entirely unstarted; everything so far is the graph branch alone.
6. **Add the explainability evaluation** (GNNExplainer/attention + fidelity metrics against known attack events) — `explainability.py` exists from the merged branch but hasn't been exercised yet.
7. **Commit and push** the current uncommitted fixes (`data_loader.py`, `feature_engine9.py`, `evaluate_on_real.py`) once reviewed, so the team isn't working from a stale merge.
8. **Housekeeping**: `explore.ipynb` still needs a decision (deprecate-and-keep vs. delete); `hour_of_day` synthetic/real mismatch is minor and low-priority.
