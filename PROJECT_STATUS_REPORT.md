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

### 6.8 [CORRECTED] The real evaluation result, and why the original diagnosis here was wrong

An earlier version of this report claimed the root cause was that "synthetic sessions are structurally simple... there's no generator mechanism for a resource to act on another resource, or a role on another role" and called for redesigning synthetic session topology. **That diagnosis was wrong**, found by direct evidence: a Neo4j query on a claimed `Resource→Resource` edge showed the *same key string* existing as two separate nodes — one correctly labeled `:Role` (from the principal side of the graph) and one mislabeled `:Resource` (from the target side). The real cause was narrower and mechanical, not topological — see 6.9.

### 6.9 Fix 1: node-identity canonicalization bug (`node_key_for_target`)

`neo4j_graph_builder.py`'s `parse_target()` only recognizes fully-qualified ARNs (`_ARN_RE`) when classifying a target as a `Role`. A bare role/user **name** appearing as a `target_resource` value (e.g. lifted from a `roleName` request parameter rather than a full ARN) fell through to a generic `Resource` node — even when that exact name was already correctly known as a `Role` from the principal side of other rows in the same dataset. This silently split one identity into two disconnected nodes, inflating the apparent real-vs-synthetic schema mismatch.

**Fix**: `node_key_for_target()` (`privilege_features.py`) now accepts optional `known_role_names`/`known_user_names` sets, built from the principal side of the same batch, and reconciles a bare-name target against them before falling back to `Resource`. Applied in both the batch graph builder and the streaming `incremental_updater.py` (there, using an incrementally-grown, causally-correct set of names seen so far in the stream). `test_incremental_updater.py`'s batch/incremental equivalence tests (asserting the two pipelines produce identical graphs) all still pass after the change.

**Result**: real-graph excluded-edge percentage dropped from ~35% to **14.3%** (48→35 distinct real triples vs. 14 trained triples). Synthetic graph was byte-for-byte unchanged (same 7,353 nodes / 9,711 edges before and after) — confirming this was a real-data-only artifact, not a synthetic generation issue.

### 6.10 Fix 2: synthetic generator never modeled "assume role, then act as it"

Inspecting the *trained* model's edge-type schema after fix 1 showed `ASSUMES` only ever targets `Resource`, never `Role`, in synthetic training data — meaning the model had never seen a single labeled-attack example of `User/Role --ASSUMES--> Role --READ/WRITE/PERMISSIONS_MANAGEMENT--> Resource`, arguably the canonical AWS privilege-escalation shape. Root cause: `generate_attack_session()` in `generate_synthetic_data.py` hardcoded `principal_arn` to the same static IAM user for every row in an attack chain, even for steps after an `AssumeRole` step. Separately, only 1 of 8 attack chains (`update_assume_role_policy`) even contained an `AssumeRole` step, and it was the chain's last step, so there was nothing downstream to attribute to the assumed role anyway.

**Fix**:
- Added an identity-pivot mechanism to `generate_attack_session()`: once a step's `event_name` is in `{AssumeRole, AssumeRoleWithSAML, AssumeRoleWithWebIdentity}`, all subsequent rows in that chain (including trailing benign noise) adopt the assumed role's identity (`arn:aws:sts::ACCOUNT:assumed-role/<name>/<session>`), reusing the exact name generated for that step's own target, so the graph builder's fix 1 correctly merges it into one canonical `Role` node.
- Extended `update_assume_role_policy` with a follow-up `GetSecretValue` step, and `full_kill_chain` with an inserted `AssumeRole` step between `AttachRolePolicy` and `GetSecretValue` — so the mechanism has real chains to act on.

**Verification**: rebuilt synthetic graph confirmed `Role`-sourced labeled-attack edges now exist (40 `READ`, 20 `WRITE`, 20 `PERMISSIONS_MANAGEMENT`, 40 `ASSUMES→Role`; zero before). Retrained GraphSAGE on the corrected graph: **synthetic held-out test P=0.960 R=0.910 F1=0.934 AUC=0.999** (16 populated triples, up from 14).

### 6.11 Real-data result after both fixes — the actual current blocker

| Stage | Excluded (out-of-schema) | Real P / R / F1 / AUC |
|---|---|---|
| Before this session | ~35% | 0.035 / 0.009 / 0.014 / 0.542 |
| After fix 1 (canonicalization) | 14.3% | 0.035 / 0.009 / 0.014 / 0.542 |
| After fix 2 (generator identity pivot + retrain) | 13.9% | 0.173 / 0.015 / **0.027** / **0.546** |

Both fixes were real, correctly diagnosed, and independently verified (synthetic test F1=0.934 proves fix 2 worked; the exclusion-percentage drop proves fix 1 worked). **Neither meaningfully moved real-data performance.** The one remaining schema gap (`UnresolvedPrincipal→ASSUMES→Role`, 3,703 edges) is mostly AWS-internal noise (RDS/EC2 service-linked-role credential rotation, where the raw log's `source_node` is genuinely null) — only 65 of 4,284 real attack edges fall in it, so closing it further would not move recall much.

**The decisive diagnostic**: raw predicted-probability distributions on real data (n=25,984 in-schema real edges) show real attack edges scoring *no higher* than real benign edges — median 0.0307 (attack) vs. 0.0311 (benign), mean 0.052 vs. 0.071. This is not a threshold/calibration problem (no threshold recovers good recall from an inverted ranking); it is a **genuine synthetic→real generalization failure**. Even for edge types the model was trained on, whatever pattern it learned to call "attack" on synthetic data does not appear the same way in real Stratus data. This is now the central open question for the project — see Section 7.

---

## 7. Key Finding: The Synthetic→Real Generalization Gap

This is the most important result of the project to date, and should be treated as a first-class finding rather than a bug to quietly patch away:

- A GraphSAGE model trained purely on procedurally-generated synthetic CloudTrail sessions achieves **near-perfect held-out synthetic performance** (F1=0.934, AUC=0.999).
- The same model, evaluated on real, red-team-generated CloudTrail data (Stratus Red Team, 4 independent AWS accounts, 11 MITRE-mapped techniques) with rigorous train/test discipline (no real data ever touched during training) and honest schema-coverage accounting, performs **at chance level** (AUC=0.546, F1=0.027) — far below the simple 11-rule GuardDuty-style baseline (F1=0.732).
- Two structural/schema-coverage hypotheses for this gap were tested, fixed, and verified — and **neither explains the gap**. The remaining explanation is a genuine feature/structural distribution shift between how the synthetic generator constructs attack sessions and how real attacker (and real background-noise) behavior actually looks in CloudTrail.

This finding is *itself* a legitimate and durable research contribution if analyzed rigorously (see Section 9) — many well-cited security-ML papers are built around precisely this kind of honest, carefully-diagnosed negative result (e.g., work on concept drift and unrealistic evaluation assumptions in security ML). The next phase of the project should treat closing (or rigorously explaining) this gap as the primary objective, not a side quest.

---

## 8. Current Repository State

**Committed this session** (branch `stratus_dataset`, on top of the prior `gnn-real-eval-integration` push):
- `privilege_features.py`, `graph_construction/neo4j_graph_builder.py`, `incremental_updater.py`, `test_incremental_updater.py` — node-identity canonicalization fix (6.9)
- `datasets/privilege-escalation/generate_synthetic_data.py` — identity-pivot fix (6.10)
- Regenerated `synthetic_cloudtrail.csv`, `cloudtrail_structural.csv`, `cloudtrail_temporal.csv`
- Retrained `checkpoints/best_GraphSAGE.pt` / `best_GraphSAGE_wrapped.pt`

**Still not done**: GAT retrain + real-eval, dev-set threshold sweep, sequence/ensemble branch, explainability validation — see Section 9.

---

## 9. Path to Publication

The goal is a paper that holds up in a strong venue for 5–10+ years, not just a working prototype. That requires closing specific, checkable gaps — listed here in priority order, split into what's needed regardless of how the generalization gap resolves, and the two possible framings depending on how it resolves.

### 9.1 Immediate technical next steps (do these regardless of framing)

1. **Diagnose the generalization gap properly**, before deciding whether to fix or report it:
   - Compare *structural* feature distributions (out-degree, fan-out, hop-count, `abnormal_path_frequency`) for attack-labeled edges specifically, synthetic vs. real — reuse the `validate_synthetic_vs_real.py` chi-square/KS methodology already built, but on graph-derived features, not just raw CloudTrail fields.
   - Visualize learned node/edge embeddings (UMAP or t-SNE) colored by synthetic/real and by label — check whether attack/benign separate *within* either domain alone, and whether the two domains occupy the same embedding region. This distinguishes "the model can't separate attack from benign at all" from "it separates them, but on axes that don't transfer."
   - Check whether the real dataset's attack representation is simply too sparse relative to synthetic's (167 real attack sessions vs. thousands of synthetic ones) to have learned anything comparable — a straightforward statistical-power explanation that would need to be ruled in or out before concluding "distribution shift."
2. **Threshold-tune on `real_dataset_dev.csv`** — still not done. Do it regardless of the diagnosis above; it's cheap and gives an honest best-case number, even though the probability-inversion finding (6.11) means it's unlikely to fix recall on its own.
3. **Evaluate GAT on real data** — never done. It outperformed GraphSAGE on synthetic-only evaluation before; check whether that holds after both fixes, and whether it shows the same real-data collapse.
4. **Fix the unit-of-analysis mismatch**: the GNN is evaluated **edge-level** (25,984 in-schema real edges); the GuardDuty-style rule baseline (F1=0.732) is very likely evaluated **session-level** (397 sessions). These are not the same metric and the direct comparison currently printed by `evaluate_on_real.py` ("Compare against GuardDuty-style rule baseline: F1=0.732") is comparing two different units. Before this number goes in a paper, either (a) aggregate GNN edge predictions up to a session-level decision (e.g., "session flagged if any edge crosses threshold") and re-baseline both methods at the same granularity, or (b) report both units separately with the discrepancy explicitly acknowledged. A reviewer will catch this if it isn't addressed first.

### 9.2 If the gap is fixable: mitigation path

- Make synthetic sessions structurally richer where real data actually differs (informed by 9.1's diagnosis, not guessed) — e.g., more realistic background noise, more varied attacker behavior within a technique, cross-session identity reuse patterns.
- Consider light, carefully-scoped domain adaptation: fine-tuning on a *small slice* of `real_dataset_dev.csv` (never `real_dataset_test.csv`) with feature-alignment or a few-shot fine-tune, clearly disclosed as such. This is legitimate as long as `real_dataset_test.csv` stays untouched until final reporting, exactly as it has been throughout.
- Re-run the full train→wrap→real-eval cycle after any such change and confirm the effect is real (not an artifact of a smaller change like this session's two fixes, which were both real but insufficient).

### 9.3 If the gap persists: reframe as the paper's contribution

A rigorously diagnosed, honestly reported negative result — "a GNN trained on realistic synthetic AWS privilege-escalation data achieves near-perfect synthetic performance but fails to generalize to real red-team data, and here is exactly why, verified two ways" — is a legitimate and citable contribution, particularly at a venue that values methodological rigor in security ML. This would reposition the paper's core claim from "we built a detector" to "we show current synthetic-data evaluation practice for this class of problem is unreliable, and quantify why" — which is arguably a *more* durable 10-year contribution than a marginal detection-accuracy improvement, since it's actionable for anyone else building similar systems.

### 9.4 Core rigor needed either way

- **Non-graph baseline under the identical protocol**: train a flat-feature classifier (XGBoost/Random Forest, or an isolation forest for unsupervised anomaly detection) on the exact same synthetic-train/real-test split. This isolates whether graph structure specifically helps or hurts transfer — informative regardless of which way the result goes, and reviewers will ask for it either way.
- **Ablations**: `train.py` already has `--ablation` (feature ablation) — use it. Add edge-type ablation (does dropping `UNKNOWN_ACTION` help or hurt?) and GNN-depth/hop-count ablation.
- **Statistical comparison, not point estimates**: use bootstrap CIs or McNemar's test when comparing GNN vs. rule-baseline F1, accounting for within-session correlation (edges from the same session aren't independent samples).
- **Explainability validation, not just execution**: `explainability.py` exists but hasn't been run. Don't just report "we ran GNNExplainer" — validate explanation *fidelity* against the dataset's own ground-truth `attack_technique` labels (already present in the data): do the top-weighted edges/features for a flagged session actually correspond to the documented MITRE technique for that session? This is what separates a real explainability evaluation from a demo.
- **The ensemble** (GNN + sequence branch): entirely unstarted. Build a session-level sequence model (LSTM/Transformer) over `cloudtrail_temporal.csv`, a documented fusion strategy (late fusion is simplest to justify), and an ablation showing the ensemble's effect vs. either branch alone — whatever that effect turns out to be.
- **Related work**: position against cloud-log anomaly detection (GuardDuty, academic CloudTrail work), provenance/host-graph GNN intrusion detection (the DARPA Transparent Computing lineage — Unicorn, ThreaTrace, Flash, etc.), and synthetic-to-real transfer in security ML specifically. If the paper ends up centered on the generalization gap (9.3), the "Dos and Don'ts of Machine Learning in Computer Security"-style literature on unrealistic security-ML evaluation is the most directly relevant prior work to engage with.
- **Reproducibility**: seeds are already fixed (seed=42) and the pipeline is scriptable end-to-end — consolidate into a documented one-command repro path, pin `requirements.txt` exactly, and decide what's safe to release publicly (real Stratus data contains real AWS account IDs — scrub before any public dataset release; synthetic data and code are release-safe as-is).
- **Ethics statement**: straightforward here (defensive detection research, red-teaming performed by the team against their own AWS accounts using an established open-source tool, no offensive tooling released) but should be stated explicitly, since security venues expect it.

### 9.5 Realistic venue targets

Given team size and current stage, a workshop or mid-tier systems-security venue is the realistic, durable target — not a stretch for top-tier on the first pass:
- **Workshops** (strong first-publication fit, still durable/citable): ACM CCS workshops (AISec), IEEE S&P workshop on Deep Learning and Security (DLS), USENIX CSET.
- **Mid-tier conferences** (achievable with the rigor items above done): RAID, DIMVA, ACSAC, ESORICS.
- **Top-tier** (stretch goal, needs substantially more novelty/results than current stage): IEEE S&P, USENIX Security, ACM CCS, NDSS.

If the paper ends up centered on the generalization-gap finding (9.3), that framing fits RAID/DIMVA/ACSAC or a security-ML workshop especially well — those venues specifically value rigorous, honest empirical findings about *why* a plausible approach fails, not just wins.

### 9.6 Housekeeping (low priority, do whenever convenient)

- Commit and push the fixes from this session once reviewed.
- `explore.ipynb` still needs a decision (deprecate-and-keep vs. delete).
- `hour_of_day` synthetic/real mismatch (KS=0.11, borderline) is minor and low-priority relative to everything above.
