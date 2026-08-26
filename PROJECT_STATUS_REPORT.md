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

> ### ⚠️ CORRECTED — the baseline must be read per split, not pooled
>
> An earlier version of this report quoted the **combined (dev+test, 397-session)** baseline of F1=0.732 as "the number to beat," and then compared it against a model scored on the **test split only (238 sessions)**. Those are different populations and the comparison was invalid. The rule set is measurably harder to beat on test alone. All three splits, recomputed:

| Method | Split | P | R | F1 | 95% CI |
|---|---|---|---|---|---|
| Minimal SIEM (3 rules) | combined (397) | 0.881 | 0.353 | 0.504 | [0.421, 0.581] |
| **GuardDuty-style (11 rules)** | combined (397) | 0.889 | 0.623 | 0.732 | [0.669, 0.787] |
| Post-incident (23 rules, unfair upper bound) | combined (397) | 0.916 | 0.910 | 0.913 | [0.879, 0.943] |
| Minimal SIEM (3 rules) | **test (238)** | 0.864 | 0.380 | 0.528 | [0.427, 0.622] |
| **GuardDuty-style (11 rules)** | **test (238)** | 0.878 | 0.650 | **0.747** | **[0.667, 0.811]** |
| Post-incident (23 rules, unfair upper bound) | **test (238)** | 0.912 | 0.930 | 0.921 | [0.879, 0.957] |
| GuardDuty-style (11 rules) | dev (159) | 0.907 | 0.582 | 0.709 | [0.602, 0.800] |

**GuardDuty-style F1=0.747 on the test split is the number any model reported on that split needs to beat.** Use 0.732 only when describing the baseline over the full real dataset, never as the comparator for a test-split model score.

Two independently-bootstrapped confidence intervals are also **not** a significance test. Because both systems score the same sessions, the difference must be resampled jointly — `evaluate_session_level.py` now computes the rule baseline on whatever sessions it just scored and reports a paired bootstrap on the difference, so this class of mismatch cannot recur.

The gap from synthetic (F1=0.889) to real (F1=0.732 combined / 0.747 test) is expected and traced to a real cause: GuardDuty-style's 11 rules cover 5 of the 11 collected techniques (IAM-focused); the 6 it misses are credential-access-flavored (`GetPasswordData`, `GetSecretValue`, `DescribeParameters`/`GetParameters`, etc.) — a rule set tuned for one attack category missing an adjacent one.

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

**Fix**: `node_key_for_target()` (`privilege_features.py`) now accepts optional `known_role_names`/`known_user_names` sets, built from the principal side of the same batch, and reconciles a bare-name target against them before falling back to `Resource`. Applied in both the batch graph builder and the streaming `incremental_updater.py` (there, using an incrementally-grown, causally-correct set of names seen so far in the stream).

> ### ⚠️ CORRECTED — the equivalence tests do NOT all pass
>
> An earlier version of this section claimed `test_incremental_updater.py`'s batch/incremental equivalence tests "all still pass after the change." **They do not.** Re-run three times, identical result each time: `Ran 13 tests — FAILED (failures=3)`. All three failures are in `TestBatchIncrementalEquivalence`:
>
> | Test | Failure |
> |---|---|
> | `test_same_node_and_edge_counts` | `7520 != 7480` — streaming builds 40 more nodes than batch on the same 9,860 rows |
> | `test_distance_to_sensitive_resource_converges_regardless_of_order` | returns a non-empty mismatch list where `[]` is asserted — the propagation-correction algorithm does not converge independently of insertion order |
> | `test_hop_count_mismatches_are_causally_explainable_not_arbitrary` | `log_id=synthetic_cloudtrail.csv:159` mismatched with no later `AssumeRole` grant to explain it — the test's own message calls this "an unexplained bug, not the expected pattern" |
>
> Not a failure, despite appearances: the `abnormal_path_frequency` drift printout (9,859/9,860 edges differ, max |diff| = 3.22) comes from `test_abnormal_path_frequency_is_NOT_identical_and_that_is_documented`, which deliberately asserts *bounded* drift and passes.
>
> **Impact on reported results: none.** The batch path (`build_graph.py` → `neo4j_graph_builder.build_graph()` → `data_loader.py`) never imports `incremental_updater.py`, so every number in this report is unaffected. What is affected is the **streaming** pipeline: `hop_count` and `distance_to_sensitive_resource` are both live model inputs, so the streaming path currently feeds the model different feature values than the batch path it was trained against. Combined with the `infer.py` schema desync (§6.16), the real-time path is broken in two independent ways and must not be claimed until fixed.

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

### 6.12 GAT trained and evaluated on real data

Retrained GAT on the fixed synthetic graph: **synthetic held-out test P=0.902 R=0.949 F1=0.925 AUC=0.995** (comparable to GraphSAGE). Real-data edge-level result is worse than GraphSAGE's, not better: **P=0.336 R=0.011 F1=0.021 AUC=0.378** — an AUC *below* 0.5 means GAT's real-data ranking is anti-correlated with the true label, not merely uninformative.

### 6.13 Fixed the edge-level vs. session-level metric mismatch (9.1 item 4)

Confirmed via `evaluate_baselines.py` (`rule_predict`, `build_sessions`) that the GuardDuty-style F1=0.732 is a **session-level** metric (a session is flagged if any of its events trips a rule). `evaluate_on_real.py`'s F1 is **edge-level** — not the same unit, not directly comparable. Built `evaluate_session_level.py`, which aggregates the GNN's per-edge probabilities up to session level (session score = max edge probability in that session, mirroring the rule baseline's "any event triggers" logic exactly), joining edges back to sessions via `log_id`'s embedded raw-row-index (never assumes positional alignment between CSVs).

Also fixed a real bug found while building this: `data_loader.py`'s loader silently skipped creating a feature tensor for any node type with zero rows in the current graph, which crashes the model at inference time whenever an eval graph (e.g. the smaller dev split) happens to have zero nodes of a type the model was trained on (here, `Policy`). Fixed by giving such types a correctly-shaped empty tensor instead of omitting them, scoped to the inference path (`fit_artifacts is not None`) so training-time behavior is unchanged.

**Session-level results** (real test set, 238 sessions, threshold=0.5):

| Model | P | R | F1 |
|---|---|---|---|
| GraphSAGE | 0.794 | 0.500 | **0.613** |
| GAT | 0.806 | 0.290 | 0.426 |
| GuardDuty-style rule baseline | 0.889 | 0.623 | **0.732** |

Substantially better than the edge-level numbers (0.027/0.021) suggested, but **a critical control test shows this still isn't reliable evidence of transferred signal**: a trivial rule using *only* session length ("flag attack if session has ≥24 events, ignoring the model's output entirely") scores **P=0.737 R=0.700 F1=0.718** — beating GraphSAGE's session-level score outright. Real attack sessions in this dataset are structurally much longer than benign ones (median 50.5 events vs. 6.0), almost certainly because Stratus detonations generate many more logged actions per run than typical background noise — an artifact of how the dataset was collected, not necessarily how real-world attacker sessions look. Max-pooling a per-edge score over more edges partly just recovers this length signal, independent of whether the model learned anything real. On the dev set the picture is less clear-cut (GraphSAGE's best-threshold F1=0.848 does beat the dev-set length-only baseline of F1=0.789; GAT's best F1=0.761 does not) — inconsistent enough across splits that "the model adds value over a trivial length heuristic" is not something this project can currently claim with confidence.

### 6.14 Distribution-shift diagnosis — likely root cause found

Compared graph-structural features (out-degree, in-degree, unique-targets/actions, `abnormal_path_frequency`, `hop_count`, `resource_sensitivity`) between synthetic and real edges via two-sample KS tests, split by attack/benign label. Result: **synthetic and real data occupy almost completely disjoint regions of structural-feature space, for both attack and benign edges alike** (KS statistics 0.7–1.0 on most features, where 1.0 = fully disjoint distributions):

| Feature | KS: synthetic-attack vs. real-attack | Synthetic attack median | Real attack median |
|---|---|---|---|
| `src_out_degree` | 1.000 | 10 | 8,441 |
| `src_unique_targets` | 1.000 | 7 | 328 |
| `dst_in_degree` | 0.819 | 2 | 9,204 |
| `abnormal_path_freq` | 0.912 | 3.56 | 1.15 |
| `hop_count` | 0.091 | 1 | 1 |
| `resource_sensitivity` | 0.166 | 1 | 1 |

The real graph has far fewer distinct nodes (1,196) carrying far more edges (30,189) than the synthetic graph (7,480 nodes / 9,860 edges) — real AWS resources and roles get reused constantly across repeated Stratus runs, while the synthetic generator mints a fresh random entity name per session, so synthetic nodes rarely accumulate degree. This means the frozen `StandardScaler` (fit on synthetic's tiny degree values, per the checkpoint-wrapping design in 6.7) is almost certainly mapping real data's degree features to extreme, never-seen z-scores when evaluating real data — plausibly wrecking the model's decision boundary on exactly the structural signals a GNN relies on most. Two features (`hop_count`, `resource_sensitivity`) transfer reasonably well and are not implicated.

This is now the leading, concrete, mechanistic explanation for the generalization gap — not a vague "distribution shift" but a specific, checkable claim: raw degree-based features are on incompatible scales between domains because of a topology difference (entity reuse rate), not an attacker-behavior difference. **Not yet verified as causal** — the natural next check is whether log-transforming or rank-normalizing degree-based features (instead of raw-value z-scoring) closes some of this gap, since that would directly test the hypothesis.

### 6.15 Tested the scale-mismatch hypothesis directly — result: made things worse, reverted

Applied `log1p()` to all six count-like columns implicated in 6.14 (`out_degree`, `unique_targets`, `unique_actions`, `role_transition_count`, `in_degree`, `unique_principals`) plus edge-level `abnormal_path_frequency`, before scaling, in `data_loader.py`. Retrained GraphSAGE from scratch (synthetic test performance held: F1=0.937, matching the untransformed model's 0.934) and re-evaluated on real data:

| | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Before (raw z-scored features) | 0.173 | 0.015 | 0.027 | 0.546 |
| After (log1p-transformed features) | 0.068 | 0.109 | 0.084 | **0.157** |

**AUC got substantially worse, not better** — the scale-mismatch hypothesis, at least as directly tested here, is not supported. Reverted (`git checkout -- checkpoints/best_GraphSAGE.pt checkpoints/best_GraphSAGE_wrapped.pt`; the `data_loader.py` diff was hand-edited to remove only the log1p hunks, keeping the unrelated empty-node-type fix from 6.13); confirmed the revert reproduces the original F1=0.027/AUC=0.546 exactly.

One detail worth carrying forward: **AUC=0.157 is not "no signal" — it's a strong, systematic inversion** (flipping every prediction would score AUC=0.843). That's different in kind from the untransformed model's near-0.5 result, and suggests the model can pick up *something* real and reproducible in real data's log-compressed structural features, just with the wrong sign relative to what "attack" meant on synthetic data. This is not itself a usable fix (inverting predictions to chase a held-out test set's AUC would be curve-fitting, not a real solution) but it's a more specific, and more tractable, phenomenon to investigate than undifferentiated noise — e.g., checking whether a single dominant feature's attack/benign relationship flips sign between domains, rather than assuming the whole feature space is uninformative.

**Updated conclusion at the time**: the degree-scale mismatch documented in 6.14 was real (verified via direct KS statistics), but a straightforward log-transform wasn't the fix. See 6.16 — a different transform of the same underlying insight did work.

### 6.16 Fix found: rank-normalization instead of scaling — the gap substantially closes

log1p compresses scale but a "high" value still isn't *comparable in meaning* between a 7,480-node synthetic graph and a 1,196-node real graph. The corrected approach: express each node's degree/count features as a **percentile rank within its own graph's population**, computed fresh on every graph (train, dev, test, or a future live one) rather than fit once and reused — this is legitimate under the "never fit scalers on eval data" rule because there is no cross-graph statistic being learned here, only a deterministic, label-free structural quantity local to whichever graph is being processed. Applied to the same six node-level count columns as 6.15 (`out_degree`, `unique_targets`, `unique_actions`, `role_transition_count`, `in_degree`, `unique_principals`) plus edge-level `abnormal_path_frequency`, all rank-normalized rather than z-scored, concatenated alongside the (still z-scored) remaining features (`data_loader.py`, `_rank_normalize`).

**Methodological correction made at the same time**: `real_dataset_test.csv` had already been evaluated against repeatedly across 6.9–6.15 (each time to check a structural bug fix, not to tune a threshold or feature choice — but repeated exposure nonetheless). From this point on, **all iteration uses `real_dataset_dev.csv` exclusively**; `real_dataset_test.csv` is touched exactly once more, at the end, to confirm.

**Dev-set result** (`real_dataset_dev.csv`, 159 sessions):
- Edge-level: P=0.260, R=0.055, F1=0.091, AUC=**0.648** (up from 0.546; contrast with 6.15's log1p attempt, which drove AUC to 0.157)
- Session-level threshold sweep: broad plateau F1=0.86–0.90 across thresholds 0.25–0.45; best F1=**0.901** (P=0.922, R=0.881) at threshold=0.35 — clearing the dev-set's own length-only baseline (F1=0.789)
- Confound check: Spearman correlation between session length and model score = 0.552 (real, but far from ≈1.0) — median score 0.045 for benign sessions vs. 0.926 for attack, a much sharper split than length alone produces. Not simply a disguised length-counter.

**Final test-set check** (`real_dataset_test.csv`, 238 sessions, threshold=0.35 fixed from dev, not re-swept):

| | Precision | Recall | F1 |
|---|---|---|---|
| GraphSAGE, session-level (this fix) | 0.859 | 0.790 | **0.823** |
| GuardDuty-style rule baseline | 0.889 | 0.623 | 0.732 [95% CI: 0.672, 0.790] |

**F1=0.823 clears the rule baseline outright — above even the baseline's own CI upper bound.** Robustness checks on this specific number: threshold stability (F1 stays in 0.814–0.823 across thresholds 0.25–0.45 on test, not a fragile spike) and a bootstrap 95% CI over session resamples: **[0.766, 0.875]** — the CI's lower bound is still above the baseline's point estimate.

**Honest caveats, to state plainly rather than bury**:
- Edge-level performance on test is still weak (AUC=0.537, F1=0.062) — the model is not accurately classifying most individual edges. The session-level win comes from correctly scoring at least one edge per attack session highly while staying quiet on benign sessions, not from precise per-edge classification. Describe this mechanism precisely in the paper, not as "accurate edge-level detection."
- `real_dataset_test.csv` was touched multiple times earlier in the diagnostic process (6.9–6.15), for structural bug verification, before the dev-only discipline above was adopted. The specific model/threshold behind the F1=0.823 number was selected using dev data only, but the session's overall dev/test hygiene was not perfect from the very start — worth disclosing as a limitation.
- Verified for GraphSAGE only so far. GAT has not yet been retrained/re-evaluated with this fix.
- `infer.py`'s live single-event streaming path builds edge features independently of `data_loader.py` and has **not** been updated to match the new rank-normalized schema — it is currently out of sync and would silently produce wrong results if run as-is. Needs fixing before any live-streaming claim.
- No non-graph baseline yet confirms the *graph structure* itself (versus the rank-normalization feature engineering alone) is what's adding value.

### 6.17 Independent audit, three correctness fixes, and a retrain — current headline numbers

An adversarial audit re-executed the whole pipeline rather than reading this report. It **reproduced F1=0.823 exactly** from the shipped checkpoint (edge AUC 0.5372, 25,984 in-schema edges, threshold-stability band 0.814–0.823 — every figure matched), and confirmed there is no fabricated metric, no train/test leakage (0 shared sessions, 0 shared `principal_arn`, 0 shared `target_resource` between synthetic training data and real test data), and no label leakage (`is_known_attacker_identity` is label-derived but provably excluded — node feature dims 4/4/4/5/4 match the label-free schema exactly). It also found four defects that had to be fixed before the number was defensible.

**Fix A — the baseline was measured on a different population than the model.** `evaluate_baselines.py` computes F1=0.732 on `real_dataset_combined.csv` (397 dev+test sessions); the model is scored on 238 test sessions. On matched sessions the same rule set scores **0.747**. Comparing two independently-bootstrapped CIs is also not a significance test. `evaluate_session_level.py` now computes the rule baseline on whatever sessions it just scored and reports a **paired** bootstrap on the difference, so this cannot recur. See §3.

**Fix B — 70.5% of real edges were fed a fabricated action name.** The `edge_type` encoder was fit on 67 synthetic actions; the real test graph has 645. Every unseen one was mapped to `classes_[0]` — alphabetically `AddUserToGroup`, itself a privilege-escalation action. Sweeping that arbitrary choice across other classes moved test F1 between **0.756 and 0.823**, with the shipped default landing on the maximum. Fixed by fitting a reserved `<UNK>` class at training time.

**Fix C — `edge_type` was fed to the network as a raw unscaled ordinal 0–66**, inventing an ordering over nominal AWS actions and dominating the z-scored features by one to two orders of magnitude. Occlusion showed it was actively harmful (collapsing it to a constant *raised* session AUC 0.912→0.934). Now one-hot encoded; `edge_feat_dim` 8 → 75, no model-side change needed.

**Fix D — a live scaler re-fit on evaluation data.** `_node_features` only registered a fitted `StandardScaler` when a node type had >1 node at training. The synthetic graph had exactly one `Policy` node, so no `Policy` scaler existed, and at inference the code silently fitted one on the *real* graph's Policy nodes. Now scalers are fit unconditionally, and a missing scaler for a type the model consumes raises instead of falling through.

**Retrained with all four fixes, following the protocol strictly** — synthetic held-out first, threshold re-selected on **dev only**, test touched exactly once:

| Stage | Result |
|---|---|
| Synthetic held-out test | P=0.973 R=0.923 **F1=0.947** AUC=0.9997 (was 0.934) |
| Dev threshold sweep (159 sessions) | best **threshold=0.65**, F1=0.887 (P=0.894, R=0.881) |
| **Real test, session-level (238 sessions, thr=0.65 fixed from dev)** | **P=0.874 R=0.830 F1=0.851** [95% CI 0.794, 0.900] |
| GuardDuty-style baseline, *same 238 sessions* | P=0.878 R=0.650 F1=0.747 [0.667, 0.811] |
| **Paired bootstrap on the difference** | **+0.104 F1, 95% CI [+0.040, +0.171], p=0.0008 — significant** |

Threshold stability on test across 0.50–0.70: F1 = 0.845 / 0.857 / 0.882 / 0.851 / 0.802 — a plateau, not a spike.

**Two confound controls, both passed** (the second answers §6.13's own objection more convincingly than the Spearman check did):
- **Permutation test** — permuting per-edge probabilities across the graph while preserving every session's size exactly destroys the edge→session association but leaves the "max over more edges" length effect intact. Permuted session AUC: mean 0.733, max 0.782 over 200 draws. Observed: **0.921 — beats all 200 (p<0.005).**
- **Within-length-strata AUC** — 0.998 / 0.970 / 0.872 / 0.759 across length bands, where session length alone is uninformative within strata (0.30–0.55). Length correlation also *fell* (Spearman 0.42, down from 0.51). The win is not a length artifact.

**Non-graph baseline (closes the §9.4 gap).** Logistic regression over a bag-of-actions vector, same protocol (train on synthetic, tune threshold on dev, test once): **F1=0.647, AUC=0.658** — well below the graph pipeline's 0.851/0.921. Adding session length made it worse (0.473); length alone scores 0.254. The graph pipeline earns its place.

**Ablations.** Zeroing `is_privilege_escalation_technique` (the hand-coded Rhino action list) changed the pre-fix result by *nothing* — the model is not covertly replaying the rule baseline. Zeroing all rank-normalized node features dropped session AUC 0.912→0.808 and pushed edge AUC below chance, supporting §6.16's claim that rank-normalization is the operative mechanism.

**⚠️ Architectural finding — principal nodes receive no messages at all.** Surfaced by a PyG `UserWarning` while writing the model tests, then confirmed directly against the trained checkpoint's schema:

| | Node types |
|---|---|
| Appear as edge **source** | `User`, `Role`, `UnresolvedPrincipal` |
| Appear as edge **destination** | `Resource`, `Policy`, `Role` |
| **Never a destination** | **`User`, `UnresolvedPrincipal`** |

The privilege-propagation graph is directed principal→target, so `User` and `UnresolvedPrincipal` are source-only. They therefore receive zero messages during aggregation: their embeddings are a linear projection of their four rank-normalized features and nothing more. For the typical scored edge `(User, READ, Resource)`, only the *destination* side carries any neighbourhood information — `h_src` involves no graph structure whatsoever, and `User` is the source of most labeled attack edges.

This is a partial mechanistic explanation for weak edge-level performance, and it is the kind of thing a reviewer will ask about directly ("what is the GNN actually aggregating?"). It also points at the cheapest high-value experiment left: **add reverse edges** (PyG's `ToUndirected`, or explicit reverse relations in the loader) so principals aggregate from the resources they touch. That is a small change and would, for the first time, give the principal side of every edge an actual graph-derived representation. Untested so far.

**⚠️ Open question — edge-level ranking is now strongly inverted.** Test edge AUC fell from 0.537 to **0.260** (dev: 0.399) even as session-level performance improved. Pooled over all 25,984 in-schema edges, real attack edges score *lower* than real benign edges, yet the per-session maximum separates the classes better than before. This echoes the inversion first seen in §6.15 (AUC=0.157 under log1p) and is not understood. It does not invalidate the session-level result — which survives the permutation and length-strata controls above — but it must be reported, not buried, and it rules out describing this system as an edge-level detector.

---

## 7. Key Finding: A Verified Fix for the Synthetic→Real Generalization Gap

This supersedes the previous version of this section (preserved below in spirit but corrected in conclusion — see 6.16 for the full evidence trail):

- A GraphSAGE model trained purely on procedurally-generated synthetic CloudTrail sessions achieves **near-perfect held-out synthetic performance** (F1=0.934, AUC=0.999 — §6.10). *(An earlier version of this line read "F1=0.926", which appears nowhere else as a synthetic score; 0.926 is the median attack-session score on dev from §6.16, copied here in error.)* Note this synthetic split is **transductive** — `stratified_edge_split` is a random split of edges over one shared graph, with degree features computed across the whole graph including held-out edges. It is not an inductive generalization estimate; the real-data numbers below are.
- Two earlier structural/schema-coverage bug fixes (6.9, 6.10) and one feature-scaling attempt (6.15, log1p) did not close the real-data gap — one made it actively worse.
- **A corrected feature-normalization approach (6.16, rank-normalization instead of z-scoring or log-scaling) did close it**, and four correctness fixes from an independent audit (6.17) then strengthened it. Current numbers, all on the same 238 held-out test sessions, dev-selected threshold, test touched once: **session-level F1=0.851 [CI 0.794, 0.900] vs. the GuardDuty-style rule baseline's F1=0.747 [CI 0.667, 0.811], paired difference +0.104 [+0.040, +0.171], p=0.0008**.
- The mechanism is specific and should be described precisely, not oversold: the model wins at the session level by correctly flagging at least one edge per attack session, not by accurately classifying individual actions. Edge-level AUC on test is **0.260** — not merely uninformative but strongly *inverted* (see 6.17's open question).
- Remaining work before this is a complete result: confirm on GAT, add a non-graph baseline to isolate the graph structure's contribution, and fix `infer.py`'s now-desynced streaming feature builder.

This finding is *itself* a legitimate and durable research contribution if analyzed rigorously (see Section 9) — many well-cited security-ML papers are built around precisely this kind of honest, carefully-diagnosed negative result (e.g., work on concept drift and unrealistic evaluation assumptions in security ML). The next phase of the project should treat closing (or rigorously explaining) this gap as the primary objective, not a side quest.

---

## 8. Current Repository State

**Committed previously this session** (branch `gnn-real-eval-integration`):
- `privilege_features.py`, `graph_construction/neo4j_graph_builder.py`, `incremental_updater.py`, `test_incremental_updater.py` — node-identity canonicalization fix (6.9)
- `datasets/privilege-escalation/generate_synthetic_data.py` — identity-pivot fix (6.10)
- Regenerated `synthetic_cloudtrail.csv`, `cloudtrail_structural.csv`, `cloudtrail_temporal.csv`
- Trained `checkpoints/best_GraphSAGE.pt` / `best_GraphSAGE_wrapped.pt` (pre-6.16; superseded)

**Uncommitted as of this writing** (see Section 9.0 for the commit/push plan):
- `data_loader.py` — empty-node-type inference fix (6.13) **and** the rank-normalization fix (6.16, the one behind F1=0.823)
- New: `evaluate_session_level.py`, `build_graph.py`, `DEMO_GUIDE.md`
- New: `checkpoints/best_GAT.pt`, `checkpoints/best_GAT_wrapped.pt` (pre-6.16 GAT; not yet re-verified with the rank-normalization fix)
- Retrained `checkpoints/best_GraphSAGE.pt` / `best_GraphSAGE_wrapped.pt` (post-6.16, the version behind the F1=0.823 result — this is the one to keep)

**Still not done**: GAT re-verification with the 6.16 fix, sequence/ensemble branch, explainability validation, non-graph baseline, `infer.py`'s streaming-path desync fix — see Section 9.

---

## 9. Path to Publication

The goal is a paper that holds up in a strong venue for 5–10+ years, not just a working prototype. Section 6.16 delivered the core result — a verified, statistically-checked win over the rule baseline. What's below is what's left to turn that into a complete, submission-ready paper.

### 9.0 Immediate next steps (do these first)

1. **Re-verify GAT with the 6.16 rank-normalization fix** — GAT's current numbers (6.12) predate this fix and are not representative of what GAT can actually do; retrain and re-evaluate before drawing any GraphSAGE-vs-GAT conclusion.
2. **Fix `infer.py`'s streaming feature builder** (flagged in 6.16) — it independently constructs edge features and was not updated for the new rank-normalized schema; would silently produce wrong results if run as-is right now.
3. **Commit and push everything** — the checkpoint behind F1=0.823 currently exists only in the local working tree.
4. **Non-graph baseline** (see 9.4) — the single most important remaining check, since it's the one that tells you whether the graph structure itself is earning its place in the paper, or whether the rank-normalized degree features alone would do just as well in a simpler model.

### 9.1 What's now resolved vs. still open

**Resolved, with evidence**: the synthetic→real generalization gap has a verified fix (6.16, hardened by 6.17) — session-level F1=0.851 [CI 0.794, 0.900] vs. the rule baseline's F1=0.747 [CI 0.667, 0.811] on the same sessions, paired difference +0.104 [+0.040, +0.171], p=0.0008. Threshold selected purely from dev data; test touched once. Confirmed not to be a disguised session-length heuristic by a permutation test (observed session AUC 0.921 beats all 200 size-preserving permutations) and within-length-strata AUCs of 0.998/0.970/0.872/0.759 — a stronger control than the Spearman check previously cited.

**Still open**:
- Edge-level accuracy remains weak (AUC=0.537 on test) — the win is a session-level aggregation effect, not precise per-action classification. This needs to be described precisely in the paper, not overstated.
- GAT unconfirmed with this fix (9.0.1).
- ~~Whether the graph structure specifically matters, versus the feature engineering alone, is untested.~~ **Closed (6.17)**: a bag-of-actions logistic regression under the identical protocol scores F1=0.647 / AUC=0.658 vs. the graph pipeline's 0.851 / 0.921.
- `real_dataset_test.csv`'s dev/test hygiene wasn't perfect from the very start of the session (touched repeatedly during earlier bug-verification rounds, 6.9–6.15) — disclose as a limitation; the fix itself was properly dev-validated, but the paper should be upfront about this rather than implying pristine single-touch discipline throughout.

### 9.2 If GAT and the non-graph baseline both come back favorably

Then the paper's core claim is straightforward and strong: a heterogeneous GNN trained on properly-constructed synthetic privilege-escalation graphs, with a domain-appropriate feature normalization, generalizes to real red-team data and beats an established rule-based baseline — with the full diagnostic journey (two real bugs found and fixed, one hypothesis tested and correctly rejected, the actual fix identified and rigorously validated) as supporting methodological contribution.

### 9.3 If the non-graph baseline matches the GNN

Still a publishable, honest result — reframe the contribution around the rank-normalization insight itself (a general lesson for synthetic-to-real transfer in graph-structured security data) rather than claiming the GNN architecture specifically is what mattered. Worth stating either way, since a reviewer will ask this question regardless of which way it goes.

### 9.4 Core rigor needed either way

- **Non-graph baseline under the identical protocol**: train a flat-feature classifier (XGBoost/Random Forest, or an isolation forest for unsupervised anomaly detection) on the exact same synthetic-train/real-test split. This isolates whether graph structure specifically helps or hurts transfer — informative regardless of which way the result goes, and reviewers will ask for it either way.
- **Ablations**: `train.py` already has `--ablation` (feature ablation) — use it. Add edge-type ablation (does dropping `UNKNOWN_ACTION` help or hurt?) and GNN-depth/hop-count ablation.
- **Statistical comparison, not point estimates**: use bootstrap CIs or McNemar's test when comparing GNN vs. rule-baseline F1, accounting for within-session correlation (edges from the same session aren't independent samples).
- **Explainability validation, not just execution**: `explainability.py` exists but hasn't been run. Don't just report "we ran GNNExplainer" — validate explanation *fidelity* against the dataset's own ground-truth `attack_technique` labels (already present in the data): do the top-weighted edges/features for a flagged session actually correspond to the documented MITRE technique for that session? This is what separates a real explainability evaluation from a demo.
- **The ensemble** (GNN + sequence branch): entirely unstarted. Build a session-level sequence model (LSTM/Transformer) over `cloudtrail_temporal.csv`, a documented fusion strategy (late fusion is simplest to justify), and an ablation showing the ensemble's effect vs. either branch alone — whatever that effect turns out to be.
- **Related work**: position against cloud-log anomaly detection (GuardDuty, academic CloudTrail work), provenance/host-graph GNN intrusion detection (the DARPA Transparent Computing lineage — Unicorn, ThreaTrace, Flash, etc.), and synthetic-to-real transfer in security ML specifically. The diagnostic journey (6.9–6.16) is itself worth a related-work nod to the "Dos and Don'ts of Machine Learning in Computer Security"-style literature on unrealistic security-ML evaluation, even though the paper's core claim is now a positive result rather than a pure negative one.
- **Reproducibility**: seeds are already fixed (seed=42) and the pipeline is scriptable end-to-end — consolidate into a documented one-command repro path (see `DEMO_GUIDE.md`), pin `requirements.txt` exactly, and decide what's safe to release publicly (real Stratus data contains real AWS account IDs — scrub before any public dataset release; synthetic data and code are release-safe as-is).
- **Ethics statement**: straightforward here (defensive detection research, red-teaming performed by the team against their own AWS accounts using an established open-source tool, no offensive tooling released) but should be stated explicitly, since security venues expect it.

### 9.5 Realistic venue targets

The verified result (6.16) makes the mid-tier tier genuinely achievable, not just a fallback:
- **Workshops** (strong fit, still durable/citable): ACM CCS workshops (AISec), IEEE S&P workshop on Deep Learning and Security (DLS), USENIX CSET.
- **Mid-tier conferences** (the realistic primary target now that there's a verified positive result plus a rigorous diagnostic narrative): RAID, DIMVA, ACSAC, ESORICS.
- **Top-tier** (stretch goal — would need the full rigor list in 9.4 done, GAT confirmed, and the non-graph baseline result to also support the graph-structure claim): IEEE S&P, USENIX Security, ACM CCS, NDSS.

### 9.6 Housekeeping (low priority, do whenever convenient)

- `explore.ipynb` still needs a decision (deprecate-and-keep vs. delete).
- `hour_of_day` synthetic/real mismatch (KS=0.11, borderline) is minor and low-priority relative to everything above.
