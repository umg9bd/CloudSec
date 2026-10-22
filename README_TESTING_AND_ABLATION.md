# Testing Coverage & Ablation Study (`realtime-pipeline`)

Worktree: `c:\Users\nanda\OneDrive\Desktop\Capstone-2027\CloudSec-rt-test` (detached from `origin/realtime-pipeline` @ `96d0667`).

**Rule followed:** no existing application source code was modified. All additions are new test files, new ablation scripts, and opt-in flags on `tests/run_tests.py`. Genuine bugs found by the new tests are pinned with `@unittest.expectedFailure` (`KNOWN DEFECT:` in the docstring) rather than hidden or worked around.

---

## 1. What Was Done

### A. Test coverage added (211 new tests across 10 files)

| Category | New file | Tests | Pass | Skip | Known defects (`expectedFailure`) |
|---|---|---:|---:|---:|---:|
| P0 Unit — blast radius | `tests/test_blast_radius.py` | 34 | 31 | 0 | 3 |
| P0 Unit — GNN scorer | `tests/test_gnn_scorer.py` | 28 | 28 | 0 | 0 |
| P0 Unit — graph builder | `tests/test_build_graph.py` | 16 | 14 | 0 | 2 |
| P0 Unit — streaming inference features | `tests/test_infer_features.py` | 25 | 16 | 0 | 9 |
| P0 Unit — Streamlit dashboard | `tests/test_dashboard.py` | 22 | 20 | 0 | 2 |
| P1 Data validation | `tests/test_data_validation.py` | 30 | 26 | 1 | 3 |
| P1 Data leakage | `tests/test_leakage_extra.py` | 17 | 12 | 2 | 3 |
| P1 Reliability | `tests/test_reliability_e2e.py` | 19 | 12 | 0 | 7 |
| P1 Explanation fidelity | `tests/test_explanation_fidelity.py` | 12 | 11 | 0 | 1 |
| P2 Performance benchmarks | `tests/performance/test_benchmarks.py` | 8 | 8 | 0 | 0 |
| **Total new (`realtime-pipeline`)** | | **211** | **178** | **3** | **30** |

Existing suites on this branch:
- **Fast tier (`python tests/run_tests.py`):** 390/390 pass.
- **Fast + new coverage (`python tests/run_tests.py --coverage`):** 593 tests run in ~94 s — **OK (skipped=3, expected failures=30)**.
- **Slow equivalence suite (`test_incremental_updater.py`):** 13 tests, **9 pass, 4 fail** (was 3 in the status report; details in Section 3).

### B. Controlled ablation study (Experiments A–G)

New scripts under `datasets/privilege-escalation/ablation/`:
- `ablation_groups.py` — defines the 5 graph feature/relation groups (E) and 4 temporal feature groups (F), plus a context manager that masks a graph group inside `OfflineGraphLoader.load()` at both training and inference time without editing `data_loader.py` or `offline_graph.py`.
- `train_hgt_ablated.py` — retrains 3 HGT seeds (`1, 2, 3`) per group using `graph_construction/train.py` with the same hyperparameters as the live checkpoints (`hidden=128, layers=2, heads=4, dropout=0.3, focal loss`). Includes a `none` control (same retrain, nothing removed).
- `train_lstm_ablated.py` — retrains LSTM v6.3 (`train_lstm_transformer_v6_3.py --drop ...`) and runs head-only L2-SP fine-tuning (`finetune_lstm_v6_3.py --l2sp 1`) per temporal group, plus a `control` run with nothing extra dropped.
- `ablation_study.py` — replays dev (546 sessions) and test (821 sessions, 107 attack) through `pipeline.Pipeline` in 250-event chunks (`evaluate_pipeline.replay`), selects weights/thresholds on dev only, replays test once with frozen configs, and computes 95% bootstrap CIs and paired bootstrap p-values (`evaluate_ml_baselines.f1_ci`, `paired_bootstrap`).

Outputs saved to `results/ablation/`:
- `ablation.md` and `ablation.json` — full tables, CIs, paired comparisons, and dev selections.
- `dev_selection_frozen.json` — dev parameters frozen before any test replay ran.
- `replays/*_{dev,test}.csv` — cached per-event `p_graph` and `p_sequence` from each pipeline replay.
- `hgt_<group>/best_HGT_seed{1,2,3}_wrapped.pt` and `lstm_<group>_ft/temporal_lstm_transformer.pt` — all retrained checkpoints.

---

## 2. How to Run

From `c:\Users\nanda\OneDrive\Desktop\Capstone-2027\CloudSec-rt-test`:

```powershell
# Fast existing suites only (390 tests)
python tests/run_tests.py

# Fast suites + all new audit coverage suites (593 tests, ~95 s)
python tests/run_tests.py --coverage

# Performance benchmarks (~4.5 min; checks against tests/performance/thresholds.json)
python tests/run_tests.py --perf

# Slow batch-vs-streaming equivalence suite (must run from datasets/privilege-escalation due to relative CSV path)
cd datasets/privilege-escalation
$env:PYTHONPATH="$PWD\..\..;$PWD\..\..\graph_construction;$PWD\..\..\temporal-analysis;$PWD\..\..\tests"
python -m unittest -v test_incremental_updater
cd ..\..

# Ablation study (uses cached replays in results/ablation/replays/; <1 min once cached)
python datasets/privilege-escalation/ablation/ablation_study.py --dev-only
python datasets/privilege-escalation/ablation/ablation_study.py
```

---

## 3. Test Outcomes & Defects Found

### Existing slow suite (`test_incremental_updater.py`) — 4 failures out of 13
1. `test_same_node_and_edge_counts`: streaming builds **7,910 nodes vs 7,736** in batch.
2. `test_distance_to_sensitive_resource_converges_regardless_of_order`: reverse distance propagation does not converge to the batch distances.
3. `test_hop_count_mismatches_are_causally_explainable_not_arbitrary`: `synthetic_cloudtrail.csv:1` (`AWSServiceRoleForResourceExplorer`) mismatches with no later `AssumeRole` grant.
4. `test_unique_targets_identical_for_every_node` *(new 4th failure)*: `('UnresolvedPrincipal', 'unknown_principal')` has `unique_targets` **671 in streaming vs 665 in batch**.
- Also note: `test_incremental_updater.py` hardcodes `./cloudtrail_structural.csv`, so `python tests/run_tests.py --all` from the repo root errors with `FileNotFoundError` unless `cwd` is `datasets/privilege-escalation/`.

### Confirmed P0 blocker in `graph_construction/infer.py` (9 `expectedFailure` in `test_infer_features.py`)
`SubgraphLoader` in `infer.py` was not updated when `data_loader.py` and the HGT checkpoints moved to the rank-normalised schema (commit `e8144eb`):
- **Edge features:** `infer.py` builds **8 columns** (`hop_count, privilege_gain, privilege_gain_defined, abnormal_path_frequency, log1p(action_global_frequency), is_priv_esc, is_read_only` + 1 ordinal `edge_type`) and passes 7 of them to `edge_scaler`, which was fitted on **5 columns**. The trained HGT expects **83 columns** (5 scaled + 2 rank-normalised + 76 one-hot `edge_type`).
- **`action_global_frequency`:** `infer.py` computes `log1p(count)` instead of within-graph percentile rank, so it is not window-invariant.
- **Principal nodes:** `infer.py` passes raw count columns (`out_degree`, etc.) instead of percentile ranks in `(0, 1]`.
- **Resource/Policy nodes:** `infer.py` passes all 4 numeric columns to a `StandardScaler` fitted on 2 columns (`ValueError: X has 4 features, but StandardScaler is expecting 2`).

### Other defects pinned by the new suites (21 `expectedFailure`)
- **Blast radius (`test_blast_radius.py`, 3):**
  1. `administrator_reachable` only inspects the first edge between two nodes; a benign `GetRole` followed by `AttachRolePolicy` on the same `(User, Role)` misses the escalation.
  2. `BlastRadiusEngine` indexes target sensitivity once in `__init__`; targets added incrementally later are scored as sensitivity 0 even after `cache.invalidate()`.
  3. `BlastRadiusConfig.__post_init__` checks that tier weights sum to 1.0 but allows negative weights.
- **Graph builder (`test_build_graph.py`, 2):**
  4. `neo4j_graph_builder.build_graph` writes the `_GraphProvenance` stamp before loading nodes/edges; a mid-load failure leaves a partial graph that passes `check_graph_provenance`.
  5. An empty CSV with valid headers raises a misleading `KeyError: "missing ['log_id']"` instead of a zero-row error.
- **Dashboard (`test_dashboard.py`, 2):**
  6. `render_explanation` falls back to `"No single factor stands out; both models scored it moderately high"` even on low-score fast-lane events (e.g. HGT 0.05, LSTM 0.40, risk 2.25/10).
  7. `cloudsec_dashboard.py` calls `st.dataframe(..., width="stretch")`, which crashes on Streamlit 1.42 (`requirements-ui.txt` allows `>=1.32.0`; Docker pins 1.64.0).
- **Data validation (`test_data_validation.py`, 3):**
  8. Duplicate CloudTrail records (same `eventID`) are not deduplicated; they are scored twice and inflate velocity features.
  9. Non-0/1 labels such as `"1.0"` or `"yes"` are silently coerced to benign `0` when fitting risk priors.
  10. A missing `event_name` column produces `edge_type=None` instead of the intended `'unknown_action'` fallback.
- **Data leakage (`test_leakage_extra.py`, 3 + 2 disclosure skips):**
  11. The committed `adaptive_risk_priors.json` was fit on all 11,753 synthetic rows, including val/test rows of `splits/campaign_family_seed42.csv` (affects within-synthetic validation only, not real dev/test).
  12. Within a 250-event chunk, `p_graph` for an early event changes by up to 0.28 depending on future events in the same chunk (the 24 h graph window is built per chunk, not causally per event). `p_sequence` (LSTM) is strictly causal.
  13. Four attacker principals (`stratus-red-team-ec2-get-password-data-role`, `stratus-red-team-eks-ANA-role`, `stratus-red-team-rds-cp-role`, `nexus-user-1`) appear in both `real_dataset_dev.csv` and `real_dataset_test.csv` (split is session-disjoint, not principal-disjoint).
  - *Disclosures (skipped with reason):* `real_dataset_test.csv` was accessed during earlier debugging rounds (status report §6.9–6.15); `lstm_transformer_v6_3_ft` heads were fine-tuned on `real_dataset_dev.csv`.
- **Reliability (`test_reliability_e2e.py`, 7):**
  14–15. Neo4j outage mid-`build_graph` leaves a provenance-stamped partial graph and leaks an unclosed driver.
  16. `infer.InferenceEngine.process_event` does not catch Neo4j errors during `load_neighbourhood`.
  17. Pointing `lstm_checkpoint` at an HGT `.pt` file raises a bare `KeyError('vocab_size')` naming neither the file nor the mismatch.
  18. A crash after writing `alerts/` but before `state_dir` saves causes duplicate alerts when restarted (18 alerted events instead of 9).
  19. Re-delivering a file with the same name appends duplicate rows to `output/risk_scores.csv`.
  20. Multiple `watch_folder` workers on one shared directory race on Windows (20/100 files scored twice, 30/100 valid files moved to `failed/`).
- **Explanation fidelity (`test_explanation_fidelity.py`, 1):**
  21. LSTM leave-one-event-out attribution misses causal events whose effect flows throughfeaturised state (e.g. an earlier privilege-escalation write that flips `pe_write_recent` on the final event).
  - *Passing fidelity metrics:* on 52 dev attack windows, LSTM top-1 event hit rate is **0.44** (random baseline 0.25) and top-3 is **0.58** (random 0.54). Feature deletion drops the logit by **0.76 vs 0.11** for random on the LSTM (top feature wins 95% of windows) and **0.52 vs 0.32** on the HGT. No explanation across TP/FP/FN/low-confidence cases invented an action absent from the session.

### Performance benchmark baseline (`tests/performance/baseline_results.json`)
Measured on CPU (3-seed HGT ensemble + LSTM v6.3-ft, 50-event files):
- **End-to-end latency per 50-event file:** p50 **2.5–3.0 s**, p95 **3.1–3.7 s** (~51–60 ms/event).
- **Throughput:** **17–20 events/s** (1 worker); **19 / 24 / 26 events/s** at 1 / 2 / 4 threads (~1.35x scaling due to the GIL).
- **Component breakdown:** 3-seed HGT scoring **2.2–3.2 s** (dominates latency); graph build (100–1,000 edges) **0.65–0.94 s**; single HGT forward **0.07–0.13 s**.
- **Memory & startup:** RSS ~800 MB (peak ~900 MB), Python heap peak 53 MB; checkpoint load 0.21 s/HGT seed, 0.015 s LSTM, 0.6 s pipeline init in-process.

---

## 4. Ablation Study Outcomes (821 Test Sessions, 107 Attack)

All configurations were selected on `real_dataset_dev.csv` (546 sessions, 71 attack) and evaluated once on `real_dataset_test.csv`. Full table in `results/ablation/ablation.md`.

### Core experiments (A–D, G)

| Exp | Configuration | Dev F1 | Test P | Test R | Test F1 [95% CI] | Test AUC | Test AP | dF1 vs A [95% CI], p |
|---|---|---:|---:|---:|---|---:|---:|---|
| **A** | **Full HGT + LSTM ensemble** (`w=0.8, thr=0.1849`) | 0.922 | 0.856 | 0.888 | **0.872** [0.821, 0.916] | **0.991** | 0.920 | reference |
| **B** | HGT only (3-seed, `thr=0.0760`) | 0.918 | 0.818 | 0.925 | **0.868** [0.819, 0.912] | 0.987 | 0.911 | -0.003 [-0.031, +0.024], p=0.803 |
| **C** | LSTM only (v6.3-ft, `thr=0.1629`) | 0.853 | 0.814 | 0.897 | **0.853** [0.800, 0.900] | 0.978 | 0.903 | -0.018 [-0.069, +0.031], p=0.470 |
| **D-avg** | Unweighted average (`w=0.5, thr=0.1783`) | 0.895 | 0.841 | 0.888 | **0.864** [0.812, 0.910] | 0.991 | **0.924** | -0.008 [-0.040, +0.023], p=0.652 |
| **D-or** | OR rule (B alerts OR C alerts) | — | 0.782 | 0.972 | **0.867** [0.817, 0.910] | — | — | -0.005 [-0.045, +0.035], p=0.811 |
| **G-RF** | Random Forest (tuned on dev) | — | 0.856 | 0.888 | **0.872** [0.822, 0.916] | — | — | +0.000 [-0.036, +0.036], p=1.000 |
| **G-XGB** | XGBoost (tuned on dev) | — | 0.869 | 0.869 | **0.869** [0.817, 0.914] | — | — | -0.002 [-0.048, +0.042], p=0.922 |

### E — Graph feature / relation ablations (3-seed HGT retrained per group)

| Exp | Removed from HGT | Test F1 [95% CI] | dF1 vs A (p) | dF1 vs E-control (p) | dF1 vs E-control @ fixed `w=0.8` (p) |
|---|---|---|---|---|---|
| **E-control** | Nothing (fresh 3-seed retrain) | 0.869 [0.818, 0.914] | -0.003 (0.893) | reference | reference (0.868) |
| **E-action_identity** | 76-dim `edge_type` one-hot | **0.830** [0.773, 0.881] | **-0.041 (0.026)** | **-0.038 (0.019)** | **-0.037 (0.031)** |
| **E-privilege** | `privilege_gain`, `privilege_gain_defined`, `is_priv_esc` | 0.848 [0.795, 0.895] | -0.023 (0.121) | -0.020 (0.113) | **-0.028 (0.046)** |
| **E-frequency** | `abnormal_path_frequency_rank`, `action_global_frequency_rank` | 0.852 [0.798, 0.899] | -0.020 (0.185) | -0.017 (0.308) | -0.005 (0.705) |
| **E-node_structure** | Node degree / sensitivity / distance numeric cols | 0.856 [0.804, 0.902] | -0.016 (0.261) | -0.013 (0.404) | -0.022 (0.117) |
| **E-rel_assumes** | All `ASSUMES` relation edges (falls back to LSTM) | 0.872 [0.821, 0.916] | +0.000 (1.000) | +0.003 (0.884) | +0.004 (0.818) |

### F — Temporal feature ablations (LSTM v6.3 + L2-SP fine-tune retrained per group)

Because `finetune_lstm_v6_3.py` fits its heads on `real_dataset_dev.csv`, re-tuning `w` on dev over-weights the retrained LSTM (`w` shifts to 0.0–0.5). Both re-tuned `w` and fixed `w=0.8` (`@wA`) are reported:

| Exp | Dropped from LSTM | Test F1 (re-tuned `w`) | dF1 vs F-control (p) | Test F1 (`@w=0.8`) | dF1 vs F-control`@w=0.8` (p) |
|---|---|---|---|---|---|
| **F-control** | Nothing extra (base 3 leaky cols only) | 0.830 [0.772, 0.881] | reference | 0.842 [0.787, 0.889] | reference |
| **F-timing** | 8 velocity / duration / time-of-day / `log_secs_since_pe` cols | 0.808 [0.751, 0.860] | -0.022 (0.409) | **0.812** [0.754, 0.863] | **-0.030 (0.040)** |
| **F-permissions** | 9 IAM policy / wildcard / coverage / delta cols | 0.855 [0.802, 0.901] | +0.024 (0.306) | **0.820** [0.763, 0.870] | **-0.022 (0.037)** |
| **F-risk_priors** | `action_risk_prior`, `principal_type_prior_risk` | 0.824 [0.768, 0.875] | -0.006 (0.809) | 0.827 [0.771, 0.876] | -0.015 (0.122) |
| **F-causal_lineage** | `causal_depth`, `lineage_enabling_steps`, `principal_handoff`, `pe_write_recent` | 0.835 [0.780, 0.884] | +0.005 (0.861) | 0.835 [0.780, 0.884] | -0.007 (0.671) |

### Key takeaways
1. **A reproduces the live benchmark exactly:** dev F1 `0.922` (`w=0.8, thr=0.1849`) and test F1 `0.872 [0.821, 0.916]`, AUC `0.991`.
2. **Single branches and baselines are statistically tied with A on session F1:** HGT alone (`0.868`), LSTM alone (`0.853`), unweighted average (`0.864`), OR rule (`0.867`), Random Forest (`0.872`), and XGBoost (`0.869`) are all within paired-bootstrap noise (`p >= 0.47`), because session max-aggregation masks event-level differences. The ensemble's gain over single branches is in ranking quality (AUC `0.991`, AP `0.920–0.924` vs `0.903–0.911`).
3. **Graph features that matter (E):** removing `action_identity` (the 76-action one-hot on edges) causes the largest, statistically significant drop (`-0.038` F1 vs E-control, `p=0.019`). Removing `privilege` features drops F1 by `-0.028` (`p=0.046`) when the ensemble weight is held fixed at `w=0.8`. Dropping `ASSUMES` edges has zero impact because the pipeline falls back to the LSTM on unscored triples.
4. **Temporal features that matter (F):** holding the ensemble weight fixed at `w=0.8` (avoiding in-sample dev weight drift), dropping `timing` (`-0.030`, `p=0.040`) and `permissions` (`-0.022`, `p=0.037`) significantly degrades F1 against F-control`@w=0.8`, and all four temporal removals drop significantly vs the live model A (`-0.037` to `-0.060`, `p <= 0.008`).
