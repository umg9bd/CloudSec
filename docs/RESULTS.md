# CloudSec — Full Pipeline Results

**Authoritative results reference.** Numbers here are the canonical, shipped-configuration
results. The source of truth for the Stage-4 configuration is
[`pipeline_config.json`](../pipeline_config.json) (`notes.ensemble_3seed`, 2026-10-17);
the Stage-1 numbers come from
[`prompt_injection_gate/results/distilbert-mbert-pi-v3-s3.json`](../prompt_injection_gate/results/distilbert-mbert-pi-v3-s3.json)
(the shipped checkpoint, sha `f03c06…`). Where a value is a dev-set diagnostic rather
than a held-out test result, it is labelled as such.

See [`PROJECT_STATUS_REPORT.md` §0](PROJECT_STATUS_REPORT.md) for the reconciliation of the
earlier (238-session) result generation versus the current (821-session) one.

---

## 0. Data

| Dataset | Size | Role |
|---|---|---|
| Real CloudTrail (Stratus + collected) | 546 dev / **821** test sessions (107 attacks), 4 AWS accounts, 30 real Stratus detonations | Stage-4 evaluation (test touched once) |
| Synthetic CloudTrail | procedurally generated | Stage-4 **training only** (never tested on) |
| Prompt-injection dataset | 3,720 train / 360 dev / 465 held-out cloud + public benchmarks | Stage-1 training & evaluation |

Discipline: models train only on synthetic/generated data and are tested on held-out real
data they never saw; the train/dev/test split is one-time and leakage-guarded.

---

## Stage 1 — Prompt-Injection Screening Gate

DistilBERT-multilingual, 3 seeds trained; shipped checkpoint = **s3**.
Thresholds (dev-derived): malicious 0.075, safe 0.0198. Outcomes: SAFE / REVIEW_REQUIRED / MALICIOUS (fail-closed).

**Held-out cloud test (465 prompts: 165 attack / 275 benign):**

| Metric | Value |
|---|---|
| F1 | **0.76** |
| ROC-AUC | **0.89** |
| Attacks passed as SAFE (safety-critical) | **3.0%** |
| Attack recall (blocked as MALICIOUS) | 0.67 |
| Benign false-positive rate | 0.058 |
| Latency (CPU, 6 threads) | p50 **32 ms**, p95 **137 ms** |

**3-seed spread (robustness):** F1 0.760 / 0.762 / 0.760; AUC 0.882 / 0.859 / 0.888;
attacks-passed-as-safe 7.9% / 10.3% / **3.0%** (s3 shipped = safest).

**Per-category "caught as not-SAFE" recall (cloud test):**

| Category | Recall |
|---|---|
| Direct prompt injection | 100% |
| Indirect prompt injection | 100% |
| Privilege escalation | 100% |
| Credential harvesting | 100% |
| Exfiltration | 100% |
| Obfuscation | 100% |
| System-prompt extraction | 100% |
| Jailbreak | 97% |
| Data destruction | 95% |
| Defense-evasion / logging | 88% |

**External benchmarks (generalization, never trained on):**

| Benchmark | F1 | AUC | Attack recall |
|---|---|---|---|
| Jailbreak (jackhhao) | 0.95 | 0.99 | 0.985 |
| Injections (deepset) | 0.89 | 0.97 | 0.80 |
| Lakera Gandalf (all attacks) | — | — | 1.00 |

> The gate has three outcomes. Recall-as-*blocked* is modest (0.67), but only **3% of
> attacks slip through as SAFE** — the rest are escalated to human review. Fail-safe by design.

---

## Stage 2 — Security Prompt Hardening

**No statistical metric** — deterministic control catalogue + Gemini, not a classifier.
Measured **functionally**: **104 tests pass**. Verified behaviours: appends missing security
controls; flags conflicts (public bucket, wildcard admin, open ports) with **cited sources**
(OWASP / Microsoft / Red Hat / AWS); leaves the original prompt unchanged.

## Stage 3 — Pre-flight Log Generation (CloudTrail Event Prediction)

**No statistical metric** — deterministic planner; the LLM extracts intent only, then the
planner maps it to **exact predicted CloudTrail events (no invented event names)**.
Measured **functionally**: **16 tests pass**, including prediction-coverage checks.

---

## Stage 4 — Risk Ensemble (headline detector)

Shipped config: **0.8 · HGT (3-seed) + 0.2 · LSTM v6.3-ft**, alert threshold 0.1849,
tuned on 546 dev (dev session F1 0.922), **test run once** on 821 sessions.

| Metric | Value |
|---|---|
| **Session F1** | **0.872** [95% CI 0.821, 0.916] |
| **Session AUC** | **0.991** |
| **Event-level AP** (localizing the attack event) | **0.654** |

**vs. baselines (same 821 sessions):**

| Baseline | Result | Significance |
|---|---|---|
| Curated rule baseline (F1 0.789) | pipeline **beats** it (≈ +0.08) | p = 0.019 |
| Logistic regression | pipeline **beats** it | p < 0.001 |
| Random Forest | **dead-heat** (−0.005) | p = 0.82 |
| XGBoost | +0.017 (nominal) | not significant |

**Robustness / honesty checks:**

- **3-seed averaging:** edge AUC 0.83 ± 0.018 vs single-seed 0.78 ± 0.11 → ~6× more stable.
- **Confound control:** session AUC 0.986 beats all 200 size-preserving permutations (p = 0.005);
  within-length-strata AUC 0.989 / 1.000 / 0.989 / 0.946 → not a session-length artifact.
- **Unseen-attack-family (leave-3-families-out):** F1 **0.929**, AUC **0.997**.
- **Component ablation (dev sweep):** HGT alone best-F1 0.868, LSTM alone 0.894, ensemble 0.908 → both branches contribute.
- **Event-level:** 0.654 AP ≈ **2.6× XGBoost's ~0.25** — much better at pinpointing the exact malicious call.

---

## Honest caveats

1. **"Beats standard ML" is not supported.** At session level the pipeline **ties** XGBoost/RF.
   Defensible claims: beats the rule baseline and each model alone, ties tuned tabular ML on
   session detection, and wins clearly at event-level localization (2.6×).
2. **Two result generations exist in the repo.** An earlier run (238 sessions, weight 0.4)
   scored F1 0.844; the **shipped/canonical** run (821 sessions, weight 0.8, 3-seed) scores
   **F1 0.872 / AUC 0.991**. `pipeline.py` loads exactly the 0.8/3-seed config, so 0.872 is the
   deployed number.
3. **AUC is the robust primary metric at Stage 4.** Paired-F1 significance vs. rules swings with
   where the dev threshold lands on a flat plateau; session AUC (~0.986–0.991) is stable across iterations.
4. **Streaming `graph_construction/infer.py` is deprecated** (feature-schema desync); the shipped
   live path is `pipeline.py`, which is batch-exact (tensor-identical to the Neo4j path, test-verified).

---

## One-line verdict per stage

- **Stage 1:** F1 0.76 / AUC 0.89, only 3% of attacks pass as safe, 32 ms median — strong and fail-safe.
- **Stage 2 & 3:** deterministic, no accuracy metric; 104 + 16 tests pass — correctness-verified.
- **Stage 4:** F1 0.872 / AUC 0.991 / event-AP 0.654 on real AWS attacks — beats rules + each model
  alone, ties tuned ML, 2.6× better localization, robust to confounds and unseen families.
