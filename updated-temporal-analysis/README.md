# updated-temporal-analysis — LSTM + Transformer on the new data

The LSTM + Transformer (temporal / sequence model) retrained on the team's current
`feature_engine9` data, with a real-data fine-tune and explainability. Graph (HGT) code is not
part of this folder.

## Contents

| Path | What |
|---|---|
| `train_lstm_transformer_v6_3.py` | v6.3 trainer (recipes `v6.3` default, `v6.2`, `v5`) |
| `train_lstm_transformer.py` | shared model class and helpers (v5 code base, with the fixes below) |
| `finetune_lstm_v6_3.py` | head-only fine-tune of v6.3 on real dev |
| `lstm_explain.py` | explanations for v5 and v6.3 alerts (`--model v5 \| v6.3`) |
| `test_lstm_explain.py` | 4 unit tests for `lstm_explain.py` |
| `splits/campaign_family_seed42.csv` | train/val/test split by attack campaign family (`campaign_split.py`) |
| `artifacts/lstm_transformer_v6_3/` | **v6.3 model**, metrics, training history, explanations |
| `artifacts/lstm_transformer_v6_3_ft/` | v6.3 + fine-tuned heads, `finetune_report.json` |
| `artifacts/lstm_transformer_v5_live/` | the live pipeline's v5 (`lstm_transformer_clean`), explanations |
| `artifacts/lstm_transformer_v5_newdata/` | v5 recipe retrained on the new data |

The scripts import `feature_engine9`, `campaign_split` and the datasets from the team repo, so run
them from a `realtime-pipeline` checkout with this folder's files in `temporal-analysis/`.

## Results

Real dev = `real_dataset_dev.csv` (546 sessions, 71 attacks). LSTM alone, no graph model.

| Model | Session AUC-PR | Session best F1 | Session AUC | Event AUC-PR |
|---|---|---|---|---|
| Live v5 (`lstm_transformer_clean`) | 0.805 | **0.863** | **0.970** | 0.190 |
| v6.2 recipe on the new data | 0.534 | 0.515 | 0.764 | 0.206 |
| v5 recipe on the new data | 0.825 | 0.816 | 0.954 | 0.328 |
| **v6.3** | **0.852** | 0.829 | 0.961 | 0.314 |
| v6.3 fine-tuned (5-fold out-of-fold) | 0.894 | 0.853 | 0.970 | 0.625 |
| Random forest baseline | 0.769 | 0.795 | 0.957 | 0.433 |

Real test = `real_dataset_test.csv` (821 sessions, 107 attacks), scored **once** by
`finetune_lstm_v6_3.py --test` with thresholds frozen on dev. Nothing was chosen on test.

| Model | Session AUC-PR | Session best F1 | Session AUC | Event AUC-PR |
|---|---|---|---|---|
| Live v5 | **0.895** | **0.869** | **0.978** | 0.165 |
| v6.3 | 0.822 | 0.796 | 0.957 | 0.251 |
| v6.3 fine-tuned | 0.870 | 0.836 | 0.977 | **0.493** |

Takeaway: v5 still ranks sessions best. v6.3-ft is close on sessions and 3x better at pointing to
the actual attack events, and it does it without the label-leaking features (see Explainability).

## What changed and why

### 1. New data (why a retrain was needed)
The team moved the pipeline to `feature_engine9`: 8 new identity / permission features
(`privilege_delta`, `permission_expansion_score`, `target_permission_coverage`,
`actor_permission_coverage`, `principal_handoff`, `new_permission_count_log`,
`causal_depth_normalized`, `lineage_enabling_steps_normalized`), new risk priors, and a new event
vocabulary. Models trained on the old features can't use the new ones.

- Training data: `datasets/privilege-escalation/cloudtrail_temporal.csv` (fe9 default run).
- Real dev is featurised exactly like `pipeline.Pipeline.featurize` (frozen vocab and priors, fresh
  per-principal state), so training and live serving see the same features.
- Split by whole attack campaign families, with an assert that no user is in two splits.

### 2. v6.2 → v6.3 (why v6.2 failed on the new data, and the fix)
v6.2 as-is scored 0.534 session AUC-PR on real dev. Ablations, one change at a time:

| v6.2 variant | Session AUC-PR | Best F1 |
|---|---|---|
| as-is | 0.534 | 0.515 |
| + `<UNK>` training off | 0.610 | 0.566 |
| + campaign relabel | 0.809 | 0.762 |
| **+ campaign relabel + secret positives x4 (= v6.3)** | **0.852** | **0.829** |

- **Campaign relabel:** a secret read / `AssumeRole` / `CreateSecret` within 10 min after a
  privilege-escalation write is labelled an attack, in the **synthetic training data only**. Real dev
  and test labels are never changed. Without it the model never learns the steps after escalation.
- **Secret positives weighted x4** in the sampler (v6.2 used x2): secret theft is rare and was
  under-scored.

Dropping the leaky features is **not** what hurt v6.2. Keeping them made it worse (0.381).

### 3. Fixes carried over (the audit issues, from v6.2)
- **Label-leaking features dropped:** `no_mfa`, `mfa_absent` and `params_length_normalized` separate
  classes on synthetic data but not on real data (`--audit` prints the gap).
- **`<UNK>` training:** the current event's name is hidden 15% of the time, so unseen API names still
  get a sensible score.
- **Epoch picked on real dev**, not on synthetic validation (which saturates near 1.0). Only epochs ≥ 4
  count, because epoch 1 can spike by chance.
- **Exact best-F1 threshold** with a small recall-side margin (the old 0.05–0.95 grid saturated).
- **Timestamp casting:** real-dev timestamps are parsed as mixed formats and cast to
  `datetime64[ns, UTC]` before any `int64` math, so time gaps are right whatever the input unit.
- **Numeric event ids outside the vocabulary range map to `<UNK>` (0)** instead of indexing past
  the embedding.
- **Real test is never read** by the trainer (`real_test_scored: false` in `metrics.json`).

### 4. Fine-tune (`finetune_lstm_v6_3.py`)
Only the 3 scoring heads are refit on real dev labels. The embedding, BiLSTM and Transformer stay
frozen. An L2-SP penalty pulls the heads back to their v6.3 weights, so 71 attack sessions can't drag
them far. The penalty strength (1.0) is picked by 5-fold cross-validation over dev sessions, and the
thresholds come from out-of-fold scores.

### 5. Explainability (`lstm_explain.py`)
For each alert:
- **Top earlier events:** each earlier event in the 10-min window is removed and the event
  re-scored. The effect is in logit units, so it stays readable when scores are near 1.
- **Top features:** Integrated Gradients per feature and for the event name. The contributions sum
  to the score's logit change (checked: error ≤ 0.04).
- **Global:** permutation importance on real dev (drop in session / event AUC-PR).

What it found on real dev:

| Model | Relies on |
|---|---|
| v6.3 | `privileged_action_reach`, `target_permission_coverage`, `action_risk_prior`, event name, event timing |
| live v5 | **`params_length_normalized`**, `no_mfa`, `time_sin`: the features flagged as label-leaking |

## Tried, did not help (LSTM alone)
- **Hiding read-only event names during training** (`--unk-read`): improved HGT + LSTM, but not
  the LSTM alone (0.852 → 0.848 → 0.822 as the rate rises). Off by default.
- **`<UNK>` training off:** 0.610 against 0.852 with it on (both with relabel).
- **Keeping the leaky features:** 0.381.

## Live pipeline
`pipeline_config.json` (team repo) still serves the live v5. On real dev, HGT + LSTM session F1 is
0.882 with v5 and 0.824 with a retrained LSTM. Only its alert threshold was re-tuned on current
dev (0.5139 → 0.5161).

## Known issues
- The committed `real_dataset_dev_temporal.csv` predates the 8 new features, so the batch check in
  `evaluate_pipeline.py` raises `KeyError` until that file is regenerated.
- The same real dev set picks the epoch and the thresholds, so dev numbers are a little optimistic.
  Real test is the honest number.
- The live v5 depends on label-leaking features (see Explainability), so its high scores may not
  hold on new traffic.

## Run
```
python temporal-analysis/train_lstm_transformer_v6_3.py --audit    # feature check: synthetic vs real dev
python temporal-analysis/train_lstm_transformer_v6_3.py            # train v6.3
python temporal-analysis/train_lstm_transformer_v6_3.py --compare  # v6.3 vs live v5 vs RF on dev
python temporal-analysis/finetune_lstm_v6_3.py [--test]            # head fine-tune (+ one test run)
python temporal-analysis/lstm_explain.py --model v6.3 --alerts 5 --global
```
