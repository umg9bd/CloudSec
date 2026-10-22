# Stage-1 ablation study

No-fine-tune models only -> these are the **bar a fine-tuned DeBERTa must beat**.
Binary gate ATTACK vs BENIGN; threshold on a grouped dev slice (5% FPR), frozen;
reported separately on cloud_test and each public_test (out-of-distribution).
Data: 4211 rows. F1 shown; OOD-avg = mean of the 3 public sets.

### A. Representation / model  (train = cloud + public)

| config | cloud_test | deepset | jackhhao | lakera | OOD-avg F1 | mean FPR |
|---|---|---|---|---|---|---|
| word TF-IDF + LR | 0.00 | 0.10 | 0.62 | 0.72 | **0.48** | 0.02 |
| char TF-IDF + LR | 0.24 | 0.24 | 0.84 | 0.96 | **0.68** | 0.01 |
| word+char + LR | 0.07 | 0.18 | 0.79 | 0.89 | **0.62** | 0.02 |
| word+char + LinearSVM | 0.00 | 0.24 | 0.80 | 0.81 | **0.62** | 0.03 |
| frozen embeddings + LR (semantic) | 0.35 | 0.26 | 0.79 | 0.97 | **0.67** | 0.08 |

_attack-recall (same configs):_

| config | cloud_test | deepset | jackhhao | lakera |
|---|---|---|---|---|
| word TF-IDF + LR | 0.00 | 0.05 | 0.48 | 0.57 |
| char TF-IDF + LR | 0.13 | 0.13 | 0.75 | 0.92 |
| word+char + LR | 0.04 | 0.10 | 0.70 | 0.81 |
| word+char + LinearSVM | 0.00 | 0.13 | 0.73 | 0.68 |
| frozen embeddings + LR (semantic) | 0.23 | 0.15 | 0.68 | 0.94 |

### B. Training data  (model = word+char + LR)

| config | cloud_test | deepset | jackhhao | lakera | OOD-avg F1 | mean FPR |
|---|---|---|---|---|---|---|
| cloud only | 0.23 | 0.03 | 0.03 | 0.04 | **0.03** | 0.00 |
| cloud + public | 0.07 | 0.18 | 0.79 | 0.89 | **0.62** | 0.02 |
| public only | 0.85 | 0.79 | 0.76 | 1.00 | **0.85** | 0.68 |
| cloud+public, no hard-safe | 0.82 | 0.60 | 0.96 | 0.84 | **0.80** | 0.14 |
| cloud+public, no general-benign | 0.44 | 0.18 | 0.88 | 0.94 | **0.67** | 0.00 |

_attack-recall (same configs):_

| config | cloud_test | deepset | jackhhao | lakera |
|---|---|---|---|---|
| cloud only | 0.13 | 0.02 | 0.01 | 0.02 |
| cloud + public | 0.04 | 0.10 | 0.70 | 0.81 |
| public only | 0.99 | 0.93 | 1.00 | 1.00 |
| cloud+public, no hard-safe | 0.81 | 0.43 | 0.92 | 0.72 |
| cloud+public, no general-benign | 0.28 | 0.10 | 0.78 | 0.89 |

### C. Channel  (model = word+char + LR, data = cloud+public)

| config | cloud_test | deepset | jackhhao | lakera | OOD-avg F1 | mean FPR |
|---|---|---|---|---|---|---|
| all channels | 0.07 | 0.18 | 0.79 | 0.89 | **0.62** | 0.02 |
| user_prompt only | 0.61 | 0.50 | 0.93 | 0.98 | **0.81** | 0.05 |

## What the numbers say (for the fine-tune decision)

- Best **lexical** model: `char TF-IDF + LR` -> OOD-avg F1 **0.68**, cloud_test F1 0.24.
- **Frozen semantic embeddings (no fine-tune)** -> OOD-avg F1 **0.67**, cloud_test F1 0.35. Gain over best lexical: -0.00 OOD-avg F1.
- **Read:** if frozen embeddings already beat lexical by a clear margin, *semantic* signal is what matters -> a **fine-tuned** transformer (which adapts those semantics to this task) is likely worth it. If the gain is small, the bottleneck is DATA, not the model.
- **Data merge is non-negotiable:** cloud-only OOD-avg F1 0.03 vs cloud+public 0.62. A model trained on the cloud slice alone does not generalise at all -- confirms the review's 45-85% FP finding.
- **Hard-safe examples work:** removing them raises mean benign-FPR from 0.02 to 0.14 (F1 looks higher only because it stops being penalised for over-blocking). Keep them.
- **'public only' is a trap:** OOD-avg F1 0.85 but mean FPR 0.68 -- it flags almost everything. The cloud hard-safe + benign are what hold the false-positive rate down.
- **Channel matters a lot:** user_prompt-only OOD-avg F1 0.81 vs all-channels 0.62. Indirect injection (retrieved_content / tool_output -- long docs with a hidden instruction) is a MUCH harder sub-problem; consider a separate path / features for it rather than one classifier over all channels.

## Recommendation
1. **Do the data work first, it's the biggest lever and model-agnostic:** train on cloud+public, KEEP hard-safe and general-benign. This alone moves OOD-avg F1 from ~0.03 to ~0.62.
2. **Semantic vs lexical is a WASH on OOD here** (char-TFIDF ~0.68 ~= frozen embeddings ~0.67). So fine-tuning is NOT an automatic win for public-OOD generalisation -- a cheap char n-gram model matches a frozen transformer there.
3. **Fine-tuning IS justified for the hardest axis:** held-out cloud *templates* (unseen attack phrasings) cap at F1 ~0.24-0.35 for every no-fine-tune model, with the semantic one best. A fine-tuned DeBERTa should beat that -- run it and compare on cloud_test specifically. If it doesn't beat char-TFIDF's 0.68 OOD / 0.24 cloud, the compute isn't worth it.
4. **Handle indirect injection separately** -- it's the sub-problem dragging the all-channel score down; a classifier tuned on retrieved/tool content (or structural features like 'instruction-like text in a data channel') will help more than a bigger model.
5. **Protocol for the fine-tune run:** identical to here -- grouped dev threshold at 5% FPR, frozen, reported on cloud_test AND each public_test. Put its row next to these for the decision.
