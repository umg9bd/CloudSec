# CloudSec Temporal Analysis (`P_seq`)

Sequence-model track for **AWS privilege-escalation detection**. Per-user CloudTrail events are scored, then pooled into **one probability per 10-minute window** (stride 2 minutes) for fusion with the graph model (`P_graph`).

Privilege-escalation is a **chain** of individually-plausible API calls (`CreateRole` → `AttachRolePolicy` → `AssumeRole` → `GetSecretValue`). A single event often looks benign; the attack is visible in **order + context**. This track owns the temporal score. The GNN track owns `P_graph`. Streamlit does **not** fuse the two.

**Fusion contract:** `P_seq = max(P_event)` inside each window.

**Working directory for every command below:** `temporal-analysis/` (this folder).

| Item | Path |
|------|------|
| This README | `CloudSec/temporal-analysis/README.md` |
| Notebook (plots + writeup) | [`temporal-lstm-transformer.ipynb`](temporal-lstm-transformer.ipynb) |
| Train / infer | [`train_lstm_transformer.py`](train_lstm_transformer.py) |
| Streamlit tester | [`streamlit_app.py`](streamlit_app.py) |
| Checkpoint | [`artifacts/lstm_transformer/temporal_lstm_transformer.pt`](artifacts/lstm_transformer/temporal_lstm_transformer.pt) |
| Train table | [`data/lstm/train_temporal_aug.csv`](data/lstm/train_temporal_aug.csv) |
| Vocab | [`data/lstm/event_name_vocab.json`](data/lstm/event_name_vocab.json) |

---

## Why LSTM + Transformer (not LSTM-only, not Transformer-only)

CloudTrail is an **ordered time series**. The detector has to answer two different questions:

1. **Local order** — does this look like a PE write chain? (`CreateUser` then `AttachUserPolicy` is different from the reverse, or from isolated recon.)
2. **Long-range campaign** — is this `GetSecretValue` loot **after** a PE write, even when dozens of other events sit in between?

| Approach | Why we tried it | Why it is not the final model |
|----------|-----------------|-------------------------------|
| Rules / flat anomaly | Easy baseline | Ignores sequence; GuardDuty-style rules miss plausible-looking chains |
| UniLSTM, last hidden (v1) | Small sequential baseline | Last-step only; left-pad + `pack_padded_sequence` reads PAD first; no attention to earlier writes |
| Masked BiLSTM + attention, bag of 5 (v2) | Stronger window classifier | Window-level max-label; still truncates busy users at T=32/128; bagging helps variance, not the loot-after-write problem |
| Transformer-only | Self-attention for long range | ~2 real Invictus attackers; a deep encoder overfits identities instead of techniques |
| Mixture-of-Experts | Capacity on mixed sources | Rejected: too much capacity for this data; bagging was enough for v2 |
| **BiLSTM + 1-layer Transformer (v5)** | LSTM encodes local API order; one attention layer lets loot attend to earlier PE writes; tabular skip keeps IAM flags from going through the sequence bottleneck; secrets head treats loot APIs separately from IAM writes | Current model |

v4 (same hybrid, **original CSV labels**) still scored bert-jan `GetSecretValue` at ~0.28 and flagged nearby `AssumeRole` / `CreateSecret` as false positives. Cause: **T=32 drops the PE write off a busy timeline**, so loot looks like a benign read. v5 keeps PE context on the **full user timeline** and adds a campaign label + secrets head.

A pure LSTM cannot look far enough back once history is truncated. A large Transformer memorizes `bert-jan`. The hybrid is ~500KB and is selected on val F1, not on saturating train AP.

---

## Models we built (history)

Tried in order. Only **v5** is production / Streamlit / `prod/`.

| Version | What it was | Data | Sequence | Status |
|---------|-------------|------|----------|--------|
| BOTSv3 notebooks | Row-count windows (`WINDOW_SIZE = 10`), then 10-min majority-vote | Splunk BOTSv3 | — | **Ruled out.** Almost no PE chains, empty `requestParameters`, `user_name` mostly NaN, `is_anomaly` labels. Smoke-test only. Notebooks removed from this branch. |
| v1 uniLSTM | `Embedding → concat feats → LSTM → last hidden → binary head` | Invictus only (`invictus_temporal.csv`) | T=128, **left-pad** | Archive. Do not serve. Last-hidden misses mid-window writes; left-pad fights `pack_padded_sequence`. |
| v2 masked BiLSTM + attention | Bag of 5, attention pool, window label = `max(event label)` | Invictus + fe-final (`train_temporal.csv`) | T=32/128, user-disjoint split | Archive notebook: [`temporal-lstm.ipynb`](temporal-lstm.ipynb), weights `artifacts/temporal_lstm_v2.pt`. Good window ranker; fails bert-jan loot. |
| v3 / v4 LSTM–Transformer | Event-level scores, `P_seq = max(P_event)` | + synthetic `syn:` chains | T=32, **right-pad** | v4 bert-jan (orig labels): precision 0.38, recall 0.43, F1 0.40, AUC-PR 0.52. Missed all `GetSecretValue`. |
| **v5 LSTM–Transformer** | v4 + full-timeline PE context + campaign relabel + secrets head | `train_temporal_aug.csv` | T=32, right-pad | **Current.** Schema `lstm_transformer_v5.0`. |

Locked choices that survived every version:

- Sessions are **per username**, never global, never mixed `inv:` / `fe:` / `syn:` in one window
- Windows are **10 minutes, stride 2 minutes** (not row-count)
- Primary ranking metric is **AUC-PR**, not accuracy
- Honest test user is **`inv:bert-jan`** (the long human-like chain). Stratus is 29 `GetPasswordData` events and is too easy. `fe:` / `syn:` templates are not the report metric.

---

## Vocabulary

File: [`data/lstm/event_name_vocab.json`](data/lstm/event_name_vocab.json)

| Item | Value |
|------|--------|
| Size | **281** (index `0` + APIs `1–280`) |
| Index `0` | `<UNK>` / PAD (same id). OOV `event_name` at serve time → `0` |
| fe-final IDs | **1–67 frozen** from the fe-final vocab |
| Invictus | Original 260 names remapped onto the **union**; 213 Invictus-only names occupy 68–280 |
| Token column | `event_name_idx` (integer). Training does not rebuild ids from strings each run |
| Serve | `event_name` string **or** `event_name_idx`; strings look up this JSON |

Rebuild the union with [`prepare_lstm_dataset.py`](prepare_lstm_dataset.py). Do not train a second vocab for Streamlit — the checkpoint also stores the same map.

Username prefixes (so windows never mix sources):

- `inv:` Invictus IR (July 2023 incident)
- `fe:` fe-final CloudTrail templates
- `syn:` synthetic PE chains from [`augment_attack_chains.py`](augment_attack_chains.py) — **train/val only**, never in the bert-jan test split

---

## v5 architecture

`LSTMTransformerModel` in [`train_lstm_transformer.py`](train_lstm_transformer.py).

```text
event_name_idx (B, T=32)     35 tabular feats + 2 PE-context feats
        │                              │
        ▼                              │
 Embedding dim=16, padding_idx=0       │
        │                              │
        └──────── concat ──────────────┘
                    │
                    ▼
         BiLSTM hidden=48 (d_model=96)
                    │
                    ▼
         LayerNorm → TransformerEncoder
         1 layer, 4 heads, GELU, prenorm
                    │
        ┌───────────┼──────────────┐
        ▼           ▼              ▼
   seq head    tabular skip   secrets head
   (h + emb)   (last feats)   (secret APIs only)
        └───────────┴──────────────┘
                    ▼
              P_event ∈ (0, 1)
                    ▼
         P_seq = max(P_event) in 10-min window
```

| Piece | Detail |
|-------|--------|
| Pad | **Right-pad.** `pack_padded_sequence` reads the first `length` steps, so PAD must be on the right (left-pad was a v1 bug). |
| PE context | `pe_write_recent`, `log_secs_since_pe` — past-only, computed on the **full user timeline**, not the truncated T=32 slice |
| Campaign labels | `GetSecretValue` / `Decrypt` / `AssumeRole` / `CreateSecret` = attack if a PE write happened in the last 10 minutes. Orig Invictus still labels 178 in-campaign `Decrypt` as 0. |
| Secrets head | Extra logit only on `GetSecretValue`, `Decrypt`, `GetPasswordData`, `GenerateDataKey` |
| Token dropout | 50% UNK-drop in train, **never on the current (last) event** |
| Loss | Event-level BCE; select checkpoint on **val F1** (stratus-vs-benjamin AP saturates at 1.0) |
| Split | Test = `inv:bert-jan`. Val = `inv:stratus-red-team-ec2-get-password-data-role` + `inv:benjamin`. `syn:` train/val only. `fe:` positives subsampled 25% in train. |
| Thresholds | Triage **0.625**, alert **0.85** |

Tabular features (35) are the shared temporal schema: MFA/principal flags, velocity, time-of-day, IAM/recon/secrets indicators, wildcard policy flags, etc. Plus the two PE-context columns at train time.

---

## Steps taken to build this model

1. **Drop BOTSv3** as training data (no real PE chains).
2. **Start from Invictus temporal** (`invictus_temporal.csv`, 2,900 events, two attackers: `bert-jan` and stratus). Window **per user by time**, not by row count.
3. **v1 uniLSTM** on Invictus only. Learned that left-pad + last-hidden and a 70/30 window split leak or fail leave-one-attacker-out.
4. **Merge fe-final CloudTrail** (`prepare_lstm_dataset.py`): recover `event_name`, build **union vocab (281)**, prefix `inv:` / `fe:`, write `train_temporal.csv` (12,611 rows).
5. **v2 bagged BiLSTM + attention** on merged data, window label = any attack event. Strong in-distribution AUC-PR; still not the bert-jan loot detector.
6. **Switch supervision to events** so fusion can use `P_seq = max(P_event)`. Right-pad T=32. Add **synthetic chains** (`augment_attack_chains.py` → `train_temporal_aug.csv`) so train sees more than two PE templates.
7. **v3/v4 hybrid LSTM + Transformer.** v4 still missed bert-jan `GetSecretValue` because T=32 truncated the PE write.
8. **v5:** PE context on the full timeline, campaign relabel, secrets head, no UNK-drop on the last token. Report **campaign-label** metrics on held-out bert-jan.
9. **Serve** the same checkpoint from Streamlit and `prod/` (CLI + FastAPI). Dual thresholds for triage vs alert.

---

## Headline metrics (report these)

bert-jan, **campaign labels**, event threshold **0.625**:

| Setting | Precision | Recall | F1 | AUC-PR |
|---------|-----------|--------|----|--------|
| v4, original CSV labels | 0.38 | 0.43 | 0.40 | 0.52 |
| **v5, campaign labels** | **0.73** | **0.98** | **0.83** | **0.95** |
| v5, original CSV labels | 0.23 | 0.99 | 0.37 | 0.39 |

Use the **v5 campaign** row in the report. v5 on orig CSV looks like precision 0.23 because `Decrypt` / `AssumeRole` / `CreateSecret` are still labeled 0 in Invictus, not because ranking failed.

Window AUC near 1.0 is inflated by max-pool. **Do not report Streamlit window AUC as the result.** Train/val event F1 ~1.00 is an easy split. This is **not SOC-ready**; there are only two real Invictus attackers.

Demo (Streamlit): PE chain ~0.97 ALERT; short benign slice ~0.19 OK.

---

## Dataset files

| Source | File | Notes |
|--------|------|--------|
| Invictus | `data/lstm/invictus_temporal.csv` | 2,900 events; `bert-jan` 107 orig. attack events; stratus 29 |
| fe-final CloudTrail | `data/lstm/cloudtrail_temporal.csv` | 9,711 template PE rows |
| Merged (no synth) | `data/lstm/train_temporal.csv` | 12,611 rows |
| **Train (v5)** | **`data/lstm/train_temporal_aug.csv`** | Merged + `syn:` chains |
| Vocab | `data/lstm/event_name_vocab.json` | 281 ids |
| fe-final-only vocab | `data/lstm/event_name_vocab_fe_final.json` | IDs 1–67 (frozen, lineage) |
| Lineage names | `invictus_enriched.csv` | Used to recover event-name strings; not the train table |

A window is labeled attack if **any** event in it is attack (after campaign relabel at train time).

---

## How to run

Python 3.10+. All commands from `temporal-analysis/`.

### 1. Install

```bash
pip install -r requirements-ui.txt
```

CPU-only API/CLI (no Streamlit):

```bash
pip install -r requirements-prod.txt
```

### 2. Rebuild data (optional — CSVs are already in git)

```bash
python prepare_lstm_dataset.py
python augment_attack_chains.py
```

### 3. Train v5

```bash
python train_lstm_transformer.py
```

Writes `artifacts/lstm_transformer/temporal_lstm_transformer.pt` and `test_metrics.json`.

### 4. Streamlit tester

```bash
python -m streamlit run streamlit_app.py
```

Upload a CloudTrail-style CSV (`username`, `timestamp`, `event_name` or `event_name_idx`) or use the built-in samples.

### 5. Production scorer

```bash
python -m prod.cli --csv data/lstm/sample_custom_events.csv --out artifacts/P_seq_prod.csv
python -m prod.smoke_test
uvicorn prod.app:app --host 0.0.0.0 --port 8000
```

API: `GET /health`, `GET /model`, `POST /score/csv`, `POST /score/json`.

### 6. Notebook

Open [`temporal-lstm-transformer.ipynb`](temporal-lstm-transformer.ipynb) with this folder as the kernel cwd. It loads the same script + checkpoint and produces the plots used in the writeup.

---

## Layout

```text
temporal-analysis/
  README.md                         ← you are here
  temporal-lstm-transformer.ipynb   # v5 notebook
  train_lstm_transformer.py         # v5 train + infer
  prepare_lstm_dataset.py           # Invictus + fe-final → train_temporal.csv
  augment_attack_chains.py          # + syn: chains → train_temporal_aug.csv
  streamlit_app.py
  prod/                             # CLI + FastAPI
  data/lstm/event_name_vocab.json
  data/lstm/train_temporal_aug.csv
  artifacts/lstm_transformer/temporal_lstm_transformer.pt
  temporal-lstm.ipynb               # archive v2
  artifacts/temporal_lstm_v2.pt     # archive v2 weights
```

Related dataset notes: [`../datasets/LINKS.txt`](../datasets/LINKS.txt).
