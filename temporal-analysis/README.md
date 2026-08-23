# CloudSec Temporal Analysis

Sequence-model track (`P_seq`) for AWS privilege-escalation detection. Per-user CloudTrail events are scored, then pooled into **one probability per 10-minute window** (stride 2 minutes) for fusion with the graph model (`P_graph`).

Fusion contract: **`P_seq = max(P_event)`** inside each window. Streamlit does not fuse with the GNN.

## Working pipeline (v5)

| Item | Path |
|------|------|
| Notebook (plots + writeup) | [`temporal-lstm-transformer.ipynb`](temporal-lstm-transformer.ipynb) |
| Train / infer | [`train_lstm_transformer.py`](train_lstm_transformer.py) |
| Streamlit tester | [`streamlit_app.py`](streamlit_app.py) |
| Checkpoint | [`artifacts/lstm_transformer/temporal_lstm_transformer.pt`](artifacts/lstm_transformer/temporal_lstm_transformer.pt) |
| Train table | [`data/lstm/train_temporal_aug.csv`](data/lstm/train_temporal_aug.csv) |

Run everything with the working directory set to this folder (`temporal-analysis/`).

```bash
pip install -r requirements-ui.txt
python train_lstm_transformer.py
python -m streamlit run streamlit_app.py
```

Production scorer (same checkpoint):

```bash
pip install -r requirements-prod.txt
python -m prod.cli --csv data/lstm/sample_custom_events.csv --out artifacts/P_seq_prod.csv
python -m prod.smoke_test
```

### Older notebooks (archive)

- [`temporal-lstm.ipynb`](temporal-lstm.ipynb) — bagged masked BiLSTM v2 (`temporal_lstm_v2.pt`)

## Why v5

v2 / v4 missed bert-jan `GetSecretValue` (mean score ~0.28) and flagged nearby IAM as false positives. On busy users, **T=32 drops the privilege-escalation write** off the visible history, so loot looks like a benign read.

v5:

- Past-only `pe_write_recent` / `log_secs_since_pe` over the **full user timeline** (not just the T=32 window)
- Campaign relabel: secrets + `AssumeRole` / `CreateSecret` count as attack if a PE write happened in the last 10 minutes
- Secrets head on secret APIs; last event is never UNK-dropped
- Architecture: right-pad T=32 → BiLSTM (h=48) + 1-layer Transformer (4 heads) + tabular skip + secrets head

**Honest test user:** `inv:bert-jan` (held out). Val: `inv:stratus-red-team-ec2-get-password-data-role` + `inv:benjamin`. `syn:` rows are train/val only. `fe:` positives are subsampled 25% in train.

bert-jan is the long human-like Invictus chain. The other Invictus attacker (stratus) is 29 `GetPasswordData` events and is too easy. Template `fe:` / `syn:` rows are not the report metric.

## Headline metrics (report these)

bert-jan, **campaign labels**, event threshold **0.625**:

| Setting | Precision | Recall | F1 | AUC-PR |
|---------|-----------|--------|----|--------|
| v4, original CSV labels | 0.38 | 0.43 | 0.40 | 0.52 |
| **v5, campaign labels** | **0.73** | **0.98** | **0.83** | **0.95** |
| v5, original CSV labels | 0.23 | 0.99 | 0.37 | 0.39 |

Campaign labels mark `GetSecretValue` / `Decrypt` / `AssumeRole` / `CreateSecret` as attack if they follow a PE write within 10 minutes. Original Invictus labels under-count in-campaign `Decrypt` (178 events still labeled 0). Use the **v5 campaign** row in the report. Window AUC near 1.0 is inflated by max-pool — do not report Streamlit window AUC as the result.

Train/val event F1 is ~1.00 (easy split). This is **not SOC-ready**; there are only two real Invictus attackers.

Triage / alert defaults: **0.625** / **0.85**. Demo PE chain scores ~0.97 (ALERT); a short benign slice ~0.19 (OK).

## Dataset

Invictus IR CloudTrail plus fe-final CloudTrail, union event-name vocab, usernames prefixed `inv:` / `fe:` so windows never mix sources. `syn:` rows are labeled synthetic PE chains from [`augment_attack_chains.py`](augment_attack_chains.py).

| Source | File | Notes |
|--------|------|--------|
| Invictus | `data/lstm/invictus_temporal.csv` | Real-ish attacker + benign users |
| fe-final CloudTrail | `data/lstm/cloudtrail_temporal.csv` | Template PE traffic |
| Merged (no synth) | `data/lstm/train_temporal.csv` | Rebuild with `prepare_lstm_dataset.py` |
| **Train (v5)** | **`data/lstm/train_temporal_aug.csv`** | Merged + `syn:` chains |
| Vocab | `data/lstm/event_name_vocab.json` | 281 APIs (`event_name_idx` 1–280; PAD/UNK = 0) |

Rebuild steps:

```bash
python prepare_lstm_dataset.py
python augment_attack_chains.py
python train_lstm_transformer.py
```

A window is labeled attack if any event in it is attack. Primary event metric is AUC-PR on **held-out bert-jan**, not overall accuracy.

## Model

| Item | Value |
|------|--------|
| Architecture | Embed APIs → concat tabular feats + PE-context → BiLSTM → Transformer → binary + secrets heads |
| Schema | `lstm_transformer_v5.0` |
| Sequence length | 32 (right-pad) |
| Split | Leave-one-attacker-out: bert-jan test |
| Checkpoint | `artifacts/lstm_transformer/temporal_lstm_transformer.pt` |

## Layout

```text
temporal-analysis/
  temporal-lstm-transformer.ipynb   # v5 notebook
  train_lstm_transformer.py
  streamlit_app.py
  prepare_lstm_dataset.py
  augment_attack_chains.py
  prod/                             # CLI + FastAPI scorer
  data/lstm/train_temporal_aug.csv
  artifacts/lstm_transformer/temporal_lstm_transformer.pt
  temporal-lstm.ipynb               # archive v2
  README.md
```

Related dataset notes: [`../datasets/LINKS.txt`](../datasets/LINKS.txt).
