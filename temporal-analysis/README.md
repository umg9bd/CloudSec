# CloudSec Temporal Analysis

Sequence-model track for AWS privilege-escalation detection. Per-user CloudTrail events are turned into 10-minute windows and scored by an LSTM. The output is one attack probability per window (`P_seq`) for fusion with the graph model (`P_graph`).

## Working notebook

**Use this:** [`temporal-lstm.ipynb`](temporal-lstm.ipynb)

That notebook is the current pipeline: load merged data, window events, train/evaluate `temporal_lstm_v2` (masked BiLSTM + attention, bag of 5), and plot metrics.

Set `RUN_TRAIN = True` to train. Set it `False` to load a local checkpoint at `artifacts/temporal_lstm_v2.pt` (weights are not in git; see below).

Run it with the working directory set to this folder (`temporal-analysis/`) so `data/lstm/train_temporal.csv` resolves.

### Older notebooks (archive)

These are earlier BOTSv3 experiments, not the production sequence track:

- `capstone-temporal-analyst.ipynb` — row-count windows (`WINDOW_SIZE = 10`)
- `capstone-temporal-analyst  v2.ipynb` — 10-minute windows, majority-vote labels
- `capstone-temporal-analyst v(2)-new.ipynb` — further BOTSv3 iteration

## Merged dataset

**Use this:** [`data/lstm/train_temporal.csv`](data/lstm/train_temporal.csv)

Official Invictus IR CloudTrail plus fe-final CloudTrail, joined on a union event-name vocab. Usernames are prefixed `inv:` / `fe:` so windows never mix the two sources.

| Source | File | Rows |
|--------|------|------|
| Invictus | `data/lstm/invictus_temporal.csv` | 2,900 |
| fe-final CloudTrail | `data/lstm/cloudtrail_temporal.csv` | 9,711 |
| **Merged (train)** | **`data/lstm/train_temporal.csv`** | **12,611** |

Supporting files:

- `data/lstm/event_name_vocab.json` — union vocab (`event_name_idx` 1–280; PAD/UNK = 0)
- `data/lstm/event_name_vocab_fe_final.json` — fe-final IDs 1–67 (frozen)
- `data/lstm/manifest.json` — merge stats and windowing notes
- `data/lstm/cloudtrail_structural.csv` — structural companion table
- `invictus_enriched.csv` / `invictus_temporal.csv` — Invictus source copies

A window is labeled attack if **any** event in it has `label == 1`. Windows are 10 minutes with a 2-minute stride, per username. Primary metric is AUC-PR (not accuracy).

## Model

| Item | Value |
|------|--------|
| Architecture | Embedding → concat numeric features → masked BiLSTM → attention pool → binary head |
| Schema | `temporal_lstm_v2.1` |
| Sequence length | 32 (pad / truncate) |
| Split | User-disjoint `GroupShuffleSplit` |
| Checkpoint | [`artifacts/temporal_lstm_v2.pt`](artifacts/temporal_lstm_v2.pt) |

The working v2 weights are in git. Other artifacts (plots, score CSVs, v1 / fe-final-only checkpoints) stay gitignored. Set `RUN_TRAIN = False` in the notebook to load this file.

## Requirements

Python 3.10+.

```bash
pip install numpy pandas torch scikit-learn matplotlib jupyter
```

## Layout

```text
temporal-analysis/
  temporal-lstm.ipynb          # working notebook
  data/lstm/train_temporal.csv # merged training table
  data/lstm/                   # vocabs, unmerged sources, manifest
  invictus_enriched.csv
  invictus_temporal.csv
  README.md
```

Related dataset notes: [`../datasets/LINKS.txt`](../datasets/LINKS.txt).
