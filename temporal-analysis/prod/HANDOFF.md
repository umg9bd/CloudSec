# Ensemble handoff — P_seq → fusion with P_graph

## Purpose

The sequence track delivers **per-window** `P_seq` scores for privilege-escalation
likelihood. The GNN track delivers `P_graph`. Fusion joins them; this service does
**not** implement fusion.

## Model

- Checkpoint: `artifacts/temporal_lstm_v2.pt`
- `model_id`: `temporal_lstm_v2`
- `schema_version`: from checkpoint (currently `temporal_lstm_v2.1`)

## Output columns (API + CSV)

| Column | Type | Meaning |
|--------|------|---------|
| `username` | str | Principal key (join key with graph) |
| `window_start` | ISO-8601 UTC | Window start (join / nearest-match key) |
| `window_end` | ISO-8601 UTC | Window end (start + 10 min) |
| `P_seq` | float `[0,1]` | Bag-averaged attack probability |
| `pred_triage` | 0/1 | `P_seq >= 0.55` — SOC review queue |
| `pred_alert` | 0/1 | `P_seq >= thr_alert` (ckpt, ~0.70) — high confidence only |
| `raw_len` | int | Events in window before pad |
| `schema_version` | str | Scorer schema |
| `model_id` | str | Model identifier |

## Join recipe for fusion

1. Align on **`username`** (exact string match).
2. Align windows on **`window_start`** when both sides use the same 10 min / 2 min grid.
3. If graph sessions use a different clock grid, join each graph session to the
   sequence window with maximum overlap, or nearest `window_start` within ±2 min.
4. Example fuse (team-owned):  
   `P_final = 0.5 * P_seq + 0.5 * P_graph` (or learned weights).

## Threshold policy

| Mode | Threshold | Use |
|------|-----------|-----|
| Triage | 0.55 | Broad review / dashboard highlight |
| Alert | ~0.70 (val-tuned in ckpt) | High-confidence flag only — **not** sole pager |

Do not page on `pred_triage` alone; precision at alert threshold is still ~0.5.

## How to produce handoff CSV

```bash
python -m prod.cli --csv data/lstm/train_temporal.csv --out artifacts/P_seq_prod.csv
```

Or `POST /score/csv` / `POST /score/json` on the FastAPI service.
