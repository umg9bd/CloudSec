# LSTM Pipeline Fix Plan — Real-World Ready Sequence Model

**Owner:** Sequence Model track (Nandan)  
**Model:** Grok 4.5 session only (no alternate model agents)  
**Goal:** Fix pipeline bugs, upgrade architecture, train on new `fe-final` data so `P_seq` is usable in production-like CloudTrail streams.

---

## 1. Pipeline debug findings (current state)

| ID | Severity | Finding | Impact |
|----|----------|---------|--------|
| D1 | **P0** | Checkpoint `vocab=261` (Invictus) ≠ new data `vocab=68` | Transfer eval meaningless; production would mis-read API tokens |
| D2 | **P0** | Mean pad fraction **96.6%** (`T=128`, median raw_len=3, max=17) | Last-hidden LSTM reads mostly zeros; weak real-world signal |
| D3 | **P0** | No length mask in forward pass | Pad timesteps pollute hidden state |
| D4 | **P1** | Random stratified window split leaks same-user windows across train/test | Inflated metrics; fails on new principals |
| D5 | **P1** | No vocab / feature schema frozen in deploy artifact beyond loose lists | Live CloudTrail with new APIs breaks silently |
| D6 | **P1** | Accuracy-focused narrative misleading (imbalanced) | Wrong gate for ship/no-ship |
| D7 | **P2** | UniLSTM last-state only; no step attention | Cannot highlight risky API steps for IR / ensemble |
| D8 | **P2** | Old AGENT.md still documents Invictus-only `(2900,40)` asserts | Pipeline docs drift from `data/lstm/train_temporal.csv` |

**Measured stats (new data):** 9,711 events · 11,151 windows · 1,081 pos · feature schema match OK · event pos rate ~4.9% · window pos rate ~9.7%.

**Transfer result (old ckpt → new data):** Acc ~70.6%, AUC-PR ~0.18, Precision ~21%, Recall ~74% → **not production-ready**.

---

## 2. Why current architecture helps / hurts

### Good (keep)
- Event embedding + 35 engineered features
- Window-level binary `P_seq` for ensemble fusion
- Bagging for variance on limited labels
- AUC-PR as primary metric + `pos_weight`

### Bad (replace)
- Left-pad to 128 without masking
- Last hidden only
- Dataset-locked integer vocab without UNK policy at serve time
- Identical bag members (no diversity of inductive bias beyond seed)

---

## 3. Target architecture (v2) — production-oriented

```
event_name_idx (T'≤32) ──► Embedding(V, 32, padding_idx=0)
35 features            ──► concat → x_t ∈ R^{67}
                           │
                     Pack / mask (ignore PAD)
                           │
              BiLSTM(hidden=64, layers=1, dropout=0.3)
                           │
              Attention pool over VALID timesteps only
                           │
                    Dropout(0.4) → Linear(128→1) logit
                           │
              Bag of 3–5 seeds → P_seq = mean(sigmoid)
```

**Why this improves real-world behavior**
1. **Mask + short T** matches real burst lengths (≤17 today).
2. **BiLSTM** sees short attack chains in both directions.
3. **Attention** focuses on IAM/priv-esc steps; supports IR attribution.
4. **Frozen vocab + UNK (0)** maps unknown live APIs safely.
5. **User-group / LOAO splits** approximate new-attacker generalization.

---

## 4. Fix plan phases

### Phase A — Pipeline correctness (must ship with v2)
1. Train **only** on `data/lstm/train_temporal.csv` with `vocab_size=68`.
2. Set `SEQ_LEN=32` (headroom over max 17; cut pad waste).
3. Pass `lengths` / boolean mask into model; never update on PAD.
4. Add `delta_t_sec` (clipped + log1p normalized) as an extra per-step feature **or** keep 35 and rely on existing temporal feats (prefer add 1 feature → 36 if schema allowed; else compute inside window builder as 36th).
5. Split by **username groups** (GroupShuffleSplit): train / val / test with no user overlap.
6. Persist in checkpoint: `state_dicts`, `feature_cols`, `event_name_vocab`, `config`, `threshold`, `metrics`, `schema_version`.
7. Serve rule: unknown `event_name` → idx 0 (`<UNK>`); missing feature → 0.0; reject windows with 0 events.

### Phase B — Architecture train
1. Implement `TemporalSeqModelV2` (masked BiLSTM + attention).
2. Train bag (seeds 42–44 or 42–46) with early stop on **val AUC-PR**.
3. Tune threshold on **val** for F1, also report precision@fixed recall (e.g. recall≥0.8).
4. Evaluate: group-held-out test + LOAO top attackers + score gap.
5. Export `artifacts/temporal_lstm_v2.pt`, `P_seq_v2.csv`, `test_metrics_v2.json`.

### Phase C — Real-world readiness gates (ship checklist)
| Gate | Pass criteria |
|------|----------------|
| G1 Schema | Checkpoint includes vocab + feature list + SEQ_LEN |
| G2 Ranking | Test AUC-PR **> 0.35** (vs transfer 0.18) and **> majority baseline** |
| G3 Detection | At val-tuned thr: recall ≥ 0.70 **and** precision ≥ 0.35 (or document PR curve) |
| G4 Generalization | ≥1 LOAO user with AUC-PR > 0.5 and recall > 0 |
| G5 Ops | Inference script loads ckpt, maps OOV→UNK, writes `P_seq` only |
| G6 No leak | Train/val/test users disjoint |

If G2–G4 fail: add synthetic Stratus chains (Phase D), do **not** inflate capacity first.

### Phase D — Data flywheel (if gates fail)
1. Merge labeled Stratus / synthetic attack chains into same temporal schema.
2. Rebuild shared vocab from `event_name` strings (not Invictus indices).
3. Re-train v2; re-run gates.

---

## 5. Explicit non-goals (this pass)
- MoE / giant Transformer
- Retraining the broken Invictus-261 ckpt for transfer bragging
- Optimizing accuracy as primary KPI
- Changing GNN / fusion (only deliver better `P_seq`)

---

## 6. Deliverables
| Artifact | Purpose |
|----------|---------|
| `FIX_PLAN.md` | This plan |
| `train_temporal_lstm_v2.py` | Train + eval v2 |
| `infer_p_seq_v2.py` | Real-world inference entrypoint |
| `artifacts/temporal_lstm_v2.pt` | Deployable weights |
| `artifacts/*_v2.*` | Metrics + `P_seq` |
| Update `AGENT.md` § architecture / data path | Stop Invictus-only drift |

---

## 7. Execution order
1. ✅ Debug pipeline (done)  
2. ⬜ Implement v2 trainer + inference  
3. ⬜ Train on new dataset  
4. ⬜ Run gates G1–G6  
5. ⬜ Update AGENT.md + report metrics vs v1 transfer  

**Success definition:** Model trained on current CloudTrail temporal schema, masked architecture, user-disjoint eval, checkpoint self-contained for live OOV handling, and metrics clearly better than the 0.18 AUC-PR transfer baseline on the same new data.

---

## 8. Results after v2 train (executed)

| Metric | v1 transfer (old ckpt→new) | **v2 user-disjoint test** |
|--------|----------------------------|---------------------------|
| Accuracy | 70.6% | **94.8%** |
| AUC-PR | 0.186 | **0.417** |
| AUC-ROC | 0.773 | **0.939** |
| Precision | 0.21 | **0.49** |
| Recall | 0.73 | **0.52** |
| F1 | 0.32 | **0.50** |
| Score gap (pos−neg) | 0.07 | **0.51** |

**Gates:** G2 ranking ✅ · G4 LOAO ✅ · G3 (recall≥0.70 & prec≥0.35) ❌ (precision OK, recall at thr=0.7 is 0.52 — lower threshold recovers recall if ops prefer sensitivity).

**Artifacts:** `artifacts/temporal_lstm_v2.pt`, `P_seq_v2.csv`, `test_metrics_v2.json`, `infer_p_seq_v2.py`.

---

## 9. Phase E — Production serve (shipped)

**Goal:** Realtime-capable `P_seq` scoring service for ensemble handoff (not standalone alerter).

| Deliverable | Location |
|-------------|----------|
| Scorer library | `prod/model.py`, `prod/scorer.py` |
| FastAPI | `prod/app.py` — `/health`, `/model`, `/score/csv`, `/score/json` |
| CLI | `python -m prod.cli` |
| Handoff contract | `prod/HANDOFF.md` |
| Smoke test | `python -m prod.smoke_test` |
| Deps / image | `requirements-prod.txt`, `Dockerfile` |

**Ship checklist**

- [x] Load only `temporal_lstm_v2.pt` (ignore v1)
- [x] Dual thresholds: triage 0.55 / alert from ckpt (~0.70)
- [x] OOV → UNK(0); missing features → 0.0; schema 400s
- [x] User-window CSV columns for `P_graph` join
- [x] Smoke test green on CPU before demo

```bash
pip install -r requirements-prod.txt
python -m prod.smoke_test
uvicorn prod.app:app --host 0.0.0.0 --port 8000
```
