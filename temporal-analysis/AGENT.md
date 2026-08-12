# agent.md — Sequence Model Track (AWS Privilege Escalation Detection)

## 1. Project Overview

**Project name:** AWS Privilege Escalation Detection using GNN + Sequence Model Fusion

**Core idea:** AWS privilege escalation attacks happen through chains of individually-plausible API calls (e.g. `CreateRole` → `AttachRolePolicy` → `AssumeRole` → `GetSecretValue`). No single call looks malicious; the attack is only visible across the full sequence and the relational structure of IAM permissions. Rule-based tools (GuardDuty) and flat anomaly detection both fail on this because they ignore either the sequence or the structure.

**Solution:** Combine two models, evaluated on real CloudTrail data from a documented incident:
- A **GNN** that models the IAM permission graph (who can reach what — relational signal)
- A **sequence model (LSTM/Transformer)** that models the order of API calls within a time window (temporal signal)
- Both outputs are fused into a single alert score

**Classification type:** Binary — benign session vs. attack session, evaluated at session level (not per-event).

---

## 2. Team & Roles

| Person | Role | Owns |
|---|---|---|
| Vanshitha | Data Engineering Lead | Raw CloudTrail JSON → `invictus_enriched.csv`; synthetic data generation |
| Akshaya | GNN Lead | Graph construction (NetworkX → PyTorch Geometric); GNN training → `P_graph` |
| Udita | Feature Architect | Shared feature design → `invictus_temporal.csv` (engineered features for sequence track) |
| **Nandan (me)** | **Sequence Model Lead** | **Session windowing; LSTM training; sequence classification → `P_seq`** |

The ensemble/fusion layer and final evaluation is a joint task across all four — not solely my responsibility, but I need to deliver a clean, working `P_seq` output for it to consume.

---

## 3. My Specific Job

**Goal:** Build session windows from `invictus_temporal.csv` and train an LSTM that outputs **one probability score per session** (`P_seq`, range 0–1) representing how likely that session contains a privilege escalation attack.

**What I am NOT doing:** graph construction, GNN training, fusion logic, or final ensemble evaluation. My deliverable is a single trained model + a clean `P_seq` output per session that gets handed off to the shared ensemble step.

**Locked spec for this track:**
- **Input:** ordered sequence of `event_name_idx` tokens + per-timestep engineered numeric features (+ `delta_t_log1p`), windowed by time per username
- **Primary data (current):** `data/lstm/train_temporal.csv` from CloudSec `fe-final` (`cloudtrail_temporal.csv`)
- **Architecture (v2):** Embedding → concat features → **masked BiLSTM** → **attention pool** → binary head (bag of 5). See `FIX_PLAN.md`, `train_temporal_lstm_v2.py`, checkpoint `artifacts/temporal_lstm_v2.pt`
- **Legacy v1:** uniLSTM last-hidden (`train_temporal_lstm.py` / `temporal_lstm.pt`) — Invictus-era; do not use for new-data production
- **Task:** session-level (window-level) binary classification
- **Supervision:** window is positive iff it contains **at least one** event with `label == 1` (`max(label) == 1`). No ±5 min soft `session_label`. Do **not** join enriched for baseline training.
- **Eval:** user-disjoint GroupShuffleSplit + LOAO; primary metric AUC-PR (not accuracy)
- **Serve (production):** `prod/` package — FastAPI + CLI; OOV event → UNK idx 0; dual thresholds (triage 0.55 / alert ~0.70). See §10.

---

## 4. Datasets

### 4.1 Primary — `invictus_temporal.csv` (USE THIS AS THE BASE, ALWAYS)

- Source: Invictus IR public incident response case, July 2023 — engineered feature view of the same 2,900 events
- Shape: `(2900, 40)`, zero NaNs
- ~55-minute time window (2023-07-10 11:42–12:37 UTC)
- Class balance at event level: 2764 benign / 136 attack (~95.3% / 4.7%)
- Attackers: `bert-jan` (107 attack events) and `stratus-red-team-ec2-get-password-data-role` (29 attack events)
- `log_id` format `invictus_enriched.csv:<row_index>` — lineage pointer only; **do not require joining enriched for the baseline**

**Columns:**
```
log_id, username, timestamp,
no_mfa, mfa_absent, principal_type_prior_risk, principal_type_idx, has_access_key,
action_velocity, is_new_action, session_duration_normalized, events_per_minute_normalized,
time_sin, time_cos, is_weekend, is_off_hours, action_risk_prior,
event_name_idx, event_source_idx,
is_write_action, read_only_absent, has_error, is_access_denied, is_iam_event,
is_recon_action, is_defense_evasion, is_get_caller_identity, is_malicious_user_agent,
is_public_ip, params_length_normalized, targets_sensitive_resource, is_non_default_region,
is_create_key, is_secrets_or_kms, is_permission_modification,
policy_statement_count_normalized, has_wildcard_action, has_wildcard_resource,
privileged_action_reach, label
```

**Columns I actually need:**
| Column | Role |
|---|---|
| `username` | groupby key — sessions are strictly per-principal, never global |
| `timestamp` | sort key + time-window boundary |
| `event_name_idx` | sequence tokens (integer IDs, 260 unique values, range 1–260; PAD=0) |
| engineered features (35 cols) | per-timestep numeric vector concatenated with the embedding |
| `label` | event-level attack flag; **window label** = `1 if max(label)==1 else 0` |

**Not in this file:** raw `event_name`, `session_label`, `attack_technique`. Attribution at inference uses `event_name_idx` lists (Counter report). No auxiliary technique head.

### 4.2 Lineage — `invictus_enriched.csv` (reference only)

- Original raw/enriched CloudTrail extraction that `invictus_temporal.csv` was built from
- Not used as the training base for this track
- Available if a future step needs the `event_name_idx → event_name` string mapping for human-readable reports

### 4.3 Ruled out — BOTSv3 (`botsv3_enriched_features_behavioral.csv`)

- Splunk Boss of the SOC v3 — only usable for **pipeline smoke-testing**, never for real training
- Confirmed unusable: only 1 `CreateAccessKey` and 1 `CreateUser` event, no real privilege escalation chains, `requestParameters` empty, `user_name` is NaN for most rows, label column is `is_anomaly` (not aligned with this track)

---

## 5. Pipeline — Step by Step

### Step 0 — Validate before touching anything
- Confirm shape: `(2900, 40)` for `invictus_temporal.csv`
- Confirm `timestamp` parses to real datetime (UTC)
- Confirm zero NaNs in `username`, `timestamp`, `event_name_idx`, `label`
- Confirm `event_name_idx.nunique() == 260`, `min == 1`, `max == 260` (PAD=0 free)
- Confirm `label` positive count == 136
- **Manually inspect `bert-jan`'s sorted `event_name_idx` sequence** before trusting downstream code

### Step 1 — Vocabulary / token IDs
- Use existing `event_name_idx` as-is
- `vocab_size = 261` (index 0 = `<PAD>`, indices 1–260 = events)
- Do **not** rebuild a string vocab from raw event names

### Step 2 — Build sliding windows (per user, time-based, NOT row-count)
- `groupby('username')` — never mix users in one window
- Sort by `timestamp` within each user group
- Slide a **10-minute time window** across each user's sorted events (**stride = 2 min**, documented)
- Window = events where `timestamp ∈ [t, t + 10min)`
- Label per window: `1 if max(label in window) == 1 else 0` (strict: must contain a real attack event)
- Per window store: `event_name_idx` list, feature matrix `(L, F)`, username, window start, window label
- Abort training if positive windows &lt; 3

### Step 3 — Fix sequence length
- Pad short windows with `<PAD>` (index 0) at the start (pre-padding); PAD feature rows = zeros
- Truncate long windows to most recent N events
- Fixed length: **T = 128**

### Step 4 — Split (stratified by window, not row)
- **70/30** train/test stratified by window label; validation = 20% held out from the 70% train pool (~56/14/30 overall)
- **All events from one window stay in the same split** — no leakage
- Also run leave-one-attacker-out (`bert-jan` vs `stratus-...`) to measure actor-specific overfitting

### Step 5 — Model architecture
```python
class TemporalSeqModel(nn.Module):
    def __init__(self, vocab_size=261, embed_dim=32, n_features=35, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + n_features, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, event_idx, feats):
        # event_idx: (B, T), feats: (B, T, F)
        x = torch.cat([self.embedding(event_idx), feats], dim=-1)
        _, (h, _) = self.lstm(x)
        return self.head(self.dropout(h.squeeze(0))).squeeze(-1)
```

No technique auxiliary head (column absent). Train a **bag of 5** models (seeds 42–46).

### Step 6 — Loss & training
- `BCEWithLogitsLoss(pos_weight=...)` where `pos_weight = n_negative / n_positive` on **window** counts — mandatory given imbalance
- Regularize hard: `hidden_dim=64`, `dropout=0.5`, `weight_decay=1e-3`, early stopping on validation AUC-PR (patience 15)
- Tune decision threshold on **validation** to maximize F1 (AUC-PR remains primary ranking metric)

### Step 7 — Evaluate
- **Primary metric: AUC-PR** (not accuracy)
- Secondary: F1, precision, recall, AUC-ROC (at val-tuned threshold)
- **Gate:** beat prior baseline test AUC-PR (0.70) **and** LOAO `bert-jan` recall &gt; 0 with AUC-PR &gt; 0.5
- Also report LOAO for `stratus-...` (tiny n — secondary)

### Step 8 — Output
- `P_seq = mean_i sigmoid(logit_i)` across the bag, per window — final deliverable for ensemble fusion

---

## 6. Key Decisions Already Locked

- Primary dataset is **`invictus_temporal.csv` only** — no join to enriched for training labels
- Window label = **`max(label) == 1`** (contains a real attack event) — no ±5 min soft session labels
- Sessions are **per-user, per-time-window** — never global, never row-count based
- Window **10 min**, stride **2 min**, sequence length **T=128**
- Bag of 5 small LSTMs; `P_seq` is averaged probability
- `event_name_idx` → `nn.Embedding` (PAD=0); concat engineered features per timestep
- `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- AUC-PR is the primary eval metric, not accuracy
- Event attribution via `event_name_idx` list + Counter report — no attention mechanism
- **MoE explicitly rejected** — bagging used instead for limited data

## 7. Known Risk: Small / Single-Incident Data

- All 2,900 events come from **one incident** (Invictus IR, July 2023), with only **2 attacker identities**
- Real risk: model memorizes `bert-jan`'s specific sequence rather than learning general privilege-escalation technique patterns
- **Observed ceiling (improved recipe):** stratified test AUC-PR can reach ~0.92, but LOAO hold-out of `bert-jan` leaves ~1 positive window (stratus) for training — thresholded recall stays 0 even when ranking AUC-PR &gt; 0.5. This is a **data limit**, not a training-bug.
- Mitigations, in priority order:
  1. Leave-one-attacker-out validation (done — reveals the ceiling)
  2. Strict window labels + longer sequences + bagging (done in `train_temporal_lstm.py`)
  3. Aggressive regularization + small model capacity
  4. Long-term real fix: synthetic attack chain augmentation (Vanshitha's parametric simulator) — required for LOAO generalization

## 8. Open Items / Dependencies on Teammates

- Optional later: obtain `event_name_idx → event_name` mapping from Udita/enriched for human-readable attribution
- Fusion weights / final `P_final` still owned jointly — handoff schema is locked in [`prod/HANDOFF.md`](prod/HANDOFF.md)

## 9. My Deliverables

- Windowing function (validated, per-user, time-based) on temporal CSV
- Trained LSTM v2 (+ `artifacts/temporal_lstm_v2.pt`)
- Sequence model notebook / training scripts; val/test AUC-PR, F1
- `P_seq` output per session/window, ready for ensemble handoff
- Production scorer (`prod/`) + smoke test
- Contribution to shared Implementation/References/Appendix sections in the paper draft

## 10. Production serve

| Piece | Path / command |
|-------|----------------|
| Checkpoint | `artifacts/temporal_lstm_v2.pt` |
| Library | `prod/scorer.py`, `prod/model.py` |
| CLI batch | `python -m prod.cli --csv data/lstm/train_temporal.csv --out artifacts/P_seq_prod.csv` |
| API | `uvicorn prod.app:app --host 0.0.0.0 --port 8000` |
| Endpoints | `GET /health`, `GET /model`, `POST /score/csv`, `POST /score/json` |
| Ensemble contract | [`prod/HANDOFF.md`](prod/HANDOFF.md) |
| Smoke gate | `python -m prod.smoke_test` (must pass before demo) |
| Docker | `docker build -t pseq-scorer .` then `docker run -p 8000:8000 pseq-scorer` |

**Threshold policy:** `pred_triage` if `P_seq >= 0.55`; `pred_alert` if `P_seq >= thr_alert` (~0.70 from ckpt). Use as scoring/triage signal — not sole pager.
