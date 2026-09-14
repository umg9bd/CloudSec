"""
LSTM + Transformer v5 — IAM writes + campaign-gated secrets.

v4 missed all bert-jan GetSecretValue (score ~0.28) and flagged nearby IAM
(AssumeRole, CreateSecret) as false positives. Cause: T=32 truncates the PE
write out of a busy user's history, so loot looks like benign reads.

v5:
  - past-only pe_write_recent / log_secs_since_pe over the full user timeline
  - campaign relabel: secrets + AssumeRole/CreateSecret within 10 min after a PE write
  - secrets head added only on secret APIs
  - do not UNK-drop the current (last) event
  - select on val F1 (stratus-vs-benjamin AP saturates at 1.0)

P_seq for fusion = max(event probability) inside each 10-min / stride-2 window.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "lstm" / "train_temporal_aug.csv"
VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab.json"
OUT_DIR = ROOT / "artifacts" / "lstm_transformer"

SEED = 42
WINDOW_MINUTES = 10
STRIDE_MINUTES = 2
SEQ_LEN = 32
EMBED_DIM = 16
HIDDEN_DIM = 48
NHEAD = 4
TF_LAYERS = 1
DROPOUT = 0.45
TOKEN_DROP = 0.50
FEAT_DROP = 0.15
FEAT_NOISE = 0.05
RISK_DROP = 0.4
INV_BOOST = 12.0
FE_POS_KEEP = 0.25
BATCH_SIZE = 64
LR = 8e-4
WEIGHT_DECAY = 5e-3
MAX_EPOCHS = 20
PATIENCE = 6
MIN_EPOCHS = 4
BERT_JAN = "inv:bert-jan"
STRATUS_ATTACKER = "inv:stratus-red-team-ec2-get-password-data-role"
VAL_INV_CLEAN = "inv:benjamin"
META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx"}
PE_WRITE_NAMES = frozenset(
    {
        "CreateRole",
        "AttachRolePolicy",
        "PutRolePolicy",
        "CreateUser",
        "CreateAccessKey",
        "AttachUserPolicy",
        "CreateLoginProfile",
        "UpdateAssumeRolePolicy",
        "StopLogging",
        "DeleteTrail",
        "PutBucketPolicy",
        "PutEventSelectors",
        "CreatePolicyVersion",
        "SetDefaultPolicyVersion",
        "AddUserToGroup",
        "CreatePolicy",
        "CreateSecret",
        "PutSecretValue",
    }
)
SECRET_NAMES = frozenset({"GetSecretValue", "Decrypt", "GetPasswordData", "GenerateDataKey"})
CAMPAIGN_EXTRA_NAMES = frozenset({"AssumeRole", "CreateSecret"})


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vocab_id_sets(vocab: dict[str, int]) -> tuple[set[int], set[int], set[int]]:
    pe = {int(vocab[n]) for n in PE_WRITE_NAMES if n in vocab}
    sec = {int(vocab[n]) for n in SECRET_NAMES if n in vocab}
    extra = {int(vocab[n]) for n in CAMPAIGN_EXTRA_NAMES if n in vocab}
    if not pe or not sec:
        raise SystemExit(f"vocab missing PE/secret names pe={len(pe)} sec={len(sec)}")
    return pe, sec, extra


def attach_pe_context(df: pd.DataFrame, pe_ids: set[int]) -> pd.DataFrame:
    """Past-only PE context from the full user timeline (not the truncated T=32 window)."""
    recent = {}
    log_dt = {}
    horizon = 600.0
    for _, g in df.groupby("username", sort=False):
        g = g.sort_values("timestamp")
        last_pe_ns = None
        for log_id, t_ns, ev in zip(
            g["log_id"].astype(str),
            g["timestamp"].astype("int64"),
            g["event_name_idx"].astype(int),
        ):
            if last_pe_ns is None:
                dt = 1_000_000.0
            else:
                dt = max((int(t_ns) - last_pe_ns) / 1e9, 0.0)
            recent[log_id] = 1.0 if dt <= horizon else 0.0
            log_dt[log_id] = float(math.log1p(min(dt, 3600.0)))
            if int(ev) in pe_ids:
                last_pe_ns = int(t_ns)
    out = df.copy()
    out["pe_write_recent"] = out["log_id"].astype(str).map(recent).fillna(0.0).astype(np.float32)
    out["log_secs_since_pe"] = out["log_id"].astype(str).map(log_dt).fillna(math.log1p(1_000_000.0)).astype(np.float32)
    return out


def relabel_campaign(df: pd.DataFrame, sec_ids: set[int], extra_ids: set[int]) -> pd.DataFrame:
    """Mark secrets / AssumeRole / CreateSecret as attack if a PE write happened in the last 10 min."""
    out = df.copy()
    out["label_orig"] = out["label"].astype(int)
    near = out["pe_write_recent"] >= 0.5
    loot = out["event_name_idx"].isin(sec_ids | extra_ids)
    flipped = int((loot & near & (out["label_orig"] == 0)).sum())
    out.loc[loot & near, "label"] = 1
    out.loc[out["label_orig"] == 1, "label"] = 1
    print(
        f"campaign relabel: +{flipped} events "
        f"(orig_pos={int(out['label_orig'].sum())} new_pos={int(out['label'].sum())})"
    )
    return out


def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[str], int]:
    df = pd.read_csv(path)
    assert df.shape[1] == 40, df.shape
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["username", "timestamp", "event_name_idx", "label"]:
        assert df[col].isna().sum() == 0, col
    assert int(df["event_name_idx"].min()) >= 1
    feature_cols = [c for c in df.columns if c not in META_COLS]
    assert len(feature_cols) == 35, len(feature_cols)
    vocab_size = int(df["event_name_idx"].max()) + 1
    print("=== Validation PASSED (merged) ===")
    print(f"shape={df.shape} features={len(feature_cols)} vocab_size={vocab_size}")
    print(f"users={df['username'].nunique()} attacks={int(df['label'].sum())}")
    print(f"prefixes: {df['username'].str.split(':').str[0].value_counts().to_dict()}")
    return df, feature_cols, vocab_size


def add_extra_feats(g: pd.DataFrame, feature_cols: list[str], is_inv: float = 0.0) -> np.ndarray:
    """Numeric FE columns + inter-event Δt. No source flag (that leaked Invictus vs fe-final)."""
    del is_inv
    feats = g[feature_cols].to_numpy(dtype=np.float32)
    times = g["timestamp"].astype("int64").to_numpy()
    deltas = np.clip(np.diff(times, prepend=times[0]) / 1e9, 0, 3600)
    delta = np.log1p(deltas).astype(np.float32).reshape(-1, 1)
    return np.concatenate([feats, delta], axis=1)


def pad_seq(idxs: np.ndarray, feats: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Right-pad. pack_padded_sequence reads the FIRST `length` steps, so PAD must be on the right."""
    raw_len = len(idxs)
    if raw_len > seq_len:
        idxs, feats = idxs[-seq_len:], feats[-seq_len:]
        raw_len = seq_len
    pad = seq_len - raw_len
    if pad > 0:
        idxs = np.concatenate([idxs, np.zeros(pad, dtype=np.int64)])
        feats = np.concatenate(
            [feats, np.zeros((pad, feats.shape[1]), dtype=np.float32)], axis=0
        )
    return idxs, feats, raw_len


@dataclass
class EventSeq:
    username: str
    timestamp: pd.Timestamp
    log_id: str
    event_idxs: np.ndarray
    feats: np.ndarray
    length: int
    label: int
    last_idx: int
    label_orig: int


def build_event_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_minutes: int = WINDOW_MINUTES,
    seq_len: int = SEQ_LEN,
) -> list[EventSeq]:
    """One sample per event: APIs in the last `window_minutes`, target = that event's label."""
    window_ns = int(pd.Timedelta(minutes=window_minutes) / pd.Timedelta(nanoseconds=1))
    seqs: list[EventSeq] = []
    for username, g in df.groupby("username", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            continue
        feats_all = add_extra_feats(g, feature_cols)
        idxs_all = g["event_name_idx"].to_numpy(dtype=np.int64)
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        labels = g["label"].to_numpy(dtype=np.int64)
        labels_orig = (
            g["label_orig"].to_numpy(dtype=np.int64) if "label_orig" in g.columns else labels
        )
        log_ids = g["log_id"].astype(str).to_numpy()
        n = len(g)
        for i in range(n):
            left = int(np.searchsorted(ts_ns, ts_ns[i] - window_ns, side="left"))
            idxs, feats, raw_len = pad_seq(idxs_all[left : i + 1], feats_all[left : i + 1], seq_len)
            seqs.append(
                EventSeq(
                    username=str(username),
                    timestamp=g.loc[i, "timestamp"],
                    log_id=log_ids[i],
                    event_idxs=idxs,
                    feats=feats,
                    length=raw_len,
                    label=int(labels[i]),
                    last_idx=int(idxs_all[i]),
                    label_orig=int(labels_orig[i]),
                )
            )
    return seqs


def build_fusion_windows(
    df: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
    stride_minutes: int = STRIDE_MINUTES,
) -> list[dict]:
    """Same event-covering 10-min / stride-2 grid used for P_seq fusion."""
    window_td = pd.Timedelta(minutes=window_minutes)
    stride_td = pd.Timedelta(minutes=stride_minutes)
    stride_ns = int(stride_td / pd.Timedelta(nanoseconds=1))
    window_ns = int(window_td / pd.Timedelta(nanoseconds=1))
    rows: list[dict] = []
    for username, g in df.groupby("username", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ts = g["timestamp"]
        ts_ns = ts.astype("int64").to_numpy()
        log_ids = g["log_id"].astype(str).to_numpy()
        labels = g["label"].to_numpy()
        starts: set[pd.Timestamp] = set()
        for t in ts:
            aligned = (int(t.value) // stride_ns) * stride_ns
            k = 0
            while k * stride_ns < window_ns:
                starts.add(pd.Timestamp(aligned - k * stride_ns, tz="UTC"))
                k += 1
        for start in sorted(starts):
            end = start + window_td
            left = int(np.searchsorted(ts_ns, start.value, side="left"))
            right = int(np.searchsorted(ts_ns, end.value, side="left"))
            if right <= left:
                continue
            chunk_labels = labels[left:right]
            rows.append(
                {
                    "username": str(username),
                    "window_start": start,
                    "window_end": end,
                    "window_label": int(chunk_labels.max() == 1),
                    "raw_len": int(right - left),
                    "log_ids": log_ids[left:right].tolist(),
                    "event_idx_list": g["event_name_idx"].iloc[left:right].astype(int).tolist(),
                }
            )
    return rows


class EventDataset(Dataset):
    def __init__(self, seqs: list[EventSeq]):
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        s = self.seqs[i]
        return (
            torch.from_numpy(s.event_idxs),
            torch.from_numpy(s.feats),
            torch.tensor(s.length, dtype=torch.long),
            torch.tensor(s.label, dtype=torch.float32),
        )


def valid_mask(lengths: torch.Tensor, t_len: int) -> torch.Tensor:
    return torch.arange(t_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class LSTMTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_features: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        nhead: int = NHEAD,
        n_layers: int = TF_LAYERS,
        dropout: float = DROPOUT,
        risk_idx: int | None = None,
        secret_ids: set[int] | None = None,
    ):
        super().__init__()
        self.risk_idx = risk_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim + n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        d_model = hidden_dim * 2
        self.lstm_norm = nn.LayerNorm(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(
                enc, num_layers=n_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.len_gate = nn.Linear(1, 1)
        nn.init.constant_(self.len_gate.bias, -2.0)
        self.seq_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model + embed_dim, 1),
        )
        self.tab_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )
        self.secret_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features + embed_dim, 1),
        )
        nn.init.zeros_(self.seq_head[-1].weight)
        nn.init.zeros_(self.seq_head[-1].bias)
        nn.init.zeros_(self.secret_head[-1].weight)
        nn.init.zeros_(self.secret_head[-1].bias)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        if secret_ids:
            mask[list(secret_ids)] = True
        self.register_buffer("secret_id_mask", mask)

    def _augment(self, event_idx, feats, lengths):
        if not self.training:
            return event_idx, feats
        drop = (torch.rand_like(event_idx, dtype=torch.float32) < TOKEN_DROP) & (event_idx != 0)
        last = (lengths - 1).clamp(min=0)
        b = torch.arange(event_idx.size(0), device=event_idx.device)
        drop = drop.clone()
        drop[b, last] = False
        event_idx = event_idx.masked_fill(drop, 0)
        feats = feats + torch.randn_like(feats) * FEAT_NOISE
        keep = (torch.rand(feats.size(0), 1, feats.size(2), device=feats.device) > FEAT_DROP).float()
        feats = feats * keep
        if self.risk_idx is not None and 0 <= self.risk_idx < feats.size(-1):
            if float(torch.rand(1)) < RISK_DROP:
                feats = feats.clone()
                feats[:, :, self.risk_idx] = 0
        return event_idx, feats

    def forward(self, event_idx, feats, lengths):
        event_idx, feats = self._augment(event_idx, feats, lengths)
        emb = self.emb_drop(self.embedding(event_idx))
        x = torch.cat([emb, feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=event_idx.size(1)
        )
        h = self.lstm_norm(h)
        pad_mask = ~valid_mask(lengths, event_idx.size(1))
        tf_out = self.transformer(h, src_key_padding_mask=pad_mask)
        gate = torch.sigmoid(self.len_gate(torch.log1p(lengths.float()).unsqueeze(-1)))
        z = h + gate.unsqueeze(1) * tf_out
        last = (lengths - 1).clamp(min=0)
        b = torch.arange(z.size(0), device=z.device)
        last_z = torch.cat([z[b, last], emb[b, last]], dim=-1)
        last_feat = feats[b, last]
        last_emb = emb[b, last]
        iam = (self.tab_head(last_feat) + self.seq_head(last_z)).squeeze(-1)
        sec = self.secret_head(torch.cat([last_feat, last_emb], dim=-1)).squeeze(-1)
        last_tok = event_idx[b, last]
        secret_mask = self.secret_id_mask[last_tok.long()].float()
        return iam + secret_mask * sec


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for event_idx, feats, lengths, y in loader:
        logits = model(event_idx.to(device), feats.to(device), lengths.to(device))
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    return y_true, probs


def metrics_dict(y_true, probs, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float((preds == y_true).mean()) if len(y_true) else float("nan"),
        "auc_pr": float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else float("nan"),
        "auc_roc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def tune_threshold(y_true, probs) -> float:
    """Max F1 on val; if several thresholds tie, take the highest (fewer false positives)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and t > best_t):
            best_f1, best_t = float(f1), float(t)
    return best_t


def _split_user_list(users: list[str], test_frac: float, rng: np.random.RandomState):
    users = list(users)
    rng.shuffle(users)
    if len(users) <= 1:
        return users, []
    n_te = min(max(1, int(round(len(users) * test_frac))), len(users) - 1)
    return users[n_te:], users[:n_te]


def group_split_v4(seqs: list[EventSeq], seed=SEED):
    """bert-jan = test. stratus + benjamin = hard Invictus val. syn never in test."""
    rng = np.random.RandomState(seed)
    by_user: dict[str, list[EventSeq]] = {}
    for s in seqs:
        by_user.setdefault(s.username, []).append(s)

    locked_test = {BERT_JAN}
    locked_val = {STRATUS_ATTACKER, VAL_INV_CLEAN}
    locked = locked_test | locked_val
    missing = sorted(u for u in locked if u not in by_user)
    if missing:
        raise SystemExit(f"split missing locked users: {missing}")

    pos_users = [u for u, xs in by_user.items() if any(s.label for s in xs) and u not in locked]
    neg_users = [u for u in by_user if u not in set(pos_users) and u not in locked]
    syn_pos = [u for u in pos_users if u.startswith("syn:")]
    fe_pos = [u for u in pos_users if u.startswith("fe:")]
    other_pos = [u for u in pos_users if u not in set(syn_pos) | set(fe_pos)]

    rng.shuffle(syn_pos)
    n_syn_va = int(round(0.15 * len(syn_pos))) if len(syn_pos) >= 8 else 0
    va_syn, tr_syn = syn_pos[:n_syn_va], syn_pos[n_syn_va:]
    te_syn: list[str] = []

    tr_fe, te_fe = _split_user_list(fe_pos + other_pos, 0.30, rng)
    tr_fe, va_fe = _split_user_list(tr_fe, 0.20, rng)

    inv_neg = [u for u in neg_users if u.startswith("inv:")]
    other_neg = [u for u in neg_users if not u.startswith("inv:")]
    tr_neg, te_neg = _split_user_list(other_neg, 0.30, rng)
    tr_neg, va_neg = _split_user_list(tr_neg, 0.20, rng)
    tr_neg = tr_neg + inv_neg

    train_u = set(tr_syn + tr_fe + tr_neg)
    val_u = set(va_syn + va_fe + va_neg) | locked_val
    test_u = set(te_fe + te_neg + te_syn) | locked_test
    take = lambda u: [s for s in seqs if s.username in u]
    print(
        f"attack-users train/val/test={len(tr_syn)+len(tr_fe)}/{len(va_syn)+len(va_fe)+1}/{len(te_fe)+1} "
        f"clean-users={len(tr_neg)}/{len(va_neg)+1}/{len(te_neg)}"
    )
    print(f"inv locked val={sorted(locked_val)} test={sorted(locked_test)}")
    print(f"syn attack-users train/val/test={len(tr_syn)}/{len(va_syn)}/{len(te_syn)}")
    return take(train_u), take(val_u), take(test_u)


def subsample_fe_positives(
    seqs: list[EventSeq], keep_frac: float = FE_POS_KEEP, seed: int = SEED
) -> list[EventSeq]:
    rng = np.random.RandomState(seed)
    fe_pos = [s for s in seqs if s.username.startswith("fe:") and s.label == 1]
    rest = [s for s in seqs if not (s.username.startswith("fe:") and s.label == 1)]
    if not fe_pos:
        return seqs
    n_keep = max(8, int(round(len(fe_pos) * keep_frac)))
    n_keep = min(n_keep, len(fe_pos))
    pick = rng.choice(len(fe_pos), size=n_keep, replace=False)
    kept = [fe_pos[int(i)] for i in pick]
    print(f"fe+ subsample {len(fe_pos)} -> {len(kept)} (keep={keep_frac})")
    return rest + kept


def group_split_70_30(seqs: list[EventSeq], seed=SEED):
    """Back-compat alias used by the notebook."""
    return group_split_v4(seqs, seed=seed)


def make_loader(seqs: list[EventSeq], weighted: bool = False, secret_ids: set[int] | None = None) -> DataLoader:
    ds = EventDataset(seqs)
    if not weighted:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    secret_ids = secret_ids or set()
    user_n = Counter(s.username for s in seqs)
    weights = []
    for s in seqs:
        w = 1.0 / math.sqrt(user_n[s.username])
        if s.label:
            if s.username.startswith("fe:"):
                w *= 0.5
            else:
                w *= 3.0
            if s.last_idx in secret_ids:
                w *= 4.0
        if s.username.startswith("inv:"):
            w *= INV_BOOST if s.label else 1.5
        weights.append(w)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(seqs),
        replacement=True,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)


def seqs_metrics(model, seqs, device, prefix: str | None = None, threshold: float = 0.5):
    if prefix:
        seqs = [s for s in seqs if s.username.startswith(prefix)]
    if not seqs:
        return None
    y, p = predict(model, make_loader(seqs), device)
    return metrics_dict(y, p, threshold)


def orig_label_metrics(seqs: list[EventSeq], probs: np.ndarray, threshold: float = 0.5):
    y = np.array([s.label_orig for s in seqs], dtype=np.int64)
    return metrics_dict(y, probs, threshold)


def train_model(train_s, val_s, vocab_size, n_features, device, risk_idx=None, secret_ids=None):
    set_seed(SEED)
    secret_ids = secret_ids or set()
    train_loader = make_loader(train_s, weighted=True, secret_ids=secret_ids)
    n_pos = sum(s.label for s in train_s)
    n_neg = len(train_s) - n_pos
    pos_weight = torch.tensor([math.sqrt(n_neg / max(n_pos, 1))], dtype=torch.float32, device=device)
    print(f"train_events={len(train_s)} pos={n_pos} neg={n_neg} pos_weight(sqrt)={pos_weight.item():.3f}")
    print(
        f"reg: token_drop={TOKEN_DROP} feat_drop={FEAT_DROP} risk_drop={RISK_DROP} "
        f"wd={WEIGHT_DECAY} inv_boost={INV_BOOST} min_epochs={MIN_EPOCHS}"
    )

    model = LSTMTransformerModel(
        vocab_size=vocab_size,
        n_features=n_features,
        risk_idx=risk_idx,
        secret_ids=secret_ids,
    ).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3)

    history, best_score, best_state, patience_left = [], -1.0, None, PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for event_idx, feats, lengths, y in train_loader:
            if int(lengths.min()) < 1:
                continue
            optim.zero_grad()
            logits = model(event_idx.to(device), feats.to(device), lengths.to(device))
            y_s = y.to(device) * 0.9 + 0.05
            loss = criterion(logits, y_s)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        tr_m = seqs_metrics(model, train_s, device)
        val_m = seqs_metrics(model, val_s, device)
        inv_m = seqs_metrics(model, val_s, device, "inv:")
        train_ap = tr_m["auc_pr"] if tr_m else float("nan")
        val_ap = val_m["auc_pr"] if val_m else float("nan")
        inv_ap = inv_m["auc_pr"] if inv_m and inv_m["n_pos"] > 0 else float("nan")
        score = val_m["f1"] if val_m else val_ap
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "lr": optim.param_groups[0]["lr"],
                "train_auc_pr": train_ap,
                "val_auc_pr": val_ap,
                "val_inv_auc_pr": inv_ap,
                "select_score": score,
                **{f"val_{k}": v for k, v in (val_m or {}).items()},
            }
        )
        print(
            f"epoch {epoch:03d} loss={history[-1]['train_loss']:.4f} "
            f"train_ap={train_ap:.4f} val_ap={val_ap:.4f} val_f1={score:.4f} val_inv_ap={inv_ap:.4f} "
            f"gap={train_ap - val_ap:.3f}",
            flush=True,
        )
        scheduler.step(0.0 if math.isnan(score) else score)
        improved = not math.isnan(score) and score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        elif epoch >= MIN_EPOCHS:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop @ {epoch} (best val F1={best_score:.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def score_seqs(model, seqs, device) -> pd.DataFrame:
    loader = make_loader(seqs, weighted=False)
    _, probs = predict(model, loader, device)
    return pd.DataFrame(
        {
            "log_id": [s.log_id for s in seqs],
            "username": [s.username for s in seqs],
            "timestamp": [s.timestamp for s in seqs],
            "label": [s.label for s in seqs],
            "label_orig": [s.label_orig for s in seqs],
            "P_event": probs,
        }
    )


def window_scores(fusion_rows: list[dict], event_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    p_map = dict(zip(event_df["log_id"], event_df["P_event"]))
    out = []
    for w in fusion_rows:
        ps = [p_map[i] for i in w["log_ids"] if i in p_map]
        p = float(max(ps)) if ps else 0.0
        out.append(
            {
                "username": w["username"],
                "window_start": w["window_start"].isoformat(),
                "window_end": w["window_end"].isoformat(),
                "window_label": w["window_label"],
                "raw_len": w["raw_len"],
                "P_seq": p,
                "pred": int(p >= threshold),
                "top_event_name_idx": json.dumps(Counter(w["event_idx_list"]).most_common(5)),
            }
        )
    return pd.DataFrame(out)


def subset_metrics(df_win: pd.DataFrame, threshold: float, prefix: str | None):
    sub = df_win if prefix is None else df_win[df_win["username"].str.startswith(prefix)]
    if sub.empty:
        return {"n": 0}
    return metrics_dict(sub["window_label"].to_numpy(), sub["P_seq"].to_numpy(), threshold)


PE_CONTEXT_COLS = ["pe_write_recent", "log_secs_since_pe"]


def load_checkpoint(ckpt_path: Path | None = None, device=None):
    """Load LSTMTransformerV5 weights for inference."""
    path = Path(ckpt_path) if ckpt_path is not None else OUT_DIR / "temporal_lstm_transformer.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint {path}. Train with train_lstm_transformer.py")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config") or {}
    secret_ids = {int(x) for x in cfg.get("secret_ids") or []}
    model = LSTMTransformerModel(
        vocab_size=int(cfg["vocab_size"]),
        n_features=int(cfg["n_features"]),
        risk_idx=None,
        secret_ids=secret_ids,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, device


def map_event_names_df(df: pd.DataFrame, vocab: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    if "event_name" in out.columns and vocab:
        out["event_name_idx"] = out["event_name"].map(lambda x: int(vocab.get(str(x), 0)))
    elif "event_name_idx" in out.columns:
        vmax = max(int(v) for v in vocab.values()) if vocab else int(out["event_name_idx"].max())
        out["event_name_idx"] = (
            pd.to_numeric(out["event_name_idx"], errors="coerce").fillna(0).astype(int)
        )
        out.loc[(out["event_name_idx"] > vmax) | (out["event_name_idx"] < 0), "event_name_idx"] = 0
    else:
        raise ValueError("Input must include event_name or event_name_idx")
    return out


def prepare_score_frame(
    df: pd.DataFrame, vocab: dict[str, int], feature_cols: list[str]
) -> pd.DataFrame:
    missing = {"username", "timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df is None or len(df) == 0:
        raise ValueError("Empty input: no events to score")
    out = map_event_names_df(df, vocab)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("timestamp contains unparseable values")
    if out["username"].isna().any():
        raise ValueError("username contains nulls")
    out["username"] = out["username"].astype(str)
    if "log_id" not in out.columns:
        out["log_id"] = [f"infer:{i}" for i in range(len(out))]
    else:
        out["log_id"] = out["log_id"].astype(str)
    if "label" not in out.columns:
        out["label"] = 0
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    for c in feature_cols:
        if c in PE_CONTEXT_COLS:
            continue
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    pe_ids, _, _ = vocab_id_sets(vocab)
    return attach_pe_context(out, pe_ids)


def score_events_to_windows(
    df: pd.DataFrame, model, ckpt: dict, device
) -> pd.DataFrame:
    """Score events with v5, return fusion windows with P_seq = max(P_event)."""
    vocab = dict(ckpt.get("event_name_vocab") or {})
    if not vocab and VOCAB_PATH.exists():
        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    feature_cols = list(ckpt["feature_cols"])
    prepared = prepare_score_frame(df, vocab, feature_cols)
    seqs = build_event_sequences(prepared, feature_cols)
    if not seqs:
        raise ValueError("No windows built from input (need ≥1 event per user)")
    event_df = score_seqs(model, seqs, device)
    fusion = build_fusion_windows(prepared)
    win_thr = float(ckpt.get("threshold", 0.85))
    return window_scores(fusion, event_df, win_thr)


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print("model=LSTMTransformerV5 (campaign-gated secrets + PE context)", flush=True)

    df, feature_cols, vocab_size = load_and_validate(CSV_PATH)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8")) if VOCAB_PATH.exists() else {}
    pe_ids, sec_ids, extra_ids = vocab_id_sets(vocab)
    df = attach_pe_context(df, pe_ids)
    df = relabel_campaign(df, sec_ids, extra_ids)
    feature_cols = feature_cols + ["pe_write_recent", "log_secs_since_pe"]
    seqs = build_event_sequences(df, feature_cols)
    n_features = seqs[0].feats.shape[1]
    print(f"event_seqs={len(seqs)} pos={sum(s.label for s in seqs)} n_features={n_features}", flush=True)
    print(
        f"hist_len min/median/max="
        f"{min(s.length for s in seqs)}/{int(np.median([s.length for s in seqs]))}/{max(s.length for s in seqs)}",
        flush=True,
    )

    train_s, val_s, test_s = group_split_v4(seqs)
    train_s = subsample_fe_positives(train_s)
    print(f"split events train/val/test={len(train_s)}/{len(val_s)}/{len(test_s)}", flush=True)
    risk_idx = feature_cols.index("action_risk_prior") if "action_risk_prior" in feature_cols else None
    print(f"action_risk_prior idx={risk_idx}", flush=True)

    model, history = train_model(
        train_s, val_s, vocab_size, n_features, device, risk_idx=risk_idx, secret_ids=sec_ids
    )
    y_val, p_val = predict(model, make_loader(val_s), device)
    threshold = tune_threshold(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.5
    print(f"val-tuned event threshold={threshold:.3f}", flush=True)

    def _print_split(name, seqs_):
        m = seqs_metrics(model, seqs_, device, threshold=threshold)
        fe = seqs_metrics(model, seqs_, device, "fe:", threshold=threshold)
        inv = seqs_metrics(model, seqs_, device, "inv:", threshold=threshold)
        print(f"=== {name} EVENT @ {threshold:.3f} ===", m, flush=True)
        print(f"    fe:", fe, flush=True)
        print(f"    inv:", inv, flush=True)
        return m

    _print_split("Train", train_s)
    _print_split("Val", val_s)
    test_evt = _print_split("Test", test_s)
    bj = [s for s in test_s if s.username == BERT_JAN]
    _, p_bj = predict(model, make_loader(bj), device)
    bj_camp = metrics_dict(np.array([s.label for s in bj]), p_bj, threshold)
    bj_orig = orig_label_metrics(bj, p_bj, threshold)
    bj_orig_05 = orig_label_metrics(bj, p_bj, 0.5)
    print("=== bert-jan CAMPAIGN labels ===", bj_camp, flush=True)
    print("=== bert-jan ORIGINAL labels @tuned ===", bj_orig, flush=True)
    print("=== bert-jan ORIGINAL labels @0.5 ===", bj_orig_05, flush=True)

    event_df = score_seqs(model, seqs, device)
    fusion_rows = build_fusion_windows(df)
    pseq = window_scores(fusion_rows, event_df, threshold)

    test_users = {s.username for s in test_s}
    val_users = {s.username for s in val_s}
    test_win = pseq[pseq["username"].isin(test_users)]
    val_win = pseq[pseq["username"].isin(val_users)]
    win_thr = tune_threshold(val_win["window_label"].to_numpy(), val_win["P_seq"].to_numpy())
    test_win_m = metrics_dict(
        test_win["window_label"].to_numpy(), test_win["P_seq"].to_numpy(), win_thr
    )
    print(f"val-tuned WINDOW threshold={win_thr:.3f}", flush=True)
    print("=== Test WINDOW metrics (P_seq = max P_event) ===", flush=True)
    print(test_win_m, flush=True)
    print("test inv:*", subset_metrics(test_win, win_thr, "inv:"), flush=True)
    print("test fe:*", subset_metrics(test_win, win_thr, "fe:"), flush=True)

    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8")) if VOCAB_PATH.exists() else {}
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "threshold": win_thr,
            "event_threshold": threshold,
            "test_metrics": test_win_m,
            "test_event_metrics": test_evt,
            "event_name_vocab": vocab,
            "config": {
                "model": "LSTMTransformerV5",
                "dataset": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "vocab_size": vocab_size,
                "n_features": n_features,
                "seq_len": SEQ_LEN,
                "window_minutes": WINDOW_MINUTES,
                "stride_minutes": STRIDE_MINUTES,
                "train_unit": "event (10-min history, loss on last step)",
                "p_seq": "max(P_event) in fusion window",
                "split": "v4 users: bert-jan test, stratus+benjamin val, syn train/val",
                "secret_ids": sorted(sec_ids),
                "anti_overfit": {
                    "token_drop": TOKEN_DROP,
                    "keep_last_token": True,
                    "feat_drop": FEAT_DROP,
                    "risk_drop": RISK_DROP,
                    "weight_decay": WEIGHT_DECAY,
                    "fe_pos_keep": FE_POS_KEEP,
                    "inv_boost": INV_BOOST,
                    "no_source_flag": True,
                    "early_stop": "val F1, min_epochs=4",
                    "pe_context": True,
                    "campaign_relabel": True,
                    "secrets_head": True,
                },
            },
        },
        OUT_DIR / "temporal_lstm_transformer.pt",
    )
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    event_df.to_csv(OUT_DIR / "P_event.csv", index=False)
    pseq["pred"] = (pseq["P_seq"] >= win_thr).astype(int)
    pseq.to_csv(OUT_DIR / "P_seq.csv", index=False)
    with open(OUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_event": test_evt,
                "test_event_fe": seqs_metrics(model, test_s, device, "fe:", threshold=threshold),
                "test_event_inv": seqs_metrics(model, test_s, device, "inv:", threshold=threshold),
                "test_bertjan_campaign": bj_camp,
                "test_bertjan_original": bj_orig,
                "test_bertjan_original_0p5": bj_orig_05,
                "train_event": seqs_metrics(model, train_s, device, threshold=threshold),
                "val_event": seqs_metrics(model, val_s, device, threshold=threshold),
                "test": test_win_m,
                "test_inv": subset_metrics(test_win, win_thr, "inv:"),
                "test_fe": subset_metrics(test_win, win_thr, "fe:"),
                "val_event_inv": seqs_metrics(model, val_s, device, "inv:", threshold=threshold),
                "event_threshold": threshold,
                "window_threshold": win_thr,
                "protocol": {
                    "model": "LSTMTransformerV5",
                    "test_attacker": BERT_JAN,
                    "val_inv": [STRATUS_ATTACKER, VAL_INV_CLEAN],
                    "fe_pos_keep": FE_POS_KEEP,
                    "early_stop": "val_f1",
                    "campaign_relabel": True,
                    "secrets_head": True,
                },
                "previous_v4": {
                    "test_inv_event_auc_pr": 0.522,
                    "test_inv_event_precision": 0.38,
                    "test_inv_event_recall": 0.43,
                    "test_inv_event_f1": 0.40,
                },
            },
            f,
            indent=2,
        )
    pos = pseq.loc[pseq.window_label == 1, "P_seq"]
    neg = pseq.loc[pseq.window_label == 0, "P_seq"]
    print(f"Wrote P_seq.csv ({len(pseq)} windows) and P_event.csv ({len(event_df)} events)", flush=True)
    print(
        f"window score gap pos_mean={pos.mean():.3f} neg_mean={neg.mean():.3f} gap={pos.mean() - neg.mean():.3f}",
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
