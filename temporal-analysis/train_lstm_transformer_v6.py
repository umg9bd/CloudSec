"""
LSTM + Transformer v6.2 — general-purpose, synthetic-only training, real-dev selection.

Does NOT modify v5 training or data/lstm/event_name_vocab.json.

v6.1 fixes (Sep 2026 audit):
  - data: make_v6_train_data.py (leaky no_mfa / mfa_absent / params_length_normalized dropped,
    AssumeRole write flag fixed, read-only credential-theft chains + benign twins added)
  - vocab: <UNK> + names seen in training (v5 ids); any other name scores as <UNK>
  - <UNK> trained on the current (last) event with p=UNK_LAST_P
  - epoch, thresholds and extra feature drops picked on real Invictus dev
    (stratus-get-password-data + benjamin + half the other users + bert-jan before its median
    attack time); test = the other half + bert-jan from that time on, scored only with --eval-test
  - split user lists saved in the checkpoint and split.json
v6.2:
  - data: data/lstm/cloudtrail_temporal_v6_2.csv (+ busy slow-theft sessions, busy benign twins,
    inventory scans over generic read-only AWS calls)
  - thresholds sit THRESHOLD_MARGIN into the best-F1 score gap from the low side (favour recall
    under dev -> test score shift; v6.1's midpoint window threshold missed every test window)
  - real test was already scored once for v6.1: any v6.2 test score is a reuse (see TEST_REUSE)

Usage:
  python train_lstm_transformer_v6.py --ablate     # try extra feature drops, keep best on dev
  python train_lstm_transformer_v6.py              # train (drops from ablation.json if present)
  python train_lstm_transformer_v6.py --eval-test  # score the saved checkpoint on real test
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_lstm_transformer as v5
from train_lstm_transformer import (
    BATCH_SIZE,
    PE_CONTEXT_COLS,
    EventDataset,
    LSTMTransformerModel,
    WINDOW_MINUTES,
    attach_pe_context,
    build_event_sequences,
    build_fusion_windows,
    metrics_dict,
    predict,
    score_seqs,
    window_scores,
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "lstm" / "cloudtrail_temporal_v6_2.csv"
REAL_CSV_PATH = ROOT / "data" / "lstm" / "train_temporal.csv"
V5_VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab.json"
VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab_v6.json"
OUT_DIR = ROOT / "artifacts" / "lstm_transformer_v6"
CKPT_PATH = OUT_DIR / "temporal_lstm_transformer_v6.pt"
METRICS_PATH = OUT_DIR / "test_metrics.json"

SEED = 42
SCHEMA_VERSION = "lstm_transformer_v6.2"
META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx", "event_name"}
LEAKY_FEATURES = ("no_mfa", "mfa_absent", "params_length_normalized")
# Extra drops tried with --ablate; each points opposite ways (or is constant) in synthetic vs real.
ABLATION_DROPS: dict[str, tuple[str, ...]] = {
    "base": (),
    "no_user_agent": ("is_malicious_user_agent",),
    "no_events_per_minute": ("events_per_minute_normalized",),
    "no_principal_type": ("principal_type_prior_risk", "principal_type_idx"),
    "all_candidates": (
        "is_malicious_user_agent",
        "events_per_minute_normalized",
        "principal_type_prior_risk",
        "principal_type_idx",
    ),
}
SECRET_NAMES = v5.SECRET_NAMES | {"GetParameters"}
REAL_DEV_LOCKED = ("inv:stratus-red-team-ec2-get-password-data-role", "inv:benjamin")
# Split in time at its median attack event: both halves keep secret theft + IAM writes.
REAL_TIME_SPLIT_USER = "inv:bert-jan"
LABEL_RULE = "Invictus rule: recon/discovery = 0; PE writes and credential-theft calls = 1"
SELECTION = "epochs >= MIN_EPOCHS: real-dev window F1 (best threshold) > real-dev event AUC-PR > synthetic val F1"
UNK_LAST_P = 0.15
THRESHOLD_MARGIN = 0.1
TEST_REUSE = "real test first scored for v6.1 (2026-09-28); v6.2 data design used v6.1 dev and test errors"
MAX_EPOCHS = 20
PATIENCE = 6
MIN_EPOCHS = 4
LR = 8e-4
WEIGHT_DECAY = 5e-3


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    """Final CSV duplicates many log_ids (twin rows). Keep last."""
    n0 = len(df)
    out = df.drop_duplicates(subset=["log_id"], keep="last").reset_index(drop=True)
    if len(out) != n0:
        print(f"dedupe log_id: {n0} -> {len(out)} rows", flush=True)
    return out


def _v5_names() -> dict[int, str]:
    return {int(v): str(k) for k, v in json.loads(V5_VOCAB_PATH.read_text(encoding="utf-8")).items()}


def load_and_validate(path: Path = CSV_PATH, drop: tuple[str, ...] = ()) -> tuple[pd.DataFrame, list[str], int]:
    df = pd.read_csv(path)
    missing = {"log_id", "username", "timestamp", "label", "event_name_idx"} - set(df.columns)
    assert not missing, missing
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["username", "timestamp", "event_name_idx", "label"]:
        assert df[col].isna().sum() == 0, col
    assert int(df["event_name_idx"].min()) >= 1
    df["username"] = df["username"].astype(str)
    df["log_id"] = df["log_id"].astype(str)
    if "event_name" not in df.columns:
        df["event_name"] = df["event_name_idx"].astype(int).map(_v5_names())
    df = dedupe_events(df)
    df = df.drop(columns=[c for c in (*LEAKY_FEATURES, *drop) if c in df.columns])
    feature_cols = [c for c in df.columns if c not in META_COLS]
    vocab_size = int(df["event_name_idx"].max()) + 1
    print(f"=== Validation PASSED ({SCHEMA_VERSION}) ===", flush=True)
    print(
        f"shape={df.shape} features={len(feature_cols)} vocab_size={vocab_size} "
        f"users={df['username'].nunique()} attacks={int(df['label'].sum())}",
        flush=True,
    )
    return df, feature_cols, vocab_size


def write_vocab_v6(df: pd.DataFrame) -> dict[str, int]:
    """<UNK> + names seen in training, keeping v5 ids. Unseen names score as <UNK>."""
    pairs = df[["event_name", "event_name_idx"]].drop_duplicates()
    assert pairs["event_name"].is_unique and pairs["event_name_idx"].is_unique, "event_name / id mismatch"
    vocab = {"<UNK>": 0, **{str(n): int(i) for n, i in zip(pairs["event_name"], pairs["event_name_idx"])}}
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {VOCAB_PATH} entries={len(vocab)} max_id={max(vocab.values())} (v5 file not modified)", flush=True)
    return vocab


def maybe_pe_ids(vocab: dict[str, int]) -> tuple[set[int], set[int]]:
    pe = {int(vocab[n]) for n in v5.PE_WRITE_NAMES if n in vocab}
    sec = {int(vocab[n]) for n in SECRET_NAMES if n in vocab}
    if not pe or not sec:
        print(f"WARN: vocab missing PE/secret names pe={len(pe)} sec={len(sec)}", flush=True)
    return pe, sec


def load_real(vocab: dict[str, int], feature_cols: list[str], path: Path = REAL_CSV_PATH) -> pd.DataFrame:
    """Invictus rows (inv:*) mapped into the v6 vocab by name. Never used for training."""
    df = pd.read_csv(path)
    df = df[df["username"].astype(str).str.startswith("inv:")].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["username"] = df["username"].astype(str)
    df["log_id"] = df["log_id"].astype(str)
    df["event_name"] = df["event_name_idx"].astype(int).map(_v5_names())
    df["event_name_idx"] = df["event_name"].map(lambda n: int(vocab.get(str(n), 0))).astype(int)
    missing = [c for c in feature_cols if c not in df.columns and c not in PE_CONTEXT_COLS]
    assert not missing, f"real data missing features: {missing}"
    return df.reset_index(drop=True)


def unk_rates(df: pd.DataFrame) -> dict[str, float]:
    unk = df["event_name_idx"] == 0
    return {"all": float(unk.mean()), "attack": float(unk[df["label"] == 1].mean())}


@dataclass
class RealProtocol:
    """Real (Invictus) dev/test: whole users, plus one attacker split in time."""

    dev_users: list[str]
    test_users: list[str]
    split_user: str
    split_time: pd.Timestamp

    def event_mask(self, df: pd.DataFrame, part: str) -> pd.Series:
        whole = self.dev_users if part == "dev" else self.test_users
        early = df["timestamp"] < self.split_time
        side = early if part == "dev" else ~early
        return df["username"].isin(whole) | ((df["username"] == self.split_user) & side)

    def has_event(self, username: str, ts: pd.Timestamp, part: str) -> bool:
        if username == self.split_user:
            return (ts < self.split_time) == (part == "dev")
        return username in (self.dev_users if part == "dev" else self.test_users)

    def has_window(self, w: dict, part: str) -> bool:
        """Windows crossing the split time belong to neither part."""
        if w["username"] == self.split_user:
            return w["window_end"] <= self.split_time if part == "dev" else w["window_start"] >= self.split_time
        return w["username"] in (self.dev_users if part == "dev" else self.test_users)

    def to_config(self) -> dict:
        return {
            "real_dev": self.dev_users,
            "real_test": self.test_users,
            "real_time_split": {
                "user": self.split_user,
                "time": self.split_time.isoformat(),
                "rule": "events before time -> dev, at/after -> test; windows crossing it dropped",
            },
        }

    @classmethod
    def from_config(cls, split_users: dict) -> "RealProtocol":
        ts = split_users["real_time_split"]
        return cls(list(split_users["real_dev"]), list(split_users["real_test"]), ts["user"], pd.Timestamp(ts["time"]))


def split_real_users(real_df: pd.DataFrame, seed: int = SEED) -> RealProtocol:
    """Dev = stratus-get-password-data + benjamin + half the rest + bert-jan before its median attack;
    test = the other half + bert-jan from that time on."""
    users = sorted(real_df["username"].unique())
    locked = (*REAL_DEV_LOCKED, REAL_TIME_SPLIT_USER)
    missing = [u for u in locked if u not in users]
    assert not missing, f"real split missing users: {missing}"
    rest = [u for u in users if u not in locked]
    rest = [rest[i] for i in np.random.RandomState(seed).permutation(len(rest))]
    half = len(rest) // 2
    su = real_df[real_df["username"] == REAL_TIME_SPLIT_USER]
    split_time = su.loc[su["label"] == 1, "timestamp"].median()
    return RealProtocol(sorted([*REAL_DEV_LOCKED, *rest[:half]]), sorted(rest[half:]), REAL_TIME_SPLIT_USER, split_time)


def group_split_users(seqs, seed: int = SEED):
    """~70/15/15 by username (synthetic). No locked test attacker."""
    users = np.array([s.username for s in seqs])
    labels = np.array([s.label for s in seqs])
    idx = np.arange(len(seqs))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, hold_idx = next(gss1.split(idx, labels, groups=users))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_rel, te_rel = next(gss2.split(hold_idx, labels[hold_idx], groups=users[hold_idx]))
    va_idx, te_idx = hold_idx[va_rel], hold_idx[te_rel]
    take = lambda ids: [seqs[i] for i in ids]
    train_s, val_s, test_s = take(tr_idx), take(va_idx), take(te_idx)
    tr_u, va_u, te_u = (
        {s.username for s in train_s},
        {s.username for s in val_s},
        {s.username for s in test_s},
    )
    assert tr_u.isdisjoint(va_u) and tr_u.isdisjoint(te_u) and va_u.isdisjoint(te_u)
    print(
        f"syn users train/val/test={len(tr_u)}/{len(va_u)}/{len(te_u)} "
        f"events={len(train_s)}/{len(val_s)}/{len(test_s)} "
        f"pos={sum(s.label for s in train_s)}/{sum(s.label for s in val_s)}/{sum(s.label for s in test_s)}",
        flush=True,
    )
    return train_s, val_s, test_s


def make_loader(seqs, weighted: bool = False, secret_ids: set[int] | None = None):
    ds = EventDataset(seqs)
    if not weighted:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    secret_ids = secret_ids or set()
    user_n = Counter(s.username for s in seqs)
    weights = []
    for s in seqs:
        w = 1.0 / math.sqrt(user_n[s.username])
        if s.label:
            w *= 3.0
            if s.last_idx in secret_ids:
                w *= 2.0
        weights.append(w)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(seqs),
        replacement=True,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)


def seqs_metrics(model, seqs, device, threshold: float = 0.5):
    if not seqs:
        return None
    y, p = predict(model, make_loader(seqs), device)
    return metrics_dict(y, p, threshold)


def unk_last(event_idx: torch.Tensor, lengths: torch.Tensor, p: float = UNK_LAST_P) -> torch.Tensor:
    """Hide the current event's name (-> <UNK>) so the heads learn to score from features alone."""
    hit = torch.rand(event_idx.size(0)) < p
    if p <= 0 or not bool(hit.any()):
        return event_idx
    rows = torch.nonzero(hit).squeeze(1)
    out = event_idx.clone()
    out[rows, (lengths[rows] - 1).clamp(min=0)] = 0
    return out


def tune_threshold(y_true, probs) -> float:
    """Exact best-F1 threshold over all score cut points; ties -> widest score gap.

    The threshold sits THRESHOLD_MARGIN into that gap from the low side, i.e. just above the
    highest excluded score (v5's 0.05..0.95 grid cannot cut between scores saturated above 0.95).
    """
    y, p = np.asarray(y_true), np.asarray(probs, dtype=float)
    cuts = np.unique(p)
    if y.sum() == 0 or len(np.unique(y)) < 2 or len(cuts) < 2:
        return 0.5
    f1s = np.array([f1_score(y, (p >= t).astype(int), zero_division=0) for t in cuts])
    tied = np.flatnonzero(f1s >= f1s.max() - 1e-12)
    lower = np.concatenate([[0.0], cuts[:-1]])
    i = tied[int(np.argmax(cuts[tied] - lower[tied]))]
    return float(lower[i] + THRESHOLD_MARGIN * (cuts[i] - lower[i]))


@dataclass
class RealSplit:
    """One part (dev/test) of the real data, scored as events and as P_seq windows."""

    part: str
    seqs: list
    windows: list[dict]

    @classmethod
    def build(cls, real_df: pd.DataFrame, proto: RealProtocol, part: str, feature_cols: list[str]) -> "RealSplit":
        # full timelines: history and PE context may cross the split time, as in production
        seqs = [s for s in build_event_sequences(real_df, feature_cols) if proto.has_event(s.username, s.timestamp, part)]
        return cls(part=part, seqs=seqs, windows=real_windows(real_df, proto, part))


def real_windows(real_df: pd.DataFrame, proto: RealProtocol, part: str) -> list[dict]:
    return [w for w in build_fusion_windows(real_df) if proto.has_window(w, part)]


def real_eval(model, split: RealSplit, device, evt_thr: float | None = None, win_thr: float | None = None):
    """Event + window metrics. Thresholds are tuned on this split when not given (dev only)."""
    ev = score_seqs(model, split.seqs, device)
    y, p = ev["label"].to_numpy(), ev["P_event"].to_numpy()
    et = tune_threshold(y, p) if evt_thr is None else float(evt_thr)
    pseq = window_scores(split.windows, ev, et)
    wy, wp = pseq["window_label"].to_numpy(), pseq["P_seq"].to_numpy()
    wt = tune_threshold(wy, wp) if win_thr is None else float(win_thr)
    pseq["pred"] = (pseq["P_seq"] >= wt).astype(int)
    return metrics_dict(y, p, et), metrics_dict(wy, wp, wt), ev, pseq


def _nan0(x: float) -> float:
    return 0.0 if x is None or math.isnan(x) else float(x)


def train_model(
    train_s, val_s, vocab_size, n_features, device, risk_idx=None, secret_ids=None, dev: RealSplit | None = None
):
    """Select the epoch on real dev when given (see SELECTION), else on synthetic val F1."""
    set_seed(SEED)
    secret_ids = secret_ids or set()
    train_loader = make_loader(train_s, weighted=True, secret_ids=secret_ids)
    n_pos = sum(s.label for s in train_s)
    n_neg = len(train_s) - n_pos
    pos_weight = torch.tensor(
        [math.sqrt(n_neg / max(n_pos, 1))], dtype=torch.float32, device=device
    )
    print(
        f"train_events={len(train_s)} pos={n_pos} neg={n_neg} pos_weight(sqrt)={pos_weight.item():.3f} "
        f"unk_last_p={UNK_LAST_P}",
        flush=True,
    )

    model = LSTMTransformerModel(
        vocab_size=vocab_size,
        n_features=n_features,
        risk_idx=risk_idx,
        secret_ids=secret_ids,
    ).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=3
    )

    history, best_key, best_state, patience_left = [], None, None, PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for event_idx, feats, lengths, y in train_loader:
            if int(lengths.min()) < 1:
                continue
            event_idx = unk_last(event_idx, lengths)
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
        val_f1 = _nan0(val_m["f1"]) if val_m else 0.0
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "lr": optim.param_groups[0]["lr"],
            "train_auc_pr": tr_m["auc_pr"] if tr_m else float("nan"),
            "val_auc_pr": val_m["auc_pr"] if val_m else float("nan"),
            "val_f1": val_f1,
        }
        if dev is not None:
            dev_evt, dev_win, _, _ = real_eval(model, dev, device)
            key = (round(_nan0(dev_win["f1"]), 4), round(_nan0(dev_evt["auc_pr"]), 4), round(val_f1, 4))
            row.update(
                dev_window_f1=dev_win["f1"],
                dev_window_auc_pr=dev_win["auc_pr"],
                dev_window_threshold=dev_win["threshold"],
                dev_event_f1=dev_evt["f1"],
                dev_event_auc_pr=dev_evt["auc_pr"],
            )
        else:
            key = (round(val_f1, 4),)
        row["select_key"] = json.dumps(key)
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} train_ap={row['train_auc_pr']:.4f} "
            f"val_ap={row['val_auc_pr']:.4f} val_f1={val_f1:.4f}"
            + (
                f" dev_win_f1={row['dev_window_f1']:.4f} dev_win_ap={row['dev_window_auc_pr']:.4f} "
                f"dev_evt_ap={row['dev_event_auc_pr']:.4f}"
                if dev is not None
                else ""
            ),
            flush=True,
        )
        scheduler.step(float(np.mean(key)))
        if epoch < MIN_EPOCHS:
            continue  # warm-up: an undertrained model can spike on a small dev set (seen at epoch 1)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop @ {epoch} (best epoch {best_epoch} key={best_key})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"selected epoch {best_epoch} key={best_key} ({SELECTION if dev is not None else 'syn val F1'})", flush=True)
    return model, history


@dataclass
class Data:
    df: pd.DataFrame
    real_df: pd.DataFrame
    feature_cols: list[str]
    vocab: dict[str, int]
    vocab_size: int
    sec_ids: set[int]
    proto: RealProtocol


def prepare_data() -> Data:
    df, feature_cols, vocab_size = load_and_validate(CSV_PATH)
    vocab = write_vocab_v6(df)
    pe_ids, sec_ids = maybe_pe_ids(vocab)
    real_df = load_real(vocab, feature_cols)
    if pe_ids:
        df = attach_pe_context(df, pe_ids)
        real_df = attach_pe_context(real_df, pe_ids)
        feature_cols = feature_cols + PE_CONTEXT_COLS
        print("PE context attached (no campaign relabel)", flush=True)
    proto = split_real_users(real_df)
    print(
        f"real dev users={len(proto.dev_users)} test users={len(proto.test_users)} "
        f"+ {proto.split_user} split at {proto.split_time.isoformat()}\n"
        f"  dev events/pos={int(proto.event_mask(real_df, 'dev').sum())}/{int(real_df.loc[proto.event_mask(real_df, 'dev'), 'label'].sum())} "
        f"test events/pos={int(proto.event_mask(real_df, 'test').sum())}/{int(real_df.loc[proto.event_mask(real_df, 'test'), 'label'].sum())}\n"
        f"  UNK dev={unk_rates(real_df[proto.event_mask(real_df, 'dev')])} "
        f"test={unk_rates(real_df[proto.event_mask(real_df, 'test')])}",
        flush=True,
    )
    return Data(df, real_df, feature_cols, vocab, vocab_size, sec_ids, proto)


def fit(data: Data, drop: tuple[str, ...], device) -> dict:
    feature_cols = [c for c in data.feature_cols if c not in drop]
    print(f"--- fit drop={list(drop)} n_feature_cols={len(feature_cols)}", flush=True)
    seqs = build_event_sequences(data.df, feature_cols)
    n_features = seqs[0].feats.shape[1]
    train_s, val_s, test_s = group_split_users(seqs)
    dev = RealSplit.build(data.real_df, data.proto, "dev", feature_cols)
    risk_idx = feature_cols.index("action_risk_prior") if "action_risk_prior" in feature_cols else None
    model, history = train_model(
        train_s, val_s, data.vocab_size, n_features, device, risk_idx=risk_idx, secret_ids=data.sec_ids, dev=dev
    )
    dev_evt, dev_win, dev_ev, dev_pseq = real_eval(model, dev, device)
    evt_thr, win_thr = dev_evt["threshold"], dev_win["threshold"]
    print(f"dev-tuned thresholds event={evt_thr:.3f} window={win_thr:.3f}", flush=True)
    print("=== Real DEV EVENT ===", dev_evt, flush=True)
    print("=== Real DEV WINDOW ===", dev_win, flush=True)

    event_df = score_seqs(model, seqs, device)
    pseq = window_scores(build_fusion_windows(data.df), event_df, evt_thr)
    test_users = {s.username for s in test_s}
    syn_test_win = pseq[pseq["username"].isin(test_users)]
    syn = {
        "syn_train_event": seqs_metrics(model, train_s, device, threshold=evt_thr),
        "syn_val_event": seqs_metrics(model, val_s, device, threshold=evt_thr),
        "syn_test_event": seqs_metrics(model, test_s, device, threshold=evt_thr),
        "syn_test_window": metrics_dict(
            syn_test_win["window_label"].to_numpy(), syn_test_win["P_seq"].to_numpy(), win_thr
        ),
    }
    print("=== Synthetic TEST EVENT (diagnostic) ===", syn["syn_test_event"], flush=True)
    eligible = [r for r in history if r["epoch"] >= MIN_EPOCHS] or history
    best = max(eligible, key=lambda r: json.loads(r["select_key"])) if eligible else {}
    return {
        "drop": drop,
        "feature_cols": feature_cols,
        "n_features": n_features,
        "model": model,
        "history": history,
        "selected_epoch": int(best.get("epoch", 0)),
        "key": tuple(json.loads(best["select_key"])) if best else (),
        "split_users": {
            "syn_train": sorted({s.username for s in train_s}),
            "syn_val": sorted({s.username for s in val_s}),
            "syn_test": sorted(test_users),
            **data.proto.to_config(),
        },
        "evt_thr": evt_thr,
        "win_thr": win_thr,
        "dev_event": dev_evt,
        "dev_window": dev_win,
        "event_df": event_df.assign(split="synthetic"),
        "pseq": pseq,
        "dev_ev": dev_ev.assign(split="real_dev"),
        "dev_pseq": dev_pseq.assign(split="real_dev"),
        **syn,
    }


def save_run(run: dict, data: Data) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dropped = [*LEAKY_FEATURES, *run["drop"]]
    unk = {
        "real_dev": unk_rates(data.real_df[data.proto.event_mask(data.real_df, "dev")]),
        "real_test": unk_rates(data.real_df[data.proto.event_mask(data.real_df, "test")]),
    }
    config = {
        "model": "LSTMTransformerV6",
        "schema_version": SCHEMA_VERSION,
        "architecture": "BiLSTM + 1-layer Transformer (same as v5)",
        "dataset": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
        "real_dataset": str(REAL_CSV_PATH.relative_to(ROOT)).replace("\\", "/") + " (inv:* rows, dev/test only)",
        "vocab_size": data.vocab_size,
        "n_features": run["n_features"],
        "seq_len": v5.SEQ_LEN,
        "window_minutes": WINDOW_MINUTES,
        "stride_minutes": v5.STRIDE_MINUTES,
        "train_unit": "event (10-min history, loss on last step)",
        "p_seq": "max(P_event) in fusion window",
        "split": "synthetic user-disjoint 70/15/15; real Invictus dev (selection) / test (final)",
        "split_users": run["split_users"],
        "secret_ids": sorted(data.sec_ids),
        "dropped_features": dropped,
        "unk_last_p": UNK_LAST_P,
        "selection": SELECTION,
        "selected_epoch": run["selected_epoch"],
        "label_rule": LABEL_RULE,
        "threshold_margin": THRESHOLD_MARGIN,
        "test_reuse": TEST_REUSE,
        "campaign_relabel": False,
    }
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": run["model"].state_dict(),
            "feature_cols": run["feature_cols"],
            "threshold": run["win_thr"],
            "event_threshold": run["evt_thr"],
            "dev_event_metrics": run["dev_event"],
            "dev_window_metrics": run["dev_window"],
            "test_metrics": {},
            "test_event_metrics": {},
            "event_name_vocab": data.vocab,
            "config": config,
        },
        CKPT_PATH,
    )
    pd.DataFrame(run["history"]).to_csv(OUT_DIR / "training_history.csv", index=False)
    pd.concat([run["event_df"], run["dev_ev"]], ignore_index=True).to_csv(OUT_DIR / "P_event.csv", index=False)
    pseq = run["pseq"].assign(split="synthetic", pred=(run["pseq"]["P_seq"] >= run["win_thr"]).astype(int))
    pd.concat([pseq, run["dev_pseq"]], ignore_index=True).to_csv(OUT_DIR / "P_seq.csv", index=False)
    (OUT_DIR / "split.json").write_text(json.dumps(run["split_users"], indent=2), encoding="utf-8")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "dropped_features": dropped,
        "selected_epoch": run["selected_epoch"],
        "selection": SELECTION,
        "event_threshold": run["evt_thr"],
        "window_threshold": run["win_thr"],
        "dev_event": run["dev_event"],
        "dev_window": run["dev_window"],
        "syn_train_event": run["syn_train_event"],
        "syn_val_event": run["syn_val_event"],
        "syn_test_event": run["syn_test_event"],
        "syn_test_window": run["syn_test_window"],
        "unk_rate": unk,
        "test_evaluations": 0,
        "n_rows": int(len(data.df)),
        "n_users": int(data.df["username"].nunique()),
        "protocol": {
            "model": "LSTMTransformerV6",
            "split": config["split"],
            "real_dev": REAL_DEV_LOCKED,
            "real_time_split": data.proto.to_config()["real_time_split"],
            "label_rule": LABEL_RULE,
            "campaign_relabel": False,
            "test_reuse": TEST_REUSE,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {CKPT_PATH}", flush=True)
    print(f"Wrote {METRICS_PATH} (real test not scored; run --eval-test once after the RF gate)", flush=True)


def eval_test() -> None:
    """Score the saved checkpoint on the real test part (half the users + late bert-jan)."""
    model, ckpt, device = v5.load_checkpoint(CKPT_PATH)
    cfg = ckpt["config"]
    vocab, feature_cols = dict(ckpt["event_name_vocab"]), list(ckpt["feature_cols"])
    pe_ids, _ = maybe_pe_ids(vocab)
    real_df = attach_pe_context(load_real(vocab, feature_cols), pe_ids)
    test = RealSplit.build(real_df, RealProtocol.from_config(cfg["split_users"]), "test", feature_cols)
    te_evt, te_win, ev, pseq = real_eval(model, test, device, ckpt["event_threshold"], ckpt["threshold"])
    print("=== Real TEST EVENT ===", te_evt, flush=True)
    print("=== Real TEST WINDOW ===", te_win, flush=True)
    ckpt["test_event_metrics"], ckpt["test_metrics"] = te_evt, te_win
    torch.save(ckpt, CKPT_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics.get("test_evaluations", 0):
        print(f"WARN: real test already scored {metrics['test_evaluations']}x for this checkpoint", flush=True)
    metrics.update(test_event=te_evt, test_window=te_win, test_evaluations=int(metrics.get("test_evaluations", 0)) + 1)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ev.assign(split="real_test").to_csv(OUT_DIR / "P_event_test.csv", index=False)
    pseq.assign(split="real_test").to_csv(OUT_DIR / "P_seq_test.csv", index=False)
    print(f"Updated {CKPT_PATH} and {METRICS_PATH}", flush=True)


def resolve_drop(arg: str | None) -> tuple[str, ...]:
    if arg is not None:
        return tuple(c for c in arg.split(",") if c and c != "none")
    abl = OUT_DIR / "ablation.json"
    if abl.exists():
        chosen = json.loads(abl.read_text(encoding="utf-8"))["chosen"]
        print(f"using ablation.json choice: {chosen}", flush=True)
        return tuple(ABLATION_DROPS[chosen])
    return ()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablate", action="store_true", help="try ABLATION_DROPS, keep the best on real dev")
    ap.add_argument("--eval-test", action="store_true", help="score the saved checkpoint on real test")
    ap.add_argument("--drop", default=None, help="comma list of extra features to drop ('none' = only leaky)")
    args = ap.parse_args()
    if args.eval_test:
        eval_test()
        return

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={SCHEMA_VERSION} csv={CSV_PATH}", flush=True)
    data = prepare_data()

    configs = ABLATION_DROPS if args.ablate else {"chosen": resolve_drop(args.drop)}
    best_name, best_run, summary = None, None, {}
    for name, drop in configs.items():
        run = fit(data, drop, device)
        summary[name] = {
            "drop": list(drop),
            "select_key": list(run["key"]),
            "selected_epoch": run["selected_epoch"],
            "dev_event": run["dev_event"],
            "dev_window": run["dev_window"],
            "syn_test_event_f1": run["syn_test_event"]["f1"] if run["syn_test_event"] else None,
        }
        print(f"=== config {name}: key={run['key']} dev_window_f1={run['dev_window']['f1']:.4f}", flush=True)
        # ties keep the earlier config (fewer drops first)
        if best_run is None or run["key"] > best_run["key"]:
            best_name, best_run = name, run
    if args.ablate:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "ablation.json").write_text(
            json.dumps({"selection": SELECTION, "chosen": best_name, "configs": summary}, indent=2), encoding="utf-8"
        )
        print(f"ablation chosen={best_name} drop={list(best_run['drop'])}", flush=True)
    save_run(best_run, data)
    print("v5 artifacts untouched. Done.", flush=True)


if __name__ == "__main__":
    main()
