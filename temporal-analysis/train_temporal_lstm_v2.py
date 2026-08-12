"""
Temporal LSTM v2 — real-world-oriented sequence model for privilege escalation.

Fixes vs v1:
- Train on data/lstm/train_temporal.csv (official Invictus + fe-final, union vocab)
- SEQ_LEN=32 (matches short bursts; was 128 with ~97% pad)
- Masked BiLSTM + attention over valid timesteps only
- User-group disjoint train/val/test (no window leakage)
- Checkpoint stores vocab + feature schema for live OOV→UNK
- Primary metric: AUC-PR; threshold tuned on val
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
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "lstm" / "train_temporal.csv"
VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab.json"
MANIFEST_PATH = ROOT / "data" / "lstm" / "manifest.json"
OUT_DIR = ROOT / "artifacts"

SEED = 42
BAG_SEEDS = [42, 43, 44, 45, 46]
WINDOW_MINUTES = 10
STRIDE_MINUTES = 2
SEQ_LEN = 32
EMBED_DIM = 32
HIDDEN_DIM = 64
LSTM_LAYERS = 1
DROPOUT = 0.4
ATTN_DROPOUT = 0.1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 60
PATIENCE = 12
MIN_POSITIVES = 3
LOAO_TOP_K = 3
SCHEMA_VERSION = "temporal_lstm_v2.1"

META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Window:
    username: str
    start: pd.Timestamp
    end: pd.Timestamp
    event_idxs: np.ndarray  # (T,)
    feats: np.ndarray  # (T, F)
    length: int
    label: int
    event_idx_list: list[int]


class WindowDataset(Dataset):
    def __init__(self, windows: list[Window]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        w = self.windows[i]
        return (
            torch.from_numpy(w.event_idxs),
            torch.from_numpy(w.feats),
            torch.tensor(w.length, dtype=torch.long),
            torch.tensor(w.label, dtype=torch.float32),
        )


class TemporalSeqModelV2(nn.Module):
    """Masked BiLSTM + attention pooling over valid timesteps."""

    def __init__(
        self,
        vocab_size: int,
        n_features: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim + n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        attn_in = hidden_dim * 2
        self.attn = nn.Linear(attn_in, 1)
        self.attn_drop = nn.Dropout(ATTN_DROPOUT)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(attn_in, 1)

    def forward(
        self,
        event_idx: torch.Tensor,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        return_attn: bool = False,
    ):
        # event_idx/feats: (B, T, ...), lengths: (B,)
        emb = self.embedding(event_idx)
        x = torch.cat([emb, feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=event_idx.size(1)
        )  # (B, T, 2H)

        scores = self.attn(self.attn_drop(out)).squeeze(-1)  # (B, T)
        T = event_idx.size(1)
        arange = torch.arange(T, device=event_idx.device).unsqueeze(0)
        mask = arange < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        if return_attn:
            return logits, weights
        return logits


def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[str], int, dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run prepare_lstm_dataset.py first.")
    df = pd.read_csv(path)
    assert df.shape[1] == 40, f"Expected 40 cols, got {df.shape}"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["username", "timestamp", "event_name_idx", "label"]:
        assert df[col].isna().sum() == 0, f"NaNs in {col}"
    assert int(df["event_name_idx"].min()) >= 1
    assert int(df["label"].sum()) > 0

    feature_cols = [c for c in df.columns if c not in META_COLS]
    assert len(feature_cols) == 35, f"Expected 35 features, got {len(feature_cols)}"
    vocab_size = int(df["event_name_idx"].max()) + 1

    vocab = {}
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab = json.load(f)

    print("=== Validation PASSED ===")
    print(
        f"shape={df.shape} features={len(feature_cols)} vocab_size={vocab_size} "
        f"users={df['username'].nunique()} pos_events={int(df['label'].sum())}"
    )
    return df, feature_cols, vocab_size, vocab


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_minutes: int = WINDOW_MINUTES,
    stride_minutes: int = STRIDE_MINUTES,
    seq_len: int = SEQ_LEN,
) -> list[Window]:
    """Event-covering stride grid + right-aligned content with left PAD."""
    window_td = pd.Timedelta(minutes=window_minutes)
    stride_td = pd.Timedelta(minutes=stride_minutes)
    stride_ns = int(stride_td / pd.Timedelta(nanoseconds=1))
    window_ns = int(window_td / pd.Timedelta(nanoseconds=1))
    windows: list[Window] = []

    for username, g in df.groupby("username", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            continue
        ts = g["timestamp"]
        starts: set[pd.Timestamp] = set()
        for t in ts:
            aligned = (int(t.value) // stride_ns) * stride_ns
            k = 0
            while k * stride_ns < window_ns:
                starts.add(pd.Timestamp(aligned - k * stride_ns, tz="UTC"))
                k += 1

        for start in sorted(starts):
            end = start + window_td
            chunk = g.loc[(ts >= start) & (ts < end)]
            if chunk.empty:
                continue

            idxs = chunk["event_name_idx"].to_numpy(dtype=np.int64)
            feats = chunk[feature_cols].to_numpy(dtype=np.float32)
            # Extra real-world signal: log1p delta seconds between events (clipped).
            times = chunk["timestamp"].astype("int64").to_numpy()
            deltas = np.diff(times, prepend=times[0]) / 1e9
            deltas = np.clip(deltas, 0, 3600)
            delta_feat = np.log1p(deltas).astype(np.float32).reshape(-1, 1)
            feats = np.concatenate([feats, delta_feat], axis=1)

            y = 1 if int(chunk["label"].max()) == 1 else 0
            raw_len = len(idxs)
            if raw_len > seq_len:
                idxs = idxs[-seq_len:]
                feats = feats[-seq_len:]
                raw_len = seq_len
            pad = seq_len - raw_len
            if pad > 0:
                idxs = np.concatenate([np.zeros(pad, dtype=np.int64), idxs])
                feats = np.concatenate(
                    [np.zeros((pad, feats.shape[1]), dtype=np.float32), feats], axis=0
                )

            windows.append(
                Window(
                    username=str(username),
                    start=start,
                    end=end,
                    event_idxs=idxs,
                    feats=feats,
                    length=raw_len,
                    label=y,
                    event_idx_list=chunk["event_name_idx"].astype(int).tolist(),
                )
            )
    return windows


def group_split(
    windows: list[Window], seed: int = SEED
) -> tuple[list[Window], list[Window], list[Window]]:
    """User-disjoint train/val/test (~70/15/15)."""
    users = np.array([w.username for w in windows])
    labels = np.array([w.label for w in windows])
    idx = np.arange(len(windows))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, hold_idx = next(gss1.split(idx, labels, groups=users))
    hold_users = users[hold_idx]
    hold_labels = labels[hold_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_rel, te_rel = next(gss2.split(hold_idx, hold_labels, groups=hold_users))
    va_idx = hold_idx[va_rel]
    te_idx = hold_idx[te_rel]

    def take(ids):
        return [windows[i] for i in ids]

    train_w, val_w, test_w = take(tr_idx), take(va_idx), take(te_idx)
    # Sanity: disjoint users
    su, sv, st = set(w.username for w in train_w), set(w.username for w in val_w), set(
        w.username for w in test_w
    )
    assert su.isdisjoint(sv) and su.isdisjoint(st) and sv.isdisjoint(st)
    return train_w, val_w, test_w


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, y_all = [], []
    for event_idx, feats, lengths, y in loader:
        event_idx = event_idx.to(device)
        feats = feats.to(device)
        lengths = lengths.to(device)
        logits = model(event_idx, feats, lengths)
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return y_true, probs


@torch.no_grad()
def predict_bag(models: list[nn.Module], loader: DataLoader, device: torch.device):
    all_probs = []
    y_true = None
    for model in models:
        y, p = predict(model, loader, device)
        all_probs.append(p)
        y_true = y
    return y_true, np.mean(np.stack(all_probs, axis=0), axis=0)


def metrics_dict(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    acc = float((preds == y_true).mean()) if len(y_true) else float("nan")
    return {
        "accuracy": acc,
        "auc_pr": float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else float("nan"),
        "auc_roc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def train_one(
    train_w: list[Window],
    val_w: list[Window],
    vocab_size: int,
    n_features: int,
    device: torch.device,
    seed: int,
    verbose: bool = True,
) -> tuple[TemporalSeqModelV2, list[dict]]:
    set_seed(seed)
    train_loader = DataLoader(WindowDataset(train_w), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_w), batch_size=BATCH_SIZE, shuffle=False)

    n_pos = sum(w.label for w in train_w)
    n_neg = len(train_w) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    if verbose:
        print(f"[seed={seed}] train={len(train_w)} pos={n_pos} neg={n_neg} pos_weight={pos_weight.item():.2f}")

    model = TemporalSeqModelV2(vocab_size=vocab_size, n_features=n_features).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_auc = -1.0
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for event_idx, feats, lengths, y in train_loader:
            event_idx = event_idx.to(device)
            feats = feats.to(device)
            lengths = lengths.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(event_idx, feats, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        y_val, p_val = predict(model, val_loader, device)
        val_m = metrics_dict(y_val, p_val)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                **{f"val_{k}": v for k, v in val_m.items()},
            }
        )
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"[seed={seed}] epoch {epoch:03d} loss={history[-1]['train_loss']:.4f} "
                f"val_auc_pr={val_m['auc_pr']:.4f} val_f1={val_m['f1']:.4f} "
                f"val_p={val_m['precision']:.4f} val_r={val_m['recall']:.4f}"
            )

        score = val_m["auc_pr"]
        if not math.isnan(score) and score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[seed={seed}] early stop @ {epoch} best_val_auc_pr={best_auc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_bag(
    train_w, val_w, vocab_size, n_features, device, seeds=BAG_SEEDS
) -> tuple[list[TemporalSeqModelV2], list[dict]]:
    models, history = [], []
    for seed in seeds:
        m, h = train_one(train_w, val_w, vocab_size, n_features, device, seed)
        models.append(m)
        history.extend(h)
    return models, history


def leave_one_attacker_out(
    windows: list[Window],
    vocab_size: int,
    n_features: int,
    device: torch.device,
    attackers: list[str],
) -> dict:
    results = {}
    for held in attackers:
        train_pool = [w for w in windows if w.username != held]
        test_w = [w for w in windows if w.username == held]
        if not test_w or sum(w.label for w in train_pool) < 2:
            results[held] = {"error": "insufficient data"}
            continue
        # Re-split train_pool by groups excluding held user
        try:
            tr, va, _ = group_split(train_pool, seed=SEED)
            if sum(w.label for w in tr) == 0 or len(va) == 0:
                raise ValueError("bad split")
        except Exception:
            # Fallback: 80/20 random on remaining
            idx = np.arange(len(train_pool))
            rng = np.random.RandomState(SEED)
            rng.shuffle(idx)
            cut = int(0.8 * len(idx))
            tr = [train_pool[i] for i in idx[:cut]]
            va = [train_pool[i] for i in idx[cut:]]

        print(f"\n=== LOAO held={held} train={len(tr)} val={len(va)} test={len(test_w)} ===")
        models, _ = train_bag(tr, va, vocab_size, n_features, device, seeds=BAG_SEEDS[:3])
        y_va, p_va = predict_bag(models, DataLoader(WindowDataset(va), batch_size=BATCH_SIZE), device)
        thr = tune_threshold(y_va, p_va) if y_va.sum() > 0 else 0.5
        y, p = predict_bag(models, DataLoader(WindowDataset(test_w), batch_size=BATCH_SIZE), device)
        m = metrics_dict(y, p, threshold=thr)
        results[held] = m
        print(f"LOAO {held}: {m}")
    return results


def export_p_seq(models, windows, device, path: Path, threshold: float) -> pd.DataFrame:
    loader = DataLoader(WindowDataset(windows), batch_size=BATCH_SIZE, shuffle=False)
    _, probs = predict_bag(models, loader, device)
    rows = []
    for w, p in zip(windows, probs):
        rows.append(
            {
                "username": w.username,
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "window_label": w.label,
                "raw_len": w.length,
                "P_seq": float(p),
                "pred": int(p >= threshold),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def evaluate_gates(test_m: dict, loao: dict, transfer_auc_pr: float = 0.186) -> dict:
    auc = test_m.get("auc_pr", float("nan"))
    prec = test_m.get("precision", 0.0)
    rec = test_m.get("recall", 0.0)
    g2 = (not math.isnan(auc)) and auc > max(0.35, transfer_auc_pr)
    g3 = rec >= 0.70 and prec >= 0.35
    g4 = False
    for m in loao.values():
        if isinstance(m, dict) and "error" not in m:
            if (not math.isnan(m.get("auc_pr", float("nan")))) and m["auc_pr"] > 0.5 and m.get("recall", 0) > 0:
                g4 = True
                break
    return {
        "G2_ranking": g2,
        "G3_detection": g3,
        "G4_loao": g4,
        "passed_core": bool(g2),
        "test_auc_pr": auc,
        "test_precision": prec,
        "test_recall": rec,
        "notes": "G1/G5/G6 enforced by code (schema in ckpt, OOV infer script, user-disjoint split).",
    }


def main() -> None:
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(
        f"v2 config: T={SEQ_LEN} BiLSTM(h={HIDDEN_DIM}) attn bag={BAG_SEEDS} "
        f"window={WINDOW_MINUTES}m stride={STRIDE_MINUTES}m"
    )

    df, feature_cols, vocab_size, vocab = load_and_validate(CSV_PATH)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        print(
            f"manifest: rows={manifest.get('n_rows')} "
            f"fe={manifest.get('n_rows_fe_final')} inv={manifest.get('n_rows_invictus')} "
            f"vocab_size={manifest.get('vocab_size')} "
            f"extra_names={manifest.get('n_invictus_only_event_names')}"
        )
        print(f"notes: {manifest.get('notes', '')}")
    # +1 delta-time feature appended in build_windows
    n_features = len(feature_cols) + 1
    feature_cols_ext = feature_cols + ["delta_t_log1p"]

    windows = build_windows(df, feature_cols)
    n_pos = sum(w.label for w in windows)
    print(f"windows={len(windows)} pos={n_pos} neg={len(windows) - n_pos}")
    if n_pos < MIN_POSITIVES:
        raise SystemExit(f"Abort: only {n_pos} positive windows")

    train_w, val_w, test_w = group_split(windows)
    print(
        f"split users: train={len({w.username for w in train_w})} "
        f"val={len({w.username for w in val_w})} test={len({w.username for w in test_w})}"
    )
    print(
        f"split windows: train={len(train_w)}/{sum(w.label for w in train_w)}pos "
        f"val={len(val_w)}/{sum(w.label for w in val_w)}pos "
        f"test={len(test_w)}/{sum(w.label for w in test_w)}pos"
    )

    models, history = train_bag(train_w, val_w, vocab_size, n_features, device)
    val_loader = DataLoader(WindowDataset(val_w), batch_size=BATCH_SIZE, shuffle=False)
    y_val, p_val = predict_bag(models, val_loader, device)
    threshold = tune_threshold(y_val, p_val) if y_val.sum() > 0 else 0.5
    val_m = metrics_dict(y_val, p_val, threshold=threshold)
    print("\n=== VAL ===")
    print(json.dumps(val_m, indent=2))

    test_loader = DataLoader(WindowDataset(test_w), batch_size=BATCH_SIZE, shuffle=False)
    y_te, p_te = predict_bag(models, test_loader, device)
    test_m = metrics_dict(y_te, p_te, threshold=threshold)
    print("\n=== TEST (user-disjoint) ===")
    print(json.dumps(test_m, indent=2))

    pos_mean = float(p_te[y_te == 1].mean()) if (y_te == 1).any() else float("nan")
    neg_mean = float(p_te[y_te == 0].mean()) if (y_te == 0).any() else float("nan")
    print(f"test score gap pos-neg = {pos_mean - neg_mean:.4f}")

    # LOAO on full window set (expensive but needed for G4)
    counts = Counter(w.username for w in windows if w.label == 1)
    loao_users = [u for u, _ in counts.most_common(LOAO_TOP_K)]
    print(f"LOAO users: {loao_users}")
    loao = leave_one_attacker_out(windows, vocab_size, n_features, device, loao_users)
    gates = evaluate_gates(test_m, loao)

    ckpt_path = OUT_DIR / "temporal_lstm_v2.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dicts": [m.state_dict() for m in models],
            "seeds": BAG_SEEDS,
            "feature_cols": feature_cols_ext,
            "base_feature_cols": feature_cols,
            "event_name_vocab": vocab,
            "threshold": threshold,
            "config": {
                "vocab_size": vocab_size,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "lstm_layers": LSTM_LAYERS,
                "bidirectional": True,
                "dropout": DROPOUT,
                "seq_len": SEQ_LEN,
                "window_minutes": WINDOW_MINUTES,
                "stride_minutes": STRIDE_MINUTES,
                "n_features": n_features,
                "label_rule": "any(label==1)",
                "split": "GroupShuffleSplit user-disjoint 70/15/15",
                "architecture": "Masked BiLSTM + attention pool",
            },
            "val_metrics": val_m,
            "test_metrics": test_m,
            "loao_metrics": loao,
            "gates": gates,
        },
        ckpt_path,
    )
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history_v2.csv", index=False)
    with open(OUT_DIR / "test_metrics_v2.json", "w", encoding="utf-8") as f:
        json.dump(
            {"val": val_m, "test": test_m, "loao": loao, "gates": gates},
            f,
            indent=2,
        )
    pseq = export_p_seq(models, windows, device, OUT_DIR / "P_seq_v2.csv", threshold)

    print(f"\nWrote {ckpt_path}")
    print(f"Wrote {OUT_DIR / 'P_seq_v2.csv'} ({len(pseq)} rows)")
    print("\n=== GATES ===")
    print(json.dumps(gates, indent=2))
    if gates["passed_core"]:
        print("CORE GATE PASSED (G2 ranking)")
    else:
        print("CORE GATE NOT PASSED — see FIX_PLAN Phase D (more attack chains).")
    print("Done.")


if __name__ == "__main__":
    main()
