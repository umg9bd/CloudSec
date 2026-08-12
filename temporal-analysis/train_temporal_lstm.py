"""
Temporal LSTM on prepared CloudTrail temporal CSV (improved limited-data recipe)
- Default data: data/lstm/train_temporal.csv (from CloudSec fe-final)
- Per-user 10-min windows (stride 2 min), T=128
- Window label = 1 if any(label==1) in window
- Bag of 5 small LSTMs; P_seq = mean(sigmoid)
- Val F1 threshold tune; LOAO gate
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "lstm" / "manifest.json"
CSV_PATH = ROOT / "data" / "lstm" / "train_temporal.csv"
OUT_DIR = ROOT / "artifacts"
SEED = 42
BAG_SEEDS = [42, 43, 44, 45, 46]
WINDOW_MINUTES = 10
STRIDE_MINUTES = 2
SEQ_LEN = 128
EMBED_DIM = 32
HIDDEN_DIM = 64
DROPOUT = 0.5
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 80
PATIENCE = 15
VOCAB_SIZE = 68  # overwritten from data/manifest (PAD=0)
MIN_POSITIVES = 3
LOAO_TOP_K = 3

BASELINE_METRICS = {
    "test_auc_pr": 0.7,
    "loao_bert_jan_recall": 0.0,
    "loao_bert_jan_auc_pr": 0.7713789682539682,
    "note": "prior soft-session_label baseline (T=32, stride=5, single model)",
}

META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx"}
# Fallback only; LOAO attackers are inferred from positive windows when possible.
ATTACKERS = {
    "bert-jan",
    "stratus-red-team-ec2-get-password-data-role",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Step 0: Validate ────────────────────────────────────────────────────────
def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[str]]:
    global VOCAB_SIZE
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run: python prepare_lstm_dataset.py"
        )

    df = pd.read_csv(path)
    assert df.shape[1] == 40, f"Expected 40 columns, got {df.shape}"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["username", "timestamp", "event_name_idx", "label"]:
        assert df[col].isna().sum() == 0, f"NaNs in {col}"
    assert df["event_name_idx"].min() >= 1
    assert int(df["label"].sum()) > 0

    feature_cols = [c for c in df.columns if c not in META_COLS]
    assert len(feature_cols) == 35, f"Expected 35 features, got {len(feature_cols)}"

    VOCAB_SIZE = int(df["event_name_idx"].max()) + 1
    manifest = load_manifest()
    if manifest.get("vocab_size"):
        VOCAB_SIZE = max(VOCAB_SIZE, int(manifest["vocab_size"]))

    print("=== Step 0: Validation PASSED ===")
    print(f"shape={df.shape}, features={len(feature_cols)}, vocab_size={VOCAB_SIZE}")
    print(f"label counts:\n{df['label'].value_counts().to_string()}")
    print(
        f"usernames={df['username'].nunique()}, "
        f"time=[{df['timestamp'].min()} -> {df['timestamp'].max()}]"
    )
    print(
        f"event_name_idx nunique/min/max="
        f"{df['event_name_idx'].nunique()}/{df['event_name_idx'].min()}/{df['event_name_idx'].max()}"
    )

    top_pos = (
        df.loc[df["label"] == 1, "username"]
        .value_counts()
        .head(5)
        .to_dict()
    )
    print(f"top positive usernames: {top_pos}")
    return df, feature_cols


def infer_loao_attackers(windows: list[Window], top_k: int = LOAO_TOP_K) -> list[str]:
    counts: Counter[str] = Counter()
    for w in windows:
        if w.label == 1:
            counts[w.username] += 1
    if not counts:
        return [a for a in sorted(ATTACKERS) if any(w.username == a for w in windows)]
    return [u for u, _ in counts.most_common(top_k)]


# ── Step 2–3: Windowing ─────────────────────────────────────────────────────
@dataclass
class Window:
    username: str
    start: pd.Timestamp
    end: pd.Timestamp
    event_idxs: np.ndarray  # (T,)
    feats: np.ndarray  # (T, F)
    label: int
    raw_len: int
    event_idx_list: list[int]  # unpadded, for attribution


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_minutes: int = WINDOW_MINUTES,
    stride_minutes: int = STRIDE_MINUTES,
    seq_len: int = SEQ_LEN,
) -> list[Window]:
    """Window positive iff it contains at least one event with label==1.

    Uses an event-covering stride grid so sparse year-long users do not
    expand into millions of empty window checks.
    """
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
            t_ns = int(t.value)
            aligned = (t_ns // stride_ns) * stride_ns
            k = 0
            while k * stride_ns < window_ns:
                starts.add(pd.Timestamp(aligned - k * stride_ns, tz="UTC"))
                k += 1

        for start in sorted(starts):
            end = start + window_td
            mask = (ts >= start) & (ts < end)
            chunk = g.loc[mask]
            if len(chunk) == 0:
                continue
            idxs = chunk["event_name_idx"].to_numpy(dtype=np.int64)
            feats = chunk[feature_cols].to_numpy(dtype=np.float32)
            y = 1 if int(chunk["label"].max()) == 1 else 0
            raw_len = len(idxs)

            if raw_len > seq_len:
                idxs = idxs[-seq_len:]
                feats = feats[-seq_len:]
            pad = seq_len - len(idxs)
            if pad > 0:
                idxs = np.concatenate([np.zeros(pad, dtype=np.int64), idxs])
                feats = np.concatenate(
                    [np.zeros((pad, feats.shape[1]), dtype=np.float32), feats],
                    axis=0,
                )

            windows.append(
                Window(
                    username=str(username),
                    start=start,
                    end=end,
                    event_idxs=idxs,
                    feats=feats,
                    label=y,
                    raw_len=raw_len,
                    event_idx_list=chunk["event_name_idx"].astype(int).tolist(),
                )
            )

    return windows


# ── Dataset ──────────────────────────────────────────────────────────────────
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
            torch.tensor(w.label, dtype=torch.float32),
        )


# ── Model ────────────────────────────────────────────────────────────────────
class TemporalSeqModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        n_features: int = 35,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + n_features, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, event_idx: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.embedding(event_idx), feats], dim=-1)
        _, (h, _) = self.lstm(x)
        return self.head(self.dropout(h.squeeze(0))).squeeze(-1)


# ── Train / Eval ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, y_all = [], []
    for event_idx, feats, y in loader:
        event_idx = event_idx.to(device)
        feats = feats.to(device)
        logits = model(event_idx, feats)
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return y_true, probs


@torch.no_grad()
def predict_bag(models: list[nn.Module], loader: DataLoader, device: torch.device):
    """Average sigmoid probs across bag members."""
    all_probs = []
    y_true = None
    for model in models:
        y, p = predict(model, loader, device)
        all_probs.append(p)
        y_true = y
    probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    return y_true, probs


def metrics_dict(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    out = {
        "auc_pr": float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else float("nan"),
        "auc_roc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }
    return out


def tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on validation."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t


def train_model(
    train_windows: list[Window],
    val_windows: list[Window],
    n_features: int,
    device: torch.device,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[TemporalSeqModel, list[dict]]:
    set_seed(seed)
    train_loader = DataLoader(WindowDataset(train_windows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(val_windows), batch_size=BATCH_SIZE, shuffle=False)

    n_pos = sum(w.label for w in train_windows)
    n_neg = len(train_windows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    if verbose:
        print(
            f"[seed={seed}] train={len(train_windows)} pos={n_pos} neg={n_neg} "
            f"pos_weight={pos_weight.item():.3f}"
        )

    model = TemporalSeqModel(vocab_size=VOCAB_SIZE, n_features=n_features).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_auc_pr = -1.0
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for event_idx, feats, y in train_loader:
            event_idx = event_idx.to(device)
            feats = feats.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(event_idx, feats)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        y_val, p_val = predict(model, val_loader, device)
        val_m = metrics_dict(y_val, p_val)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)
        if verbose:
            print(
                f"[seed={seed}] epoch {epoch:03d} loss={row['train_loss']:.4f} "
                f"val_auc_pr={val_m['auc_pr']:.4f} val_f1={val_m['f1']:.4f} "
                f"val_p={val_m['precision']:.4f} val_r={val_m['recall']:.4f}"
            )

        score = val_m["auc_pr"]
        if not math.isnan(score) and score > best_auc_pr:
            best_auc_pr = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[seed={seed}] Early stop @ {epoch} (best val AUC-PR={best_auc_pr:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_bag(
    train_windows: list[Window],
    val_windows: list[Window],
    n_features: int,
    device: torch.device,
    seeds: list[int] = BAG_SEEDS,
) -> tuple[list[TemporalSeqModel], list[dict]]:
    models: list[TemporalSeqModel] = []
    all_history: list[dict] = []
    for seed in seeds:
        model, hist = train_model(
            train_windows, val_windows, n_features, device, seed=seed, verbose=True
        )
        models.append(model)
        all_history.extend(hist)
    return models, all_history


def stratified_split(windows: list[Window], seed: int = SEED):
    """70/30 train/test; validation = 20% held out from the 70% train pool."""
    labels = [w.label for w in windows]
    idx = np.arange(len(windows))

    def _split(ids, labs, test_size):
        try:
            if labs is not None and sum(labs) >= 2 and (len(labs) - sum(labs)) >= 2:
                return train_test_split(
                    ids, test_size=test_size, random_state=seed, stratify=labs
                )
        except ValueError:
            pass
        return train_test_split(ids, test_size=test_size, random_state=seed)

    # 70% train_pool / 30% test
    train_pool_i, test_i = _split(idx, labels, test_size=0.3)
    pool_labels = [labels[i] for i in train_pool_i]
    # Val from train pool (20% of the 70% → ~14% overall)
    train_i, val_i = _split(train_pool_i, pool_labels, test_size=0.2)

    def take(ids):
        return [windows[i] for i in ids]

    return take(train_i), take(val_i), take(test_i)


def export_p_seq(
    models: list[nn.Module],
    windows: list[Window],
    device: torch.device,
    out_path: Path,
    threshold: float,
) -> pd.DataFrame:
    loader = DataLoader(WindowDataset(windows), batch_size=BATCH_SIZE, shuffle=False)
    _, probs = predict_bag(models, loader, device)
    rows = []
    for w, p in zip(windows, probs):
        top_events = Counter(w.event_idx_list).most_common(5)
        rows.append(
            {
                "username": w.username,
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "window_label": w.label,
                "raw_len": w.raw_len,
                "P_seq": float(p),
                "pred": int(p >= threshold),
                "top_event_name_idx": json.dumps(top_events),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def leave_one_attacker_out(
    windows: list[Window],
    n_features: int,
    device: torch.device,
    attackers: list[str] | None = None,
) -> dict:
    held_users = attackers if attackers is not None else sorted(ATTACKERS)
    results = {}
    for held_out in held_users:
        train_pool = [w for w in windows if w.username != held_out]
        test_w = [w for w in windows if w.username == held_out]
        if not test_w:
            results[held_out] = {"error": "no windows"}
            continue

        labels = [w.label for w in train_pool]
        idx = np.arange(len(train_pool))
        try:
            tr_i, va_i = train_test_split(
                idx, test_size=0.2, random_state=SEED, stratify=labels
            )
        except ValueError:
            tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=SEED)
        tr = [train_pool[i] for i in tr_i]
        va = [train_pool[i] for i in va_i]

        print(f"\n=== LOAO held_out={held_out} train={len(tr)} val={len(va)} test={len(test_w)} ===")
        models, _ = train_bag(tr, va, n_features, device, seeds=BAG_SEEDS)
        val_loader = DataLoader(WindowDataset(va), batch_size=BATCH_SIZE, shuffle=False)
        y_val, p_val = predict_bag(models, val_loader, device)
        thr = tune_threshold(y_val, p_val) if y_val.sum() > 0 else 0.5

        y, p = predict_bag(
            models, DataLoader(WindowDataset(test_w), batch_size=BATCH_SIZE), device
        )
        m = metrics_dict(y, p, threshold=thr)
        results[held_out] = m
        print(f"LOAO {held_out}: {m}")
    return results


def evaluate_gate(test_m: dict, loao: dict) -> dict:
    test_auc = test_m.get("auc_pr", float("nan"))
    beat_auc = (not math.isnan(test_auc)) and test_auc > BASELINE_METRICS["test_auc_pr"]

    loao_ok_any = False
    for _name, m in loao.items():
        if not isinstance(m, dict) or "error" in m:
            continue
        auc = m.get("auc_pr", float("nan"))
        recall = m.get("recall", 0.0)
        if (not math.isnan(auc)) and auc > 0.5 and recall > 0.0:
            loao_ok_any = True
            break

    passed = bool(beat_auc and loao_ok_any)
    ceiling_note = None
    if beat_auc and not loao_ok_any:
        ceiling_note = (
            "Stratified test beat baseline, but LOAO thresholded recall stays 0 for "
            "held-out attackers. Ranking AUC-PR/ROC may still be >0.5; transferable "
            "attack patterns may need more diverse attack chains."
        )

    return {
        "passed": passed,
        "beat_test_auc_pr": beat_auc,
        "loao_ok_any": loao_ok_any,
        "bert_jan_loao_ok": loao_ok_any,  # backward-compatible key
        "data_ceiling": ceiling_note is not None,
        "ceiling_note": ceiling_note,
        "baseline": BASELINE_METRICS,
        "improved": {
            "test_auc_pr": test_auc,
            "test_f1": test_m.get("f1"),
            "test_recall": test_m.get("recall"),
            "test_precision": test_m.get("precision"),
            "threshold": test_m.get("threshold"),
            "loao": loao,
        },
    }


def main():
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(
        f"config: window={WINDOW_MINUTES}m stride={STRIDE_MINUTES}m T={SEQ_LEN} "
        f"dropout={DROPOUT} wd={WEIGHT_DECAY} bag={BAG_SEEDS}"
    )

    df, feature_cols = load_and_validate(CSV_PATH)
    windows = build_windows(df, feature_cols)
    labels = [w.label for w in windows]
    lengths = [w.raw_len for w in windows]
    n_pos = sum(labels)
    print("\n=== Window stats ===")
    print(f"n_windows={len(windows)} pos={n_pos} neg={len(labels) - n_pos}")
    print(
        f"raw_len min/median/max={min(lengths)}/{int(np.median(lengths))}/{max(lengths)} "
        f"pct_over_T={100 * np.mean(np.array(lengths) > SEQ_LEN):.1f}%"
    )
    if n_pos < MIN_POSITIVES:
        raise SystemExit(f"Abort: only {n_pos} positive windows (need >= {MIN_POSITIVES})")

    train_w, val_w, test_w = stratified_split(windows)
    n_all = len(windows)
    print(
        f"split train/val/test = {len(train_w)}/{len(val_w)}/{len(test_w)} "
        f"({100*len(train_w)/n_all:.1f}%/{100*len(val_w)/n_all:.1f}%/{100*len(test_w)/n_all:.1f}%) "
        f"[70/30 train/test; val from train]"
    )
    print(
        f"pos rates: train={np.mean([w.label for w in train_w]):.3f} "
        f"val={np.mean([w.label for w in val_w]):.3f} "
        f"test={np.mean([w.label for w in test_w]):.3f}"
    )

    models, history = train_bag(train_w, val_w, n_features=len(feature_cols), device=device)

    # Tune threshold on validation bag probs
    val_loader = DataLoader(WindowDataset(val_w), batch_size=BATCH_SIZE, shuffle=False)
    y_val, p_val = predict_bag(models, val_loader, device)
    threshold = tune_threshold(y_val, p_val) if y_val.sum() > 0 else 0.5
    print(f"\nVal-tuned threshold={threshold:.3f}")

    test_loader = DataLoader(WindowDataset(test_w), batch_size=BATCH_SIZE, shuffle=False)
    y_test, p_test = predict_bag(models, test_loader, device)
    test_m = metrics_dict(y_test, p_test, threshold=threshold)
    print("\n=== Test metrics (bagged) ===")
    print(test_m)

    # Save bag weights
    weights_path = OUT_DIR / "temporal_lstm.pt"
    torch.save(
        {
            "state_dicts": [m.state_dict() for m in models],
            "seeds": BAG_SEEDS,
            "feature_cols": feature_cols,
            "threshold": threshold,
            "config": {
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "weight_decay": WEIGHT_DECAY,
                "seq_len": SEQ_LEN,
                "window_minutes": WINDOW_MINUTES,
                "stride_minutes": STRIDE_MINUTES,
                "n_features": len(feature_cols),
                "label_rule": "any(label==1)",
                "bag_seeds": BAG_SEEDS,
            },
            "test_metrics": test_m,
        },
        weights_path,
    )
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    pseq_df = export_p_seq(models, windows, device, OUT_DIR / "P_seq.csv", threshold)
    print(f"Wrote {weights_path}")
    print(f"Wrote {OUT_DIR / 'P_seq.csv'} ({len(pseq_df)} rows)")

    # Separation diagnostics
    pos_scores = pseq_df.loc[pseq_df["window_label"] == 1, "P_seq"]
    neg_scores = pseq_df.loc[pseq_df["window_label"] == 0, "P_seq"]
    print(
        f"score gap: pos_mean={pos_scores.mean():.3f} neg_mean={neg_scores.mean():.3f} "
        f"gap={pos_scores.mean() - neg_scores.mean():.3f}"
    )

    loao_users = infer_loao_attackers(windows)
    print(f"LOAO users: {loao_users}")
    loao = leave_one_attacker_out(
        windows,
        n_features=len(feature_cols),
        device=device,
        attackers=loao_users,
    )
    compare = evaluate_gate(test_m, loao)

    with open(OUT_DIR / "loao_metrics.json", "w", encoding="utf-8") as f:
        json.dump(loao, f, indent=2)
    with open(OUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_m, f, indent=2)
    with open(OUT_DIR / "compare_baseline.json", "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)

    print("\n=== Gate vs baseline ===")
    print(json.dumps(compare, indent=2))
    if compare["passed"]:
        print("GATE PASSED")
    else:
        print("GATE NOT PASSED — data ceiling may still limit LOAO; artifacts saved for review.")

    print("\nDone.")


if __name__ == "__main__":
    main()
