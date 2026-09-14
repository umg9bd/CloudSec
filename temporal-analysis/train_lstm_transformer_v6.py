"""
LSTM + Transformer v6 — general-purpose (user-disjoint split).

Does NOT modify v5 trainer, weights, or data/lstm/event_name_vocab.json.

Data: data/lstm/cloudtrail_temporal_final.csv
Split: GroupShuffleSplit by username (~70/15/15). No locked bert-jan/stratus.
Labels: CSV labels only (no Invictus campaign relabel).
Architecture: same BiLSTM + 1-layer Transformer as v5 (imported, not copied-and-forked in v5 file).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_lstm_transformer as v5
from train_lstm_transformer import (
    BATCH_SIZE,
    EventDataset,
    LSTMTransformerModel,
    WINDOW_MINUTES,
    attach_pe_context,
    build_event_sequences,
    build_fusion_windows,
    metrics_dict,
    predict,
    score_seqs,
    tune_threshold,
    window_scores,
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "lstm" / "cloudtrail_temporal_final.csv"
V5_VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab.json"
VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab_v6.json"
OUT_DIR = ROOT / "artifacts" / "lstm_transformer_v6"

SEED = 42
SCHEMA_VERSION = "lstm_transformer_v6.0"
META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx"}
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
    """Final CSV currently duplicates many log_ids (twin rows). Keep last."""
    n0 = len(df)
    out = df.drop_duplicates(subset=["log_id"], keep="last").reset_index(drop=True)
    print(f"dedupe log_id: {n0} -> {len(out)} rows", flush=True)
    return out


def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[str], int]:
    df = pd.read_csv(path)
    assert df.shape[1] == 40, df.shape
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ["username", "timestamp", "event_name_idx", "label"]:
        assert df[col].isna().sum() == 0, col
    assert int(df["event_name_idx"].min()) >= 1
    df["username"] = df["username"].astype(str)
    df["log_id"] = df["log_id"].astype(str)
    df = dedupe_events(df)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    assert len(feature_cols) == 35, len(feature_cols)
    vocab_size = int(df["event_name_idx"].max()) + 1
    print("=== Validation PASSED (v6 general) ===", flush=True)
    print(
        f"shape={df.shape} features={len(feature_cols)} vocab_size={vocab_size} "
        f"users={df['username'].nunique()} attacks={int(df['label'].sum())}",
        flush=True,
    )
    return df, feature_cols, vocab_size


def write_vocab_v6(vocab_size: int) -> dict[str, int]:
    vocab: dict[str, int] = {"<UNK>": 0}
    if V5_VOCAB_PATH.exists():
        src = json.loads(V5_VOCAB_PATH.read_text(encoding="utf-8"))
        for k, v in src.items():
            iv = int(v)
            if 0 <= iv < vocab_size:
                vocab[str(k)] = iv
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {VOCAB_PATH} entries={len(vocab)} (v5 file not modified)", flush=True)
    return vocab


def maybe_pe_ids(vocab: dict[str, int]) -> tuple[set[int], set[int]]:
    try:
        pe, sec, _extra = v5.vocab_id_sets(vocab)
        return pe, sec
    except SystemExit:
        print("WARN: vocab missing PE/secret names — skip PE context + empty secrets head", flush=True)
        return set(), set()


def group_split_users(seqs, seed: int = SEED):
    """~70/15/15 by username. No locked test attacker."""
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
        f"users train/val/test={len(tr_u)}/{len(va_u)}/{len(te_u)} "
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


def train_model(train_s, val_s, vocab_size, n_features, device, risk_idx=None, secret_ids=None):
    set_seed(SEED)
    secret_ids = secret_ids or set()
    train_loader = make_loader(train_s, weighted=True, secret_ids=secret_ids)
    n_pos = sum(s.label for s in train_s)
    n_neg = len(train_s) - n_pos
    pos_weight = torch.tensor(
        [math.sqrt(n_neg / max(n_pos, 1))], dtype=torch.float32, device=device
    )
    print(
        f"train_events={len(train_s)} pos={n_pos} neg={n_neg} pos_weight(sqrt)={pos_weight.item():.3f}",
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
        train_ap = tr_m["auc_pr"] if tr_m else float("nan")
        val_ap = val_m["auc_pr"] if val_m else float("nan")
        score = val_m["f1"] if val_m else val_ap
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "lr": optim.param_groups[0]["lr"],
                "train_auc_pr": train_ap,
                "val_auc_pr": val_ap,
                "select_score": score,
                **{f"val_{k}": v for k, v in (val_m or {}).items()},
            }
        )
        print(
            f"epoch {epoch:03d} loss={history[-1]['train_loss']:.4f} "
            f"train_ap={train_ap:.4f} val_ap={val_ap:.4f} val_f1={score:.4f} "
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


def main() -> None:
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print("model=LSTMTransformerV6 (general user-disjoint, CSV labels)", flush=True)
    print(f"csv={CSV_PATH}", flush=True)

    df, feature_cols, vocab_size = load_and_validate(CSV_PATH)
    vocab = write_vocab_v6(vocab_size)
    pe_ids, sec_ids = maybe_pe_ids(vocab)
    if pe_ids:
        df = attach_pe_context(df, pe_ids)
        feature_cols = feature_cols + ["pe_write_recent", "log_secs_since_pe"]
        print("PE context attached (no campaign relabel)", flush=True)

    seqs = build_event_sequences(df, feature_cols)
    n_features = seqs[0].feats.shape[1]
    print(
        f"event_seqs={len(seqs)} pos={sum(s.label for s in seqs)} n_features={n_features}",
        flush=True,
    )

    train_s, val_s, test_s = group_split_users(seqs)
    risk_idx = feature_cols.index("action_risk_prior") if "action_risk_prior" in feature_cols else None

    model, history = train_model(
        train_s, val_s, vocab_size, n_features, device, risk_idx=risk_idx, secret_ids=sec_ids
    )
    y_val, p_val = predict(model, make_loader(val_s), device)
    threshold = tune_threshold(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.5
    print(f"val-tuned event threshold={threshold:.3f}", flush=True)

    train_evt = seqs_metrics(model, train_s, device, threshold=threshold)
    val_evt = seqs_metrics(model, val_s, device, threshold=threshold)
    test_evt = seqs_metrics(model, test_s, device, threshold=threshold)
    print(f"=== Train EVENT @ {threshold:.3f} ===", train_evt, flush=True)
    print(f"=== Val EVENT @ {threshold:.3f} ===", val_evt, flush=True)
    print(f"=== Test EVENT (held-out users) @ {threshold:.3f} ===", test_evt, flush=True)

    event_df = score_seqs(model, seqs, device)
    fusion_rows = build_fusion_windows(df)
    pseq = window_scores(fusion_rows, event_df, threshold)

    test_users = {s.username for s in test_s}
    val_users = {s.username for s in val_s}
    test_win = pseq[pseq["username"].isin(test_users)]
    val_win = pseq[pseq["username"].isin(val_users)]
    win_thr = (
        tune_threshold(val_win["window_label"].to_numpy(), val_win["P_seq"].to_numpy())
        if len(val_win) and int(val_win["window_label"].sum()) > 0
        else threshold
    )
    test_win_m = metrics_dict(
        test_win["window_label"].to_numpy(), test_win["P_seq"].to_numpy(), win_thr
    )
    print(f"val-tuned WINDOW threshold={win_thr:.3f}", flush=True)
    print("=== Test WINDOW (P_seq = max P_event, held-out users) ===", test_win_m, flush=True)

    ckpt_path = OUT_DIR / "temporal_lstm_transformer_v6.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "threshold": win_thr,
            "event_threshold": threshold,
            "test_metrics": test_win_m,
            "test_event_metrics": test_evt,
            "event_name_vocab": vocab,
            "config": {
                "model": "LSTMTransformerV6",
                "architecture": "BiLSTM + 1-layer Transformer (same as v5)",
                "dataset": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
                "vocab_size": vocab_size,
                "n_features": n_features,
                "seq_len": v5.SEQ_LEN,
                "window_minutes": WINDOW_MINUTES,
                "stride_minutes": v5.STRIDE_MINUTES,
                "train_unit": "event (10-min history, loss on last step)",
                "p_seq": "max(P_event) in fusion window",
                "split": "GroupShuffleSplit user-disjoint 70/15/15",
                "secret_ids": sorted(sec_ids),
                "campaign_relabel": False,
                "locked_test_user": None,
            },
        },
        ckpt_path,
    )
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    event_df.to_csv(OUT_DIR / "P_event.csv", index=False)
    pseq["pred"] = (pseq["P_seq"] >= win_thr).astype(int)
    pseq.to_csv(OUT_DIR / "P_seq.csv", index=False)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "train_event": train_evt,
        "val_event": val_evt,
        "test_event": test_evt,
        "test_window": test_win_m,
        "event_threshold": threshold,
        "window_threshold": win_thr,
        "n_rows_deduped": int(len(df)),
        "n_users": int(df["username"].nunique()),
        "protocol": {
            "model": "LSTMTransformerV6",
            "split": "user-disjoint 70/15/15",
            "test_attacker": None,
            "campaign_relabel": False,
        },
    }
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {ckpt_path}", flush=True)
    print(f"Wrote {OUT_DIR / 'test_metrics.json'}", flush=True)
    print("v5 artifacts untouched.", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
