"""
Validate the trained temporal LSTM bag on the new fe-final dataset.

Loads artifacts/temporal_lstm.pt (trained on invictus_temporal) and evaluates
on data/lstm/train_temporal.csv without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_temporal_lstm import (
    BATCH_SIZE,
    OUT_DIR,
    ROOT,
    TemporalSeqModel,
    WindowDataset,
    build_windows,
    load_and_validate,
    metrics_dict,
    predict_bag,
    tune_threshold,
)

CHECKPOINT = OUT_DIR / "temporal_lstm.pt"
NEW_CSV = ROOT / "data" / "lstm" / "train_temporal.csv"
OLD_METRICS = OUT_DIR / "test_metrics.json"
OUT_JSON = OUT_DIR / "validate_new_data_metrics.json"
OUT_CSV = OUT_DIR / "P_seq_new_data.csv"
SEED = 42


def load_models(ckpt: dict, device: torch.device) -> list[TemporalSeqModel]:
    cfg = ckpt["config"]
    models: list[TemporalSeqModel] = []
    for state in ckpt["state_dicts"]:
        model = TemporalSeqModel(
            vocab_size=int(cfg["vocab_size"]),
            embed_dim=int(cfg["embed_dim"]),
            n_features=int(cfg["n_features"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def export_scores(
    windows,
    probs: np.ndarray,
    threshold: float,
    out_path: Path,
) -> pd.DataFrame:
    rows = []
    for w, p in zip(windows, probs):
        rows.append(
            {
                "username": w.username,
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "window_label": w.label,
                "raw_len": w.raw_len,
                "P_seq": float(p),
                "pred_saved_thr": int(p >= threshold),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"Missing checkpoint: {CHECKPOINT}")
    if not NEW_CSV.exists():
        raise SystemExit(f"Missing dataset: {NEW_CSV}. Run prepare_lstm_dataset.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    saved_thr = float(ckpt.get("threshold", 0.5))
    train_feature_cols = list(ckpt.get("feature_cols", []))

    print("=== Checkpoint ===")
    print(f"path={CHECKPOINT}")
    print(f"trained_vocab_size={ckpt['config']['vocab_size']}")
    print(f"saved_threshold={saved_thr}")
    print(f"bag_size={len(ckpt['state_dicts'])}")

    df, feature_cols = load_and_validate(NEW_CSV)
    if train_feature_cols and train_feature_cols != feature_cols:
        print("WARNING: feature column order differs from checkpoint; using dataset order.")

    new_idx_max = int(df["event_name_idx"].max())
    trained_vocab = int(ckpt["config"]["vocab_size"])
    if new_idx_max >= trained_vocab:
        raise SystemExit(
            f"New event_name_idx max ({new_idx_max}) exceeds trained vocab ({trained_vocab})."
        )

    # Mapping caveat: same index may mean different API events across datasets.
    overlap_note = (
        "Checkpoint trained on invictus_temporal (vocab=261). New data uses a "
        "different event_name_idx mapping (67 APIs). Embedding rows are reused "
        "by index, so this is a cross-dataset transfer check, not in-distribution eval."
    )
    print(f"\nNOTE: {overlap_note}\n")

    windows = build_windows(df, feature_cols)
    labels = [w.label for w in windows]
    n_pos = sum(labels)
    print("=== Windows ===")
    print(f"n_windows={len(windows)} pos={n_pos} neg={len(windows) - n_pos}")

    models = load_models(ckpt, device)
    loader = DataLoader(WindowDataset(windows), batch_size=BATCH_SIZE, shuffle=False)
    y_true, probs = predict_bag(models, loader, device)

    # Full-dataset metrics with saved training threshold.
    full_saved = metrics_dict(y_true, probs, threshold=saved_thr)
    tuned_thr = tune_threshold(y_true, probs) if y_true.sum() > 0 else 0.5
    full_tuned = metrics_dict(y_true, probs, threshold=tuned_thr)

    # Stratified hold-out (same split recipe as training script).
    idx = np.arange(len(windows))
    try:
        tr_i, te_i = train_test_split(
            idx, test_size=0.2, random_state=SEED, stratify=labels
        )
    except ValueError:
        tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=SEED)
    train_w = [windows[i] for i in tr_i]
    test_w = [windows[i] for i in te_i]
    y_tr = np.array([w.label for w in train_w], dtype=float)
    y_te = np.array([w.label for w in test_w], dtype=float)

    tr_loader = DataLoader(WindowDataset(train_w), batch_size=BATCH_SIZE, shuffle=False)
    _, p_tr = predict_bag(models, tr_loader, device)
    thr_from_train = tune_threshold(y_tr, p_tr) if y_tr.sum() > 0 else saved_thr

    te_loader = DataLoader(WindowDataset(test_w), batch_size=BATCH_SIZE, shuffle=False)
    y_te_pred, p_te = predict_bag(models, te_loader, device)
    test_saved = metrics_dict(y_te_pred, p_te, threshold=saved_thr)
    test_tuned = metrics_dict(y_te_pred, p_te, threshold=thr_from_train)

    pos_scores = probs[y_true == 1]
    neg_scores = probs[y_true == 0]
    score_gap = float(pos_scores.mean() - neg_scores.mean()) if len(pos_scores) else float("nan")

    old_metrics = {}
    if OLD_METRICS.exists():
        with open(OLD_METRICS, encoding="utf-8") as f:
            old_metrics = json.load(f)

    report = {
        "checkpoint": str(CHECKPOINT),
        "dataset": str(NEW_CSV),
        "device": str(device),
        "caveat": overlap_note,
        "checkpoint_config": ckpt["config"],
        "saved_threshold": saved_thr,
        "dataset_stats": {
            "n_events": int(len(df)),
            "n_pos_events": int(df["label"].sum()),
            "n_users": int(df["username"].nunique()),
            "event_name_idx_max": new_idx_max,
            "n_windows": len(windows),
            "n_pos_windows": n_pos,
        },
        "full_dataset": {
            "saved_threshold": full_saved,
            "oracle_threshold": full_tuned,
            "score_gap_pos_minus_neg": score_gap,
            "pos_score_mean": float(pos_scores.mean()) if len(pos_scores) else None,
            "neg_score_mean": float(neg_scores.mean()) if len(neg_scores) else None,
        },
        "stratified_test_20pct": {
            "threshold_from_train_f1": thr_from_train,
            "saved_threshold": test_saved,
            "train_tuned_threshold": test_tuned,
            "n_test": int(len(test_w)),
            "n_test_pos": int(y_te.sum()),
        },
        "old_invictus_test_metrics": old_metrics,
        "comparison_vs_old_test": {
            "old_auc_pr": old_metrics.get("auc_pr"),
            "new_test_auc_pr_saved_thr": test_saved.get("auc_pr"),
            "new_test_auc_pr_tuned_thr": test_tuned.get("auc_pr"),
            "old_f1": old_metrics.get("f1"),
            "new_test_f1_saved_thr": test_saved.get("f1"),
            "new_test_f1_tuned_thr": test_tuned.get("f1"),
        },
    }

    export_scores(windows, probs, saved_thr, OUT_CSV)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Full dataset (saved threshold) ===")
    print(json.dumps(full_saved, indent=2))
    print("\n=== Full dataset (oracle threshold) ===")
    print(json.dumps(full_tuned, indent=2))
    print(f"\nscore gap (pos_mean - neg_mean) = {score_gap:.4f}")
    print("\n=== Stratified 20% test (saved threshold) ===")
    print(json.dumps(test_saved, indent=2))
    print("\n=== Stratified 20% test (threshold tuned on train split) ===")
    print(json.dumps(test_tuned, indent=2))
    print("\n=== vs old Invictus test ===")
    print(json.dumps(report["comparison_vs_old_test"], indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
