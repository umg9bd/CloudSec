"""Generate Temporal LSTM learning curves, ROC/PR, confusion matrices, etc.

Reads artifacts from training and writes PNGs to artifacts/plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ART / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Visual language (shared across figures)
C_LOSS = "#2f5d50"
C_AUC_PR = "#c45c26"
C_AUC_ROC = "#3b6ea5"
C_F1 = "#7a3e65"
C_PREC = "#5c7a3e"
C_REC = "#8a6d3b"
C_ACC = "#4a5568"
SEED_COLORS = ["#2f5d50", "#c45c26", "#3b6ea5", "#7a3e65", "#8a6d3b"]


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def _load_history(name: str) -> pd.DataFrame | None:
    p = ART / name
    if not p.exists():
        print(f"  skip missing {p.name}")
        return None
    return pd.read_csv(p)


def _load_pseq(name: str) -> pd.DataFrame | None:
    p = ART / name
    if not p.exists():
        print(f"  skip missing {p.name}")
        return None
    return pd.read_csv(p)


def plot_learning_curves(hist: pd.DataFrame, tag: str, title: str) -> None:
    """Loss + val metrics over epochs (mean across seeds + per-seed faint lines)."""
    metrics = [
        ("train_loss", "Train loss", C_LOSS),
        ("val_auc_pr", "Val AUC-PR", C_AUC_PR),
        ("val_auc_roc", "Val AUC-ROC", C_AUC_ROC),
        ("val_f1", "Val F1", C_F1),
    ]
    if "val_accuracy" in hist.columns:
        metrics.append(("val_accuracy", "Val accuracy", C_ACC))

    n = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    seeds = sorted(hist["seed"].unique()) if "seed" in hist.columns else [None]

    for ax, (col, label, color) in zip(axes, metrics):
        if col not in hist.columns:
            ax.set_visible(False)
            continue
        for i, seed in enumerate(seeds):
            g = hist if seed is None else hist[hist["seed"] == seed]
            ax.plot(
                g["epoch"],
                g[col],
                color=SEED_COLORS[i % len(SEED_COLORS)],
                alpha=0.35 if len(seeds) > 1 else 0.9,
                linewidth=1.2,
                label=f"seed {seed}" if seed is not None and len(seeds) > 1 else None,
            )
        if "seed" in hist.columns and len(seeds) > 1:
            mean = hist.groupby("epoch")[col].mean()
            ax.plot(mean.index, mean.values, color=color, linewidth=2.4, label="mean")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, f"{tag}_learning_curves.png")


def plot_metrics_panel(hist: pd.DataFrame, tag: str, title: str) -> None:
    """Single multi-line chart of val precision / recall / F1 (mean)."""
    cols = [c for c in ("val_precision", "val_recall", "val_f1", "val_auc_pr") if c in hist.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {
        "val_precision": C_PREC,
        "val_recall": C_REC,
        "val_f1": C_F1,
        "val_auc_pr": C_AUC_PR,
    }
    labels = {
        "val_precision": "Precision",
        "val_recall": "Recall",
        "val_f1": "F1",
        "val_auc_pr": "AUC-PR",
    }
    if "seed" in hist.columns:
        mean = hist.groupby("epoch")[cols].mean()
        for c in cols:
            ax.plot(mean.index, mean[c], color=colors[c], linewidth=2, label=labels[c])
    else:
        for c in cols:
            ax.plot(hist["epoch"], hist[c], color=colors[c], linewidth=2, label=labels[c])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, f"{tag}_val_metrics.png")


def _eval_bundle(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "threshold": threshold,
        "auc_pr": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auc_roc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "cm": confusion_matrix(y, pred, labels=[0, 1]),
        "y": y,
        "p": p,
        "pred": pred,
    }


def plot_confusion(cm: np.ndarray, tag: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign (0)", "Attack (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, f"{tag}_confusion_matrix.png")


def plot_roc_pr(y: np.ndarray, p: np.ndarray, tag: str, title: str) -> None:
    if len(np.unique(y)) < 2:
        print(f"  skip ROC/PR for {tag}: single class")
        return

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(fpr, tpr, color=C_AUC_ROC, linewidth=2.2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(rec, prec, color=C_AUC_PR, linewidth=2.2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, f"{tag}_roc_pr.png")


def plot_score_hist(y: np.ndarray, p: np.ndarray, tag: str, title: str, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[y == 0], bins=bins, alpha=0.55, color=C_AUC_ROC, label="Benign", density=True)
    ax.hist(p[y == 1], bins=bins, alpha=0.55, color=C_AUC_PR, label="Attack", density=True)
    ax.axvline(threshold, color=C_LOSS, linestyle="--", linewidth=1.8, label=f"thr={threshold:.3f}")
    ax.set_xlabel("P_seq score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, f"{tag}_score_distribution.png")


def plot_threshold_sweep(y: np.ndarray, p: np.ndarray, tag: str, title: str) -> None:
    if len(np.unique(y)) < 2:
        return
    thrs = np.linspace(0.05, 0.95, 37)
    f1s, precs, recs = [], [], []
    for t in thrs:
        pred = (p >= t).astype(int)
        f1s.append(f1_score(y, pred, zero_division=0))
        precs.append(precision_score(y, pred, zero_division=0))
        recs.append(recall_score(y, pred, zero_division=0))
    best_i = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thrs, precs, color=C_PREC, label="Precision")
    ax.plot(thrs, recs, color=C_REC, label="Recall")
    ax.plot(thrs, f1s, color=C_F1, linewidth=2.2, label="F1")
    ax.axvline(thrs[best_i], color=C_LOSS, linestyle="--", label=f"best F1 @ {thrs[best_i]:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, f"{tag}_threshold_sweep.png")


def plot_metrics_bars(metrics: dict, tag: str, title: str) -> None:
    keys = ["auc_pr", "auc_roc", "f1", "precision", "recall"]
    keys = [k for k in keys if k in metrics and metrics[k] is not None]
    if not keys:
        return
    vals = [float(metrics[k]) for k in keys]
    colors = [C_AUC_PR, C_AUC_ROC, C_F1, C_PREC, C_REC][: len(keys)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, vals, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, f"{tag}_metrics_bars.png")


def plot_v1_vs_v2_compare() -> None:
    """Side-by-side metric comparison if both metric JSONs exist."""
    m1_path, m2_path = ART / "test_metrics.json", ART / "test_metrics_v2.json"
    if not (m1_path.exists() and m2_path.exists()):
        return
    m1 = json.loads(m1_path.read_text())
    m2 = json.loads(m2_path.read_text())
    # v2 nests test under "test"
    if "test" in m2:
        m2 = m2["test"]

    keys = ["auc_pr", "auc_roc", "f1", "precision", "recall"]
    x = np.arange(len(keys))
    w = 0.36
    v1 = [float(m1.get(k, np.nan)) for k in keys]
    v2 = [float(m2.get(k, np.nan)) for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w / 2, v1, w, color=C_AUC_ROC, label="v1 (bag LSTM)")
    ax.bar(x + w / 2, v2, w, color=C_AUC_PR, label="v2")
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Test metrics — v1 vs v2")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "compare_v1_v2_test_metrics.png")


def plot_loss_only(hist: pd.DataFrame, tag: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    seeds = sorted(hist["seed"].unique()) if "seed" in hist.columns else [None]
    for i, seed in enumerate(seeds):
        g = hist if seed is None else hist[hist["seed"] == seed]
        ax.plot(
            g["epoch"],
            g["train_loss"],
            color=SEED_COLORS[i % len(SEED_COLORS)],
            alpha=0.45 if len(seeds) > 1 else 0.95,
            linewidth=1.3,
            label=f"seed {seed}" if seed is not None else "train loss",
        )
    if "seed" in hist.columns and len(seeds) > 1:
        mean = hist.groupby("epoch")["train_loss"].mean()
        ax.plot(mean.index, mean.values, color=C_LOSS, linewidth=2.6, label="mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, f"{tag}_train_loss.png")


def run_pseq_suite(csv_name: str, tag: str, title: str, threshold: float) -> None:
    df = _load_pseq(csv_name)
    if df is None:
        return
    y = df["window_label"].to_numpy(dtype=int)
    p = df["P_seq"].to_numpy(dtype=float)
    bundle = _eval_bundle(y, p, threshold)
    print(
        f"  [{tag}] n={len(y)} pos={int(y.sum())} "
        f"AUC-PR={bundle['auc_pr']:.3f} F1={bundle['f1']:.3f} thr={threshold}"
    )
    plot_confusion(bundle["cm"], tag, f"{title} — confusion matrix (thr={threshold:.2f})")
    plot_roc_pr(y, p, tag, f"{title} — ROC / PR")
    plot_score_hist(y, p, tag, f"{title} — score distribution", threshold)
    plot_threshold_sweep(y, p, tag, f"{title} — threshold sweep")
    plot_metrics_bars(
        {k: bundle[k] for k in ("auc_pr", "auc_roc", "f1", "precision", "recall")},
        tag,
        f"{title} — metrics",
    )


def main() -> None:
    print(f"Output -> {OUT}")

    # --- Learning curves ---
    hist_v1 = _load_history("training_history.csv")
    if hist_v1 is not None:
        print("v1 training history")
        plot_learning_curves(hist_v1, "v1", "Temporal LSTM v1 — learning curves")
        plot_metrics_panel(hist_v1, "v1", "Temporal LSTM v1 — validation metrics (mean)")
        plot_loss_only(hist_v1, "v1", "Temporal LSTM v1 — train loss")

    hist_v2 = _load_history("training_history_v2.csv")
    if hist_v2 is not None:
        print("v2 training history")
        plot_learning_curves(hist_v2, "v2", "Temporal LSTM v2 — learning curves")
        plot_metrics_panel(hist_v2, "v2", "Temporal LSTM v2 — validation metrics (mean)")
        plot_loss_only(hist_v2, "v2", "Temporal LSTM v2 — train loss")

    # --- Prediction-based graphs ---
    thr_v1 = 0.55
    thr_v2 = 0.70
    tm1 = ART / "test_metrics.json"
    tm2 = ART / "test_metrics_v2.json"
    if tm1.exists():
        thr_v1 = float(json.loads(tm1.read_text()).get("threshold", thr_v1))
    if tm2.exists():
        raw = json.loads(tm2.read_text())
        thr_v2 = float((raw.get("test") or raw).get("threshold", thr_v2))

    print("v1 P_seq suite")
    run_pseq_suite("P_seq.csv", "v1", "Temporal LSTM v1 (all windows)", thr_v1)

    print("v2 P_seq suite")
    run_pseq_suite("P_seq_v2.csv", "v2", "Temporal LSTM v2 (all windows)", thr_v2)

    print("new-data transfer suite")
    run_pseq_suite(
        "P_seq_new_data.csv",
        "new_data",
        "Transfer check on new data",
        thr_v1,
    )

    print("v1 vs v2 comparison")
    plot_v1_vs_v2_compare()

    # Optional LOAO bars
    loao_path = ART / "loao_metrics.json"
    if loao_path.exists():
        # Allow NaN in this artifact (non-strict JSON from numpy dumps)
        loao = json.loads(loao_path.read_text().replace("NaN", "null"))
        if "loao" in loao and isinstance(loao["loao"], dict):
            loao = loao["loao"]
        users, aucs = [], []
        for u, m in loao.items():
            if isinstance(m, dict) and m.get("auc_pr") is not None:
                users.append(u)
                aucs.append(float(m["auc_pr"]))
        if users:
            fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(users) + 2), 4))
            ax.bar(users, aucs, color=C_AUC_PR)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("AUC-PR")
            ax.set_title("LOAO AUC-PR by held-out attacker")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            _save(fig, "loao_auc_pr.png")

    print("Done.")
    print(f"All plots in: {OUT}")


if __name__ == "__main__":
    main()
