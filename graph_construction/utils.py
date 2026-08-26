"""
utils.py  (v3 — Privilege Propagation Graph)
===============================================
Shared training/evaluation utilities:
  - FocalLoss
  - evaluate()  — updated for the heterogeneous, multi-triple model output
  - print_comparison_table / print_confusion_matrix

MOVED, NOT REMOVED:
  - GNNExplainerWrapper / FeatureAblation → explainability.py (rewritten
    there with a real PyG GNNExplainer/PGExplainer path plus the same
    gradient-based method as before, now also updated for the multi-triple
    structure — see that file's module docstring).

REMOVED, WITH REASON:
  - HybridSAGELSTM (session-level GraphSAGE+LSTM hybrid). This was already
    gated/non-functional in the previous version — it required a
    `session_label` and a verified temporal ordering that this dataset
    has never had (see neo4j_graph_builder.py / privilege_features.py
    module docstrings). Session- and time-aware reasoning now belongs to
    the adaptive/incremental framework being designed separately (risk-
    aware neighbourhood expansion, incremental graph updates), which is
    the more appropriate home for it going forward, rather than a
    parallel, still-inapplicable LSTM path living here.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch_geometric.data import HeteroData

from data_loader import flatten_mask_dict, global_labels

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]


# ── Focal Loss (unchanged) ────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss. FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).
    gamma > 0 reduces the relative loss for easy negatives, focusing
    training on hard/misclassified attack events — useful given the
    ~4.7% attack rate. Reference: Lin et al., 2017 (RetinaNet).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt      = torch.exp(-bce)
        focal_w = self.alpha * (1 - pt) ** self.gamma
        return (focal_w * bce).mean()


# ── Evaluation — updated for the heterogeneous, multi-triple model ──────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: HeteroData,
    mask_dict: Dict[EdgeTriple, torch.Tensor],
    threshold: float = 0.5,
    return_probs: bool = False,
) -> dict:
    """
    Full evaluation suite, operating on the model's FLAT logits (ordered
    by sorted(data.edge_types) — see model_graphsage.py/model_gat.py) and
    a correspondingly flattened y / mask (via data_loader.py's
    global_labels / flatten_mask_dict), rather than a single relation's
    tensors as in the earlier bipartite design.
    """
    model.eval()
    logits = model(data)
    probs  = torch.sigmoid(logits)

    y_true_full = global_labels(data)
    mask_full   = flatten_mask_dict(data, mask_dict).cpu().numpy()

    y_true = y_true_full[mask_full]
    y_prob = probs.cpu().numpy()[mask_full]
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=["normal", "attack"], output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": report["attack"]["precision"],
        "recall":    report["attack"]["recall"],
        "f1":        report["attack"]["f1-score"],
        "roc_auc":   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "report":    report,
    }
    if return_probs:
        metrics["probs"], metrics["labels"] = y_prob, y_true

    log.info(
        "Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f  AUC=%.4f  (n=%d)",
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["roc_auc"], mask_full.sum(),
    )
    return metrics


def print_comparison_table(sage_metrics: dict, gat_metrics: dict):
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'Metric':<15} {'GraphSAGE':>12} {'GAT':>12}  {'Δ (SAGE-GAT)':>14}"
    print("\n" + "=" * 60)
    print("  GraphSAGE vs GAT — Performance Comparison")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for k in keys:
        sv, gv = sage_metrics.get(k, 0.0), gat_metrics.get(k, 0.0)
        print(f"  {k:<13} {sv:>12.4f} {gv:>12.4f}  {sv-gv:>+14.4f}")
    print("=" * 60 + "\n")


def print_confusion_matrix(cm: list, labels=("normal", "attack")):
    print("\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for l in labels:
        print(f"  {'Pred_' + l:>14}", end="")
    print()
    for i, row_label in enumerate(labels):
        print(f"  {'True_' + row_label:<10}", end="")
        for val in cm[i]:
            print(f"  {val:>14}", end="")
        print()
    print()
