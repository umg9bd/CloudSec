"""
utils.py
========
Shared utilities:
  - FocalLoss (handles class imbalance better than weighted BCE)
  - evaluate()          — accuracy / precision / recall / F1 / AUC / confusion
  - GNNExplainerWrapper — feature importance for a single edge prediction
  - FeatureAblation     — which edge features matter most?
  - HybridSAGELSTM      — session-level architecture, GATED for this dataset
                           (see class docstring — no verified temporal/session
                           signal exists in invictus_structural.csv)
  - print_confusion_matrix
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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

log = logging.getLogger(__name__)

# Order MUST match the concatenation order in data_loader.py's
# _build_edges(): EDGE_NUM_COLS + [label-encoded EDGE_CAT_COLS].
# EDGE_NUM_COLS = ["is_read_only", "is_privilege_sensitive", "action_global_frequency_log"]
# EDGE_CAT_COLS = ["edge_type"]
EDGE_FEATURE_NAMES = [
    "is_read_only",
    "is_privilege_sensitive",
    "action_global_frequency_log",
    "edge_type",
]


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    γ > 0 reduces the relative loss for easy (well-classified) negatives,
    so the model focuses on hard, misclassified attack events. Useful here
    given the ~4.7% attack rate. Reference: Lin et al., 2017 (RetinaNet).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce      = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt       = torch.exp(-bce)
        focal_w  = self.alpha * (1 - pt) ** self.gamma
        loss     = focal_w * bce
        return loss.mean()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: HeteroData,
    mask: torch.Tensor,
    threshold: float = 0.5,
    return_probs: bool = False,
) -> dict:
    model.eval()
    logits = model(data)
    probs  = torch.sigmoid(logits)

    y_true = data["principal", "invoked", "target"].y[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(y_true, y_pred, target_names=["normal", "attack"],
                                   output_dict=True, zero_division=0)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": report["attack"]["precision"],
        "recall":    report["attack"]["recall"],
        "f1":        report["attack"]["f1-score"],
        "roc_auc":   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion":  confusion_matrix(y_true, y_pred).tolist(),
        "report":    report,
    }

    if return_probs:
        metrics["probs"]  = y_prob
        metrics["labels"] = y_true

    log.info(
        "Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f  AUC=%.4f",
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1"], metrics["roc_auc"],
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
        sv = sage_metrics.get(k, 0.0)
        gv = gat_metrics.get(k, 0.0)
        print(f"  {k:<13} {sv:>12.4f} {gv:>12.4f}  {sv-gv:>+14.4f}")
    print("=" * 60 + "\n")


# ── Explainability ────────────────────────────────────────────────────────────

class GNNExplainerWrapper:
    """
    Lightweight edge-level explanation via gradient × input attribution
    over the four features listed in EDGE_FEATURE_NAMES.
    """

    def __init__(self, model: nn.Module, edge_feat_names: List[str] = None):
        self.model = model
        self.feat_names = edge_feat_names or EDGE_FEATURE_NAMES
        try:
            from torch_geometric.explain import Explainer, GNNExplainer as PyGExplainer
            self._pyg_available = True
            log.info("PyG GNNExplainer available.")
        except ImportError:
            self._pyg_available = False
            log.info("PyG GNNExplainer not found — using gradient-based fallback.")

    def explain_edge(self, data: HeteroData, edge_idx: int) -> Dict[str, float]:
        return self._gradient_explanation(data, edge_idx)

    def _gradient_explanation(self, data: HeteroData, edge_idx: int) -> Dict[str, float]:
        self.model.eval()
        edge_attr = data["principal", "invoked", "target"].edge_attr
        edge_attr.requires_grad_(True)

        logits = self.model(data)
        logit  = logits[edge_idx]

        self.model.zero_grad()
        logit.backward()

        grads  = edge_attr.grad[edge_idx].abs()
        values = edge_attr[edge_idx].detach().abs()
        importance = (grads * values).cpu().numpy()

        total = importance.sum() + 1e-9
        importance = importance / total

        n = min(len(self.feat_names), len(importance))
        result = {self.feat_names[i]: float(importance[i]) for i in range(n)}
        result = dict(sorted(result.items(), key=lambda x: -x[1]))

        prob = torch.sigmoid(logit).item()
        log.info(
            "Edge %d | pred_prob=%.4f | top feature: %s (%.3f)",
            edge_idx, prob, list(result.keys())[0], list(result.values())[0],
        )
        return result

    def explain_top_k(self, data: HeteroData, mask: torch.Tensor, k: int = 5):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data)
            probs  = torch.sigmoid(logits)

        masked_probs = probs.clone()
        masked_probs[~mask] = -1.0
        top_k_edges = masked_probs.topk(k).indices.tolist()

        explanations = {}
        for idx in top_k_edges:
            explanations[idx] = self.explain_edge(data, idx)
        return explanations


class FeatureAblation:
    """Feature ablation: F1 drop when each edge feature is zeroed."""

    def __init__(self, model: nn.Module, feat_names: List[str] = None):
        self.model      = model
        self.feat_names = feat_names or EDGE_FEATURE_NAMES

    def run(self, data: HeteroData, mask: torch.Tensor) -> Dict[str, float]:
        baseline = evaluate(self.model, data, mask)
        baseline_f1 = baseline["f1"]

        results = {}
        edge_attr_orig = data["principal", "invoked", "target"].edge_attr.clone()

        for i, feat_name in enumerate(self.feat_names):
            if i >= edge_attr_orig.shape[1]:
                break
            ablated = edge_attr_orig.clone()
            ablated[:, i] = 0.0
            data["principal", "invoked", "target"].edge_attr = ablated

            m    = evaluate(self.model, data, mask)
            drop = baseline_f1 - m["f1"]
            results[feat_name] = float(drop)

            log.info("Ablate %-30s  Δ F1 = %+.4f", feat_name, drop)

        data["principal", "invoked", "target"].edge_attr = edge_attr_orig
        return dict(sorted(results.items(), key=lambda x: -x[1]))


# ── Session-level Hybrid GraphSAGE + LSTM — GATED FOR THIS DATASET ───────────

class HybridSAGELSTM(nn.Module):
    """
    Architecture reference for a session-level GraphSAGE + LSTM hybrid.

    ⚠ NOT APPLICABLE to invictus_structural.csv as currently constituted.
    This module is retained because it is a legitimate architecture for
    datasets that DO carry verified per-session temporal ordering (the
    "LSTM temporal branch" in the overall system architecture is meant to
    be trained on such a dataset, handled separately from the graph
    branch). Two things this dataset lacks that the class needs:

      1. A true session/temporal ordering. `log_id` is a row identifier,
         not a verified timestamp (see neo4j_graph_builder.py docstring,
         point 2). `group_edges_by_principal` below groups edges by
         principal correctly (that IS structural, from edge_index), but
         will refuse to silently order them by log_id and call that a
         "session" unless the caller explicitly acknowledges the caveat
         via `assume_log_id_order=True`.
      2. A session-level label. The original `session_label` field was a
         proxy for "this event happened during an attacker's session" —
         not derivable without a real session boundary. Callers must
         supply `session_labels` explicitly (e.g. from a different,
         genuinely time-stamped dataset); this class no longer reads a
         `y_session` field from `data`, because none is written by
         data_loader.py.
    """

    def __init__(
        self,
        sage_model: nn.Module,
        edge_embed_dim: int,
        lstm_hidden:    int  = 64,
        num_lstm_layers: int = 2,
        dropout: float       = 0.3,
    ):
        super().__init__()
        self.sage     = sage_model
        self.lstm     = nn.LSTM(
            input_size=edge_embed_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.session_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
        )

    def forward(
        self,
        data: HeteroData,
        session_sequences: List[torch.Tensor],
    ) -> torch.Tensor:
        edge_embs = self.sage.get_edge_embeddings(data)

        session_logits = []
        for seq_edge_indices in session_sequences:
            if len(seq_edge_indices) == 0:
                session_logits.append(torch.zeros(1, device=edge_embs.device))
                continue
            seq_embs = edge_embs[seq_edge_indices].unsqueeze(0)
            _, (h_n, _) = self.lstm(seq_embs)
            h_last = h_n[-1].squeeze(0)
            logit = self.session_head(h_last)
            session_logits.append(logit)

        return torch.cat(session_logits, dim=0)

    @staticmethod
    def group_edges_by_principal(
        data: HeteroData,
        assume_log_id_order: bool = False,
    ) -> Dict[int, List[int]]:
        """
        Groups edge indices by their source principal index (this part is
        purely structural — derived from edge_index, no assumption
        involved).

        Ordering WITHIN each group defaults to the order edges were
        returned by Neo4j (arbitrary, but deterministic given a fixed
        query and database state). Passing `assume_log_id_order=True`
        will instead sort by the `log_id` metadata tensor — but this is
        NOT a verified chronological order (see class docstring) and
        should only be used with that caveat stated in any write-up.
        """
        edge_index = data["principal", "invoked", "target"].edge_index
        src_nodes  = edge_index[0].cpu().numpy()

        if assume_log_id_order:
            if not hasattr(data["principal", "invoked", "target"], "log_id"):
                raise ValueError("No log_id metadata found on the edge store.")
            order_key = data["principal", "invoked", "target"].log_id.cpu().numpy()
        else:
            order_key = np.arange(len(src_nodes))

        sessions: Dict[int, List[Tuple[float, int]]] = {}
        for e_idx, (src, k) in enumerate(zip(src_nodes, order_key)):
            sessions.setdefault(int(src), []).append((float(k), e_idx))

        return {
            pid: [e for _, e in sorted(seq)]
            for pid, seq in sessions.items()
        }


# ── Confusion matrix printer ──────────────────────────────────────────────────

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
