"""
explainability.py
===================
Edge-level explainability for the Privilege Propagation Graph GraphSAGE/GAT
models, using PyTorch Geometric's real `Explainer` framework
(GNNExplainer or PGExplainer) rather than the gradient×input placeholder
utils.py previously used under a "GNNExplainerWrapper" name (that name was
always a stand-in — the codebase never actually called PyG's GNNExplainer
algorithm; this file fixes that).

AN HONEST CAVEAT ABOUT WHAT'S VERIFIED HERE
─────────────────────────────────────────────────────────────────────────
This development environment has no working `torch` / `torch_geometric`
install (verified: `pip install torch` is not reachable here), so the
`Explainer`/`GNNExplainer` path below has been written carefully against
the documented PyG API but NOT executed. PyG's heterogeneous-graph
explainability support (HeteroData, multi-relation edge-level targets)
has had real version-dependent gaps historically, more so than the
homogeneous case. Two things follow from that:

  1. `explain_edge_gradient` (this file's fallback) uses plain autograd —
     `logit.backward()`, read `.grad` — which has no PyG-version-specific
     surface at all. This is the path I'm fully confident is correct, and
     it is registered as the DEFAULT.
  2. `explain_edge_gnnexplainer` wraps PyG's real `Explainer` +
     `GNNExplainer`/`PGExplainer`. Treat this as "should work, verify
     against your installed PyG version before using its output as a
     paper's headline explainability result" rather than as something
     this environment has actually confirmed executes. If it raises or
     behaves unexpectedly on your installed version, `explain_edge` below
     will log the failure and fall back to the gradient method
     automatically, rather than silently producing wrong numbers.

WHY GNNExplainer/PGExplainer OVER SHAP
─────────────────────────────────────────────────────────────────────────
SHAP is a model-agnostic, feature-perturbation method built for tabular/
sequence inputs; it has no native notion of message-passing structure, so
applying it to a GNN either treats the graph as a fixed feature vector
(losing the structural "why did THIS neighbourhood matter" question) or
requires bespoke graph-aware adaptations. GNNExplainer/PGExplainer were
purpose-built for GNNs: they learn a soft mask over edges/features that
maximizes mutual information with the model's own prediction, directly
answering "which edges and which edge/node features this specific GNN
relied on for this specific prediction" — the right question for a
security analyst auditing a single alert.

HOW THIS FITS THE HETEROGENEOUS, MULTI-TRIPLE MODEL
─────────────────────────────────────────────────────────────────────────
Both model_graphsage.py and model_gat.py expose `forward(data) -> flat
logits ordered by sorted(data.edge_types)` (see those files' docstrings).
Explanations are requested per EDGE, identified by (triple, local_index)
rather than a single flat integer, since that is how the heterogeneous
graph is actually organised — see `TargetEdge`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]

# Order MUST match data_loader.py's EDGE_NUM_COLS + EDGE_CAT_COLS concatenation.
EDGE_FEATURE_NAMES = [
    "hop_count", "privilege_gain", "privilege_gain_defined",
    "abnormal_path_frequency", "action_global_frequency_log",
    "is_privilege_escalation_technique", "is_read_only", "edge_type",
]


@dataclass(frozen=True)
class TargetEdge:
    """Identifies one edge to explain: which (src,rel,dst) triple, and
    which row within that triple's own edge_index/edge_attr tensors."""
    triple: EdgeTriple
    local_index: int


class EdgeExplainer:
    """
    Produces feature-importance explanations for individual edge
    predictions from a GraphSAGEAnomalyDetector or GATAnomalyDetector.

    Usage:
        explainer = EdgeExplainer(model, method="gnnexplainer")  # or "gradient"
        result = explainer.explain(data, TargetEdge(("User","READ","Resource"), 12))
        # result: {feature_name: importance_score}, sorted descending
    """

    def __init__(self, model: nn.Module, feat_names: List[str] = None, method: str = "gradient"):
        self.model = model
        self.feat_names = feat_names or EDGE_FEATURE_NAMES
        self.method = method
        self._pyg_explainer = None
        if method == "gnnexplainer":
            self._pyg_explainer = self._try_build_pyg_explainer()
            if self._pyg_explainer is None:
                log.warning(
                    "Could not initialise PyG's GNNExplainer in this environment — "
                    "falling back to the gradient-based method for every call. "
                    "See explainability.py module docstring."
                )
                self.method = "gradient"

    # ── Real PyG Explainer path (GNNExplainer / PGExplainer) ────────────────

    def _try_build_pyg_explainer(self):
        try:
            from torch_geometric.explain import Explainer, GNNExplainer
        except ImportError:
            return None
        try:
            return Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=200),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="binary_classification",
                    task_level="edge",
                    return_type="raw",
                ),
            )
        except Exception as exc:
            log.warning("PyG Explainer construction failed (%s) — will use gradient fallback.", exc)
            return None

    def _explain_gnnexplainer(self, data: HeteroData, target: TargetEdge) -> Dict[str, float]:
        """
        Runs PyG's Explainer for the requested edge. See module docstring
        for the honesty caveat on this path — wrapped in try/except so a
        version mismatch degrades to the gradient method instead of
        crashing the caller.
        """
        try:
            explanation = self._pyg_explainer(
                x=data.x_dict,
                edge_index=data.edge_index_dict,
                index=target.local_index,
                target_edge_type=target.triple,
            )
            edge_mask = explanation.get(target.triple, {}).get("edge_mask")
            if edge_mask is None:
                raise RuntimeError("Explainer returned no edge_mask for this triple.")
            edge_attr = data[target.triple].edge_attr[target.local_index].detach().cpu().numpy()
            importance = np.abs(edge_attr) * float(edge_mask[target.local_index].item())
            return self._normalise(importance)
        except Exception as exc:
            log.warning("GNNExplainer run failed (%s) — falling back to gradient method for this edge.", exc)
            return self._explain_gradient(data, target)

    # ── Gradient × input fallback (no PyG-Explainer-specific API risk) ─────

    def _explain_gradient(self, data: HeteroData, target: TargetEdge) -> Dict[str, float]:
        """
        importance_i = |grad_i * input_i| for the target edge's own
        edge_attr row, w.r.t. that specific edge's predicted logit. Plain
        autograd — the same technique utils.py's GNNExplainerWrapper used
        previously, carried over here as the dependable default.
        """
        self.model.eval()
        edge_attr = data[target.triple].edge_attr
        edge_attr.requires_grad_(True)

        logits = self.model(data)
        # Map (triple, local_index) -> position in the model's flat output,
        # which is ordered by sorted(data.edge_types) — see model files.
        offset = 0
        for t in sorted(data.edge_types):
            if t == target.triple:
                global_index = offset + target.local_index
                break
            offset += data[t].y.shape[0]
        else:
            raise ValueError(f"Triple {target.triple} not found in data.edge_types")

        logit = logits[global_index]
        self.model.zero_grad()
        logit.backward()

        grads  = edge_attr.grad[target.local_index].abs()
        values = edge_attr[target.local_index].detach().abs()
        importance = (grads * values).cpu().numpy()

        prob = torch.sigmoid(logit).item()
        log.info("Edge %s#%d | pred_prob=%.4f", target.triple, target.local_index, prob)
        return self._normalise(importance)

    def _normalise(self, importance: np.ndarray) -> Dict[str, float]:
        total = importance.sum() + 1e-9
        importance = importance / total
        n = min(len(self.feat_names), len(importance))
        result = {self.feat_names[i]: float(importance[i]) for i in range(n)}
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    # ── Public API ───────────────────────────────────────────────────────────

    def explain(self, data: HeteroData, target: TargetEdge) -> Dict[str, float]:
        if self.method == "gnnexplainer" and self._pyg_explainer is not None:
            return self._explain_gnnexplainer(data, target)
        return self._explain_gradient(data, target)

    @torch.no_grad()
    def _flat_probs_and_targets(self, data: HeteroData) -> Tuple[np.ndarray, List[TargetEdge]]:
        self.model.eval()
        logits = self.model(data)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets: List[TargetEdge] = []
        for t in sorted(data.edge_types):
            n_t = data[t].y.shape[0]
            targets.extend(TargetEdge(t, i) for i in range(n_t))
        return probs, targets

    def explain_top_k(self, data: HeteroData, mask_dict: Dict[EdgeTriple, torch.Tensor], k: int = 5) -> Dict[TargetEdge, Dict[str, float]]:
        """
        Explains the k highest-confidence attack predictions within the
        given (per-triple) mask dict — mirrors the previous
        GNNExplainerWrapper.explain_top_k, adapted to the multi-triple
        structure.
        """
        probs, targets = self._flat_probs_and_targets(data)
        allowed = np.array([mask_dict.get(t.triple, torch.zeros(0, dtype=torch.bool))[t.local_index].item()
                             if t.local_index < mask_dict.get(t.triple, torch.zeros(0)).shape[0] else False
                             for t in targets])
        masked_probs = np.where(allowed, probs, -1.0)
        top_k_idx = np.argsort(-masked_probs)[:k]
        return {targets[i]: self.explain(data, targets[i]) for i in top_k_idx}


class FeatureAblation:
    """
    Feature ablation: measures F1 drop, over ALL populated triples at
    once (via the model's flattened output), when each edge feature
    column is zeroed across every triple simultaneously.
    """

    def __init__(self, model: nn.Module, feat_names: List[str] = None):
        self.model = model
        self.feat_names = feat_names or EDGE_FEATURE_NAMES

    def run(self, data: HeteroData, mask_dict: Dict[EdgeTriple, torch.Tensor], evaluate_fn) -> Dict[str, float]:
        baseline = evaluate_fn(self.model, data, mask_dict)
        baseline_f1 = baseline["f1"]

        originals = {t: data[t].edge_attr.clone() for t in data.edge_types}
        results = {}
        for i, feat_name in enumerate(self.feat_names):
            for t in data.edge_types:
                if i < data[t].edge_attr.shape[1]:
                    ablated = originals[t].clone()
                    ablated[:, i] = 0.0
                    data[t].edge_attr = ablated
            m = evaluate_fn(self.model, data, mask_dict)
            drop = baseline_f1 - m["f1"]
            results[feat_name] = float(drop)
            log.info("Ablate %-35s Δ F1 = %+.4f", feat_name, drop)

        for t in data.edge_types:
            data[t].edge_attr = originals[t]

        return dict(sorted(results.items(), key=lambda x: -x[1]))
