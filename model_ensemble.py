"""
model_ensemble.py
====================
Combines GraphSAGE (over the entire graph) and HGT (over the "important
region" sampled subgraph — see node_importance.py, model_hgt.py) into a
single threat score, plus a factory (build_ensemble_from_args) that
infer.py's load_model_from_checkpoint() dispatches to.

WEIGHTED AVERAGING HAPPENS IN LOGIT SPACE, NOT PROBABILITY SPACE
─────────────────────────────────────────────────────────────────────────
Every existing caller applies torch.sigmoid() to whatever a model
returns, OUTSIDE the model — infer.py's process_event does
`probs = torch.sigmoid(self.model(subgraph))`, and utils.py's evaluate()
does the same. Returning an already-blended PROBABILITY here would need
every caller to special-case skip that sigmoid for the ensemble;
returning a LOGIT-space blend instead means every existing call site
keeps working completely unchanged, for every model type.
FinalLogit = alpha*HGT_logit + (1-alpha)*SAGE_logit — algebraically
different from alpha*sigmoid(HGT_logit) + (1-alpha)*sigmoid(SAGE_logit).
Logit-space was chosen specifically for the compatibility reason above,
not asserted as more "correct" in the abstract — flagged as a real
choice in the extension design notes, not a formality.

WHAT HAPPENS WHEN A COMPONENT DIDN'T SCORE AN EDGE
─────────────────────────────────────────────────────────────────────────
HGT only ever sees edges inside its sampled "important region" subgraph
— by design, not every edge GraphSAGE scores has an HGT counterpart.
Rather than inventing a placeholder HGT logit for uncovered edges, this
class renormalises weights per-edge over whichever components actually
covered it — an edge only GraphSAGE covers gets alpha implicitly
redistributed to GraphSAGE for that edge, not zeroed out or crashed on.
last_component_coverage exposes which components covered which triples
after each forward() call — not because forward()'s callers need it
(they get one flat logits tensor exactly like every other model), but
because "was this particular alert HGT-informed or pure-GraphSAGE" is
exactly the kind of thing worth being able to check.

DESIGNED FOR MORE THAN TWO MODELS FROM THE START
─────────────────────────────────────────────────────────────────────────
Per the extension brief's "additional models can easily be added later":
internally this is a list of (name, model, weight) entries, not two
named attributes. Two-model alpha-blending is expressed as the special
case components=[("graphsage", sage, 1-alpha), ("hgt", hgt, alpha)] — a
third component later is a one-line addition to that list, no change to
forward()'s logic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]


class EnsembleModel(nn.Module):
    def __init__(self, components: List[Tuple[str, nn.Module, float]]):
        """
        components: list of (name, model, weight). Each `model` must
        implement the same forward(data) -> flat logits ordered by
        sorted(data.edge_types) contract as every other model in this
        repository, and expose an `edge_types` attribute (every model in
        this repo already does — see model_graphsage.py / model_gat.py /
        model_hgt.py). Weights need not sum to 1 up front — they're
        renormalised per-edge over whichever components covered that
        edge; see module docstring.
        """
        super().__init__()
        if not components:
            raise ValueError("EnsembleModel needs at least one component.")
        self.names = [c[0] for c in components]
        self.weights = [float(c[2]) for c in components]
        self.models = nn.ModuleList([c[1] for c in components])
        self.last_component_coverage: Dict[str, Dict[EdgeTriple, torch.Tensor]] = {}

    def forward(self, data: HeteroData) -> torch.Tensor:
        all_triples = sorted(data.edge_types)

        per_component: Dict[str, Dict[EdgeTriple, torch.Tensor]] = {}
        for name, model in zip(self.names, self.models):
            model_edge_types = set(getattr(model, "edge_types", all_triples))
            covered_triples = [t for t in all_triples if t in model_edge_types]
            if not covered_triples:
                per_component[name] = {}
                continue
            flat = model(data)  # ordered by sorted(data.edge_types) filtered to model_edge_types
            per_triple: Dict[EdgeTriple, torch.Tensor] = {}
            offset = 0
            for t in covered_triples:
                n_t = data[t].edge_index.shape[1]
                per_triple[t] = flat[offset:offset + n_t]
                offset += n_t
            per_component[name] = per_triple

        self.last_component_coverage = per_component

        out_chunks = []
        for triple in all_triples:
            n_t = data[triple].edge_index.shape[1]
            device = data[triple].edge_index.device
            weighted_sum = torch.zeros(n_t, device=device)
            weight_total = torch.zeros(n_t, device=device)
            for name, w in zip(self.names, self.weights):
                triple_logits = per_component.get(name, {}).get(triple)
                if triple_logits is None:
                    continue  # this component didn't cover this triple at all
                weighted_sum = weighted_sum + w * triple_logits
                weight_total = weight_total + w
            # Edges no component covered (shouldn't happen if at least one
            # model runs on the full graph, e.g. GraphSAGE) fall back to a
            # 0 logit (probability 0.5) rather than dividing by zero.
            safe_total = torch.where(weight_total > 0, weight_total, torch.ones_like(weight_total))
            blended = torch.where(weight_total > 0, weighted_sum / safe_total, torch.zeros_like(weighted_sum))
            out_chunks.append(blended)

        return torch.cat(out_chunks, dim=0) if out_chunks else torch.zeros(0)


def build_ensemble_from_args(args: dict, device: torch.device) -> EnsembleModel:
    """
    Factory for infer.py's load_model_from_checkpoint() dispatch.
    Expects args["ensemble"] = {
        "components": [
            {"name": ..., "model_type": "graphsage" | "hgt", "weight": ...,
             "state_dict": {...}, "model_args": {...}},
            ...
        ]
    } — fully self-contained (embedded state dicts, not external
    checkpoint paths), so an ensemble checkpoint never depends on other
    files staying in a fixed relative location. See extension design
    notes for why self-contained was chosen over path references.
    """
    from model_graphsage import GraphSAGEAnomalyDetector
    from model_hgt import build_hgt_from_args

    components = []
    for c in args["ensemble"]["components"]:
        if c["model_type"] == "graphsage":
            sub_args = c["model_args"]
            model = GraphSAGEAnomalyDetector(
                node_feat_dims=sub_args["node_feat_dims"],
                edge_types=sub_args["edge_types"],
                edge_feat_dim=sub_args["edge_feat_dim"],
                hidden_dim=sub_args.get("hidden_dim", 128),
                num_sage_layers=sub_args.get("num_sage_layers", 2),
                dropout=0.0,
            )
        elif c["model_type"] == "hgt":
            model = build_hgt_from_args(c["model_args"])
        else:
            raise ValueError(f"Unknown ensemble component model_type {c['model_type']!r}")
        model.load_state_dict(c["state_dict"])
        model.to(device)
        model.eval()
        components.append((c.get("name", c["model_type"]), model, float(c["weight"])))

    ensemble = EnsembleModel(components)
    ensemble.to(device)
    ensemble.eval()
    return ensemble
