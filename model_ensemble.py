"""
model_ensemble.py
=================

GraphSAGE + GAT ensemble used as a comparison baseline against the
primary HGT model.

Both GraphSAGE and GAT operate on the heterogeneous IAM graph and
produce edge-level logits using the same edge-ordering contract.

The ensemble combines the component logits using a weighted average
in logit space:

    EnsembleLogit = sum(w_i * Logit_i) / sum(w_i)

The weights are renormalised independently for each edge so that the
ensemble can safely handle components that do not cover every edge
type.
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
    """
    Generic edge-level ensemble for GraphSAGE and GAT.

    Each component must implement:

        model(data) -> flat edge logits

    and expose:

        model.edge_types

    The output ordering must follow the repository-wide convention:
    edge types are processed in sorted(data.edge_types) order, with
    only edge types supported by the component included.

    Components are represented as:

        (name, model, weight)

    Weights do not need to sum to 1. They are normalised per edge over
    the components that provide a prediction for that edge type.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float]],
    ):
        super().__init__()

        if not components:
            raise ValueError(
                "EnsembleModel needs at least one component."
            )

        self.names = [component[0] for component in components]

        self.weights = [
            float(component[2])
            for component in components
        ]

        self.models = nn.ModuleList(
            [component[1] for component in components]
        )

        # Stores the logits produced by each component for the most
        # recent forward pass. Useful for debugging/explainability.
        self.last_component_coverage: (
            Dict[str, Dict[EdgeTriple, torch.Tensor]]
        ) = {}

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Produce the final edge-level ensemble logits.

        For every edge type:

            weighted_sum =
                sum(weight_i * component_logit_i)

            final_logit =
                weighted_sum / sum(weight_i)

        Only components that support a particular edge type contribute
        to that edge.

        If no component covers an edge type, the output falls back to
        a zero logit, corresponding to probability 0.5.
        """

        # Repository-wide edge ordering contract.
        all_triples = sorted(data.edge_types)

        # -------------------------------------------------------------
        # 1. Run every component and split its flat output back into
        #    per-edge-type tensors.
        # -------------------------------------------------------------

        per_component: Dict[
            str,
            Dict[EdgeTriple, torch.Tensor]
        ] = {}

        for name, model in zip(self.names, self.models):

            model_edge_types = set(
                getattr(model, "edge_types", all_triples)
            )

            covered_triples = [
                triple
                for triple in all_triples
                if triple in model_edge_types
            ]

            if not covered_triples:
                per_component[name] = {}
                continue

            # Both GraphSAGE and GAT return a flat tensor following
            # their edge-type ordering contract.
            flat_logits = model(data)

            per_triple: Dict[
                EdgeTriple,
                torch.Tensor
            ] = {}

            offset = 0

            for triple in covered_triples:

                num_edges = data[triple].edge_index.shape[1]

                per_triple[triple] = (
                    flat_logits[
                        offset : offset + num_edges
                    ]
                )

                offset += num_edges

            per_component[name] = per_triple

        # Keep the latest component predictions available for
        # debugging, explainability, and inspection.
        self.last_component_coverage = per_component

        # -------------------------------------------------------------
        # 2. Blend component logits edge-by-edge.
        # -------------------------------------------------------------

        out_chunks = []

        for triple in all_triples:

            num_edges = data[triple].edge_index.shape[1]

            device = data[triple].edge_index.device

            weighted_sum = torch.zeros(
                num_edges,
                device=device,
            )

            weight_total = torch.zeros(
                num_edges,
                device=device,
            )

            for name, weight in zip(
                self.names,
                self.weights,
            ):

                triple_logits = (
                    per_component
                    .get(name, {})
                    .get(triple)
                )

                if triple_logits is None:
                    # This component does not support this edge type.
                    continue

                weighted_sum = (
                    weighted_sum
                    + weight * triple_logits
                )

                weight_total = (
                    weight_total
                    + weight
                )

            # Avoid division by zero.
            safe_total = torch.where(
                weight_total > 0,
                weight_total,
                torch.ones_like(weight_total),
            )

            blended = torch.where(
                weight_total > 0,
                weighted_sum / safe_total,
                torch.zeros_like(weighted_sum),
            )

            out_chunks.append(blended)

        if not out_chunks:
            return torch.zeros(
                0,
                device=next(self.parameters()).device,
            )

        return torch.cat(
            out_chunks,
            dim=0,
        )


def build_ensemble_from_args(
    args: dict,
    device: torch.device,
) -> EnsembleModel:
    """
    Build the GraphSAGE + GAT comparison ensemble from a self-contained
    checkpoint configuration.

    Expected structure:

        args["ensemble"] = {
            "components": [
                {
                    "name": "graphsage",
                    "model_type": "graphsage",
                    "weight": 0.5,
                    "state_dict": {...},
                    "model_args": {...}
                },
                {
                    "name": "gat",
                    "model_type": "gat",
                    "weight": 0.5,
                    "state_dict": {...},
                    "model_args": {...}
                }
            ]
        }

    Supported model types:

        - graphsage
        - gat

    Each component stores its state_dict and model arguments directly
    inside the ensemble checkpoint. Therefore, the ensemble does not
    depend on separate checkpoint files remaining at fixed paths.
    """

    from model_graphsage import GraphSAGEAnomalyDetector
    from model_gat import GATAnomalyDetector

    components = []

    for component_config in args["ensemble"]["components"]:

        model_type = component_config["model_type"]

        sub_args = component_config["model_args"]

        # -------------------------------------------------------------
        # GraphSAGE component
        # -------------------------------------------------------------

        if model_type == "graphsage":

            model = GraphSAGEAnomalyDetector(
                node_feat_dims=sub_args["node_feat_dims"],
                edge_types=sub_args["edge_types"],
                edge_feat_dim=sub_args["edge_feat_dim"],
                hidden_dim=sub_args.get(
                    "hidden_dim",
                    128,
                ),
                num_sage_layers=sub_args.get(
                    "num_sage_layers",
                    2,
                ),
                dropout=0.0,
            )

        # -------------------------------------------------------------
        # GAT component
        # -------------------------------------------------------------

        elif model_type == "gat":

            model = GATAnomalyDetector(
                node_feat_dims=sub_args["node_feat_dims"],
                edge_types=sub_args["edge_types"],
                edge_feat_dim=sub_args["edge_feat_dim"],
                hidden_dim=sub_args.get(
                    "hidden_dim",
                    128,
                ),
                num_gat_layers=sub_args.get(
                    "num_gat_layers",
                    2,
                ),
                heads=sub_args.get(
                    "heads",
                    4,
                ),
                dropout=0.0,
            )

        else:
            raise ValueError(
                f"Unknown ensemble component model_type "
                f"{model_type!r}. Expected 'graphsage' or 'gat'."
            )

        # -------------------------------------------------------------
        # Load pretrained component weights
        # -------------------------------------------------------------

        model.load_state_dict(
            component_config["state_dict"]
        )

        model.to(device)
        model.eval()

        components.append(
            (
                component_config.get(
                    "name",
                    model_type,
                ),
                model,
                float(
                    component_config["weight"]
                ),
            )
        )

    # -------------------------------------------------------------
    # Construct final GraphSAGE + GAT ensemble
    # -------------------------------------------------------------

    ensemble = EnsembleModel(
        components
    )

    ensemble.to(device)
    ensemble.eval()

    return ensemble
