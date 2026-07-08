"""
model_graphsage.py
==================
PRIMARY MODEL — GraphSAGE for edge-level attack detection.

Node types renamed to match the redesigned schema (see neo4j_graph_builder.py
and data_loader.py): "principal" and "target" (was "awsservice" — renamed
because target_node in the real dataset is not always a clean AWS service
identifier; see neo4j_graph_builder.py module docstring, point 5).

Why GraphSAGE for AWS CloudTrail?
──────────────────────────────────
1. INDUCTIVE LEARNING
   GraphSAGE learns an *aggregation function* over neighbours rather than
   memorising node-specific embeddings.  New IAM principals or targets
   that appear after training get meaningful embeddings by aggregating
   over their 1-hop neighbourhood — no retraining required.

2. UNSEEN PRINCIPALS / TARGETS
   Cloud accounts constantly spin up new roles and resources. Transductive
   models (GCN, basic GNN) fail on these. GraphSAGE handles them
   out-of-the-box. NOTE: this dataset has only 14 distinct principals, so
   an entity-disjoint evaluation of this property is high-variance here —
   see `principal_disjoint_split` in data_loader.py for the documented
   caveat.

3. LARGE-SCALE AWS GRAPHS
   GraphSAGE uses neighbourhood sampling: instead of full-graph message
   passing (O(N²)), it samples a fixed number of neighbours per layer,
   making it linear in the number of sampled edges.

Edge Feature Integration Strategy
───────────────────────────────────
GraphSAGE is node-centric; edges carry `is_read_only`,
`is_privilege_sensitive`, `action_global_frequency_log`, and the
label-encoded `edge_type` (see data_loader.py EDGE_NUM_COLS/EDGE_CAT_COLS).

We use a TWO-STAGE approach:
  Stage 1 — EdgeConditionalAggregation (not used by default here; SAGEConv
    aggregates node features only, edge features are reserved for Stage 2)
  Stage 2 — EdgeClassifier head
    After two SAGEConv layers produce node embeddings h_u and h_v,
    the INVOKED edge representation is:
        e = concat(h_u, h_v, edge_attr)
    fed through an MLP → logit for attack / normal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


# ── Edge feature MLP ──────────────────────────────────────────────────────────

class EdgeMLP(nn.Module):
    """Projects edge attributes to a vector the same size as node hidden dim."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── GraphSAGE encoder ─────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """
    Two-layer GraphSAGE encoder.

    Architecture
    ────────────
    Input projection  → hidden_dim
    SAGEConv layer 1  → hidden_dim   (aggregates 1-hop neighbours)
    SAGEConv layer 2  → hidden_dim   (aggregates 2-hop neighbours)

    Neighbour aggregation in this context:
      For a Principal node p:
        AGG = MEAN({ h_v : v ∈ N(p) })    (SAGEConv default = mean)
        h_p_new = MLP( concat(h_p, AGG) )

      This means an attacker-linked identity's embedding will be
      influenced by the resources it targeted — providing structural
      context for anomaly detection beyond individual API calls.
    """

    def __init__(
        self,
        in_channels_dict: dict,   # {"principal": F_p, "target": F_t}
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float  = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim)
            for ntype, fdim in in_channels_dict.items()
        })

        self.convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, aggr="mean", normalize=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_offset_dict: dict | None = None,
    ) -> dict:
        h_dict = {
            ntype: F.gelu(self.input_proj[ntype](x))
            for ntype, x in x_dict.items()
        }

        for conv, norm in zip(self.convs, self.norms):
            new_h = {}
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                src_h = h_dict[src_type]
                dst_h = h_dict[dst_type]
                out = conv((src_h, dst_h), edge_index)
                if dst_type in new_h:
                    new_h[dst_type] = new_h[dst_type] + out
                else:
                    new_h[dst_type] = out

            h_dict = {
                ntype: self.dropout(
                    norm(h_dict[ntype] + new_h.get(ntype, torch.zeros_like(h_dict[ntype])))
                )
                for ntype in h_dict
            }

        return h_dict


# ── Edge classifier head ──────────────────────────────────────────────────────

class EdgeClassifierHead(nn.Module):
    """
    Combines (h_src, h_dst, edge_attr_projected) → attack logit.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        num_classes: int = 2,
        dropout: float  = 0.3,
    ):
        super().__init__()
        self.edge_proj = EdgeMLP(edge_feat_dim, hidden_dim, dropout=dropout)

        concat_dim = hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_emb = self.edge_proj(edge_attr)
        combined = torch.cat([h_src, h_dst, edge_emb], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ── Full GraphSAGE model ──────────────────────────────────────────────────────

class GraphSAGEAnomalyDetector(nn.Module):
    """Full edge-level anomaly detector using GraphSAGE over (Principal, Target)."""

    def __init__(
        self,
        principal_feat_dim: int,
        target_feat_dim:    int,
        edge_feat_dim:      int,
        hidden_dim:         int = 128,
        num_sage_layers:    int = 2,
        dropout:            float = 0.3,
    ):
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            in_channels_dict={
                "principal": principal_feat_dim,
                "target":    target_feat_dim,
            },
            hidden_dim=hidden_dim,
            num_layers=num_sage_layers,
            dropout=dropout,
        )

        self.head = EdgeClassifierHead(
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {
            "principal": data["principal"].x,
            "target":    data["target"].x,
        }
        edge_index_dict = {
            ("principal", "invoked", "target"):
                data["principal", "invoked", "target"].edge_index
        }
        h_dict = self.encoder(x_dict, edge_index_dict)

        edge_index = data["principal", "invoked", "target"].edge_index
        edge_attr  = data["principal", "invoked", "target"].edge_attr

        h_src = h_dict["principal"][edge_index[0]]
        h_dst = h_dict["target"][edge_index[1]]

        logits = self.head(h_src, h_dst, edge_attr)
        return logits

    @torch.no_grad()
    def get_edge_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Returns the pre-classification edge embeddings [E, 3*hidden_dim]."""
        x_dict = {
            "principal": data["principal"].x,
            "target":    data["target"].x,
        }
        edge_index_dict = {
            ("principal", "invoked", "target"):
                data["principal", "invoked", "target"].edge_index
        }
        h_dict     = self.encoder(x_dict, edge_index_dict)
        edge_index = data["principal", "invoked", "target"].edge_index
        edge_attr  = data["principal", "invoked", "target"].edge_attr

        h_src    = h_dict["principal"][edge_index[0]]
        h_dst    = h_dict["target"][edge_index[1]]
        edge_emb = self.head.edge_proj(edge_attr)
        return torch.cat([h_src, h_dst, edge_emb], dim=-1)


class GraphSAGEWithSampling(GraphSAGEAnomalyDetector):
    """
    Thin wrapper documenting the recommended mini-batch usage for graphs
    with 100K+ edges (not needed at this dataset's current 2,900-edge scale,
    kept for scale-up).

        from torch_geometric.loader import NeighborLoader
        loader = NeighborLoader(
            data,
            num_neighbors=[15, 10],
            batch_size=512,
            edge_label_index=(
                ("principal", "invoked", "target"),
                data["principal", "invoked", "target"].edge_index,
            ),
            input_nodes=("principal", train_principal_mask),
        )
    """
    pass
