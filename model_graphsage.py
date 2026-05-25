"""
model_graphsage.py
==================
PRIMARY MODEL — GraphSAGE for edge-level attack detection.

Why GraphSAGE for AWS CloudTrail?
──────────────────────────────────
1. INDUCTIVE LEARNING
   GraphSAGE learns an *aggregation function* over neighbours rather than
   memorising node-specific embeddings.  New IAM principals or AWS services
   that appear after training (common in growing cloud environments) get
   meaningful embeddings by aggregating over their 1-hop neighbourhood —
   no retraining required.

2. UNSEEN PRINCIPALS / SERVICES
   Cloud accounts constantly spin up new roles and services (Lambda, ECS
   tasks, etc.).  Transductive models (GCN, basic GNN) fail on these.
   GraphSAGE handles them out-of-the-box.

3. LARGE-SCALE AWS GRAPHS
   GraphSAGE uses neighbourhood sampling: instead of full-graph message
   passing (O(N²)), it samples a fixed number of neighbours per layer,
   making it linear in the number of sampled edges.  This scales to
   100 K+ CloudTrail events without memory issues.

Edge Feature Integration Strategy
───────────────────────────────────
GraphSAGE is node-centric; edges carry the richest signals here
(privilege_score, action_encoded, is_error …).

We use a TWO-STAGE approach:
  Stage 1 — EdgeConditionalAggregation
    An MLP transforms each edge's feature vector and ADDS it to the
    source node's representation before neighbourhood aggregation.
    This is equivalent to edge-conditioned message passing.

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

      This means an attacker role's embedding will be influenced by
      the AWS services it called — providing structural context for
      anomaly detection beyond individual API calls.
    """

    def __init__(
        self,
        in_channels_dict: dict,   # {"principal": F_p, "awsservice": F_s}
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float  = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-node-type input projections (handles different feature dims)
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim)
            for ntype, fdim in in_channels_dict.items()
        })

        # SAGEConv layers (applied to each node type via to_hetero)
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
        """
        Parameters
        ----------
        x_dict          : {ntype: [N, F]} node features
        edge_index_dict : {(src,rel,dst): [2, E]}
        edge_offset_dict: pre-computed edge→node offsets (optional)

        Returns
        -------
        h_dict : {ntype: [N, hidden_dim]} node embeddings
        """
        # Input projection
        h_dict = {
            ntype: F.gelu(self.input_proj[ntype](x))
            for ntype, x in x_dict.items()
        }

        # Message passing layers
        for conv, norm in zip(self.convs, self.norms):
            new_h = {}
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                src_h = h_dict[src_type]
                dst_h = h_dict[dst_type]
                # SAGEConv expects (x_src, x_dst) for bipartite graphs
                out = conv((src_h, dst_h), edge_index)
                # Accumulate (a dst node may receive messages from multiple rels)
                if dst_type in new_h:
                    new_h[dst_type] = new_h[dst_type] + out
                else:
                    new_h[dst_type] = out

            # Residual + norm + dropout
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

    The edge feature projection is the key design choice:
      raw edge features (privilege_score, action_encoded, …) are projected
      to hidden_dim and CONCATENATED with node embeddings before classification.
      This ensures no edge signal is discarded.
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

        # Concatenate: h_src(128) + h_dst(128) + edge_proj(128) = 384
        concat_dim = hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),   # binary output
        )

    def forward(
        self,
        h_src: torch.Tensor,    # [E, hidden_dim]
        h_dst: torch.Tensor,    # [E, hidden_dim]
        edge_attr: torch.Tensor # [E, edge_feat_dim]
    ) -> torch.Tensor:
        edge_emb = self.edge_proj(edge_attr)
        combined = torch.cat([h_src, h_dst, edge_emb], dim=-1)  # [E, 3*hidden]
        return self.classifier(combined).squeeze(-1)             # [E]


# ── Full GraphSAGE model ──────────────────────────────────────────────────────

class GraphSAGEAnomalyDetector(nn.Module):
    """
    Full edge-level anomaly detector using GraphSAGE.

    Forward pass:
      1. Encode nodes via GraphSAGEEncoder (message-passing)
      2. For each INVOKED edge (src, dst):
            gather h_src, h_dst
      3. Concatenate with projected edge_attr
      4. MLP classifier → logit

    Training objective: BCEWithLogitsLoss (pos_weight handles imbalance)
    """

    def __init__(
        self,
        principal_feat_dim: int,
        service_feat_dim:   int,
        edge_feat_dim:      int,
        hidden_dim:         int = 128,
        num_sage_layers:    int = 2,
        dropout:            float = 0.3,
    ):
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            in_channels_dict={
                "principal":  principal_feat_dim,
                "awsservice": service_feat_dim,
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
        """
        Returns raw logits [E] for the INVOKED edges.
        Apply sigmoid for probabilities; use BCEWithLogitsLoss during training.
        """
        # ── 1. Node embeddings ────────────────────────────────────────────────
        x_dict = {
            "principal":  data["principal"].x,
            "awsservice": data["awsservice"].x,
        }
        edge_index_dict = {
            ("principal", "invoked", "awsservice"):
                data["principal", "invoked", "awsservice"].edge_index
        }
        h_dict = self.encoder(x_dict, edge_index_dict)

        # ── 2. Gather per-edge node embeddings ────────────────────────────────
        edge_index = data["principal", "invoked", "awsservice"].edge_index  # [2, E]
        edge_attr  = data["principal", "invoked", "awsservice"].edge_attr   # [E, F]

        h_src = h_dict["principal"][edge_index[0]]    # [E, hidden_dim]
        h_dst = h_dict["awsservice"][edge_index[1]]   # [E, hidden_dim]

        # ── 3. Classify ───────────────────────────────────────────────────────
        logits = self.head(h_src, h_dst, edge_attr)   # [E]
        return logits

    # ── Embedding extraction (for explainability / LSTM hybrid) ──────────────

    @torch.no_grad()
    def get_edge_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Returns the pre-classification edge embeddings [E, 3*hidden_dim]."""
        x_dict = {
            "principal":  data["principal"].x,
            "awsservice": data["awsservice"].x,
        }
        edge_index_dict = {
            ("principal", "invoked", "awsservice"):
                data["principal", "invoked", "awsservice"].edge_index
        }
        h_dict     = self.encoder(x_dict, edge_index_dict)
        edge_index = data["principal", "invoked", "awsservice"].edge_index
        edge_attr  = data["principal", "invoked", "awsservice"].edge_attr

        h_src    = h_dict["principal"][edge_index[0]]
        h_dst    = h_dict["awsservice"][edge_index[1]]
        edge_emb = self.head.edge_proj(edge_attr)
        return torch.cat([h_src, h_dst, edge_emb], dim=-1)  # [E, 3*hidden]


# ── Mini-batch compatible wrapper ─────────────────────────────────────────────

class GraphSAGEWithSampling(GraphSAGEAnomalyDetector):
    """
    Thin wrapper that works with NeighborLoader mini-batches.
    The underlying model is unchanged; this class documents the recommended
    mini-batch usage for graphs with 100K+ edges.

    Usage (in train.py):
        from torch_geometric.loader import NeighborLoader
        loader = NeighborLoader(
            data,
            num_neighbors=[15, 10],   # sample 15 neighbours per layer
            batch_size=512,
            edge_label_index=(
                ("principal","invoked","awsservice"),
                data["principal","invoked","awsservice"].edge_index,
            ),
            input_nodes=("principal", train_principal_mask),
        )
    The sampled subgraph is a valid HeteroData — pass it directly to forward().
    """
    pass
