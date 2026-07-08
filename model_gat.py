"""
model_gat.py
============
SECONDARY / COMPARISON MODEL — Graph Attention Network (GAT) with
multi-head attention for edge-level attack detection.

Node types renamed to match the redesigned schema: "principal" and
"target" (was "awsservice" — see neo4j_graph_builder.py module docstring,
point 5, for why the target-side node is not always a clean AWS service).

GAT vs GraphSAGE for this task:
─────────────────────────────────
• GAT learns to WEIGHT neighbours differently via attention scores.
• Downside: more parameters, slower, and transductive unless combined
  with inductive tricks (GATv2 partially addresses this).
• GraphSAGE is the primary choice for inductive + scaling reasons;
  GAT is kept here as a comparison baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData


class EdgeMLP(nn.Module):
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


class GATEncoder(nn.Module):
    """
    Two-layer GATv2 encoder. Edge features are injected as an additive
    signal to source nodes BEFORE each attention layer (edge-conditioned
    input trick), ensuring that is_privilege_sensitive, edge_type, etc.
    influence the attention coefficients.
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_dim: int  = 128,
        heads: int       = 4,
        num_layers: int  = 2,
        dropout: float   = 0.3,
        edge_feat_dim: int = None,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.heads        = heads
        self.edge_feat_dim = edge_feat_dim

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim)
            for ntype, fdim in in_channels_dict.items()
        })

        if edge_feat_dim is not None:
            self.edge_projs = nn.ModuleList([
                EdgeMLP(edge_feat_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ])

        self.convs = nn.ModuleList()
        self.proj_backs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                    share_weights=False,
                )
            )
            self.proj_backs.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict:  dict | None = None,
    ) -> dict:
        h_dict = {
            ntype: F.gelu(self.input_proj[ntype](x))
            for ntype, x in x_dict.items()
        }

        for layer_i, (conv, proj_back, norm) in enumerate(
            zip(self.convs, self.proj_backs, self.norms)
        ):
            new_h = {}
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                src_h = h_dict[src_type]

                if (
                    edge_attr_dict is not None
                    and (src_type, rel, dst_type) in edge_attr_dict
                    and self.edge_feat_dim is not None
                ):
                    edge_bias = self.edge_projs[layer_i](
                        edge_attr_dict[(src_type, rel, dst_type)]
                    )
                    src_h = src_h.clone()
                    src_h.index_add_(0, edge_index[0], edge_bias)

                dst_h = h_dict[dst_type]
                out   = conv((src_h, dst_h), edge_index)
                out   = F.gelu(proj_back(out))

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


class EdgeClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, edge_feat_dim: int, dropout: float = 0.3):
        super().__init__()
        self.edge_proj = EdgeMLP(edge_feat_dim, hidden_dim, dropout)
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
        h_src:     torch.Tensor,
        h_dst:     torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_emb = self.edge_proj(edge_attr)
        combined = torch.cat([h_src, h_dst, edge_emb], dim=-1)
        return self.classifier(combined).squeeze(-1)


class GATAnomalyDetector(nn.Module):
    """Drop-in replacement for GraphSAGEAnomalyDetector — same interface, uses GATv2."""

    def __init__(
        self,
        principal_feat_dim: int,
        target_feat_dim:    int,
        edge_feat_dim:      int,
        hidden_dim:         int   = 128,
        heads:              int   = 4,
        num_gat_layers:     int   = 2,
        dropout:            float = 0.3,
    ):
        super().__init__()

        self.encoder = GATEncoder(
            in_channels_dict={
                "principal": principal_feat_dim,
                "target":    target_feat_dim,
            },
            hidden_dim=hidden_dim,
            heads=heads,
            num_layers=num_gat_layers,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
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
        edge_attr_dict = {
            ("principal", "invoked", "target"):
                data["principal", "invoked", "target"].edge_attr
        }

        h_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)

        edge_index = data["principal", "invoked", "target"].edge_index
        edge_attr  = data["principal", "invoked", "target"].edge_attr

        h_src = h_dict["principal"][edge_index[0]]
        h_dst = h_dict["target"][edge_index[1]]

        return self.head(h_src, h_dst, edge_attr)

    @torch.no_grad()
    def get_attention_weights(self, data: HeteroData) -> dict:
        """
        Design blueprint for extracting GATv2 attention coefficients per
        layer (requires register_forward_hook + return_attention_weights=True
        on each self.encoder.convs[i]).
        """
        hooks    = []
        attn_map = {}

        def make_hook(i):
            def hook(module, inp, out):
                pass
            return hook

        for i, conv in enumerate(self.encoder.convs):
            hooks.append(conv.register_forward_hook(make_hook(i)))

        self.forward(data)

        for h in hooks:
            h.remove()

        return attn_map
