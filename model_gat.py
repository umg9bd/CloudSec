"""
model_gat.py
============
SECONDARY / COMPARISON MODEL — Graph Attention Network (GAT) with
multi-head attention for edge-level attack detection.

GAT vs GraphSAGE for this task:
─────────────────────────────────
• GAT learns to WEIGHT neighbours differently via attention scores.
  In cloud graphs, this means it can learn that calls to IAM are more
  "suspicious-neighbourhood-relevant" than calls to S3.
• Downside: more parameters, slower, and transductive unless combined
  with inductive tricks (GATv2 partially addresses this).
• GraphSAGE is the primary choice for inductive + scaling reasons;
  GAT is kept here as a comparison baseline.

Same edge-feature integration strategy as GraphSAGE:
  - edge_attr projected via EdgeMLP
  - concatenated with (h_src, h_dst) before classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData


# ── Shared edge MLP (re-used from graphsage module) ──────────────────────────

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


# ── GAT encoder ───────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Two-layer GATv2 encoder (GATv2 fixes the expressiveness limitation of
    original GAT where attention is a function of separate node features
    rather than their interaction).

    Edge features are injected as an additive signal to source nodes BEFORE
    each attention layer (edge-conditioned input trick), ensuring that
    privilege_score, action_encoded etc. influence the attention coefficients.
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

        # Input projections per node type
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim)
            for ntype, fdim in in_channels_dict.items()
        })

        # Edge feature projections (one per layer) for edge-conditioning
        if edge_feat_dim is not None:
            self.edge_projs = nn.ModuleList([
                EdgeMLP(edge_feat_dim, hidden_dim, dropout=dropout)
                for _ in range(num_layers)
            ])

        # GATv2Conv layers
        # After each layer output dim = hidden_dim * heads, projected back to hidden_dim
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
            # Project multi-head output back to hidden_dim
            self.proj_backs.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict:  dict | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        x_dict          : {ntype: [N, F]}
        edge_index_dict : {(src,rel,dst): [2, E]}
        edge_attr_dict  : {(src,rel,dst): [E, edge_feat_dim]} (optional)

        Returns
        -------
        h_dict : {ntype: [N, hidden_dim]}
        """
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

                # ── Edge-conditioned input: add projected edge signal to src ──
                if (
                    edge_attr_dict is not None
                    and (src_type, rel, dst_type) in edge_attr_dict
                    and self.edge_feat_dim is not None
                ):
                    edge_bias = self.edge_projs[layer_i](
                        edge_attr_dict[(src_type, rel, dst_type)]
                    )  # [E, hidden_dim]
                    # Scatter-add edge bias onto source nodes
                    src_h = src_h.clone()
                    src_h.index_add_(0, edge_index[0], edge_bias)

                dst_h = h_dict[dst_type]
                out   = conv((src_h, dst_h), edge_index)   # [N_dst, heads * (hidden//heads)]
                out   = F.gelu(proj_back(out))              # [N_dst, hidden_dim]

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


# ── Edge classifier head (identical structure to SAGE version) ─────────────────

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


# ── Full GAT anomaly detector ─────────────────────────────────────────────────

class GATAnomalyDetector(nn.Module):
    """
    Drop-in replacement for GraphSAGEAnomalyDetector — same interface,
    uses GATv2 instead of SAGEConv.

    Attention advantage:
      GAT can assign higher attention to "suspicious" neighbours
      (e.g., a principal that also called DeleteTrail gets a higher
      attention weight from other connected nodes), potentially
      improving recall on novel attack patterns.

    Limitation vs GraphSAGE:
      - Computationally heavier (multi-head attention per edge)
      - Less suited for truly inductive settings with unseen nodes
    """

    def __init__(
        self,
        principal_feat_dim: int,
        service_feat_dim:   int,
        edge_feat_dim:      int,
        hidden_dim:         int   = 128,
        heads:              int   = 4,
        num_gat_layers:     int   = 2,
        dropout:            float = 0.3,
    ):
        super().__init__()

        self.encoder = GATEncoder(
            in_channels_dict={
                "principal":  principal_feat_dim,
                "awsservice": service_feat_dim,
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
            "principal":  data["principal"].x,
            "awsservice": data["awsservice"].x,
        }
        edge_index_dict = {
            ("principal", "invoked", "awsservice"):
                data["principal", "invoked", "awsservice"].edge_index
        }
        edge_attr_dict = {
            ("principal", "invoked", "awsservice"):
                data["principal", "invoked", "awsservice"].edge_attr
        }

        h_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)

        edge_index = data["principal", "invoked", "awsservice"].edge_index
        edge_attr  = data["principal", "invoked", "awsservice"].edge_attr

        h_src = h_dict["principal"][edge_index[0]]
        h_dst = h_dict["awsservice"][edge_index[1]]

        return self.head(h_src, h_dst, edge_attr)

    @torch.no_grad()
    def get_attention_weights(self, data: HeteroData) -> dict:
        """
        Extract attention coefficients from every GATv2Conv layer.
        Returns {layer_i: [E, heads]} — useful for explainability.

        Note: this requires hooking into the conv layers; shown as a
        design blueprint here.  In practice, use register_forward_hook
        on each self.encoder.convs[i] to capture the attention tensor.
        """
        hooks    = []
        attn_map = {}

        def make_hook(i):
            def hook(module, inp, out):
                # GATv2Conv returns (out, attn) when return_attention_weights=True
                # For simplicity we patch with return_attention_weights in the call below
                pass
            return hook

        for i, conv in enumerate(self.encoder.convs):
            hooks.append(conv.register_forward_hook(make_hook(i)))

        self.forward(data)  # run to trigger hooks

        for h in hooks:
            h.remove()

        return attn_map
