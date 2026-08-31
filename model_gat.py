"""
model_gat.py  (v3 — Privilege Propagation Graph)
===================================================
SECONDARY / COMPARISON MODEL — Graph Attention Network (GATv2) with
multi-head attention, adapted to the heterogeneous Privilege Propagation
Graph the same way model_graphsage.py was (see that file's module
docstring for the general rationale: HeteroConv/per-relation weights
instead of one relation shared across all edges).

WHY THIS FILE USES A MANUAL PER-RELATION LOOP INSTEAD OF HeteroConv
─────────────────────────────────────────────────────────────────────────
model_graphsage.py uses PyG's `HeteroConv` wrapper directly, because
SAGEConv there needs no edge_attr — HeteroConv's plain
`{triple: conv}` -> `conv(x_dict, edge_index_dict)` pattern is
unambiguous. v2's GAT encoder additionally injects edge_attr as an
attention-conditioning signal (so privilege-sensitive / high-access-level
edges get a different attention contribution than routine ones) — that
needs per-relation edge_attr routed alongside edge_index. Rather than
depend on a specific version's exact kwarg-forwarding behaviour for
HeteroConv + edge_attr (which cannot be executed/verified in this
environment), this file keeps the per-relation loop explicit: a
ModuleDict keyed by a string (PyTorch module dict keys must be strings,
so "SRC__REL__DST" is used) holding one GATv2Conv per triple, iterated
manually with the same edge-bias-injection trick as v2, generalised from
one relation to N. This is more verbose than HeteroConv but everything in
it is standard nn.Module / tensor indexing that can be reasoned about
directly.

GLOBAL EDGE ORDER: identical contract to model_graphsage.py — forward()
returns logits ordered by `sorted(data.edge_types)`, matching
data_loader.py's `y` construction order (verified consistent — see
data_loader.py's module docstring).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData

EdgeTriple = Tuple[str, str, str]


def _triple_key(triple: EdgeTriple) -> str:
    return "__".join(triple)


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
    Multi-layer, per-relation GATv2 encoder. Edge features are injected as
    an additive signal to source-node embeddings BEFORE each relation's
    attention layer (same trick as v2, now scoped per (src,rel,dst)
    triple), so is_privilege_escalation_technique, access-level-derived
    relation type, hop_count, etc. all influence attention coefficients.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        edge_types: List[EdgeTriple],
        hidden_dim: int = 128,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        edge_feat_dim: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.edge_feat_dim = edge_feat_dim
        self.edge_types = list(edge_types)

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim) for ntype, fdim in in_channels_dict.items()
        })

        if edge_feat_dim is not None:
            self.edge_projs = nn.ModuleList([
                nn.ModuleDict({_triple_key(t): EdgeMLP(edge_feat_dim, hidden_dim, dropout=dropout)
                               for t in self.edge_types})
                for _ in range(num_layers)
            ])

        self.convs = nn.ModuleList([
            nn.ModuleDict({
                _triple_key(t): GATv2Conv(
                    in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads,
                    concat=True, dropout=dropout, add_self_loops=False, share_weights=False,
                ) for t in self.edge_types
            })
            for _ in range(num_layers)
        ])
        self.proj_backs = nn.ModuleList([
            nn.ModuleDict({_triple_key(t): nn.Linear(hidden_dim, hidden_dim) for t in self.edge_types})
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in in_channels_dict})
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeTriple, torch.Tensor],
        edge_attr_dict: Dict[EdgeTriple, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_dict = {ntype: F.gelu(self.input_proj[ntype](x)) for ntype, x in x_dict.items()}

        for layer_i, (conv_dict, proj_back_dict, norm_dict) in enumerate(
            zip(self.convs, self.proj_backs, self.norms)
        ):
            new_h: Dict[str, torch.Tensor] = {}
            for triple, edge_index in edge_index_dict.items():
                if _triple_key(triple) not in conv_dict:
                    continue  # triple present in this data but unseen at construction
                src_type, _, dst_type = triple
                src_h = h_dict[src_type]

                if edge_attr_dict is not None and triple in edge_attr_dict and self.edge_feat_dim is not None:
                    edge_bias = self.edge_projs[layer_i][_triple_key(triple)](edge_attr_dict[triple])
                    src_h = src_h.clone()
                    src_h.index_add_(0, edge_index[0], edge_bias)

                dst_h = h_dict[dst_type]
                out = conv_dict[_triple_key(triple)]((src_h, dst_h), edge_index)
                out = F.gelu(proj_back_dict[_triple_key(triple)](out))

                new_h[dst_type] = new_h.get(dst_type, torch.zeros_like(h_dict[dst_type])) + out

            h_dict = {
                ntype: self.dropout(
                    norm_dict[ntype](h_dict[ntype] + new_h.get(ntype, torch.zeros_like(h_dict[ntype])))
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

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_proj(edge_attr)
        combined = torch.cat([h_src, h_dst, edge_emb], dim=-1)
        return self.classifier(combined).squeeze(-1)


class GATAnomalyDetector(nn.Module):
    """Drop-in comparison model for GraphSAGEAnomalyDetector — same interface, uses GATv2 per relation."""

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        edge_types: List[EdgeTriple],
        edge_feat_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        num_gat_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.edge_types = sorted(edge_types)

        self.encoder = GATEncoder(
            in_channels_dict=node_feat_dims,
            edge_types=self.edge_types,
            hidden_dim=hidden_dim,
            heads=heads,
            num_layers=num_gat_layers,
            dropout=dropout,
            edge_feat_dim=edge_feat_dim,
        )
        self.head = EdgeClassifierHead(hidden_dim=hidden_dim, edge_feat_dim=edge_feat_dim, dropout=dropout)

    def _encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {ntype: data[ntype].x for ntype in self.encoder.input_proj.keys()}
        edge_index_dict = {t: data[t].edge_index for t in self.edge_types if t in data.edge_types}
        edge_attr_dict  = {t: data[t].edge_attr for t in self.edge_types if t in data.edge_types}
        return self.encoder(x_dict, edge_index_dict, edge_attr_dict)

    def forward(self, data: HeteroData) -> torch.Tensor:
        h_dict = self._encode(data)
        logits_per_triple = []
        for triple in sorted(data.edge_types):
            if triple not in self.edge_types:
                continue
            src_type, _, dst_type = triple
            edge_index = data[triple].edge_index
            edge_attr  = data[triple].edge_attr
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            logits_per_triple.append(self.head(h_src, h_dst, edge_attr))
        return torch.cat(logits_per_triple, dim=0)
