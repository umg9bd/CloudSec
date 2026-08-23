"""
model_graphsage.py  (v3 — Privilege Propagation Graph)
=========================================================
PRIMARY MODEL — GraphSAGE for edge-level attack detection over the
heterogeneous Privilege Propagation Graph (neo4j_graph_builder.py v3 /
data_loader.py v3): 6 node types, up to 20 populated
(src_type, relation, dst_type) edge triples on this dataset.

WHY THIS FILE CHANGED FROM v2, AND WHY IT'S A "MINOR" MODIFICATION
─────────────────────────────────────────────────────────────────────────
v2's graph had exactly ONE relation ("invoked") between exactly TWO node
types (principal, target), so a single SAGEConv, reapplied identically to
every edge regardless of what it represented, was a defensible choice.
The redesigned graph is genuinely heterogeneous: an ASSUMES edge and a
PERMISSIONS_MANAGEMENT edge carry very different semantics and should not
share a weight matrix. PyTorch Geometric's `HeteroConv` is the standard,
idiomatic mechanism for exactly this situation: one SAGEConv INSTANCE per
(src_type, relation, dst_type) triple, with their outputs summed per
destination node type. This is the smallest change that makes GraphSAGE
correct on a multi-relational graph — the operator is still SAGEConv, the
overall encoder/classifier-head structure is unchanged, and nothing about
GraphSAGE's neighbourhood-sampling/aggregation semantics is altered.

Because HeteroConv needs every (src,rel,dst) triple declared once at
construction time (PyTorch requires fixed module structure for correct
parameter registration), the model's constructor now takes the list of
populated triples explicitly (from `meta["populated_triples"]|`), rather
than assuming a single fixed relation name as v2 did.

GLOBAL EDGE ORDER
─────────────────────────────────────────────────────────────────────────
forward() must return logits in the SAME order as data_loader.py built
`y` — see that file's module docstring. Rather than depending on
data_loader.py's internal bookkeeping, this file independently computes
`sorted(data.edge_types)` as the canonical order; this is verified (see
development notes) to exactly match how data_loader.py groups and sorts
its edges, so the two files stay in sync without a hidden coupling to
each other's internals — only to the same well-defined sort.

THE EDGE CLASSIFIER HEAD IS SHARED ACROSS RELATIONS, DELIBERATELY
─────────────────────────────────────────────────────────────────────────
The classifier's job — given (h_src, h_dst, edge_attr), predict attack
probability — is relation-agnostic by design: it is the ENCODER's
per-relation HeteroConv weights that let the model treat an ASSUMES edge
differently from a WRITE edge structurally; the classifier head then
scores any edge the same way regardless of which relation produced its
embeddings. This also keeps `edge_attr`'s feature schema simple (one
shared EDGE_NUM_COLS/EDGE_CAT_COLS layout across all relations, as
enforced in data_loader.py) rather than needing per-relation classifier
variants.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData

EdgeTriple = Tuple[str, str, str]


# ── Edge feature MLP (unchanged from v2) ──────────────────────────────────────

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


# ── GraphSAGE encoder — now HeteroConv-based (see module docstring) ─────────

class GraphSAGEEncoder(nn.Module):
    """
    Multi-layer, per-relation GraphSAGE encoder.

    Architecture
    ────────────
    Input projection (per node TYPE)     → hidden_dim
    HeteroConv layer 1 (per (src,rel,dst) triple, own SAGEConv weights)
    HeteroConv layer 2
    ...

    A destination node type that receives edges via multiple relations
    (e.g. :Resource receives READ, WRITE, LIST, TAGGING, PERMISSIONS_
    MANAGEMENT, UNKNOWN_ACTION, and ASSUMES edges) has its per-relation
    HeteroConv outputs combined via `aggr` (default "sum" — the direct
    heterogeneous analogue of v2's manual `new_h[dst] = new_h[dst] + out`
    accumulation across relations targeting the same type).
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        edge_types: List[EdgeTriple],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_aggr: str = "sum",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_types = list(edge_types)

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim)
            for ntype, fdim in in_channels_dict.items()
        })

        self.convs = nn.ModuleList([
            HeteroConv(
                {triple: SAGEConv(hidden_dim, hidden_dim, aggr="mean", normalize=True)
                 for triple in self.edge_types},
                aggr=conv_aggr,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in in_channels_dict})
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[EdgeTriple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_dict = {ntype: F.gelu(self.input_proj[ntype](x)) for ntype, x in x_dict.items()}

        for conv, norm_dict in zip(self.convs, self.norms):
            out_dict = conv(h_dict, edge_index_dict)
            # HeteroConv only returns entries for node types that received
            # at least one edge this layer; node types with no incoming
            # edges (shouldn't happen for a fully-connected schema, but
            # guarded for robustness) keep their previous embedding.
            h_dict = {
                ntype: self.dropout(
                    norm_dict[ntype](h_dict[ntype] + out_dict.get(ntype, torch.zeros_like(h_dict[ntype])))
                )
                for ntype in h_dict
            }
        return h_dict


# ── Edge classifier head (unchanged from v2 — shared across relations) ──────

class EdgeClassifierHead(nn.Module):
    """
    Combines (h_src, h_dst, edge_attr_projected) → attack logit. Identical
    to v2's head — see module docstring for why this stays relation-
    agnostic and shared rather than being duplicated per relation.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int, dropout: float = 0.3):
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

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_proj(edge_attr)
        combined = torch.cat([h_src, h_dst, edge_emb], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ── Full GraphSAGE model ──────────────────────────────────────────────────────

class GraphSAGEAnomalyDetector(nn.Module):
    """
    Full edge-level anomaly detector over the heterogeneous Privilege
    Propagation Graph.

    Forward pass:
      1. Encode ALL node types via the HeteroConv-based GraphSAGEEncoder
         (message-passing respects each relation's own weights).
      2. For each (src, rel, dst) triple, in `sorted(data.edge_types)`
         order (see module docstring on why this exact order matters):
            gather h_src, h_dst for that triple's edges
      3. Concatenate with the triple's own edge_attr via the SHARED
         EdgeClassifierHead
      4. Concatenate all triples' logits, in that same sorted order, into
         one flat tensor — this is what aligns with data_loader.py's `y`.

    Training objective: unchanged (BCEWithLogitsLoss / FocalLoss, pos_weight
    handles imbalance) — see train.py.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        edge_types: List[EdgeTriple],
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_sage_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.edge_types = sorted(edge_types)  # canonical order, fixed at construction

        self.encoder = GraphSAGEEncoder(
            in_channels_dict=node_feat_dims,
            edge_types=self.edge_types,
            hidden_dim=hidden_dim,
            num_layers=num_sage_layers,
            dropout=dropout,
        )
        self.head = EdgeClassifierHead(hidden_dim=hidden_dim, edge_feat_dim=edge_feat_dim, dropout=dropout)

    def _encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {ntype: data[ntype].x for ntype in self.encoder.input_proj.keys()}
        edge_index_dict = {t: data[t].edge_index for t in self.edge_types if t in data.edge_types}
        return self.encoder(x_dict, edge_index_dict)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns raw logits, one flat tensor, ordered by sorted(data.edge_types)."""
        h_dict = self._encode(data)
        logits_per_triple = []
        for triple in sorted(data.edge_types):
            if triple not in self.edge_types:
                continue  # triple present in this batch but unseen at construction — skip rather than crash
            src_type, _, dst_type = triple
            edge_index = data[triple].edge_index
            edge_attr  = data[triple].edge_attr
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            logits_per_triple.append(self.head(h_src, h_dst, edge_attr))
        return torch.cat(logits_per_triple, dim=0)

    @torch.no_grad()
    def get_edge_embeddings(self, data: HeteroData) -> Dict[EdgeTriple, torch.Tensor]:
        """
        Returns {triple: [E_triple, 3*hidden_dim]} pre-classification edge
        embeddings, per triple (kept per-triple rather than concatenated,
        since downstream consumers — e.g. explainability.py — generally
        want to reason about one triple/relation at a time).
        """
        h_dict = self._encode(data)
        out = {}
        for triple in sorted(data.edge_types):
            if triple not in self.edge_types:
                continue
            src_type, _, dst_type = triple
            edge_index = data[triple].edge_index
            edge_attr  = data[triple].edge_attr
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            edge_emb = self.head.edge_proj(edge_attr)
            out[triple] = torch.cat([h_src, h_dst, edge_emb], dim=-1)
        return out


class GraphSAGEWithSampling(GraphSAGEAnomalyDetector):
    """
    Thin wrapper documenting mini-batch usage for graphs beyond this
    dataset's current ~2,900-edge scale via PyG's HeteroData-aware
    NeighborLoader:

        from torch_geometric.loader import NeighborLoader
        loader = NeighborLoader(
            data,
            num_neighbors={t: [15, 10] for t in data.edge_types},
            batch_size=512,
            input_nodes=("User", train_user_mask),
        )
    """
    pass
