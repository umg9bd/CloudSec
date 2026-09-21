"""
model_hgt.py  (v3 — Privilege Propagation Graph)
===================================================
THIRD MODEL — Heterogeneous Graph Transformer (HGTConv) for edge-level
attack detection, added alongside GraphSAGEAnomalyDetector (primary,
model_graphsage.py) and GATAnomalyDetector (comparison baseline,
model_gat.py). GraphSAGE operates over the complete graph; HGT operates
over a ranked, sampled "important-region" heterogeneous subgraph (see
node_importance.py) at training/batch-evaluation time. The two are
combined by model_ensemble.py.

WHY THIS FILE LOOKS LIKE model_gat.py, NOT model_graphsage.py
─────────────────────────────────────────────────────────────────────────
model_graphsage.py uses PyG's `HeteroConv` because SAGEConv needs no
edge_attr — one SAGEConv instance per (src,rel,dst) triple, wired
together by HeteroConv's `{triple: conv}` pattern, is unambiguous. Like
model_gat.py's GATv2Conv, HGTConv has no native edge_attr input, so —
following the exact same "inject edge_attr as an additive bias to
source-node embeddings before message passing" trick model_gat.py
already uses — edge features are routed in manually here too.

UNLIKE model_gat.py, THIS FILE DOES NOT LOOP ONE CONV PER TRIPLE
─────────────────────────────────────────────────────────────────────────
HGTConv is natively heterogeneous: a SINGLE HGTConv instance, constructed
with `metadata = (node_types, edge_types)`, internally holds its own
per-relation-type and per-node-type attention parameters and processes
every populated triple in ONE forward(x_dict, edge_index_dict) call —
that's the whole point of HGT over "GAT applied once per relation."
Consequently the edge-bias injection here builds ONE shared, per-node-
type x_dict that already reflects every triple's edge attributes before
that single HGTConv call, rather than a fresh triple-local copy per conv
call the way GATEncoder does it. Concretely: for each (src, rel, dst)
triple, that triple's edge-derived bias is index_add_-ed into a per-
layer WORKING COPY of x_dict[src] — a node type that is the source of
multiple relations accumulates contributions from all of them into the
same tensor HGTConv then consumes for every relation at once.

AN HONEST CAVEAT (same spirit as explainability.py's module docstring)
─────────────────────────────────────────────────────────────────────────
This development environment has no working torch / torch_geometric
install, so this file is written carefully against PyG's documented
HGTConv API but has NOT been executed. The specific spot most likely to
need adjustment: whether your installed HGTConv's constructor accepts a
`dropout` kwarg directly (some versions don't — `_make_hgt_conv` below
tries it and falls back rather than crashing outright) and whether
`heads` must evenly divide `hidden_dim` (true for HGTConv the same way
it's true for GATv2Conv — enforced explicitly below rather than left to
surface as a cryptic runtime error deep inside PyG).

GLOBAL EDGE ORDER — identical contract to model_graphsage.py/model_gat.py
─────────────────────────────────────────────────────────────────────────
forward() returns logits ordered by sorted(data.edge_types), matching
data_loader.py's `y` construction order and, deliberately, requiring
ZERO changes to explainability.py: EdgeExplainer's `self.model(data)`
call and its "map (triple, local_index) -> flat position via
sorted(data.edge_types)" bookkeeping work on this model exactly as they
do on the other two, with no HGT-specific branch anywhere in that file.

GRADIENT CHECKPOINTING
─────────────────────────────────────────────────────────────────────────
use_checkpointing=True wraps each HGTConv layer's forward in
torch.utils.checkpoint.checkpoint — plain PyTorch, no PyG-version
dependency, so this part is one I'm confident is correct even
unexecuted. Trades recompute for not storing per-layer activations;
worth enabling if hidden_dim/heads tuning runs into GPU memory limits on
Colab's free tier.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]


def _triple_key(triple: EdgeTriple) -> str:
    return "__".join(triple)


import re as _re


def _make_hgt_conv(hidden_dim: int, metadata, heads: int, group: str, attn_dropout: float):
    """HGTConv's exact constructor kwargs vary across PyG versions —
    verified directly (2025-era PyG, torch_geometric==2.8.0): NEITHER
    `group=` nor `dropout=` is accepted any more (both were silently
    swallowed into **kwargs and forwarded to MessagePassing.__init__,
    which rejects them with TypeError: "unexpected keyword argument
    'group'"). The original version of this function only guarded
    `dropout`, so on a current PyG install it still crashed on `group`
    — caught by test_model_hgt.py's test_build_hgt_from_args_uses_defaults
    and test_forward_shape_matches_edge_count once torch/PyG were
    actually installed and the tests run for real (see FINAL_REPORT.md).

    Fix: attempt construction, and on TypeError("unexpected keyword
    argument 'X'") drop exactly that kwarg and retry, looping until it
    succeeds or every optional kwarg has been stripped. This adapts to
    either direction of future PyG drift (a kwarg being removed, or a
    different kwarg being renamed) without hardcoding a version check,
    and it logs exactly what was dropped rather than failing silently.
    """
    optional = {"group": group, "dropout": attn_dropout}
    dropped = []
    while True:
        try:
            return HGTConv(hidden_dim, hidden_dim, metadata, heads=heads, **optional)
        except TypeError as exc:
            m = _re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if not m or m.group(1) not in optional:
                raise  # not a kwarg-compat issue we know how to handle — surface it
            bad_kwarg = m.group(1)
            optional.pop(bad_kwarg)
            dropped.append(bad_kwarg)
            log.warning(
                "Installed HGTConv (torch_geometric) does not accept %s= (%s) — "
                "constructing without it. If this is 'group', relation-aggregation "
                "is fixed internally by this PyG version rather than configurable, "
                "and the group=%r argument passed to HGTEncoder has no effect on "
                "this install. If this is 'dropout', attn_dropout=%.3f has no "
                "effect until reconciled against your PyG version.",
                bad_kwarg, exc, group, attn_dropout,
            )


# ── Edge feature MLP — byte-for-byte the same shape as model_graphsage.py /
#    model_gat.py's version, duplicated rather than imported. This
#    codebase's existing precedent (model_gat.py already duplicates
#    rather than importing from model_graphsage.py) is to keep each model
#    file self-contained rather than couple model files to each other's
#    internals — see model_graphsage.py's module docstring for the same
#    "couple to a contract, not to internals" principle applied elsewhere.

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


class HGTEncoder(nn.Module):
    """
    Multi-layer HGT encoder over the sampled "important-region"
    heterogeneous subgraph (see node_importance.py for how that subgraph
    is selected — this class itself is agnostic to how its input
    HeteroData was chosen; it will run on the full graph too, it's just
    not intended to at training/eval time given HGT's cost per node).

    Architecture
    ────────────
    Input projection (per node TYPE)     → hidden_dim
    [edge-bias injection into a shared x_dict, per module docstring]
    HGTConv layer 1  (ONE module, metadata-aware, covers every triple)
    [edge-bias injection again, using layer 2's own edge_projs]
    HGTConv layer 2
    ...
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        edge_types: List[EdgeTriple],
        hidden_dim: int = 128,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
        edge_feat_dim: Optional[int] = None,
        group: str = "sum",
        use_checkpointing: bool = False,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}) — "
                f"HGTConv, like GATv2Conv, splits hidden_dim evenly across heads."
            )
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.edge_feat_dim = edge_feat_dim
        self.edge_types = list(edge_types)
        self.node_types = list(in_channels_dict.keys())
        self.use_checkpointing = use_checkpointing
        metadata = (self.node_types, self.edge_types)

        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(fdim, hidden_dim) for ntype, fdim in in_channels_dict.items()
        })

        if edge_feat_dim is not None:
            self.edge_projs = nn.ModuleList([
                nn.ModuleDict({_triple_key(t): EdgeMLP(edge_feat_dim, hidden_dim, dropout=dropout)
                               for t in self.edge_types})
                for _ in range(num_layers)
            ])

        # ONE HGTConv per layer — not per triple; see module docstring.
        self.convs = nn.ModuleList([
            _make_hgt_conv(hidden_dim, metadata, heads, group, attn_dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in in_channels_dict})
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def _inject_edge_bias(
        self,
        layer_i: int,
        h_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeTriple, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeTriple, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Builds the shared, edge-bias-augmented x_dict HGTConv consumes
        this layer — see module docstring for why this accumulates across
        every triple sharing a source type, rather than a fresh
        per-triple copy like GATEncoder's version."""
        if edge_attr_dict is None or self.edge_feat_dim is None:
            return h_dict
        biased = {ntype: h.clone() for ntype, h in h_dict.items()}
        for triple, edge_index in edge_index_dict.items():
            if _triple_key(triple) not in self.edge_projs[layer_i]:
                continue
            if triple not in edge_attr_dict or edge_index.shape[1] == 0:
                continue
            src_type = triple[0]
            edge_bias = self.edge_projs[layer_i][_triple_key(triple)](edge_attr_dict[triple])
            biased[src_type].index_add_(0, edge_index[0], edge_bias)
        return biased

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeTriple, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeTriple, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        h_dict = {ntype: F.gelu(self.input_proj[ntype](x)) for ntype, x in x_dict.items()}

        for layer_i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            biased_dict = self._inject_edge_bias(layer_i, h_dict, edge_index_dict, edge_attr_dict)
            if self.use_checkpointing and self.training:
                out_dict = grad_checkpoint(
                    lambda bd: conv(bd, edge_index_dict), biased_dict, use_reentrant=False
                )
            else:
                out_dict = conv(biased_dict, edge_index_dict)
            h_dict = {
                ntype: self.dropout(
                    norm_dict[ntype](h_dict[ntype] + out_dict.get(ntype, torch.zeros_like(h_dict[ntype])))
                )
                for ntype in h_dict
            }
        return h_dict


class EdgeClassifierHead(nn.Module):
    """Identical in structure to model_graphsage.py / model_gat.py's head
    — duplicated per this file's self-contained-module convention."""

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


class HGTAnomalyDetector(nn.Module):
    """
    Edge-level anomaly detector over a (typically sampled/"important
    region") heterogeneous subgraph, using HGTConv. Same public contract
    as GraphSAGEAnomalyDetector/GATAnomalyDetector: forward(data) returns
    flat logits ordered by sorted(data.edge_types). get_edge_embeddings
    mirrors GraphSAGEAnomalyDetector's version, for downstream-analysis
    parity (explainability, ensemble diagnostics).

    Has NO internal node-selection/sampling logic — it runs full message
    passing over whatever HeteroData it's handed. Which subgraph it's
    handed (the node_importance.py-selected "important region" at
    training/batch-eval time, or the same small subgraph GraphSAGE
    already scores at streaming-inference time) is decided by the
    caller, deliberately, so this class stays loader-agnostic exactly
    the way GraphSAGEAnomalyDetector already is (see that file's
    GraphSAGEWithSampling docstring for the same principle).
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        edge_types: List[EdgeTriple],
        edge_feat_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        num_hgt_layers: int = 2,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
        group: str = "sum",
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.edge_types = sorted(edge_types)  # canonical order, fixed at construction

        self.encoder = HGTEncoder(
            in_channels_dict=node_feat_dims,
            edge_types=self.edge_types,
            hidden_dim=hidden_dim,
            heads=heads,
            num_layers=num_hgt_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            edge_feat_dim=edge_feat_dim,
            group=group,
            use_checkpointing=use_checkpointing,
        )
        self.head = EdgeClassifierHead(hidden_dim=hidden_dim, edge_feat_dim=edge_feat_dim, dropout=dropout)

    def _encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {ntype: data[ntype].x for ntype in self.encoder.input_proj.keys() if ntype in data.node_types}
        edge_index_dict = {t: data[t].edge_index for t in self.edge_types if t in data.edge_types}
        edge_attr_dict = {t: data[t].edge_attr for t in self.edge_types if t in data.edge_types}
        return self.encoder(x_dict, edge_index_dict, edge_attr_dict)

    def forward(self, data: HeteroData) -> torch.Tensor:
        h_dict = self._encode(data)
        logits_per_triple = []
        for triple in sorted(data.edge_types):
            if triple not in self.edge_types:
                continue  # triple present in this batch but unseen at construction — skip rather than crash
            src_type, _, dst_type = triple
            edge_index = data[triple].edge_index
            edge_attr = data[triple].edge_attr
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            logits_per_triple.append(self.head(h_src, h_dst, edge_attr))
        return torch.cat(logits_per_triple, dim=0)

    @torch.no_grad()
    def get_edge_embeddings(self, data: HeteroData) -> Dict[EdgeTriple, torch.Tensor]:
        h_dict = self._encode(data)
        out = {}
        for triple in sorted(data.edge_types):
            if triple not in self.edge_types:
                continue
            src_type, _, dst_type = triple
            edge_index = data[triple].edge_index
            edge_attr = data[triple].edge_attr
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            edge_emb = self.head.edge_proj(edge_attr)
            out[triple] = torch.cat([h_src, h_dst, edge_emb], dim=-1)
        return out


def build_hgt_from_args(args: dict) -> HGTAnomalyDetector:
    """
    Factory used by infer.py's load_model_from_checkpoint() dispatch —
    mirrors how that function already constructs GraphSAGEAnomalyDetector
    inline, just pulled out so infer.py's own diff stays a one-line
    import plus a dispatch branch. `args` is a checkpoint's `model_args`
    dict (same shape convention as the existing GraphSAGE branch uses).
    """
    return HGTAnomalyDetector(
        node_feat_dims=args["node_feat_dims"],
        edge_types=args["edge_types"],
        edge_feat_dim=args["edge_feat_dim"],
        hidden_dim=args.get("hidden_dim", 128),
        heads=args.get("heads", 4),
        num_hgt_layers=args.get("num_hgt_layers", 2),
        dropout=args.get("dropout", 0.0),  # eval mode: dropout inactive regardless
        attn_dropout=args.get("attn_dropout", 0.0),
        group=args.get("group", "sum"),
        use_checkpointing=False,  # never useful at inference (no backward pass)
    )
