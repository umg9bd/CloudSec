"""
node_importance.py
====================
Ranks nodes in the Privilege Propagation Graph using ONLY features
already present on the loaded HeteroData object — no new graph
construction, no new columns, nothing written back to Neo4j or the CSV.
Feeds HGTLoader's `input_nodes` for the "important heterogeneous
subgraph" HGT trains/evaluates on (see model_hgt.py, train_scalable.py).

WHY THIS AGGREGATES EDGE FEATURES ONTO NODES
─────────────────────────────────────────────────────────────────────────
The suggested ranking signals — privilege_gain, hop_count,
abnormal_path_frequency, is_privilege_escalation_technique — are
EDGE-level columns in data_loader.py's schema (EDGE_NUM_COLS), not
node-level; resource_sensitivity, distance_to_sensitive_resource, and
degree ARE node-level (NODE_FEATURE_SCHEMA). To rank NODES using the
edge-level signals without engineering anything new, this module takes
the MAX over each node's incident edges for each edge feature — max,
not mean, because a node touched by even one highly-privileged, highly-
abnormal edge is "important" even if most of its other edges are
routine, and because this mirrors how privilege_features.py's own
hop_count is fundamentally a per-source-node quantity (constant across a
node's outgoing edges) that data_loader.py happens to store per-edge.

WHY RANK-PERCENTILE AVERAGING, NOT A HAND-WEIGHTED SUM
─────────────────────────────────────────────────────────────────────────
The input signals live on very different scales (resource_sensitivity
is a small integer tier; hop_count is 1 or 2; abnormal_path_frequency is
an unbounded -log()). Rather than inventing weights that would need
re-tuning as the dataset grows, every signal is converted to its
within-node-type percentile rank first, then averaged (equally by
default; `weights=` overrides per-signal-name if a different combination
is wanted). This is a documented modelling convention, the same spirit
as privilege_features.py's own ACCESS_LEVEL_RANK / SERVICE_SENSITIVITY
callouts about judgment calls vs. data-derived facts.

DELIBERATELY DOES NOT CALL BlastRadiusEngine
─────────────────────────────────────────────────────────────────────────
"Blast radius" is a candidate ranking signal in the extension brief; it
is NOT used here. Based on how infer.py invokes it — BlastRadiusEngine
is constructed once and queried per-principal only AFTER a malicious
prediction, inside InferenceEngine — it reads as a relatively heavy,
per-principal report rather than a lightweight per-node prefilter, and
blast_radius.py itself was not available to verify this against
directly. distance_to_sensitive_resource and resource_sensitivity are
used instead, as the closest already-materialised proxies for "how much
sensitive stuff is exposed near this node". If blast_radius.py turns out
to expose something cheaper and more node-local, this is the one place
to wire it in.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

EdgeTriple = Tuple[str, str, str]

# Position of each edge feature within data_loader.py's EDGE_NUM_COLS
# layout (see that file / explainability.py's EDGE_FEATURE_NAMES for the
# authoritative order this must track — passed in, not hardcoded, would
# be more robust still, but every existing model file already hardcodes
# this same list independently rather than importing it, so this matches
# that precedent).
_EDGE_NUM_COLS = [
    "hop_count", "privilege_gain", "privilege_gain_defined",
    "abnormal_path_frequency", "action_global_frequency_log",
    "is_privilege_escalation_technique", "is_read_only",
]
_EDGE_SIGNALS = ["hop_count", "privilege_gain", "abnormal_path_frequency", "is_privilege_escalation_technique"]
_EDGE_SIGNAL_IDX = {name: _EDGE_NUM_COLS.index(name) for name in _EDGE_SIGNALS}


def _percentile_rank(x: torch.Tensor) -> torch.Tensor:
    """Each element's fractional rank in [0, 1] among x. Falls back to
    all-0.5 for a constant or length<=1 tensor (no ranking information
    to extract, and 0.5 keeps it neutral rather than favouring/penalising
    a singleton node type in the later average)."""
    n = x.shape[0]
    if n <= 1:
        return torch.full_like(x, 0.5)
    order = x.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(n, dtype=torch.float)
    return ranks / (n - 1)


def _aggregate_edge_signals_onto_nodes(
    data: HeteroData, edge_types: List[EdgeTriple]
) -> Dict[str, torch.Tensor]:
    """
    For every node type, returns a [num_nodes, len(_EDGE_SIGNALS)]
    tensor: the MAX (see module docstring) over that node's incident
    edges, checked in BOTH directions (as source and as destination) —
    a superset of data_loader.py's principal/target split, not a
    mismatch with it: a type that never appears as an edge's dst simply
    never receives a contribution from that direction. Nodes touched by
    zero edges of a given type get 0.0 for every signal — baseline
    importance, not missing data.
    """
    node_counts = {nt: data[nt].x.shape[0] for nt in data.node_types}
    agg = {nt: torch.zeros(node_counts[nt], len(_EDGE_SIGNALS)) for nt in data.node_types}

    for triple in edge_types:
        if triple not in data.edge_types:
            continue
        src_type, _, dst_type = triple
        edge_index = data[triple].edge_index
        edge_attr = data[triple].edge_attr
        if edge_index.shape[1] == 0:
            continue
        sig_cols = torch.stack(
            [edge_attr[:, _EDGE_SIGNAL_IDX[name]] for name in _EDGE_SIGNALS], dim=1
        )
        for node_type, idx in ((src_type, edge_index[0]), (dst_type, edge_index[1])):
            agg[node_type][idx] = torch.maximum(agg[node_type][idx], sig_cols)

    return agg


def _native_node_signal(data: HeteroData, ntype: str, col_names: List[str], name: str) -> Optional[torch.Tensor]:
    if name not in col_names:
        return None
    idx = col_names.index(name)
    return data[ntype].x[:, idx]


def compute_node_importance(
    data: HeteroData,
    edge_types: List[EdgeTriple],
    node_feature_schema: Dict[str, Tuple[List[str], List[str]]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns {node_type: [num_nodes] importance score in [0, 1]}, computed
    purely from tensors already materialised on `data` — nothing
    recomputed from raw logs, nothing written back anywhere.
    `node_feature_schema` should be data_loader.py's NODE_FEATURE_SCHEMA,
    passed in rather than imported, so this module has no import-time
    dependency on data_loader.py's internals beyond the shape of that one
    dict — matching this codebase's "couple to a well-defined contract,
    not to another file's internals" convention (see model_graphsage.py's
    module docstring for the same principle applied to sort order).
    `weights` (optional): {signal_name: weight}, e.g.
    {"privilege_gain": 2.0} — unlisted signals default to weight 1.0.
    """
    edge_agg = _aggregate_edge_signals_onto_nodes(data, edge_types)
    scores: Dict[str, torch.Tensor] = {}

    for ntype in data.node_types:
        n = data[ntype].x.shape[0]
        names: List[str] = []
        components: List[torch.Tensor] = []

        for i, name in enumerate(_EDGE_SIGNALS):
            names.append(name)
            components.append(_percentile_rank(edge_agg[ntype][:, i]))

        num_cols, _ = node_feature_schema.get(ntype, ([], []))
        for name in ("resource_sensitivity", "in_degree", "out_degree"):
            sig = _native_node_signal(data, ntype, num_cols, name)
            if sig is not None:
                names.append(name)
                components.append(_percentile_rank(sig))

        # distance_to_sensitive_resource: LOWER is more important — invert
        # the percentile rank rather than the raw value, so the sentinel
        # fill value data_loader.py/infer.py both use for "no sensitive
        # resource reachable" doesn't need to be known/imported here to
        # be handled correctly.
        dist = _native_node_signal(data, ntype, num_cols, "distance_to_sensitive_resource")
        if dist is not None:
            names.append("distance_to_sensitive_resource")
            components.append(1.0 - _percentile_rank(dist))

        if not components:
            scores[ntype] = torch.zeros(n)
            continue

        stacked = torch.stack(components, dim=1)
        w = torch.tensor([(weights or {}).get(name, 1.0) for name in names], dtype=torch.float)
        scores[ntype] = (stacked * w).sum(dim=1) / w.sum()

    return scores


def select_important_nodes(
    data: HeteroData,
    edge_types: List[EdgeTriple],
    node_feature_schema: Dict[str, Tuple[List[str], List[str]]],
    top_frac: float = 0.2,
    min_per_type: int = 1,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns {node_type: LongTensor of selected node indices} — the top
    `top_frac` of each node type by compute_node_importance's score (at
    least `min_per_type` nodes per populated type, so a small or sparse
    type isn't excluded outright by rounding top_frac*n down to 0 — same
    "don't silently drop a rare type" spirit as infer.py's
    _apply_node_scaler handling of low-cardinality types like Policy).
    Feed the result into HGTLoader's `input_nodes` — verify the exact
    multi-node-type seed format your installed PyG version's HGTLoader
    expects (see train_scalable.py's honest caveat on this extension's
    least-verified corner).
    """
    scores = compute_node_importance(data, edge_types, node_feature_schema, weights=weights)
    selected: Dict[str, torch.Tensor] = {}
    for ntype, s in scores.items():
        n = s.shape[0]
        if n == 0:
            continue
        k = min(max(min_per_type, int(round(n * top_frac))), n)
        if k == 0:
            continue
        selected[ntype] = torch.topk(s, k).indices
    return selected
