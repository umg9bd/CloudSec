
"""Read-only, UI-friendly adapter over the existing blast-radius implementation.

The graph is built with neo4j_graph_builder.compute_graph(), then analyzed by
BlastRadiusEngine. This module contains no alternate reachability or scoring
formula and never writes to the source data or live graph.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd

_GRAPH_DIR = os.path.dirname(os.path.abspath(__file__))
if _GRAPH_DIR not in sys.path:
    sys.path.insert(0, _GRAPH_DIR)

import neo4j_graph_builder as nb  # noqa: E402
from blast_radius import BlastRadiusEngine, _categorize_asset  # noqa: E402

LIMITATION = (
    "Observed reachability from CloudTrail-derived graph data only; this is not proof of all "
    "permissions granted by IAM policies. The score is a lower bound on true exposure, not "
    "a complete permission analysis."
)


def _unavailable(status: str, message: str, window_rows: int = 0) -> dict:
    """Return an explicit unavailable result, never fabricated zero counts."""
    return {
        "status": status,
        "message": message,
        "limitation": LIMITATION,
        "window_rows": int(window_rows),
        "origin": None,
        "score": None,
        "reachable_count": None,
        "critical_count": None,
        "asset_categories": {},
        "reachable_resources": [],
        "paths": [],
        "snapshot": {"nodes": [], "edges": []},
    }


def _origin_candidates(value: Any) -> list[tuple[str, str]]:
    """Resolve a persisted principal value using the existing graph-builder convention."""
    if value is None:
        return []
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return [(str(value[0]), str(value[1]))]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    info = nb.parse_principal(text)
    key = nb.pf.node_key_for_principal(
        text, info.principal_type, info.name
    )
    return [(key.label, key.key)]


def calculate_event_blast_radius(
    structural_rows: pd.DataFrame | list[dict],
    *,
    source_node: Any = None,
    principal_arn: Any = None,
    username: Any = None,
    max_snapshot_nodes: int = 16,
    max_snapshot_edges: int = 24,
    max_resource_rows: int = 15,
) -> dict:
    """Build and analyze a graph from persisted structural event rows.

    source_node is the preferred origin because it is the exact principal value
    emitted by the event pipeline.

    The pipeline score CSV does not persist the training label. When absent,
    zero is supplied only for compute_graph's non-topological attack-label
    metadata. It does not affect node keys, graph relationships, reachability,
    or the existing BlastRadiusEngine score.
    """
    try:
        df = (
            structural_rows.copy()
            if isinstance(structural_rows, pd.DataFrame)
            else pd.DataFrame(structural_rows)
        )
    except Exception as exc:
        return _unavailable(
            "invalid_input",
            f"Could not read structural event rows: {type(exc).__name__}: {exc}",
        )

    required = {"log_id", "source_node", "target_node", "edge_type"}
    missing = required - set(df.columns)
    if missing:
        return _unavailable(
            "missing_graph_data",
            "Structural event data is missing: " + ", ".join(sorted(missing)),
        )
    if df.empty:
        return _unavailable(
            "missing_graph_data",
            "No structural event rows are available in this event window.",
        )

    df = df.copy()
    for col in ("log_id", "source_node", "target_node", "edge_type"):
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=list(required))
    df = df[
        (df["log_id"] != "")
        & (df["source_node"] != "")
        & (df["target_node"] != "")
        & (df["edge_type"] != "")
    ]
    df = df.drop_duplicates("log_id", keep="first").reset_index(drop=True)

    if df.empty:
        return _unavailable(
            "missing_graph_data",
            "No usable structural event rows are available in this event window.",
        )

    if "label" not in df:
        # The label is metadata only here; do not infer attack labels.
        df["label"] = 0
    else:
        df["label"] = (
            pd.to_numeric(df["label"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    try:
        ppg = nb.compute_graph(
            df[["log_id", "source_node", "target_node", "edge_type", "label"]]
        )["ppg"]
    except Exception as exc:
        return _unavailable(
            "calculation_failed",
            f"Could not construct the event-derived graph: "
            f"{type(exc).__name__}: {exc}",
            len(df),
        )

    graph = ppg.graph
    hints = (source_node, principal_arn, username)

    origin = next(
        (
            node
            for hint in hints
            for node in _origin_candidates(hint)
            if node in graph
        ),
        None,
    )

    if origin is None and username is not None:
        # Username is a fallback only if that node actually exists in the graph.
        name = str(username).strip()
        for label in ("User", "Role", "UnresolvedPrincipal"):
            candidate = (label, name)
            if candidate in graph:
                origin = candidate
                break

    if origin is None:
        supplied = next(
            (
                str(value)
                for value in hints
                if value is not None and str(value).strip()
            ),
            "the selected principal",
        )
        return _unavailable(
            "origin_unresolved",
            f"Origin principal could not be resolved in this event window "
            f"({supplied}). No blast-radius counts are reported as zero because "
            "the origin is unknown.",
            len(df),
        )

    try:
        engine = BlastRadiusEngine(ppg)
        report = engine.compute(origin)
        reachable = engine.graph.reachable_within(
            origin, engine.config.max_traversal_depth
        )
    except Exception as exc:
        return _unavailable(
            "calculation_failed",
            f"Blast-radius calculation failed: {type(exc).__name__}: {exc}",
            len(df),
        )

    critical_nodes = set(report.critical_assets.critical_assets)
    service_by_node = engine._service_by_node
    resource_type_by_node = engine._resource_type_by_node

    def node_name(node: tuple[str, str]) -> str:
        return f"{node[0]}: {node[1]}"

    resource_items = []
    for node, depth in reachable.items():
        label, key = node
        resource_items.append(
            {
                "label": label,
                "key": key,
                "display": node_name(node),
                "depth": int(depth),
                "category": _categorize_asset(
                    node, service_by_node, resource_type_by_node
                ),
                "service": service_by_node.get(node),
                "resource_type": resource_type_by_node.get(node),
                "critical": node in critical_nodes,
                "sensitivity_tier": int(
                    engine._sensitivity_by_node.get(node, -1)
                ),
            }
        )

    resource_items.sort(
        key=lambda item: (
            not item["critical"],
            -item["sensitivity_tier"],
            item["depth"],
            item["label"],
            item["key"],
        )
    )

    paths = []
    for path in report.top_paths:
        path_edges = []
        for edge in path.edges:
            path_edges.append(
                {
                    "source": list(edge["src"]),
                    "target": list(edge["dst"]),
                    "relation": str(edge.get("relation", "RELATED")),
                    "edge_type": str(edge.get("edge_type", "unknown action")),
                }
            )
        paths.append(
            {
                "nodes": [list(node) for node in path.nodes],
                "display": " → ".join(node_name(node) for node in path.nodes),
                "impact_score": float(path.impact_score),
                "edges": path_edges,
            }
        )

    # Snapshot nodes are real nodes from the graph; no synthetic nodes are added.
    selected_nodes: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()

    def add_node(node):
        if (
            node in graph
            and node not in selected_set
            and len(selected_nodes) < max_snapshot_nodes
        ):
            selected_set.add(node)
            selected_nodes.append(node)

    add_node(origin)

    for path in report.top_paths:
        for node in path.nodes:
            add_node(node)

    for _, target, _key in graph.out_edges(origin, keys=True):
        add_node(target)

    for item in resource_items:
        add_node((item["label"], item["key"]))

    node_ids = {
        node: f"n{index}" for index, node in enumerate(selected_nodes)
    }

    snapshot_nodes = []
    for node in selected_nodes:
        snapshot_nodes.append(
            {
                "id": node_ids[node],
                "label": node[0],
                "key": node[1],
                "display": node_name(node),
                "origin": node == origin,
                "depth": 0 if node == origin else reachable.get(node),
            }
        )

    path_edge_keys = {
        (
            tuple(edge["source"]),
            tuple(edge["target"]),
            edge["edge_type"],
            edge["relation"],
        )
        for path in paths
        for edge in path["edges"]
    }

    snapshot_edges = []
    for src, dst, edge_key, attrs in graph.edges(keys=True, data=True):
        if src not in selected_set or dst not in selected_set:
            continue

        edge_type = str(attrs.get("edge_type", "unknown action"))
        relation = str(attrs.get("relation", "RELATED"))

        snapshot_edges.append(
            {
                "source_id": node_ids[src],
                "target_id": node_ids[dst],
                "source": node_name(src),
                "target": node_name(dst),
                "relation": relation,
                "edge_type": edge_type,
                "priority": (
                    src, dst, edge_type, relation
                ) in path_edge_keys,
            }
        )

    snapshot_edges.sort(
        key=lambda edge: (
            not edge["priority"],
            edge["source_id"] != node_ids[origin],
        )
    )
    snapshot_edges = snapshot_edges[:max_snapshot_edges]

    return {
        "status": "ok",
        "message": None,
        "limitation": LIMITATION,
        "window_rows": int(len(df)),
        "origin": {
            "label": origin[0],
            "key": origin[1],
            "display": node_name(origin),
        },
        "score": float(report.score),
        "score_components": {
            key: float(value)
            for key, value in report.score_components.items()
        },
        "reachable_count": int(report.reachable_assets.total),
        "critical_count": int(
            report.critical_assets.critical_asset_count
        ),
        "asset_categories": {
            key: int(value)
            for key, value in report.reachable_assets.counts.items()
            if value
        },
        "reachable_resources": resource_items[:max_resource_rows],
        "additional_reachable_resources": max(
            0, len(resource_items) - max_resource_rows
        ),
        "paths": paths,
        "snapshot": {
            "nodes": snapshot_nodes,
            "edges": snapshot_edges,
        },
        "administrator_reachable_proxy": bool(
            report.privilege_reachability.administrator_reachable
        ),
        "pass_role_observed": bool(
            report.privilege_reachability.pass_role_observed
        ),
    }
