"""
offline_graph.py
================
Neo4j-free route from a structural DataFrame to the exact HeteroData that
PrivilegePropagationGraphLoader.load() produces from a Neo4j graph built with
neo4j_graph_builder.build_graph().

Nothing is re-implemented: node/edge properties come from
neo4j_graph_builder.compute_graph()/node_properties()/edge_properties() -- the
same functions build_graph() writes to Neo4j -- and load() itself is the
unmodified parent method. Only the two storage reads (_fetch_nodes/_fetch_edges)
are swapped for in-memory tables shaped like their Cypher results.

Used by the real-time pipeline (pipeline.py), which rebuilds the graph over a
rolling window of recent events on every batch: rebuilding with the batch code
means streaming scores equal batch scores on the same events, which the old
incremental Neo4j updater could not guarantee (PROJECT_STATUS_REPORT.md §6.9).
"""
from __future__ import annotations

import collections
from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

import neo4j_graph_builder as nb
from data_loader import ALL_NODE_TYPES, PrivilegePropagationGraphLoader

# Neo4j labels each node with its specific type plus a super-label (see
# neo4j_graph_builder._NODE_MERGE_TEMPLATES); get_specific_label picks the former.
_SUPER_LABEL = {"User": "Principal", "Role": "Principal", "UnresolvedPrincipal": "Principal",
                "Service": "Target", "Resource": "Target", "Policy": "Target"}


def graph_tables(structural_df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """(node rows per node type, edge rows per relation), shaped like data_loader's
    _fetch_nodes/_fetch_edges results for a graph build_graph() made from structural_df."""
    df = structural_df.reset_index(drop=True)
    missing = nb.REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"structural rows are missing required columns {missing}")
    g = nb.compute_graph(df)

    nodes = collections.defaultdict(list)
    for n in g["ppg"].graph.nodes:
        label, key = n
        props = nb.node_properties(g, n)
        nodes[label].append({"key": key, "resource_type": None,
                             "is_known_attacker_identity": props.pop("is_known_attacker_identity", None),
                             **props})
    edges = collections.defaultdict(list)
    for i, row in df.iterrows():
        relation, src, dst, props = nb.edge_properties(g, i, row)
        edges[relation].append({
            "src_key": src.key, "src_labels": [src.label, _SUPER_LABEL[src.label]],
            "dst_key": dst.key, "dst_labels": [dst.label, _SUPER_LABEL[dst.label]],
            "log_id": props["log_id"], "edge_type": props["edge_type"],
            "hop_count": props["hop_count"], "privilege_gain": props["privilege_gain"],
            "privilege_gain_defined": props["privilege_gain_defined"],
            "abnormal_path_frequency": props["abnormal_path_frequency"],
            "action_global_frequency": props["action_global_frequency"],
            "is_privilege_escalation_technique": props["is_priv_esc"],
            "is_attack": props["is_attack"],
        })
    return ({t: pd.DataFrame(rows) for t, rows in nodes.items()},
            {r: pd.DataFrame(rows) for r, rows in edges.items()})


class _NoDriver:
    """Stands in for the Neo4j driver load() closes when it finishes."""

    def close(self):
        pass


class OfflineGraphLoader(PrivilegePropagationGraphLoader):
    """PrivilegePropagationGraphLoader over in-memory structural rows instead of Neo4j.
    Same constructor options (fit_artifacts, model_node_types, add_reverse_edges)."""

    def __init__(self, structural_df: pd.DataFrame, device: str = "cpu", fit_artifacts: dict = None,
                 model_node_types=None, add_reverse_edges: bool = False, source_name: str | None = None):
        self.driver = _NoDriver()
        self.device = torch.device(device)
        self._fit_artifacts = fit_artifacts
        self._model_node_types = set(model_node_types) if model_node_types is not None else None
        self._add_reverse_edges = add_reverse_edges
        self.label_encoders = {}
        self.node_scalers: Dict[str, StandardScaler] = {}
        self.edge_scaler = StandardScaler()
        self._source_name = source_name
        self._nodes, self._edges = graph_tables(structural_df)

    def _fetch_provenance(self):
        return self._source_name

    def _fetch_nodes(self, node_type: str) -> pd.DataFrame:
        return self._nodes.get(node_type, pd.DataFrame()).copy()

    def _fetch_edges(self, relation: str) -> pd.DataFrame:
        df = self._edges.get(relation)
        if df is None or not len(df):
            return pd.DataFrame()
        # The same derived columns _fetch_edges adds to a Cypher result.
        df = df.copy()
        df["src_type"] = df["src_labels"].str[0]
        df["dst_type"] = df["dst_labels"].str[0]
        df["is_read_only"] = df["edge_type"].str.startswith(
            ("Get", "List", "Describe", "Head", "Lookup", "Scan", "Query", "Search", "Check", "Validate")
        ).astype(int)
        return df


def load_offline(structural_df: pd.DataFrame, **kwargs) -> Tuple[HeteroData, dict]:
    """HeteroData + meta for structural_df, as PrivilegePropagationGraphLoader.load() would give."""
    return OfflineGraphLoader(structural_df, **kwargs).load()


assert set(_SUPER_LABEL) == set(ALL_NODE_TYPES)
