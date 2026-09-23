"""
offline_pipeline.py
======================
Neo4j-FREE validation harness for running REAL experiments (task
requirements 10/11) in an environment with no reachable Neo4j server.

WHY THIS EXISTS / WHY IT IS NOT A NEW GRAPH-BUILDING IMPLEMENTATION
─────────────────────────────────────────────────────────────────────────
This sandbox has no Neo4j instance and no route to one (network egress is
restricted to package registries — verified: the `neo4j` driver package
itself installs fine from PyPI, there's just nothing at bolt://localhost
:7687 or anywhere else to talk to). Task requirements 10 ("Do NOT claim
HGT is better until the experiments demonstrate it") and 11 ("prepare
experiments... compare... report test results once") are impossible to
honor with fabricated numbers. This file makes it possible to honor them
for real, by reproducing — not re-deriving — exactly what a live Neo4j
round trip would produce from the same CSV:

  - graph_construction/neo4j_graph_builder.py's build_graph() ALREADY
    computes every node and edge property PURELY IN PYTHON/networkx
    (via privilege_features.PrivilegePropagationGraph) before it ever
    opens a Neo4j session — the Neo4j session in that function starts at
    its `driver = GraphDatabase.driver(...)` line, strictly AFTER every
    feature is already sitting in local variables (ppg, node_out_degree,
    sensitivity_lookup, edge_features, attacker_principals, action_freq).
    This file's `_compute_graph_and_properties()` below reproduces that
    same pre-session computation block VERBATIM (same functions, same
    order, same variable names on purpose, so a line-by-line diff against
    build_graph() is easy) and stops right where build_graph() would
    start writing to Neo4j — instead handing the results to in-memory
    DataFrames shaped exactly like data_loader.py's _fetch_nodes() /
    _fetch_edges() Cypher results.
  - data_loader.py's PrivilegePropagationGraphLoader._node_features() /
    ._edge_features() (the actual scaling/encoding logic — StandardScaler,
    LabelEncoder) are called UNCHANGED, imported directly from that file,
    not reimplemented.

Net effect: every number produced through this file is the same feature
computation the production Neo4j pipeline performs, not an approximation
of it — the only thing swapped out is the storage round-trip in between.
data_loader.py itself is NOT modified and remains the production path
when a real Neo4j instance is available; this file is purely additive,
for offline validation.

MAINTENANCE COUPLING (stated plainly rather than hidden)
─────────────────────────────────────────────────────────────────────────
`load_offline()` below mirrors the BODY of
PrivilegePropagationGraphLoader.load() (data_loader.py) from the point
node_dfs/edge_dfs exist onward, because that body is not factored into a
piece callable independently of self._fetch_nodes/self._fetch_edges. If
that method's logic changes in the future, this function needs the same
change, or the two will silently drift. This coupling is the price of
getting real numbers in an offline sandbox; it is not present in, and
does not affect, the production import path (train.py / infer.py never
import this file).
"""

from __future__ import annotations

import collections
import logging
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_construction"))

import privilege_features as pf
import neo4j_graph_builder as nb  # PRIVILEGE_ESCALATION_TECHNIQUES, compute_attacker_principals, parse_principal, parse_target

from data_loader import (
    ALL_NODE_TYPES,
    RELATION_TYPES,
    PrivilegePropagationGraphLoader,
    UNREACHABLE_DISTANCE_SENTINEL,
)

log = logging.getLogger(__name__)


def _compute_graph_and_properties(csv_path: str):
    """Mirrors neo4j_graph_builder.build_graph() up to (not including) its
    `driver = GraphDatabase.driver(...)` line — see module docstring."""
    df = pd.read_csv(csv_path)
    missing = nb.REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}")

    resolver = pf.ActionAccessLevelResolver()

    principal_infos = df["source_node"].apply(nb.parse_principal)
    target_infos = df["target_node"].apply(nb.parse_target)

    src_keys = [
        pf.node_key_for_principal(arn, info.principal_type, info.name)
        for arn, info in zip(df["source_node"], principal_infos)
    ]
    dst_keys = [pf.node_key_for_target(t.value, t.resource_type, t.service) for t in target_infos]

    attacker_principals = nb.compute_attacker_principals(df)
    action_freq = df["edge_type"].value_counts()

    rows_for_graph = [
        {"log_id": lid, "source_key": sk, "target_key": dk, "edge_type": et, "label": int(lbl)}
        for lid, sk, dk, et, lbl in zip(df["log_id"], src_keys, dst_keys, df["edge_type"], df["label"])
    ]
    ppg = pf.PrivilegePropagationGraph(resolver).build_from_rows(rows_for_graph)
    edge_features = ppg.compute_all_edge_features().set_index("log_id")

    node_out_degree = collections.Counter()
    node_in_degree = collections.Counter()
    node_unique_targets = collections.defaultdict(set)
    node_unique_sources = collections.defaultdict(set)
    node_unique_actions = collections.defaultdict(set)
    for u, v, d in ppg.graph.edges(data=True):
        node_out_degree[u] += 1
        node_in_degree[v] += 1
        node_unique_targets[u].add(v)
        node_unique_sources[v].add(u)
        node_unique_actions[u].add(d["edge_type"])

    target_info_by_value = {t.value: t for t in target_infos}
    sensitivity_lookup = {}
    for n in ppg.graph.nodes:
        label, key = n
        if label in ("Service", "Resource", "Policy"):
            matching = target_info_by_value.get(key)
            svc = matching.service if matching else "unresolved"
            rtype = matching.resource_type if matching else "opaque"
            sensitivity_lookup[n] = pf.resource_sensitivity_score(svc, rtype)
        else:
            sensitivity_lookup[n] = -1

    return {
        "df": df, "src_keys": src_keys, "dst_keys": dst_keys, "target_infos": target_infos,
        "ppg": ppg, "edge_features": edge_features,
        "node_out_degree": node_out_degree, "node_in_degree": node_in_degree,
        "node_unique_targets": node_unique_targets, "node_unique_sources": node_unique_sources,
        "node_unique_actions": node_unique_actions, "sensitivity_lookup": sensitivity_lookup,
        "attacker_principals": attacker_principals, "action_freq": action_freq,
        "resolver": resolver,
    }


def _build_node_dfs(computed: dict) -> Dict[str, pd.DataFrame]:
    ppg = computed["ppg"]
    node_dfs: Dict[str, list] = {nt: [] for nt in ALL_NODE_TYPES}
    for n in ppg.graph.nodes:
        label, key = n
        row = {
            "key": key,
            "out_degree": computed["node_out_degree"].get(n, 0),
            "in_degree": computed["node_in_degree"].get(n, 0),
            "unique_targets": len(computed["node_unique_targets"].get(n, set())),
            "unique_principals": len(computed["node_unique_sources"].get(n, set())),
            "unique_actions": len(computed["node_unique_actions"].get(n, set())),
            "role_transition_count": ppg.role_transition_count(n),
            "resource_sensitivity": computed["sensitivity_lookup"].get(n, -1),
            "distance_to_sensitive_resource": ppg.distance_to_sensitive_resource(
                n, computed["sensitivity_lookup"]
            ),
            "resource_type": None,
            "is_known_attacker_identity": None,
        }
        if label in ("User", "Role", "UnresolvedPrincipal"):
            row["is_known_attacker_identity"] = key in computed["attacker_principals"]
        if label == "Resource":
            matching = next((t for t in computed["target_infos"] if t.value == key), None)
            row["resource_type"] = matching.resource_type if matching else "opaque"
        node_dfs[label].append(row)

    return {nt: pd.DataFrame(rows) for nt, rows in node_dfs.items() if rows}


def _build_edge_dfs(computed: dict) -> Dict[str, pd.DataFrame]:
    df = computed["df"]
    edge_features = computed["edge_features"]
    src_keys, dst_keys = computed["src_keys"], computed["dst_keys"]
    resolver = computed["resolver"]
    action_freq = computed["action_freq"]

    edge_dfs: Dict[str, list] = {rel: [] for rel in RELATION_TYPES}
    for i, row in df.iterrows():
        relation = pf.resolve_relation_type(str(row["edge_type"]), resolver)
        feats = edge_features.loc[row["log_id"]]
        edge_dfs[relation].append({
            "src_key": src_keys[i].key, "src_labels": [src_keys[i].label],
            "dst_key": dst_keys[i].key, "dst_labels": [dst_keys[i].label],
            "log_id": str(row["log_id"]), "edge_type": str(row["edge_type"]),
            "hop_count": int(feats["hop_count"]),
            "privilege_gain": float(feats["privilege_gain"]),
            "privilege_gain_defined": bool(feats["privilege_gain_defined"]),
            "abnormal_path_frequency": float(feats["abnormal_path_frequency"]),
            "action_global_frequency": int(action_freq[row["edge_type"]]),
            "is_privilege_escalation_technique": str(row["edge_type"]) in nb.PRIVILEGE_ESCALATION_TECHNIQUES,
            "is_attack": int(row["label"]),
        })

    out = {}
    for rel, rows in edge_dfs.items():
        if not rows:
            continue
        # Build derived columns without chained-assignment patterns so this
        # remains correct when pandas Copy-on-Write is enabled.
        d = (
            pd.DataFrame(rows)
            .copy()
            .assign(
                src_type=lambda x: x["src_labels"].apply(lambda l: l[0]),
                dst_type=lambda x: x["dst_labels"].apply(lambda l: l[0]),
                is_read_only=lambda x: x["edge_type"].str.startswith(
                    (
                        "Get", "List", "Describe", "Head", "Lookup",
                        "Scan", "Query", "Search", "Check", "Validate",
                    )
                ).astype(int),
            )
        )
        out[rel] = d
    return out


def load_offline(csv_path: str, device: str = "cpu") -> Tuple[HeteroData, dict]:
    """Neo4j-free equivalent of PrivilegePropagationGraphLoader.load() —
    see module docstring for exactly what is and is not reproduced."""
    computed = _compute_graph_and_properties(csv_path)
    node_dfs = _build_node_dfs(computed)
    edge_dfs = _build_edge_dfs(computed)

    # A loader instance built WITHOUT calling __init__ (which would try to
    # construct a live neo4j.GraphDatabase.driver) — we only need its
    # *_features methods and the scaler/encoder dicts they populate.
    loader = PrivilegePropagationGraphLoader.__new__(PrivilegePropagationGraphLoader)
    loader.device = torch.device(device)
    loader.label_encoders = {}
    loader.node_scalers = {}
    from sklearn.preprocessing import StandardScaler
    loader.edge_scaler = StandardScaler()

    node_idx: Dict[str, Dict[str, int]] = {}
    data = HeteroData()
    for ntype, ndf in node_dfs.items():
        if len(ndf) == 0:
            continue
        node_idx[ntype] = {key: i for i, key in enumerate(ndf["key"])}
        data[ntype].x = loader._node_features(ntype, ndf)
        data[ntype].key = list(ndf["key"])

    log.info("Node counts: %s", {k: v.x.shape[0] for k, v in data.node_items()})

    edge_order = []
    log_id_to_global_index: Dict[str, int] = {}

    all_edge_types = (
        pd.concat([d["edge_type"] for d in edge_dfs.values()]) if edge_dfs else pd.Series([], dtype=str)
    )
    edge_type_enc = LabelEncoder().fit(all_edge_types)
    loader.label_encoders["edge_type"] = edge_type_enc

    from data_loader import EDGE_NUM_COLS
    for rel, d in edge_dfs.items():
        edge_dfs[rel] = d.assign(
            action_global_frequency_log=np.log1p(
                d["action_global_frequency"].astype(float)
            )
        )
    all_num = (
        pd.concat([d[EDGE_NUM_COLS] for d in edge_dfs.values()])
        if edge_dfs else pd.DataFrame(columns=EDGE_NUM_COLS)
    )
    if len(all_num):
        loader.edge_scaler.fit(all_num.astype(float).values)

    combined = (
        pd.concat([d.assign(relation=rel) for rel, d in edge_dfs.items()], ignore_index=True)
        if edge_dfs else pd.DataFrame(columns=["src_type", "relation", "dst_type"])
    )
    for (src_type, rel, dst_type), gdf in combined.groupby(["src_type", "relation", "dst_type"], sort=True):
        gdf = (
            gdf.sort_values("log_id")
            .reset_index(drop=True)
            .copy()
            .assign(
                src_idx=lambda x: x["src_key"].map(node_idx[src_type]),
                dst_idx=lambda x: x["dst_key"].map(node_idx[dst_type]),
            )
            .dropna(subset=["src_idx", "dst_idx"])
            .copy()
            .assign(
                src_idx=lambda x: x["src_idx"].astype(int),
                dst_idx=lambda x: x["dst_idx"].astype(int),
            )
        )
        triple = (src_type, rel, dst_type)

        edge_index = torch.tensor(np.stack([gdf["src_idx"].values, gdf["dst_idx"].values]), dtype=torch.long)
        edge_attr = loader._edge_features(gdf, edge_type_enc)
        y = torch.tensor(gdf["is_attack"].astype(int).values, dtype=torch.long)
        log_ids = list(gdf["log_id"])

        data[triple].edge_index = edge_index
        data[triple].edge_attr = edge_attr
        data[triple].y = y
        data[triple].log_id = log_ids

        for local_i, lid in enumerate(gdf["log_id"].tolist()):
            log_id_to_global_index[lid] = len(edge_order)
            edge_order.append((src_type, rel, dst_type, local_i))

    populated_triples = sorted(data.edge_types)
    log.info("Populated (src,rel,dst) triples: %d | total edges: %d", len(populated_triples), len(edge_order))

    attacker_identity_by_key = {}
    for ntype in ("User", "Role", "UnresolvedPrincipal"):
        ndf = node_dfs.get(ntype)
        if ndf is not None and "is_known_attacker_identity" in ndf.columns:
            attacker_identity_by_key.update(dict(zip(ndf["key"], ndf["is_known_attacker_identity"])))

    edge_feat_dim = data[populated_triples[0]].edge_attr.shape[1] if populated_triples else 0
    meta = {
        "node_counts": {k: v.x.shape[0] for k, v in data.node_items()},
        "populated_triples": populated_triples,
        "edge_order": edge_order,
        "log_id_to_global_index": log_id_to_global_index,
        "node_idx": node_idx,
        "label_encoders": loader.label_encoders,
        "attacker_identity_by_key": attacker_identity_by_key,
        "edge_feat_dim": edge_feat_dim,
        "node_feat_dim": {k: v.x.shape[1] for k, v in data.node_items()},
        "edge_counts": {f"{s}__{r}__{d}": int(data[(s, r, d)].y.shape[0]) for (s, r, d) in populated_triples},
    }
    return data.to(loader.device), meta
