"""
infer.py  —  Real-Time Streaming Inference Orchestrator
=========================================================
Converts the offline GraphSAGE training pipeline into a continuous,
incremental inference system. Watches an incoming directory for new
CloudTrail events (or pre-featurised structural rows written by
feature_engine9), processes each one end-to-end without retraining,
reloading the whole graph, or rebuilding Neo4j.

WHAT THIS FILE DOES (AND WHAT IT DELIBERATELY DOES NOT DO)
──────────────────────────────────────────────────────────────────────────
DOES:
  • Loads a trained GraphSAGE checkpoint ONCE at startup, keeps it in
    eval() mode in memory for the process lifetime.
  • Initialises IncrementalGraphUpdater with a live Neo4j session factory
    so every incoming event is written through to Neo4j incrementally
    (not in batch, not by dropping and rebuilding the graph).
  • For every new event, queries ONLY the k-hop neighbourhood of the
    changed nodes from Neo4j — never the full graph.
  • Runs GraphSAGE inference over that subgraph only.
  • When a prediction exceeds the malicious threshold, invokes
    BlastRadiusEngine on the affected principal and emits a structured
    JSON alert to ./alerts/.
  • Persists all feature-engine state (FeatureEngineer / StateTracker /
    GraphNodeTracker / VocabIndex) so a watch-mode restart doesn't reset
    per-principal velocity/session/degree history.

DOES NOT:
  • Retrain the model at any point.
  • Reload the entire graph (Neo4j or in-memory) at any point.
  • Rebuild Neo4j from scratch.
  • Load every node / every relation from Neo4j per event (only the
    k-hop neighbourhood around changed nodes is fetched).

MODULE IMPORT MAP — where each piece comes from
──────────────────────────────────────────────────────────────────────────
  feature_engine9.py      →  FeatureEngineer, iter_input_rows,
                              normalize_cloudtrail_row,
                              process_batch_file (used for seeding only),
                              STATE_TRACKER_FILE, GRAPH_NODE_STATE_FILE,
                              EVENT_NAME_VOCAB_FILE
  incremental_updater.py  →  IncrementalGraphUpdater, CloudTrailEvent,
                              IncrementalUpdateConfig
  blast_radius.py         →  BlastRadiusEngine, BlastRadiusCache,
                              BlastRadiusConfig, BlastRadiusReport
  neo4j_graph_builder.py  →  parse_principal, parse_target,
                              _NODE_MERGE_TEMPLATES, _EDGE_CREATE_TEMPLATES,
                              PRIVILEGE_ESCALATION_TECHNIQUES,
                              is_read_only (alias READ_ONLY_PREFIXES)
  privilege_features.py   →  PrivilegePropagationGraph,
                              ActionAccessLevelResolver,
                              node_key_for_principal, node_key_for_target,
                              resolve_relation_type
  model_graphsage.py      →  GraphSAGEAnomalyDetector
  data_loader.py          →  PrivilegePropagationGraphLoader,
                              EDGE_NUM_COLS, EDGE_CAT_COLS,
                              NODE_FEATURE_SCHEMA, ALL_NODE_TYPES,
                              RELATION_TYPES, UNREACHABLE_DISTANCE_SENTINEL

CHECKPOINT FORMAT (written by train.py)
──────────────────────────────────────────────────────────────────────────
train.py saves the model's state_dict only:
    torch.save(model.state_dict(), ckpt_path)

To rebuild the model at load time we also need its construction arguments
(node_feat_dims, edge_types, edge_feat_dim, hidden_dim, num_sage_layers,
dropout). train.py derives these from `meta` (from data_loader.load()).
We save those alongside the state_dict in a companion sidecar:
    torch.save({"state_dict": ..., "model_args": ...}, ckpt_path)

If you trained with the original train.py (which saves state_dict only),
run `python infer.py --wrap-checkpoint <ckpt>` ONCE to generate the
updated sidecar; the flag is described at the bottom of this file.

SUBGRAPH LOADING — WHY data_loader.py IS NOT MODIFIED
──────────────────────────────────────────────────────────────────────────
data_loader.PrivilegePropagationGraphLoader.load() fetches the entire
graph. For full-graph inference (batch / training mode) that is exactly
right. For streaming inference we need only the k-hop neighbourhood of
the two nodes touched by one incoming event.

Rather than modifying that class (which would risk breaking training),
this file adds SubgraphLoader — a standalone helper that runs targeted
Cypher queries over the same Neo4j schema, building a minimal HeteroData
object from the neighbourhood alone. SubgraphLoader reuses the SAME
feature-normalisation scalers/encoders that were fit at training time and
saved in the checkpoint sidecar (model_args["fit_artifacts"]), so the
feature distributions the model sees at inference time are identical to
training time.

FEATURE NORMALISATION AT INFERENCE TIME
──────────────────────────────────────────────────────────────────────────
data_loader.py fits a StandardScaler over the full training dataset and
transforms edge/node features before passing them to the model. At
inference time we must apply the SAME scaler (not re-fit it on each
tiny subgraph). Those fitted scalers and label-encoders are saved inside
the checkpoint sidecar under the key "fit_artifacts". SubgraphLoader
accepts them and uses sklearn's transform() (not fit_transform()) so
feature statistics are always training-distribution-aligned.
"""

from __future__ import annotations
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import argparse
import dataclasses
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from neo4j import GraphDatabase
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import HeteroData

# ── Your existing modules ────────────────────────────────────────────────────

# Feature engine — structural row producer
from feature_engine9 import (
    EVENT_NAME_VOCAB_FILE,
    GRAPH_NODE_STATE_FILE,
    STATE_TRACKER_FILE,
    FeatureEngineer,
    iter_input_rows,
    normalize_cloudtrail_row,
)

# Incremental graph updater
from incremental_updater import (
    CloudTrailEvent,
    IncrementalUpdateConfig,
    IncrementalGraphUpdater,
)

# Blast radius
from blast_radius import (
    BlastRadiusCache,
    BlastRadiusConfig,
    BlastRadiusEngine,
    BlastRadiusReport,
)

# Graph schema / parsing
import neo4j_graph_builder as nb
import privilege_features as pf

# Model — imported but NOT modified
from model_graphsage import GraphSAGEAnomalyDetector

# data_loader constants (no full load called here)
from data_loader import (
    ALL_NODE_TYPES,
    EDGE_CAT_COLS,
    EDGE_NUM_COLS,
    NODE_FEATURE_SCHEMA,
    PRINCIPAL_NODE_TYPES,
    RELATION_TYPES,
    TARGET_NODE_TYPES,
    UNREACHABLE_DISTANCE_SENTINEL,
    get_specific_label,
)

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("infer")

EdgeTriple = Tuple[str, str, str]
Node = Tuple[str, str]


def _display_ntype(labels: Optional[List[str]]) -> str:
    """
    Best-effort specific-node-type resolution for diagnostic/log output
    ONLY (never raises, unlike get_specific_label). Neo4j's labels() has
    no guaranteed ordering (see data_loader.get_specific_label's
    docstring), so diagnostics must resolve the specific label the same
    way the real conversion logic does, rather than printing labels[0]
    — otherwise the diagnostics themselves would be misleading.
    """
    if not labels:
        return "(no label)"
    try:
        return get_specific_label(labels)
    except ValueError:
        return f"(no specific label in {labels!r})"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[GraphSAGEAnomalyDetector, dict]:
    """
    Loads a trained GraphSAGEAnomalyDetector from a checkpoint file.

    Expects the checkpoint to be a dict with keys:
        "state_dict"   : model.state_dict()
        "model_args"   : {node_feat_dims, edge_types, edge_feat_dim,
                          hidden_dim, num_sage_layers, dropout}
        "fit_artifacts": {edge_scaler, node_scalers, label_encoders}
                         — sklearn objects fitted during training

    If the checkpoint is a bare state_dict (saved by the original
    train.py before this file existed), the function raises a clear error
    directing you to run --wrap-checkpoint first.
    """
    log.info("Loading checkpoint from %s …", ckpt_path)
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
        )
    if not isinstance(ckpt, dict) or "model_args" not in ckpt:
        raise ValueError(
            f"Checkpoint at {ckpt_path!r} appears to be a bare state_dict "
            f"(no 'model_args' key). Run:\n"
            f"  python infer.py --wrap-checkpoint {ckpt_path}\n"
            f"to generate a sidecar with model construction args and fit "
            f"artifacts. See the module docstring for details."
        )

    args = ckpt["model_args"]
    model = GraphSAGEAnomalyDetector(
        node_feat_dims=args["node_feat_dims"],
        edge_types=args["edge_types"],
        edge_feat_dim=args["edge_feat_dim"],
        hidden_dim=args.get("hidden_dim", 128),
        num_sage_layers=args.get("num_sage_layers", 2),
        dropout=args.get("dropout", 0.0),  # eval mode: dropout inactive regardless
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    log.info(
        "Model loaded — %d parameters | edge_types=%d | edge_feat_dim=%d",
        sum(p.numel() for p in model.parameters()),
        len(args["edge_types"]),
        args["edge_feat_dim"],
    )
    return model, ckpt.get("fit_artifacts", {})


def wrap_checkpoint(
    original_ckpt_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pass: str,
    hidden_dim: int = 128,
    num_sage_layers: int = 2,
    dropout: float = 0.3,
    output_path: Optional[str] = None,
):
    """
    One-time migration utility: wraps a bare state_dict checkpoint
    (produced by the original train.py) into the richer sidecar format
    this file needs.

    Connects to Neo4j to discover the current graph schema (edge_types,
    node_feat_dims, edge_feat_dim) and re-fits the scalers/encoders so
    the fit_artifacts key is populated. Call this once after training;
    afterwards use the wrapped checkpoint for all inference runs.
    """
    from data_loader import PrivilegePropagationGraphLoader, compute_class_weights, stratified_edge_split
    log.info("Wrapping checkpoint %s …", original_ckpt_path)

    loader = PrivilegePropagationGraphLoader(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
    data, meta = loader.load()

    state_dict = torch.load(original_ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    wrapped = {
        "state_dict": state_dict,
        "model_args": {
            "node_feat_dims": meta["node_feat_dim"],
            "edge_types": meta["populated_triples"],
            "edge_feat_dim": meta["edge_feat_dim"],
            "hidden_dim": hidden_dim,
            "num_sage_layers": num_sage_layers,
            "dropout": dropout,
        },
        "fit_artifacts": {
            "edge_scaler": loader.edge_scaler,
            "node_scalers": loader.node_scalers,
            "label_encoders": loader.label_encoders,
        },
    }

    out = output_path or original_ckpt_path.replace(".pt", "_wrapped.pt")
    torch.save(wrapped, out)
    log.info("Wrapped checkpoint saved to %s", out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SUBGRAPH LOADER
# Fetches only the k-hop neighbourhood of the two changed nodes from Neo4j.
# data_loader.py is NOT modified; this is a standalone helper.
# ══════════════════════════════════════════════════════════════════════════════

# How many hops around the changed nodes to pull into the subgraph.
# 2 is sufficient for this dataset (max observed chain depth is 2 hops —
# see privilege_features.py / blast_radius.py module docstrings). Raising
# this to 3+ is safe but increases Neo4j query cost per event.
SUBGRAPH_HOP_RADIUS = 2


class SubgraphLoader:
    """
    Loads a minimal HeteroData subgraph containing only the nodes and
    edges reachable within SUBGRAPH_HOP_RADIUS hops from a given set of
    seed node keys, using the same Neo4j schema as
    PrivilegePropagationGraphLoader.

    Scalers and encoders are accepted from outside (not re-fitted here) so
    that feature distributions stay identical to training time.
    """

    def __init__(
        self,
        driver,
        fit_artifacts: dict,
        device: torch.device,
        hop_radius: int = SUBGRAPH_HOP_RADIUS,
    ):
        self.driver = driver
        self.device = device
        self.hop_radius = hop_radius
        # Fitted during training, passed in at startup — never re-fitted.
        self.edge_scaler: StandardScaler = fit_artifacts.get("edge_scaler")
        self.node_scalers: Dict[str, StandardScaler] = fit_artifacts.get("node_scalers", {})
        self.label_encoders: Dict[str, LabelEncoder] = fit_artifacts.get("label_encoders", {})
        # Node types for which we've already logged the "unfitted/missing
        # scaler, falling back to unscaled features" warning — logged once
        # per type (not once per event) so a long-running watch loop
        # doesn't spam this on every single inference call.
        self._unscaled_ntypes_warned: Set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_neighbourhood(self, seed_keys: Set[str]) -> Optional[HeteroData]:
        """
        Queries Neo4j for every node/edge within hop_radius of the seed
        node keys. Returns a HeteroData object formatted identically to
        what PrivilegePropagationGraphLoader.load() returns, or None if
        the neighbourhood is empty (e.g. the seed is a brand-new node
        with no neighbours yet and no edges yet written to Neo4j).

        The HeteroData object has no .y tensor (we are doing inference,
        not training), so utils.evaluate() / global_labels() should not
        be called on it — use model(data) directly.
        """
        # ── Seed diagnostics ─────────────────────────────────────────────────
        print(
            f"\n  [SubgraphLoader] Seed keys passed to Neo4j query:\n"
            f"    {sorted(seed_keys)}",
            flush=True,
        )

        # Inspect each seed key: what TYPE of value is it?
        for sk in sorted(seed_keys):
            print(
                f"    seed key repr: {sk!r}  type={type(sk).__name__}",
                flush=True,
            )
            if isinstance(sk, tuple):
                print(
                    f"    WARNING: seed key is a TUPLE {sk!r}, not a plain string. "
                    f"Neo4j will compare this against n.key (a string property) "
                    f"and will NEVER match.",
                    flush=True,
                )

        # Quick Neo4j sanity check: how many nodes exist at all, and do
        # any of them match the seed keys?
        try:
            with self.driver.session() as session:
                # Total node count
                total_count_result = session.run(
                    "MATCH (n) RETURN count(n) AS total"
                ).data()
                total_nodes_in_db = total_count_result[0]["total"] if total_count_result else 0

                # Sample up to 20 node keys from Neo4j
                sample_result = session.run(
                    "MATCH (n) WHERE n.key IS NOT NULL RETURN n.key AS key, labels(n) AS labels LIMIT 20"
                ).data()

                print(
                    f"\n  [SubgraphLoader] Neo4j graph state:\n"
                    f"    Total graph nodes (all labels): {total_nodes_in_db}",
                    flush=True,
                )

                if sample_result:
                    print("    First ≤20 node keys in Neo4j:", flush=True)
                    for row in sample_result:
                        ntype = _display_ntype(row.get("labels"))
                        print(f"      ({ntype}, {row['key']!r})", flush=True)
                else:
                    print(
                        "    WARNING: No nodes with a .key property found in Neo4j at all. "
                        "The graph may be empty or nodes may be missing the .key property.",
                        flush=True,
                    )

                # Per-seed lookup
                for sk in sorted(seed_keys):
                    # Only plain-string seeds can match n.key (a string property).
                    if isinstance(sk, tuple):
                        print(
                            f"\n    Searching for seed: {sk!r}\n"
                            f"      Exact match:        False  (seed is a tuple, not a string)\n"
                            f"      Matching graph node: NONE — seed is tuple, Neo4j key is string\n"
                            f"      ROOT CAUSE: update_result.source_key.key returned a tuple "
                            f"instead of a plain string.",
                            flush=True,
                        )
                        continue

                    match_result = session.run(
                        "MATCH (n) WHERE n.key = $k RETURN n.key AS key, labels(n) AS labels LIMIT 1",
                        k=sk,
                    ).data()
                    if match_result:
                        row = match_result[0]
                        ntype = _display_ntype(row.get("labels"))
                        print(
                            f"\n    Searching for seed: {sk!r}\n"
                            f"      Exact match:        True\n"
                            f"      Matching graph node: ({ntype}, {row['key']!r})",
                            flush=True,
                        )
                    else:
                        # Try a CONTAINS search to see if a partial match exists
                        partial_result = session.run(
                            "MATCH (n) WHERE n.key CONTAINS $k OR $k CONTAINS n.key "
                            "RETURN n.key AS key, labels(n) AS labels LIMIT 3",
                            k=sk,
                        ).data()
                        if partial_result:
                            candidates = [
                                f"({_display_ntype(r.get('labels'))}, {r['key']!r})"
                                for r in partial_result
                            ]
                            print(
                                f"\n    Searching for seed: {sk!r}\n"
                                f"      Exact match:        False\n"
                                f"      No matching graph node found.\n"
                                f"      Partial/substring candidates in Neo4j: {candidates}\n"
                                f"      MISMATCH: seed key does not exactly equal any n.key in Neo4j.",
                                flush=True,
                            )
                        else:
                            print(
                                f"\n    Searching for seed: {sk!r}\n"
                                f"      Exact match:        False\n"
                                f"      No matching graph node found.\n"
                                f"      No partial matches either — this key does not appear in Neo4j at all.",
                                flush=True,
                            )
        except Exception as diag_exc:
            print(
                f"  [SubgraphLoader] Diagnostic Neo4j query failed: {diag_exc}",
                flush=True,
            )

        # ── Normal query path ─────────────────────────────────────────────────
        node_rows, edge_rows = self._fetch_neighbourhood(seed_keys)

        print(
            f"  [SubgraphLoader] _fetch_neighbourhood returned:\n"
            f"    node_rows count : {len(node_rows)}\n"
            f"    edge_rows count : {len(edge_rows)}",
            flush=True,
        )

        if not node_rows and not edge_rows:
            log.debug("Empty neighbourhood for seeds %s — no subgraph to score.", seed_keys)
            return None
        return self._build_heterodata(node_rows, edge_rows)

    # ── Neo4j queries ─────────────────────────────────────────────────────────

    def _fetch_neighbourhood(self, seed_keys: Set[str]):
        """
        Single Cypher query: find every node within hop_radius hops of ANY
        seed key, then collect all edges between those nodes.

        Variable-length path `[*0..hop_radius]` is the standard Cypher
        idiom for bounded BFS reachability — no APOC required.

        Returns (node_rows, edge_rows) as lists of dicts.
        """
        seed_list = list(seed_keys)
        with self.driver.session() as session:
            # Step 1: collect all nodes in the neighbourhood
            node_result = session.run(
                """
                MATCH (seed)
                WHERE seed.key IN $seeds
                MATCH (seed)-[*0..{hop}]-(neighbour)
                WITH collect(DISTINCT neighbour) AS neighbourhood
                UNWIND neighbourhood AS n
                RETURN
                    n.key                           AS key,
                    labels(n)                       AS labels,
                    n.out_degree                    AS out_degree,
                    n.in_degree                     AS in_degree,
                    n.unique_targets                AS unique_targets,
                    n.unique_principals             AS unique_principals,
                    n.unique_actions                AS unique_actions,
                    n.role_transition_count         AS role_transition_count,
                    n.resource_sensitivity          AS resource_sensitivity,
                    n.distance_to_sensitive_resource AS distance_to_sensitive_resource,
                    n.resource_type                 AS resource_type
                """.format(hop=self.hop_radius),
                seeds=seed_list,
            ).data()

            if not node_result:
                return [], []

            # Step 2: collect all edges WITHIN that neighbourhood
            edge_result = session.run(
                """
                MATCH (seed)
                WHERE seed.key IN $seeds
                MATCH (seed)-[*0..{hop}]-(neighbour)
                WITH collect(DISTINCT neighbour) AS neighbourhood
                UNWIND neighbourhood AS src
                MATCH (src)-[r]->(dst)
                WHERE dst IN neighbourhood
                RETURN
                    src.key AS src_key,
                    labels(src) AS src_labels,
                    dst.key AS dst_key,
                    labels(dst) AS dst_labels,
                    type(r) AS relation,
                    r.log_id AS log_id,
                    r.edge_type AS edge_type,
                    r.hop_count AS hop_count,
                    r.privilege_gain AS privilege_gain,
                    r.privilege_gain_defined AS privilege_gain_defined,
                    r.abnormal_path_frequency AS abnormal_path_frequency,
                    r.action_global_frequency AS action_global_frequency,
                    r.is_privilege_escalation_technique AS is_privilege_escalation_technique,
                    r.is_attack AS is_attack
                """.format(hop=self.hop_radius),
                seeds=seed_list,
            ).data()

        return node_result, edge_result

    # ── HeteroData builder ────────────────────────────────────────────────────

    def _build_heterodata(self, node_rows: list, edge_rows: list) -> HeteroData:
        """
        Converts raw Cypher result rows into a HeteroData object whose
        tensor layout exactly mirrors PrivilegePropagationGraphLoader.load().

        Key differences vs. the full loader:
        - No .y tensor (inference only; the label comes from Neo4j's
          is_attack property but we don't need it for the forward pass).
        - StandardScaler.transform() is called with the training-fitted
          scaler, not fit_transform(), so the feature distribution is
          training-aligned.
        - node_scaler is also the training-fitted one.

        NOTE: the block marked [DIAG] below is temporary instrumentation
        added to trace where rows are lost during row → HeteroData
        conversion. It is purely observational — no conversion logic is
        changed. Remove once the root cause is found.
        """
        # ══════════════════ [DIAG] pre-conversion snapshot ══════════════════
        print(
            f"\n  [DIAG] Node rows returned:\n"
            f"    {len(node_rows)}\n"
            f"  [DIAG] Edge rows returned:\n"
            f"    {len(edge_rows)}",
            flush=True,
        )

        _unique_node_labels = sorted({
            _display_ntype(row.get("labels")) for row in node_rows if row.get("labels")
        })
        print("\n  [DIAG] Unique node labels:", flush=True)
        if _unique_node_labels:
            for lbl in _unique_node_labels:
                print(f"    {lbl}", flush=True)
        else:
            print("    (none — no node_row had a usable 'labels' value)", flush=True)

        _unique_edge_relations = sorted({
            row["relation"] for row in edge_rows if row.get("relation")
        })
        print("\n  [DIAG] Unique edge relations:", flush=True)
        if _unique_edge_relations:
            for rel in _unique_edge_relations:
                print(f"    {rel}", flush=True)
        else:
            print("    (none — no edge_row had a usable 'relation' value)", flush=True)

        print(
            f"\n  [DIAG] ALL_NODE_TYPES accepted by node_by_type below:\n"
            f"    {sorted(ALL_NODE_TYPES)}",
            flush=True,
        )
        # ══════════════════════════════════════════════════════════════════

        data = HeteroData()

        # ── Index nodes by type ───────────────────────────────────────────────
        # labels(n) has NO guaranteed ordering in Neo4j (order is determined
        # by internal label-token IDs, not by MERGE/CREATE order — confirmed
        # Neo4j behavior: https://github.com/neo4j/neo4j/issues/13350), so
        # labels[0] can just as easily be the generic supertype ("Principal"/
        # "Target") as the specific type. get_specific_label scans the full
        # list instead — the same defense PrivilegePropagationGraphLoader
        # already uses in data_loader.py's _fetch_edges.
        node_by_type: Dict[str, List[dict]] = {t: [] for t in ALL_NODE_TYPES}
        _nodes_accepted = 0
        _skipped_nodes: List[Tuple[dict, str]] = []  # [DIAG]
        for row in node_rows:
            try:
                ntype = get_specific_label(row["labels"])
            except ValueError:
                _skipped_nodes.append(
                    (row, f"no specific label found in labels={row.get('labels')!r}; "
                          f"expected one of {sorted(ALL_NODE_TYPES)}")
                )
                continue
            if ntype in node_by_type:
                node_by_type[ntype].append(row)
                _nodes_accepted += 1
            else:
                _skipped_nodes.append(
                    (row, f"label {ntype!r} not in ALL_NODE_TYPES {sorted(ALL_NODE_TYPES)}")
                )

        # [DIAG] node conversion summary
        print(
            f"\n  [DIAG] Nodes accepted: {_nodes_accepted}\n"
            f"  [DIAG] Nodes skipped: {len(_skipped_nodes)}",
            flush=True,
        )
        for row, reason in _skipped_nodes:
            print(
                f"    Node: (key={row.get('key')!r}, labels={row.get('labels')!r})\n"
                f"    Reason skipped: {reason}",
                flush=True,
            )

        node_idx: Dict[str, Dict[str, int]] = {}
        for ntype, rows in node_by_type.items():
            if not rows:
                continue
            node_idx[ntype] = {r["key"]: i for i, r in enumerate(rows)}
            data[ntype].x = self._node_features(ntype, rows)
            data[ntype].key = [r["key"] for r in rows]

        # ── Index edges by (src_type, relation, dst_type) ────────────────────
        # Same labels()-ordering hazard as above — src_labels[0]/dst_labels[0]
        # is not safe. Use get_specific_label for both endpoints. In the
        # (should-never-happen-given-the-schema) case where an endpoint truly
        # has no specific label, tag it with a sentinel type string so it
        # naturally falls into the existing "not in node_idx" skip path below
        # with a self-describing reason, rather than crashing conversion.
        edge_by_triple: Dict[EdgeTriple, List[dict]] = {}
        for row in edge_rows:
            try:
                src_type = get_specific_label(row["src_labels"])
            except ValueError:
                src_type = f"(unresolved src labels={row.get('src_labels')!r})"
            try:
                dst_type = get_specific_label(row["dst_labels"])
            except ValueError:
                dst_type = f"(unresolved dst labels={row.get('dst_labels')!r})"
            relation = row["relation"]
            triple = (src_type, relation, dst_type)
            edge_by_triple.setdefault(triple, []).append(row)

        _edges_accepted = 0
        _skipped_edges: List[Tuple[EdgeTriple, dict, str]] = []  # [DIAG]

        for triple, rows in edge_by_triple.items():
            src_type, _, dst_type = triple
            if src_type not in node_idx or dst_type not in node_idx:
                missing = []
                if src_type not in node_idx:
                    missing.append(f"src_type {src_type!r} has no accepted nodes (not in node_idx)")
                if dst_type not in node_idx:
                    missing.append(f"dst_type {dst_type!r} has no accepted nodes (not in node_idx)")
                reason = "; ".join(missing)
                for row in rows:
                    _skipped_edges.append((triple, row, reason))
                continue  # endpoints not in our node index — skip

            src_indices, dst_indices, attr_rows = [], [], []
            for row in rows:
                si = node_idx[src_type].get(row["src_key"])
                di = node_idx[dst_type].get(row["dst_key"])
                if si is None or di is None:
                    missing = []
                    if si is None:
                        missing.append(f"src_key {row['src_key']!r} not found in node_idx[{src_type!r}]")
                    if di is None:
                        missing.append(f"dst_key {row['dst_key']!r} not found in node_idx[{dst_type!r}]")
                    _skipped_edges.append((triple, row, "; ".join(missing)))
                    continue
                src_indices.append(si)
                dst_indices.append(di)
                attr_rows.append(row)

            if not src_indices:
                continue

            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long
            )
            edge_attr = self._edge_features(attr_rows)
            data[triple].edge_index = edge_index
            data[triple].edge_attr = edge_attr
            # Attach log_ids for traceability (explainability.py uses
            # these). Plain list, not a tensor — log_id is an opaque
            # string identifier with no numeric meaning (same convention
            # as data_loader.py's identical fix / this file's `.key`
            # attributes above); torch tensors need a numeric dtype.
            data[triple].log_id = [r.get("log_id") for r in attr_rows]
            _edges_accepted += len(attr_rows)  # [DIAG]

        # [DIAG] edge conversion summary
        print(
            f"\n  [DIAG] Edges accepted: {_edges_accepted}\n"
            f"  [DIAG] Edges skipped: {len(_skipped_edges)}",
            flush=True,
        )
        for triple, row, reason in _skipped_edges:
            print(
                f"    Edge: {triple} src_key={row.get('src_key')!r} dst_key={row.get('dst_key')!r}\n"
                f"    Reason skipped: {reason}",
                flush=True,
            )

        # ══════════════════ [DIAG] final state before return ══════════════════
        print("\n  [DIAG] Node types:", flush=True)
        _total_nodes = 0
        if data.node_types:
            for ntype in data.node_types:
                n = data[ntype].x.shape[0] if hasattr(data[ntype], "x") else 0
                _total_nodes += n
                print(f"    {ntype}: {n} nodes", flush=True)
        else:
            print("    (none — HeteroData has zero node types)", flush=True)

        print("\n  [DIAG] Edge types:", flush=True)
        _total_edges = 0
        if data.edge_types:
            for etype in data.edge_types:
                n = data[etype].edge_index.shape[1] if hasattr(data[etype], "edge_index") else 0
                _total_edges += n
                print(f"    ({etype[0]}, {etype[1]}, {etype[2]}): {n}", flush=True)
        else:
            print("    (none — HeteroData has zero edge types)", flush=True)

        print("\n  [DIAG] Tensor sizes:", flush=True)
        if data.node_types or data.edge_types:
            for ntype in data.node_types:
                if hasattr(data[ntype], "x"):
                    print(f"    {ntype}.x.shape = {tuple(data[ntype].x.shape)}", flush=True)
            for etype in data.edge_types:
                if hasattr(data[etype], "edge_attr"):
                    print(f"    {etype}.edge_attr.shape = {tuple(data[etype].edge_attr.shape)}", flush=True)
                if hasattr(data[etype], "edge_index"):
                    print(f"    {etype}.edge_index.shape = {tuple(data[etype].edge_index.shape)}", flush=True)
        else:
            print("    (none — no tensors exist on an empty HeteroData)", flush=True)

        print("\n  [DIAG] Metadata:", flush=True)
        try:
            print(f"    {data.metadata()}", flush=True)
        except Exception as meta_exc:
            print(f"    hetero.metadata() raised: {meta_exc!r}", flush=True)

        print(
            f"\n  [DIAG] Total nodes in HeteroData: {_total_nodes}\n"
            f"  [DIAG] Total edges in HeteroData: {_total_edges}",
            flush=True,
        )

        if _total_nodes == 0 and node_rows:
            _reason_counts: Dict[str, int] = {}
            for _, reason in _skipped_nodes:
                _reason_counts[reason] = _reason_counts.get(reason, 0) + 1
            if _reason_counts:
                top_reason, top_count = max(_reason_counts.items(), key=lambda kv: kv[1])
                print(
                    f"    EXPLANATION: 0 nodes survived conversion out of "
                    f"{len(node_rows)} node_rows. {top_count}/{len(node_rows)} "
                    f"were skipped for: {top_reason}",
                    flush=True,
                )

        if _total_edges == 0 and edge_rows:
            _reason_counts = {}
            for _, __, reason in _skipped_edges:
                _reason_counts[reason] = _reason_counts.get(reason, 0) + 1
            if _reason_counts:
                top_reason, top_count = max(_reason_counts.items(), key=lambda kv: kv[1])
                print(
                    f"    EXPLANATION: 0 edges survived conversion out of "
                    f"{len(edge_rows)} edge_rows. {top_count}/{len(edge_rows)} "
                    f"were skipped for: {top_reason}",
                    flush=True,
                )
            elif _total_nodes == 0:
                print(
                    "    EXPLANATION: 0 edges survived because 0 nodes survived "
                    "first — every edge's (src_type, dst_type) failed the "
                    "'not in node_idx' check since node_idx is empty.",
                    flush=True,
                )
        # ══════════════════════════════════════════════════════════════════

        return data.to(self.device)

    # ── Feature builders ──────────────────────────────────────────────────────

    def _apply_node_scaler(self, ntype: str, num_arr: np.ndarray) -> np.ndarray:
        """
        Applies the training-fitted StandardScaler for `ntype`, if one
        exists — and NEVER fits or fit_transforms at inference time.

        Two distinct "no scaling happens" states, handled differently:

          1. No scaler registered for `ntype` at all (`self.node_scalers`
             has no key for it). This is a LEGITIMATE, EXPECTED state for
             any node type that had <=1 row when data_loader.py's
             _node_features() ran — see that function: it only calls
             `self.node_scalers[ntype] = scaler` immediately after a
             successful `fit_transform`, inside the same `len(df) > 1`
             branch, so a low-cardinality type is silently skipped there
             too. Training itself passes that type's raw, unscaled values
             straight through (`else: num = df[num_cols].values`). So
             "missing scaler → use raw values" is not a workaround here;
             it is mirroring exactly what training did. No warning needed
             — this is expected, not an error condition.

          2. A scaler object IS registered for `ntype`, but it fails
             check_is_fitted (no mean_/var_/scale_/n_features_in_). This
             should not be reachable from the current data_loader.py (it
             only ever stores a scaler right after fitting it
             successfully) — seeing this means the fit_artifacts in the
             loaded checkpoint were produced by a different/older version
             of that loader, i.e. a STALE wrapped checkpoint. We log a
             loud, explicit warning (once per node type, not once per
             event) and fall back to unscaled values for that type rather
             than crashing — but this is a signal to regenerate the
             wrapped checkpoint, not a state to rely on long-term.
        """
        scaler = self.node_scalers.get(ntype)
        if scaler is None or len(num_arr) == 0:
            return num_arr

        try:
            check_is_fitted(scaler)
        except NotFittedError:
            if ntype not in self._unscaled_ntypes_warned:
                log.warning(
                    "fit_artifacts['node_scalers'][%r] exists but is UNFITTED "
                    "(no mean_/var_/scale_) — this node type's checkpoint "
                    "scaler predates the current data_loader.py, most likely "
                    "because %r has too few nodes (<=1) to ever fit a scaler "
                    "on, in which case this key should not exist at all in a "
                    "freshly-wrapped checkpoint. Falling back to UNSCALED "
                    "features for %r (matching what training does for "
                    "low-cardinality node types) instead of crashing. "
                    "Re-run `infer.py --wrap-checkpoint` to regenerate "
                    "fit_artifacts and clear this warning.",
                    ntype, ntype, ntype,
                )
                self._unscaled_ntypes_warned.add(ntype)
            return num_arr

        return scaler.transform(num_arr)

    def _node_features(self, ntype: str, rows: List[dict]) -> torch.Tensor:
        """
        Builds node feature tensors using the same column layout as
        data_loader.py's NODE_FEATURE_SCHEMA and applies the
        training-fitted StandardScaler (no re-fit — see
        _apply_node_scaler's docstring for why "no scaler" is a real,
        legitimate state for low-cardinality node types like Policy,
        rather than something to paper over with a fresh unfitted one).
        """
        num_cols, cat_cols = NODE_FEATURE_SCHEMA[ntype]
        arr = []
        for row in rows:
            nums = []
            for col in num_cols:
                v = row.get(col)
                if v is None:
                    v = UNREACHABLE_DISTANCE_SENTINEL if col == "distance_to_sensitive_resource" else 0.0
                nums.append(float(v))
            arr.append(nums)
        num_arr = np.array(arr, dtype=float)
        num_arr = self._apply_node_scaler(ntype, num_arr)

        if cat_cols:
            cat_parts = []
            for col in cat_cols:
                enc: Optional[LabelEncoder] = self.label_encoders.get(f"{ntype}.{col}")
                vals = [str(row.get(col) or "unknown") for row in rows]
                if enc is not None:
                    # Handle unseen categories gracefully (treat as 0)
                    safe = [v if v in enc.classes_ else enc.classes_[0] for v in vals]
                    cat_parts.append(enc.transform(safe).astype(float).reshape(-1, 1))
                else:
                    cat_parts.append(np.zeros((len(rows), 1), dtype=float))
            feats = np.concatenate([num_arr] + cat_parts, axis=1)
        else:
            feats = num_arr

        return torch.tensor(feats, dtype=torch.float)

    def _edge_features(self, rows: List[dict]) -> torch.Tensor:
        """
        Builds edge feature tensors matching EDGE_NUM_COLS + EDGE_CAT_COLS
        (same layout as data_loader.py's _edge_features) and applies the
        training-fitted edge_scaler.
        """
        edge_type_enc: Optional[LabelEncoder] = self.label_encoders.get("edge_type")
        num_list, cat_list = [], []

        for row in rows:
            hop_count = float(row.get("hop_count") or 1)
            privilege_gain = float(row.get("privilege_gain") or 0.0)
            privilege_gain_defined = float(bool(row.get("privilege_gain_defined")))
            abnormal_path_frequency = float(row.get("abnormal_path_frequency") or 0.0)
            # action_global_frequency_log: log1p of the raw count, matching
            # data_loader._edge_features's `np.log1p(df["action_global_frequency"])`
            action_global_frequency_log = float(
                np.log1p(row.get("action_global_frequency") or 0)
            )
            is_priv_esc = float(bool(row.get("is_privilege_escalation_technique")))
            # is_read_only: derived from edge_type name prefix, same as
            # data_loader.py line: df["edge_type"].str.startswith(READ_ONLY_PREFIXES)
            is_read_only = float(
                str(row.get("edge_type") or "").startswith(
                    ("Get", "List", "Describe", "Head", "Lookup",
                     "Scan", "Query", "Search", "Check", "Validate")
                )
            )
            num_list.append([
                hop_count, privilege_gain, privilege_gain_defined,
                abnormal_path_frequency, action_global_frequency_log,
                is_priv_esc, is_read_only,
            ])

            # Categorical: edge_type label-encoded globally
            et = str(row.get("edge_type") or "unknown")
            if edge_type_enc is not None:
                safe_et = et if et in edge_type_enc.classes_ else edge_type_enc.classes_[0]
                cat_list.append(float(edge_type_enc.transform([safe_et])[0]))
            else:
                cat_list.append(0.0)

        num_arr = np.array(num_list, dtype=float)
        if self.edge_scaler is not None and len(num_arr) > 0:
            num_arr = self.edge_scaler.transform(num_arr)

        cat_arr = np.array(cat_list, dtype=float).reshape(-1, 1)
        feats = np.concatenate([num_arr, cat_arr], axis=1)
        return torch.tensor(feats, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ALERT GENERATOR
# Serialises a BlastRadiusReport into a structured JSON alert.
# ══════════════════════════════════════════════════════════════════════════════

def _serialise_blast_report(report: BlastRadiusReport) -> dict:
    """
    Converts a BlastRadiusReport into a plain-dict JSON payload.
    All dataclasses / sets / tuples are converted to JSON-safe types.
    """
    label, key = report.principal

    top_paths = []
    for p in report.top_paths:
        top_paths.append({
            "nodes": [f"{l}:{k}" for l, k in p.nodes],
            "edge_types": [e.get("edge_type", "?") for e in p.edges],
            "impact_score": round(p.impact_score, 4),
        })

    return {
        "principal_label": label,
        "principal_key": key,
        "score": round(report.score, 4),
        "score_components": {k: round(v, 4) for k, v in report.score_components.items()},
        "is_lower_bound": report.is_lower_bound,
        "reachable_assets": {
            "total": report.reachable_assets.total,
            "by_category": report.reachable_assets.counts,
        },
        "privilege_reachability": {
            "assume_role_chain_depth": report.privilege_reachability.assume_role_chain_depth,
            "administrator_reachable": report.privilege_reachability.administrator_reachable,
            "administrator_reachable_is_proxy": report.privilege_reachability.administrator_reachable_is_proxy,
            "pass_role_observed": report.privilege_reachability.pass_role_observed,
            "cross_account_ids_reachable": sorted(
                report.privilege_reachability.cross_account_ids_reachable
            ),
        },
        "critical_assets": {
            "count": report.critical_assets.critical_asset_count,
            "exposure_score": round(report.critical_assets.exposure_score, 4),
            "assets": [f"{l}:{k}" for l, k in report.critical_assets.critical_assets],
        },
        "cross_service_count": report.cross_service_count,
        "top_propagation_paths": top_paths,
    }


def generate_alert(
    event_row: dict,
    pred_prob: float,
    principal_node: Node,
    blast_report: BlastRadiusReport,
    alert_dir: str,
) -> dict:
    """
    Builds a JSON alert dict and writes it to alert_dir.

    Alert schema:
        alert_id        : UUID4
        generated_at    : ISO-8601 UTC timestamp
        event           : the triggering CloudTrail event fields
        prediction      : {probability, principal_node}
        blast_radius    : serialised BlastRadiusReport
    """
    alert = {
        "alert_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event": {
            "timestamp": event_row.get("timestamp"),
            "principal_arn": event_row.get("principal_arn"),
            "principal_type": event_row.get("principal_type"),
            "event_name": event_row.get("event_name"),
            "event_source": event_row.get("event_source"),
            "target_resource": event_row.get("target_resource"),
            "source_ip": event_row.get("source_ip"),
            "aws_region": event_row.get("aws_region"),
            "error_code": event_row.get("error_code"),
        },
        "prediction": {
            "probability": round(float(pred_prob), 6),
            "principal_label": principal_node[0],
            "principal_key": principal_node[1],
        },
        "blast_radius": _serialise_blast_report(blast_report),
    }

    os.makedirs(alert_dir, exist_ok=True)
    fname = f"alert_{alert['alert_id']}.json"
    fpath = os.path.join(alert_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2, default=str)
    log.warning("🚨 ALERT written → %s (prob=%.4f)", fpath, pred_prob)
    return alert


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INFERENCE ENGINE
# The hot path: feature → graph update → subgraph load → model → alert.
# ══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Holds all long-lived state in memory and processes one CloudTrail
    event at a time through the full pipeline.

    All heavy initialisation (Neo4j connection, model load, PPG build,
    BlastRadiusEngine index) is done once in __init__. Per-event work
    is confined to process_event().
    """

    def __init__(
        self,
        ckpt_path: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_pass: str,
        alert_dir: str = "./alerts",
        malicious_threshold: float = 0.5,
        device: str = "cpu",
        hop_radius: int = SUBGRAPH_HOP_RADIUS,
    ):
        self.alert_dir = alert_dir
        self.threshold = malicious_threshold
        self.device = torch.device(device)

        # 1. Load model — kept in eval() mode, never switched to train().
        self.model, fit_artifacts = load_model_from_checkpoint(ckpt_path, self.device)
        self.model.eval()

        # 2. Neo4j driver — one persistent driver, sessions opened per write.
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        log.info("Neo4j connected at %s", neo4j_uri)

        def _session_factory():
            return self.driver.session()

        # 3. In-memory Privilege Propagation Graph.
        #    Seeded empty here — incremental updater will populate it as
        #    events arrive. If you want to seed from existing Neo4j state,
        #    call _seed_ppg_from_neo4j() below before starting the watch loop.
        self.ppg = pf.PrivilegePropagationGraph()

        # 4. IncrementalGraphUpdater — handles the in-memory graph and Neo4j
        #    write-through for every event. BlastRadiusCache is wired in so
        #    it gets invalidated whenever a relevant principal is updated.
        brc = BlastRadiusConfig()
        self.blast_cache = BlastRadiusCache()
        self.updater = IncrementalGraphUpdater(
            ppg=self.ppg,
            config=IncrementalUpdateConfig.from_blast_radius_config(brc),
            neo4j_session_factory=_session_factory,
            blast_radius_cache=self.blast_cache,
        )

        # 5. BlastRadiusEngine — read-only, operates on self.ppg.
        self.blast_engine = BlastRadiusEngine(self.ppg, config=brc)

        # 6. SubgraphLoader — targeted Neo4j queries for inference subgraphs.
        self.subgraph_loader = SubgraphLoader(
            driver=self.driver,
            fit_artifacts=fit_artifacts,
            device=self.device,
            hop_radius=hop_radius,
        )

        # 7. Feature engineer — stateful; persists velocity / session /
        #    degree history across restarts via STATE_TRACKER_FILE etc.
        self.feature_engineer = FeatureEngineer(
            event_name_vocab_path=EVENT_NAME_VOCAB_FILE,
            state_tracker_path=STATE_TRACKER_FILE,
            graph_state_path=GRAPH_NODE_STATE_FILE,
        )

        # Running log_id counter (just needs to be monotone per session;
        # not critical for correctness — log_id is used only for
        # traceability in UpdateResult / explainability, not for features).
        # Kept as a plain int internally for cheap `+= 1` arithmetic, but
        # every CloudTrailEvent built from it stringifies it (see
        # process_event) — log_id is a string everywhere else in the
        # project (Feature Engine batch rows use
        # "<source_file>:<row_index>"), and a "live:" prefix keeps
        # runtime-generated ids visually distinct from those batch ids
        # without the two ever colliding.
        self._log_id_counter = int(time.time() * 1000)

        log.info("InferenceEngine initialised. Threshold=%.2f", self.threshold)

    # ── Optional: seed in-memory PPG from existing Neo4j data ────────────────

    def seed_ppg_from_neo4j(self) -> int:
        """
        Populates self.ppg from the current Neo4j graph so that the
        in-memory view is consistent with what's already stored. Call this
        ONCE at startup if Neo4j already contains historical events.

        Runs the same Cypher queries as data_loader.py but feeds the
        results into ppg.graph directly via NetworkX, without building
        PyTorch tensors, so it does not trigger a full data load.

        Returns the number of edges ingested.
        """
        log.info("Seeding in-memory PPG from Neo4j …")
        count = 0
        with self.driver.session() as session:
            for rel in RELATION_TYPES:
                # NOTE: this used to do `labels(src)[0] AS src_type` /
                # `labels(dst)[0] AS dst_type` directly in Cypher. That is
                # the same unsafe-ordering bug as SubgraphLoader had, except
                # worse here: once Cypher reduces the list to a single
                # value server-side, the full labels list is gone by the
                # time the row reaches Python, so there is nothing left to
                # recover the correct type from. Fetching the full
                # labels(src)/labels(dst) list and resolving it in Python
                # via get_specific_label (same helper data_loader.py and
                # SubgraphLoader use) fixes this the same way.
                rows = session.run(
                    f"""
                    MATCH (src)-[r:{rel}]->(dst)
                    RETURN src.key AS src_key, labels(src) AS src_labels,
                           dst.key AS dst_key, labels(dst) AS dst_labels,
                           r.log_id AS log_id, r.edge_type AS edge_type,
                           r.relation AS relation, r.access_level AS access_level,
                           r.hop_count AS hop_count, r.is_attack AS is_attack,
                           r.abnormal_path_frequency AS abnormal_path_freq
                    """
                ).data()
                for row in rows:
                    try:
                        src_type = get_specific_label(row["src_labels"])
                        dst_type = get_specific_label(row["dst_labels"])
                    except ValueError:
                        log.warning(
                            "seed_ppg_from_neo4j: skipping %s edge (src_key=%r, dst_key=%r) — "
                            "could not resolve a specific node type from src_labels=%r / dst_labels=%r",
                            rel, row.get("src_key"), row.get("dst_key"),
                            row.get("src_labels"), row.get("dst_labels"),
                        )
                        continue
                    src_node = (src_type, row["src_key"])
                    dst_node = (dst_type, row["dst_key"])
                    if src_node not in self.ppg.graph:
                        self.ppg.graph.add_node(src_node)
                        self.updater._known_nodes.add(src_node)
                    if dst_node not in self.ppg.graph:
                        self.ppg.graph.add_node(dst_node)
                        self.updater._known_nodes.add(dst_node)
                    self.ppg.graph.add_edge(
                        src_node, dst_node,
                        log_id=row.get("log_id"),
                        edge_type=row.get("edge_type", ""),
                        relation=rel,
                        access_level=row.get("access_level"),
                        label=int(row.get("is_attack") or 0),
                        hop_count=int(row.get("hop_count") or 1),
                        abnormal_path_frequency=float(row.get("abnormal_path_freq") or 0.0),
                    )
                    # Keep updater's frequency counters consistent
                    et = row.get("edge_type", "")
                    self.updater._action_freq[et] = self.updater._action_freq.get(et, 0) + 1
                    self.updater._total_edges += 1
                    count += 1
        # Re-index known_nodes
        self.updater._known_nodes = set(self.ppg.graph.nodes)
        log.info("PPG seeded: %d nodes, %d edges", self.ppg.graph.number_of_nodes(), count)
        return count

    # ── Hot path: process one CloudTrail event ────────────────────────────────

    @torch.no_grad()
    def process_event(self, raw_log: dict) -> Optional[dict]:
        """
        Full pipeline for one CloudTrail event (normalised dict).

        Steps:
          1. Feature engineering (structural row from FeatureEngineer).
          2. Incremental graph update (in-memory PPG + Neo4j write-through).
          3. Subgraph load from Neo4j (neighbourhood of changed nodes only).
          4. GraphSAGE inference on the subgraph.
          5. If prediction >= threshold: blast radius + alert generation.

        Returns the alert dict if malicious, else None.
        """
        t_start = time.perf_counter()

        # Ensure diagnostic counters exist even if process_file wasn't the
        # caller (e.g. direct engine.process_event() calls in tests).
        if not hasattr(self, "_diag"):
            self._diag = {
                "events": 0,
                "subgraphs_built": 0,
                "model_inferences": 0,
                "malicious_predictions": 0,
                "blast_radius_executions": 0,
                "alerts_generated": 0,
            }

        self._diag["events"] += 1

        # ── Diagnostic header ─────────────────────────────────────────────────
        _log_id_preview = raw_log.get("log_id") or raw_log.get("eventID") or "(unknown)"
        _event_name     = raw_log.get("event_name") or raw_log.get("eventName") or "(unknown)"
        _principal      = (
            raw_log.get("principal_arn")
            or raw_log.get("userIdentity", {}).get("arn")
            or "(unknown)"
        )
        _target         = (
            raw_log.get("target_resource")
            or raw_log.get("requestParameters", {}).get("resourceArn")
            or "(unknown)"
        )
        print(
            f"\n{'='*40}\n"
            f"Event:\n"
            f"  log_id    : {_log_id_preview}\n"
            f"  event_name: {_event_name}\n"
            f"  principal : {_principal}\n"
            f"  target    : {_target}",
            flush=True,
        )

        # ── Step 1: Feature engineering ───────────────────────────────────────
        # get_structural_data() updates FeatureEngineer's internal state
        # (GraphNodeTracker degree/age/risk) and returns the edge-level
        # structural row we need to build the CloudTrailEvent.
        try:
            struct = self.feature_engineer.get_structural_data(raw_log)
        except (ValueError, KeyError) as exc:
            log.warning("Structural data extraction failed (%s) — skipping event.", exc)
            print(f"  [STEP 1] Feature engineering FAILED: {exc}", flush=True)
            return None

        print("  [STEP 1] Feature engineering succeeded", flush=True)

        source_node_str = raw_log.get("principal_arn") or struct.get("source_node") or ""
        target_node_str = raw_log.get("target_resource") or struct.get("target_node") or "aws_service"
        edge_type = raw_log.get("event_name") or struct.get("edge_type") or "Unknown"
        label = int(raw_log.get("label", 0))

        self._log_id_counter += 1
        ct_event = CloudTrailEvent(
            log_id=f"live:{self._log_id_counter}",
            source_node=source_node_str or None,
            target_node=target_node_str,
            edge_type=edge_type,
            label=label,
        )

        # ── Step 2: Incremental graph update ──────────────────────────────────
        # apply_event() is thread-safe (RLock inside updater).
        update_result = self.updater.apply_event(ct_event)
        if update_result.sync_status.value == "failed":
            log.error(
                "Neo4j write-through failed for log_id=%s: %s",
                ct_event.log_id, update_result.sync_error,
            )
            print(
                f"  [STEP 2] Incremental graph updated (Neo4j write-through FAILED: "
                f"{update_result.sync_error})",
                flush=True,
            )
            # Continue — the in-memory graph is still updated; Neo4j will
            # catch up when the connection is restored.
        else:
            print("  [STEP 2] Incremental graph updated", flush=True)

        # Identify the two canonical node keys that changed
        src_key = update_result.source_key.key
        dst_key = update_result.target_key.key
        seed_keys = {src_key, dst_key}

        print(f"           seed_keys: {seed_keys}", flush=True)

        # ── Step 3: Subgraph load ─────────────────────────────────────────────
        subgraph = self.subgraph_loader.load_neighbourhood(seed_keys)

        if subgraph is None or not subgraph.edge_types:
            # The edge was just written to Neo4j, but there are no neighbours
            # yet (brand-new isolated node) — nothing for GraphSAGE to score.
            print(
                "  [STEP 3] Subgraph loaded\n"
                "           Subgraph is empty.\n"
                "           Skipping inference.",
                flush=True,
            )
            log.debug(
                "log_id=%s: subgraph empty after update — skipping inference.",
                ct_event.log_id,
            )
            return None

        # ── Subgraph diagnostics ──────────────────────────────────────────────
        _node_types_present   = [nt for nt in ALL_NODE_TYPES if nt in subgraph.node_types]
        _total_nodes          = sum(
            subgraph[nt].x.shape[0]
            for nt in _node_types_present
            if hasattr(subgraph[nt], "x")
        )
        _edge_types_present   = list(subgraph.edge_types)
        _total_edges          = sum(
            subgraph[et].edge_index.shape[1]
            for et in _edge_types_present
            if hasattr(subgraph[et], "edge_index")
        )
        print(
            f"  [STEP 3] Subgraph loaded\n"
            f"           number of node types : {len(_node_types_present)}\n"
            f"           number of nodes      : {_total_nodes}\n"
            f"           number of edges      : {_total_edges}\n"
            f"           edge types present   : {_edge_types_present}",
            flush=True,
        )
        self._diag["subgraphs_built"] += 1

        # ── Step 4: GraphSAGE inference ───────────────────────────────────────
        # model.eval() is set at startup and never changed.
        # torch.no_grad() is set via the decorator on this method.
        print("  [STEP 4] Running GraphSAGE...", flush=True)
        try:
            logits = self.model(subgraph)
        except Exception as exc:
            log.error("Model forward() failed: %s", exc, exc_info=True)
            print(f"           Model forward() FAILED: {exc}", flush=True)
            return None

        self._diag["model_inferences"] += 1

        probs = torch.sigmoid(logits).cpu().numpy()

        print(
            f"           raw logits      : {logits.cpu().numpy().tolist()}\n"
            f"           sigmoid probs   : {probs.tolist()}",
            flush=True,
        )

        # Find the edge in the subgraph that corresponds to (src_key, dst_key).
        # We want the probability for the specific edge just added, not the
        # max over all edges in the subgraph.
        target_prob = self._find_edge_probability(subgraph, probs, src_key, dst_key)
        if target_prob is None:
            # Edge not found in the subgraph tensors; use the max probability
            # as a conservative upper bound rather than silently returning 0.
            print(
                f"           WARNING: target edge ({src_key!r} → {dst_key!r}) not found "
                f"in subgraph tensors — falling back to max probability.",
                flush=True,
            )
            target_prob = float(probs.max()) if len(probs) > 0 else 0.0

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        _prediction_label = "MALICIOUS" if target_prob >= self.threshold else "BENIGN"
        _threshold_met    = target_prob >= self.threshold

        print(
            f"           Probability : {target_prob:.4f}\n"
            f"           Threshold   : {self.threshold:.2f}\n"
            f"           Prediction  : {_prediction_label}",
            flush=True,
        )

        log.info(
            "log_id=%s | %s → %s | %s | prob=%.4f | %.1f ms",
            ct_event.log_id, src_key[:30], dst_key[:30], edge_type, target_prob, elapsed_ms,
        )

        # ── Step 5: Blast radius + alert ─────────────────────────────────────
        if not _threshold_met:
            print(
                "  [STEP 5] Blast Radius skipped because prediction < threshold.",
                flush=True,
            )
            return None

        self._diag["malicious_predictions"] += 1

        log.warning(
            "⚠  Prediction %.4f >= threshold %.2f — computing blast radius for %s/%s …",
            target_prob, self.threshold, update_result.source_key.label, src_key,
        )
        print("  [STEP 5] Blast Radius invoked.", flush=True)

        principal_node: Node = (update_result.source_key.label, src_key)
        blast_report = self.blast_cache.get_or_compute(
            principal=principal_node,
            engine=self.blast_engine,
        )

        self._diag["blast_radius_executions"] += 1

        # Re-index blast engine's target lookups to cover newly added nodes.
        # This is the one place where we do a partial re-index rather than
        # a full one — only the newly created nodes need to be added.
        if update_result.created_new_target_node:
            self.blast_engine._index_targets()

        # ── Blast radius diagnostics ──────────────────────────────────────────
        _br_score          = round(blast_report.score, 4)
        _reachable_total   = blast_report.reachable_assets.total
        _critical_count    = blast_report.critical_assets.critical_asset_count
        _propagation_count = len(blast_report.top_paths)

        print(
            f"           Blast Radius score    : {_br_score}\n"
            f"           Reachable assets      : {_reachable_total}\n"
            f"           Critical assets       : {_critical_count}\n"
            f"           Propagation paths     : {_propagation_count}",
            flush=True,
        )

        alert = generate_alert(
            event_row=raw_log,
            pred_prob=target_prob,
            principal_node=principal_node,
            blast_report=blast_report,
            alert_dir=self.alert_dir,
        )

        self._diag["alerts_generated"] += 1

        _alert_filename = f"alert_{alert['alert_id']}.json"
        print(f"  [STEP 5] Alert written: {_alert_filename}", flush=True)

        return alert

    def _find_edge_probability(
        self,
        subgraph: HeteroData,
        probs: np.ndarray,
        src_key: str,
        dst_key: str,
    ) -> Optional[float]:
        """
        Maps (src_key, dst_key) → the probability scalar for that specific
        edge in the model's flat output tensor.

        The flat output is ordered by sorted(subgraph.edge_types), then by
        local edge index within each triple — exactly the same order as
        data_loader.py's global edge ordering. We search for the first
        edge in any triple where the src node key == src_key and dst node
        key == dst_key.
        """
        offset = 0
        for triple in sorted(subgraph.edge_types):
            triple_data = subgraph[triple]
            if not hasattr(triple_data, "edge_index"):
                continue
            n_edges = triple_data.edge_index.shape[1]
            src_type, _, dst_type = triple
            src_keys = getattr(subgraph[src_type], "key", [])
            dst_keys = getattr(subgraph[dst_type], "key", [])

            for local_i in range(n_edges):
                s_idx = int(triple_data.edge_index[0, local_i])
                d_idx = int(triple_data.edge_index[1, local_i])
                s_key_i = src_keys[s_idx] if s_idx < len(src_keys) else None
                d_key_i = dst_keys[d_idx] if d_idx < len(dst_keys) else None
                if s_key_i == src_key and d_key_i == dst_key:
                    global_i = offset + local_i
                    if global_i < len(probs):
                        return float(probs[global_i])
            offset += n_edges
        return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self.feature_engineer.save_state()
        self.driver.close()
        log.info("InferenceEngine closed.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WATCH LOOP
# Watches an incoming directory for new CloudTrail files (JSON / CSV /
# NDJSON / gz) and feeds each event through InferenceEngine.process_event().
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = (
    ".csv", ".csv.gz", ".json", ".json.gz", ".jsonl", ".jsonl.gz",
    ".ndjson", ".ndjson.gz",
)


def _file_is_stable(path: str, checks: int = 2, interval: float = 0.3) -> bool:
    """Waits until the file size stops changing (identical to feature_engine9.py's
    implementation — copied here to avoid a hidden dependency on a private
    function from that module)."""
    last_size = -1
    stable_count = 0
    while stable_count < checks:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size
        time.sleep(interval)
    return True


def _load_processed_set(state_path: str) -> Set[str]:
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            return set(json.load(f).get("processed_files", []))
    return set()


def _save_processed_set(state_path: str, processed: Set[str]) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"processed_files": sorted(processed)}, f, indent=2)


def process_file(engine: InferenceEngine, path: str) -> Tuple[int, int]:
    """
    Streams every CloudTrail event in `path` through the InferenceEngine.
    Returns (total_events, total_alerts).
    """
    total, alerts = 0, 0
    # Per-file diagnostic counters – incremented inside process_event via
    # the _diag dict that InferenceEngine exposes for this purpose.
    engine._diag = {
        "events": 0,
        "subgraphs_built": 0,
        "model_inferences": 0,
        "malicious_predictions": 0,
        "blast_radius_executions": 0,
        "alerts_generated": 0,
    }
    for raw_log in iter_input_rows(path):
        result = engine.process_event(raw_log)
        total += 1
        if result is not None:
            alerts += 1
    log.info("File %s — %d events, %d alerts", os.path.basename(path), total, alerts)

    d = engine._diag
    print(
        f"\n{'='*40}\n"
        f"FILE SUMMARY: {os.path.basename(path)}\n"
        f"  Processed events       : {d['events']}\n"
        f"  Subgraphs built        : {d['subgraphs_built']}\n"
        f"  Model inferences       : {d['model_inferences']}\n"
        f"  Malicious predictions  : {d['malicious_predictions']}\n"
        f"  Blast Radius executions: {d['blast_radius_executions']}\n"
        f"  Alerts generated       : {d['alerts_generated']}\n"
        f"{'='*40}\n",
        flush=True,
    )
    return total, alerts


def watch_directory(
    engine: InferenceEngine,
    directory: str,
    state_path: str = "infer_state.json",
    poll_interval: float = 1.0,
) -> None:
    """
    Polls `directory` every poll_interval seconds for new files that match
    SUPPORTED_EXTENSIONS and have not been processed yet.

    Uses a plain polling loop rather than watchdog (removing the dependency
    on an optional package) — at the event rates of CloudTrail (one S3
    delivery every ~5 minutes per region) polling at 1-second resolution
    adds negligible CPU overhead.

    To use watchdog instead, swap the body of the loop with an
    ArrivalHandler exactly as feature_engine9.watch_folder() does —
    the process_file() call is the only interface point.
    """
    os.makedirs(directory, exist_ok=True)
    processed = _load_processed_set(state_path)
    log.info("Watching %s for new CloudTrail files (Ctrl+C to stop) …", directory)

    try:
        while True:
            for fname in sorted(os.listdir(directory)):
                if fname in processed:
                    continue
                if not any(fname.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    continue
                fpath = os.path.join(directory, fname)
                if not _file_is_stable(fpath):
                    continue
                log.info("New file: %s", fname)
                process_file(engine, fpath)
                processed.add(fname)
                _save_processed_set(state_path, processed)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        log.info("Watch loop stopped.")
    finally:
        engine.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time GraphSAGE inference over CloudTrail events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical usage
─────────────
# 1. Wrap an existing bare checkpoint (run once after training):
python infer.py --wrap-checkpoint ./checkpoints/best_sage.pt \\
    --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-pass test1234

# 2. Start the watch loop:
python infer.py \\
    --checkpoint ./checkpoints/best_sage_wrapped.pt \\
    --watch ./incoming \\
    --alert-dir ./alerts \\
    --threshold 0.5

# 3. Process a single file and exit:
python infer.py \\
    --checkpoint ./checkpoints/best_sage_wrapped.pt \\
    --input ./incoming/cloudtrail_20260101.json \\
    --threshold 0.5
""",
    )

    # Mode flags
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", metavar="DIR",
                      help="Watch DIR continuously for new CloudTrail files.")
    mode.add_argument("--input", metavar="FILE",
                      help="Process a single file and exit.")
    mode.add_argument("--wrap-checkpoint", metavar="CKPT",
                      help="Wrap a bare state_dict checkpoint with model_args "
                           "and fit_artifacts (one-time migration, requires Neo4j).")

    # Model / checkpoint
    p.add_argument("--checkpoint", default="./checkpoints/best_sage_wrapped.pt",
                   help="Path to the wrapped checkpoint file (default: %(default)s).")

    # Neo4j
    p.add_argument("--neo4j-uri",  default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="test1234")

    # Inference settings
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold above which an edge is flagged malicious.")
    p.add_argument("--alert-dir", default="./alerts",
                   help="Directory where JSON alert files are written.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device (cpu or cuda). Model stays in eval() regardless.")
    p.add_argument("--hop-radius", type=int, default=SUBGRAPH_HOP_RADIUS,
                   help="How many hops around the changed nodes to pull from Neo4j "
                        "for each inference pass. Default: %(default)d.")
    p.add_argument("--seed-from-neo4j", action="store_true",
                   help="Seed the in-memory PPG from existing Neo4j data at startup "
                        "(recommended when Neo4j already contains historical events).")
    p.add_argument("--state-file", default="infer_state.json",
                   help="Path to the watch-mode state file tracking processed files.")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="Seconds between directory polls in watch mode.")

    # Wrap-checkpoint specific
    p.add_argument("--hidden-dim",   type=int,   default=128,
                   help="For --wrap-checkpoint: hidden_dim used during training.")
    p.add_argument("--num-layers",   type=int,   default=2,
                   help="For --wrap-checkpoint: num_sage_layers used during training.")
    p.add_argument("--dropout",      type=float, default=0.3,
                   help="For --wrap-checkpoint: dropout used during training.")
    p.add_argument("--wrapped-output", default=None,
                   help="For --wrap-checkpoint: output path for the wrapped file. "
                        "Default: <original>_wrapped.pt")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Mode: wrap-checkpoint ─────────────────────────────────────────────────
    if args.wrap_checkpoint:
        out = wrap_checkpoint(
            original_ckpt_path=args.wrap_checkpoint,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_pass=args.neo4j_pass,
            hidden_dim=args.hidden_dim,
            num_sage_layers=args.num_layers,
            dropout=args.dropout,
            output_path=args.wrapped_output,
        )
        print(f"Wrapped checkpoint: {out}")
        return

    # ── Shared: build InferenceEngine ─────────────────────────────────────────
    engine = InferenceEngine(
        ckpt_path=args.checkpoint,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_pass=args.neo4j_pass,
        alert_dir=args.alert_dir,
        malicious_threshold=args.threshold,
        device=args.device,
        hop_radius=args.hop_radius,
    )

    if args.seed_from_neo4j:
        engine.seed_ppg_from_neo4j()

    # ── Mode: watch ───────────────────────────────────────────────────────────
    if args.watch:
        watch_directory(
            engine=engine,
            directory=args.watch,
            state_path=args.state_file,
            poll_interval=args.poll_interval,
        )
        return

    # ── Mode: single file ─────────────────────────────────────────────────────
    if args.input:
        total, alerts = process_file(engine, args.input)
        print(f"\n{total} events processed | {alerts} alerts generated → {args.alert_dir}/")
        engine.close()
        return


if __name__ == "__main__":
    main()
