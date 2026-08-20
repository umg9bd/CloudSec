"""
data_loader.py  (v3 — Privilege Propagation Graph)
====================================================
Loads the Neo4j Privilege Propagation Graph (see neo4j_graph_builder.py v3)
into a PyTorch Geometric HeteroData object.

Node types : User, Role, UnresolvedPrincipal, Service, Resource, Policy
Edge types : (SrcType, relation, DstType) for every relation in
             {ASSUMES, LIST, READ, WRITE, TAGGING, PERMISSIONS_MANAGEMENT,
             UNKNOWN_ACTION} that has at least one edge (verified: 20
             distinct (src_type, relation, dst_type) triples are populated
             on this dataset — see neo4j_graph_builder.py).

THE GLOBAL EDGE ORDER PROBLEM
─────────────────────────────────────────────────────────────────────────
In the previous single-relation design, "all edges" and "the tensor of
edge_index columns" were the same thing, so a single y / train_mask /
val_mask worked directly. With up to 20 populated relation triples, each
triple gets its OWN edge_index/edge_attr/y tensor in HeteroData (this is
how PyG represents heterogeneous edges — there is no such thing as a
single edge_index spanning different node-type pairs). But this
codebase's loss, evaluation, and splitting logic all need ONE flat vector
of labels to operate on.

The fix used throughout this file and in model_graphsage.py/model_gat.py:
every one of them independently computes `sorted(data.edge_types)` as the
canonical global order, and `global_labels(data)` / `flatten_mask_dict()`
below flatten each triple's own tensors in that exact order. Because this
order is a pure, deterministic function of `data` itself (not a
side-channel value handed around separately), the model's flat logits,
this file's flattened `y`, and any flattened mask are guaranteed to stay
index-aligned without model_graphsage.py/model_gat.py/utils.py needing to
import or depend on this file's internal bookkeeping — only on the same
well-defined sort applied to the same data object.
`meta["edge_order"]` / `meta["log_id_to_global_index"]` are still built
during `load()` for TRACEABILITY back to the source CSV row (e.g. for
explainability.py to report "this prediction corresponds to log_id N"),
but nothing in the split/evaluation path depends on them anymore.

A CONCRETE SUBTLETY THAT WAS CAUGHT AND FIXED DURING DEVELOPMENT: a single
relation (e.g. READ) can span MULTIPLE (src_type, dst_type) pairs —
verified on this dataset: READ alone covers 5 distinct pairs (User→
Resource, Role→Resource, UnresolvedPrincipal→Resource, User→Policy,
User→Service). An earlier draft of this loader grouped by relation alone
and read only the FIRST row's types for the whole relation, which would
have silently misassigned ~67 edges into the wrong (src,rel,dst) tensor.
The loader below groups by the full (src_type, relation, dst_type) key
directly, verified against the real CSV to produce exactly 20 correctly
populated triples summing to exactly 2,900 edges.

FEATURE PROVENANCE
─────────────────────────────────────────────────────────────────────────
Every tensor here is either:
  (a) a topology aggregate (out/in-degree, unique_targets, etc.) computed
      once over the full static graph — same category of feature as the
      previous design, not new leakage,
  (b) an access-level / privilege-propagation feature from
      privilege_features.py (hop_count, privilege_gain,
      abnormal_path_frequency, resource_sensitivity,
      distance_to_sensitive_resource) — all pure functions of graph
      structure, documented in that module, or
  (c) `is_known_attacker_identity`, fetched for descriptive
      metadata/reporting ONLY (`meta["attacker_identity_by_key"]`) and
      deliberately NOT concatenated into any node's `.x` tensor — same
      exclusion rationale as the v2 loader (a real-time detector would
      not have this post-hoc IR knowledge a priori).
No timestamp/session feature exists anywhere in this file — this dataset
still has none (see neo4j_graph_builder.py / privilege_features.py).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_URI  = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASS = "test1234"

PRINCIPAL_NODE_TYPES = ("User", "Role", "UnresolvedPrincipal")
TARGET_NODE_TYPES    = ("Service", "Resource", "Policy")
ALL_NODE_TYPES       = PRINCIPAL_NODE_TYPES + TARGET_NODE_TYPES
RELATION_TYPES       = ("ASSUMES", "LIST", "READ", "WRITE", "TAGGING",
                         "PERMISSIONS_MANAGEMENT", "UNKNOWN_ACTION")

_ALL_NODE_TYPES_SET = set(ALL_NODE_TYPES)


def get_specific_label(labels: List[str]) -> str:
    """
    Extracts the specific node type (User/Role/UnresolvedPrincipal/Service/
    Resource/Policy) from a Neo4j `labels(n)` result list.

    IMPORTANT: Neo4j does NOT guarantee any ordering for labels() — the
    order is determined by internal label-token IDs, not by the order
    labels were written in a MERGE/CREATE statement, and that ordering
    can differ across databases, restarts, and Neo4j versions (confirmed
    by Neo4j engineering: https://github.com/neo4j/neo4j/issues/13350).
    `neo4j_graph_builder.py` always writes nodes as e.g.
    `(:User:Principal)`, but `labels(n)` may return `["User","Principal"]`
    OR `["Principal","User"]` depending on the instance. Callers must
    NEVER assume `labels[0]` is the specific type — this function scans
    the full list instead.

    This is the single, shared source of truth for that lookup — both
    this file's `_fetch_edges` and infer.py's SubgraphLoader must use
    THIS function rather than each maintaining their own copy, so the
    two pipelines can never drift out of sync on this logic again.
    """
    for label in labels:
        if label in _ALL_NODE_TYPES_SET:
            return label
    raise ValueError(
        f"No specific node label found in {labels!r}; expected one of "
        f"{sorted(_ALL_NODE_TYPES_SET)}"
    )

# Distance-to-sensitive-resource is unbounded-but-capped (BFS cutoff=6 in
# privilege_features.py); nodes with no sensitive resource reachable
# within that cutoff get this explicit sentinel rather than NaN.
UNREACHABLE_DISTANCE_SENTINEL = 7

# ── Feature schemas per node type (numeric cols, categorical cols) ──────────
# Principal-like types (User/Role/UnresolvedPrincipal) share a schema: node
# TYPE itself now carries what used to be the `principal_type` categorical
# feature (User vs Role are literally different node types with their own
# weight matrices in the encoder), so it does not need to be re-encoded as
# an input feature — a direct simplification enabled by the heterogeneous
# redesign.
_PRINCIPAL_NUM_COLS = ["out_degree", "unique_targets", "unique_actions", "role_transition_count"]
_TARGET_NUM_COLS    = ["in_degree", "unique_principals", "resource_sensitivity",
                        "distance_to_sensitive_resource"]

# Raw degree/count columns -- rank-normalized within their own graph instead
# of z-scored (see _rank_normalize below). Excludes resource_sensitivity/
# distance_to_sensitive_resource (bounded scores, not counts) and hop_count
# (small bounded integer, not implicated -- see PROJECT_STATUS_REPORT.md
# section 6.14).
_COUNT_LIKE_COLS = {"out_degree", "unique_targets", "unique_actions",
                     "role_transition_count", "in_degree", "unique_principals"}

NODE_FEATURE_SCHEMA: Dict[str, Tuple[List[str], List[str]]] = {
    "User":               (_PRINCIPAL_NUM_COLS, []),
    "Role":                (_PRINCIPAL_NUM_COLS, []),
    "UnresolvedPrincipal": (_PRINCIPAL_NUM_COLS, []),
    "Service":             (_TARGET_NUM_COLS, []),
    "Policy":              (_TARGET_NUM_COLS, []),
    "Resource":            (_TARGET_NUM_COLS, ["resource_type"]),
}

EDGE_NUM_COLS = [
    "hop_count", "privilege_gain", "privilege_gain_defined",
    "action_global_frequency_log",
    "is_privilege_escalation_technique", "is_read_only",
]  # abnormal_path_frequency is handled separately (rank-normalized, not
   # scaled -- see _rank_normalize and abnormal_path_frequency_rank below)
EDGE_CAT_COLS = ["edge_type"]  # label-encoded GLOBALLY across all 260 actions,
                                # shared across every relation type, so the
                                # single shared EdgeClassifierHead sees a
                                # consistent feature schema regardless of
                                # which relation an edge belongs to.


class PrivilegePropagationGraphLoader:
    """Loads the Neo4j Privilege Propagation Graph and converts it to HeteroData."""

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        device: str = "cpu",
        fit_artifacts: dict = None,
    ):
        """fit_artifacts (optional): {"edge_scaler", "node_scalers",
        "label_encoders"} from a PRIOR load() call (e.g. saved in a
        wrapped checkpoint via infer.py's wrap_checkpoint). When given,
        every scaler/encoder is applied with .transform() only -- never
        re-fit -- so a real/held-out graph is normalized on the exact
        training-time distribution instead of its own. Without this, two
        load() calls on different graphs would each fit fresh statistics,
        making any cross-graph comparison (e.g. train-on-synthetic,
        evaluate-on-real) invalid."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = torch.device(device)
        self._fit_artifacts = fit_artifacts
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.node_scalers: Dict[str, StandardScaler] = {}
        self.edge_scaler = StandardScaler()
        log.info("Connected to Neo4j at %s", uri)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> Tuple[HeteroData, dict]:
        node_dfs = {ntype: self._fetch_nodes(ntype) for ntype in ALL_NODE_TYPES}
        edge_dfs = {rel: self._fetch_edges(rel) for rel in RELATION_TYPES}
        edge_dfs = {rel: df for rel, df in edge_dfs.items() if len(df) > 0}

        node_idx: Dict[str, Dict[str, int]] = {}
        data = HeteroData()
        for ntype, df in node_dfs.items():
            if len(df) == 0:
                # During training this type is legitimately absent from the
                # graph entirely. During inference against a pretrained
                # checkpoint, though, the model's input_proj has a layer for
                # every node type seen at training time and looks each one
                # up unconditionally -- a real eval graph simply having zero
                # instances of a trained type (e.g. no Policy nodes in a
                # smaller dev split) must not crash the model, so give it an
                # empty-but-correctly-shaped tensor instead of skipping it.
                if self._fit_artifacts is not None:
                    num_cols, cat_cols = NODE_FEATURE_SCHEMA[ntype]
                    data[ntype].x = torch.zeros((0, len(num_cols) + len(cat_cols)), dtype=torch.float)
                    data[ntype].key = []
                    node_idx[ntype] = {}
                continue
            node_idx[ntype] = {key: i for i, key in enumerate(df["key"])}
            data[ntype].x = self._node_features(ntype, df)
            data[ntype].key = list(df["key"])  # human-readable index for explainability

        log.info("Node counts: %s", {k: v.x.shape[0] for k, v in data.node_items()})

        edge_order: List[Tuple[str, str, str, int]] = []  # (src,rel,dst, local_idx)
        log_id_to_global_index: Dict[str, int] = {}  # log_id is an opaque
        # string under the Feature Engine schema (e.g.
        # "synthetic_cloudtrail.csv:0") — this dict is a lookup table, not
        # an ordering structure, so a string key works identically to the
        # old int key.

        # A global edge_type encoder fit across every relation's actions,
        # so `edge_type` is on a single consistent encoding regardless of
        # which relation bucket an edge lands in. Reused (never re-fit)
        # from _fit_artifacts when evaluating a graph against an
        # already-trained model -- see __init__ docstring.
        fitted_edge_type_enc = (self._fit_artifacts or {}).get("label_encoders", {}).get("edge_type")
        if fitted_edge_type_enc is not None:
            edge_type_enc = fitted_edge_type_enc
        else:
            all_edge_types = pd.concat([df["edge_type"] for df in edge_dfs.values()]) if edge_dfs else pd.Series([], dtype=str)
            edge_type_enc = LabelEncoder().fit(all_edge_types)
            self.label_encoders["edge_type"] = edge_type_enc

        # Fit ONE shared edge-feature scaler across ALL relations' rows, so
        # every relation's edge_attr lives on the same numeric scale before
        # being consumed by the single shared EdgeClassifierHead. Reused
        # (never re-fit) from _fit_artifacts, same rationale as above.
        for df in edge_dfs.values():
            df["action_global_frequency_log"] = np.log1p(
            df["action_global_frequency"].astype(float)
            )

        # abnormal_path_frequency rank-normalized across the WHOLE graph's
        # edges (not per-relation-group -- a rare relation would have too
        # few edges for a meaningful percentile). Computed fresh per graph,
        # positionally sliced back into each relation's df before the
        # groupby below so _edge_features sees it as a plain column.
        if edge_dfs:
            lens = [len(df) for df in edge_dfs.values()]
            all_apf = pd.concat([df["abnormal_path_frequency"] for df in edge_dfs.values()], ignore_index=True)
            all_apf_rank = self._rank_normalize(all_apf)
            offset = 0
            for df, n in zip(edge_dfs.values(), lens):
                df["abnormal_path_frequency_rank"] = all_apf_rank[offset:offset + n]
                offset += n

        fitted_edge_scaler = (self._fit_artifacts or {}).get("edge_scaler")
        if fitted_edge_scaler is not None:
            self.edge_scaler = fitted_edge_scaler
        else:
            all_num = pd.concat([df[EDGE_NUM_COLS] for df in edge_dfs.values()]) if edge_dfs else pd.DataFrame(columns=EDGE_NUM_COLS)
            self.edge_scaler.fit(all_num.astype(float).values) if len(all_num) else None

        # Single pass, grouped by the FULL (src_type, relation, dst_type) key,
        # sorted lexicographically by that key. This matters beyond tidiness:
        # model_graphsage.py's forward() independently derives its output
        # order from `sorted(data.edge_types)` (so it has no hidden coupling
        # to this loader's internals) — and Python's tuple sort on
        # (src_type, rel, dst_type) matches pandas groupby's default
        # sort=True group order on the same three columns. Building
        # `edge_order` any other way (e.g. relation-major) would silently
        # desynchronize the model's logit order from this file's `y` order.
        combined = (
            pd.concat([df.assign(relation=rel) for rel, df in edge_dfs.items()], ignore_index=True)
            if edge_dfs else pd.DataFrame(columns=["src_type", "relation", "dst_type"])
        )
        for (src_type, rel, dst_type), df in combined.groupby(["src_type", "relation", "dst_type"], sort=True):
            # sort_values("log_id") only needs to give a FIXED, DETERMINISTIC,
            # UNIQUE row order (Neo4j's MATCH does not guarantee result
            # order across runs) — it was never a claim about chronological
            # order (log_id is documented elsewhere in this file as
            # carrying no such guarantee). A string log_id sorts
            # lexicographically rather than numerically, which satisfies
            # that determinism requirement exactly as well as the old
            # integer sort did.
            df = df.sort_values("log_id").reset_index(drop=True)
            triple = (src_type, rel, dst_type)

            df = df.copy()
            df["src_idx"] = df["src_key"].map(node_idx[src_type])
            df["dst_idx"] = df["dst_key"].map(node_idx[dst_type])
            n_before = len(df)
            df = df.dropna(subset=["src_idx", "dst_idx"])
            if len(df) != n_before:
                log.warning("Dropped %d %s edges with unmapped endpoints", n_before - len(df), rel)
            df["src_idx"] = df["src_idx"].astype(int)
            df["dst_idx"] = df["dst_idx"].astype(int)

            edge_index = torch.tensor(np.stack([df["src_idx"].values, df["dst_idx"].values]), dtype=torch.long)
            edge_attr  = self._edge_features(df, edge_type_enc)
            y          = torch.tensor(df["is_attack"].astype(int).values, dtype=torch.long)
            # log_id is an opaque STRING identifier with no numeric meaning
            # to compute over, so — unlike edge_index/edge_attr/y — it is
            # kept as a plain Python list rather than a torch.Tensor
            # (tensors need a numeric dtype; strings don't fit one). This
            # mirrors the exact convention already used a few lines above
            # for node keys: `data[ntype].key = list(df["key"])`. PyG's
            # HeteroData accepts arbitrary non-tensor attributes like this
            # without issue — only tensor-valued attributes are moved by
            # `.to(device)` below; a plain list simply passes through.
            log_ids    = list(df["log_id"])

            data[triple].edge_index = edge_index
            data[triple].edge_attr  = edge_attr
            data[triple].y          = y
            data[triple].log_id     = log_ids

            for local_i, lid in enumerate(df["log_id"].tolist()):
                log_id_to_global_index[lid] = len(edge_order)
                edge_order.append((src_type, rel, dst_type, local_i))

        populated_triples = sorted(data.edge_types)  # actual (src,rel,dst) triples, not just relation names
        log.info("Populated (src,rel,dst) triples: %d | total edges: %d",
                  len(populated_triples), len(edge_order))

        attacker_identity_by_key = {}
        for ntype in PRINCIPAL_NODE_TYPES:
            df = node_dfs.get(ntype)
            if df is not None and "is_known_attacker_identity" in df.columns:
                attacker_identity_by_key.update(dict(zip(df["key"], df["is_known_attacker_identity"])))

        edge_feat_dim = data[populated_triples[0]].edge_attr.shape[1] if populated_triples else 0

        meta = {
            "node_counts": {k: v.x.shape[0] for k, v in data.node_items()},
            "populated_triples": populated_triples,
            "edge_order": edge_order,                       # global order, list of (src,rel,dst,local_idx)
            "log_id_to_global_index": log_id_to_global_index,
            "node_idx": node_idx,
            "label_encoders": self.label_encoders,
            "attacker_identity_by_key": attacker_identity_by_key,
            "edge_feat_dim": edge_feat_dim,
            "node_feat_dim": {k: v.x.shape[1] for k, v in data.node_items()},
            "edge_counts": {f"{s}__{r}__{d}": int(data[(s, r, d)].y.shape[0]) for (s, r, d) in populated_triples},
        }

        self.driver.close()
        return data.to(self.device), meta

    # ── Neo4j queries (static per node/relation type — see
    #    neo4j_graph_builder.py's Cypher note on why these are not
    #    parameterized) ──────────────────────────────────────────────────────

    def _fetch_nodes(self, node_type: str) -> pd.DataFrame:
        with self.driver.session() as s:
            rows = s.run(
                f"""
                MATCH (n:{node_type})
                RETURN n.key AS key, n.out_degree AS out_degree,
                       n.in_degree AS in_degree, n.unique_targets AS unique_targets,
                       n.unique_principals AS unique_principals,
                       n.unique_actions AS unique_actions,
                       n.role_transition_count AS role_transition_count,
                       n.resource_sensitivity AS resource_sensitivity,
                       n.distance_to_sensitive_resource AS distance_to_sensitive_resource,
                       n.resource_type AS resource_type,
                       n.is_known_attacker_identity AS is_known_attacker_identity
                """
            ).data()
        df = pd.DataFrame(rows)
        log.info("Fetched %d :%s nodes", len(df), node_type)
        return df

    def _fetch_edges(self, relation: str) -> pd.DataFrame:
        with self.driver.session() as s:
            rows = s.run(
                f"""
                MATCH (src)-[r:{relation}]->(dst)
                RETURN src.key AS src_key, labels(src) AS src_labels,
                       dst.key AS dst_key, labels(dst) AS dst_labels,
                       r.log_id AS log_id, r.edge_type AS edge_type,
                       r.hop_count AS hop_count, r.privilege_gain AS privilege_gain,
                       r.privilege_gain_defined AS privilege_gain_defined,
                       r.abnormal_path_frequency AS abnormal_path_frequency,
                       r.action_global_frequency AS action_global_frequency,
                       r.is_privilege_escalation_technique AS is_privilege_escalation_technique,
                       r.is_attack AS is_attack
                """
            ).data()
        df = pd.DataFrame(rows)
        if len(df):
            # labels(n) has NO guaranteed ordering (see get_specific_label's
            # docstring above) — never assume label[0] is the specific type.
            df["src_type"] = df["src_labels"].apply(get_specific_label)
            df["dst_type"] = df["dst_labels"].apply(get_specific_label)
            df["is_read_only"] = df["edge_type"].str.startswith(
                ("Get", "List", "Describe", "Head", "Lookup", "Scan", "Query", "Search", "Check", "Validate")
                ).astype(int)
        log.info("Fetched %d :%s edges", len(df), relation)
        return df

    # ── Feature builders ──────────────────────────────────────────────────────

    @staticmethod
    def _rank_normalize(series: pd.Series) -> np.ndarray:
        """Percentile rank of each value WITHIN this series' own population,
        in (0, 1]. Deterministic per-graph, uses no labels, fits nothing --
        computed identically and fresh on any graph (training, dev, test,
        or a future live one), so this is not subject to the "never fit on
        eval data" rule the StandardScalers below follow: there is no
        cross-graph statistic here to leak, only this graph's own topology.
        Exists because a fixed scale transform (raw or log1p) still can't
        make a node's degree comparable in MEANING between a small sparse
        synthetic graph and a small number of heavily-reused real entities
        -- see PROJECT_STATUS_REPORT.md section 6.15/6.16."""
        if len(series) <= 1:
            return np.ones(len(series), dtype=float)
        return series.rank(pct=True, method="average").to_numpy(dtype=float)

    def _node_features(self, ntype: str, df: pd.DataFrame) -> torch.Tensor:
        num_cols, cat_cols = NODE_FEATURE_SCHEMA[ntype]
        df = df.copy()
        if "distance_to_sensitive_resource" in df.columns:
            df["distance_to_sensitive_resource"] = df["distance_to_sensitive_resource"].fillna(
                UNREACHABLE_DISTANCE_SENTINEL
            )
        df[num_cols] = df[num_cols].fillna(0).astype(float)

        scaled_cols = [c for c in num_cols if c not in _COUNT_LIKE_COLS]
        rank_cols = [c for c in num_cols if c in _COUNT_LIKE_COLS]
        rank_part = np.stack([self._rank_normalize(df[c]) for c in rank_cols], axis=1) if rank_cols \
            else np.zeros((len(df), 0))

        fitted_scalers = (self._fit_artifacts or {}).get("node_scalers", {})
        fitted_scaler = fitted_scalers.get(ntype)
        if not scaled_cols:
            scaled_part = np.zeros((len(df), 0))
        elif fitted_scaler is not None:
            scaled_part = fitted_scaler.transform(df[scaled_cols].values)
        elif len(df) > 1:
            scaler = StandardScaler()
            scaled_part = scaler.fit_transform(df[scaled_cols].values)
            self.node_scalers[ntype] = scaler  # only registered once actually fitted
        else:
            scaled_part = df[scaled_cols].values
        num = np.concatenate([scaled_part, rank_part], axis=1)

        if cat_cols:
            fitted_encoders = (self._fit_artifacts or {}).get("label_encoders", {})
            cat_arrays = []
            for col in cat_cols:
                df[col] = df[col].fillna("unknown").astype(str)
                fitted_enc = fitted_encoders.get(f"{ntype}.{col}")
                if fitted_enc is not None:
                    known = set(fitted_enc.classes_)
                    safe_vals = df[col].where(df[col].isin(known), fitted_enc.classes_[0])
                    cat_arrays.append(fitted_enc.transform(safe_vals).astype(float).reshape(-1, 1))
                else:
                    enc = LabelEncoder().fit(df[col])
                    self.label_encoders[f"{ntype}.{col}"] = enc
                    cat_arrays.append(enc.transform(df[col]).astype(float).reshape(-1, 1))
            feats = np.concatenate([num] + cat_arrays, axis=1)
        else:
            feats = num
        return torch.tensor(feats, dtype=torch.float)

    def _edge_features(self, df: pd.DataFrame, edge_type_enc: LabelEncoder) -> torch.Tensor:
        df = df.copy()
        df["action_global_frequency_log"] = np.log1p(df["action_global_frequency"].astype(float))
        df["privilege_gain"] = df["privilege_gain"].fillna(0.0)
        for col in ["privilege_gain_defined", "is_privilege_escalation_technique", "is_read_only"]:
            df[col] = df[col].astype(int)

        num = df[EDGE_NUM_COLS].astype(float).values
        num = self.edge_scaler.transform(num)
        rank_part = df[["abnormal_path_frequency_rank"]].to_numpy(dtype=float)
        num = np.concatenate([num, rank_part], axis=1)
        # A real/held-out graph can contain edge_type values never seen
        # while fitting edge_type_enc on the training graph -- map those to
        # the encoder's first known class rather than letting .transform()
        # raise, same fallback used for node categorical features above.
        known = set(edge_type_enc.classes_)
        safe_edge_type = df["edge_type"].where(df["edge_type"].isin(known), edge_type_enc.classes_[0])
        cat = edge_type_enc.transform(safe_edge_type).astype(float).reshape(-1, 1)
        return torch.tensor(np.concatenate([num, cat], axis=1), dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════
# SPLIT STRATEGIES — operate on the global edge order derived from
# sorted(data.edge_types), not on any single triple's local indices, and
# are label-stratified rather than order-based (still no temporal
# assumption — same rationale as the v2 loader; AWS CloudTrail log files
# carry no guaranteed ordering and this dataset's log_id is not documented
# as chronological).
# ══════════════════════════════════════════════════════════════════════════

def global_labels(data: HeteroData) -> np.ndarray:
    """
    Flattens `y` across every populated triple in sorted(data.edge_types)
    order. This is the SAME order model_graphsage.py's and model_gat.py's
    forward() independently use for their output (see those files'
    docstrings), so `global_labels(data)` always lines up with
    `model(data)` without needing to pass `meta` around — a single,
    self-contained source of truth for "the global order" rather than a
    second copy of it living in `meta["edge_order"]`.
    """
    if not data.edge_types:
        return np.array([], dtype=int)
    return np.concatenate([data[t].y.cpu().numpy() for t in sorted(data.edge_types)])


def flatten_mask_dict(data: HeteroData, mask_dict: Dict[tuple, torch.Tensor]) -> torch.Tensor:
    """Flattens a {triple: BoolTensor[local E]} dict into one global BoolTensor, same order as global_labels/model output."""
    if not data.edge_types:
        return torch.tensor([], dtype=torch.bool)
    return torch.cat([mask_dict[t] for t in sorted(data.edge_types)])


def stratified_edge_split(
    data: HeteroData,
    train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42,
) -> Tuple[Dict[tuple, torch.Tensor], Dict[tuple, torch.Tensor], Dict[tuple, torch.Tensor]]:
    """
    Random split stratified on the label, computed once over the global
    edge order (`global_labels`) and then projected back into a per-triple
    boolean mask dict (since PyG needs a mask aligned to each triple's own
    local edge_index).

    This is the DEFAULT split. It makes no ordering assumption, is fully
    reproducible given `seed`, and preserves the ~4.7% attack-event rate
    in each split. It is a TRANSDUCTIVE evaluation: the same 13 principal-
    side identities appear across train/val/test — see
    `principal_disjoint_split` for the inductive alternative and its
    limitations at this dataset's scale.
    """
    triples = sorted(data.edge_types)
    triple_lengths = [data[t].y.shape[0] for t in triples]
    offsets = np.cumsum([0] + triple_lengths[:-1])
    n = sum(triple_lengths)
    y = global_labels(data)
    idx = np.arange(n)

    train_idx, rest_idx = train_test_split(idx, train_size=train_ratio, random_state=seed, stratify=y)
    rel_val = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(rest_idx, train_size=rel_val, random_state=seed, stratify=y[rest_idx])

    def _project(global_idx_subset):
        masks = {t: torch.zeros(l, dtype=torch.bool) for t, l in zip(triples, triple_lengths)}
        for gi in global_idx_subset:
            ti = int(np.searchsorted(offsets, gi, side="right") - 1)
            local_i = int(gi - offsets[ti])
            masks[triples[ti]][local_i] = True
        return masks

    train_masks, val_masks, test_masks = _project(train_idx), _project(val_idx), _project(test_idx)
    log.info(
        "Stratified split (seed=%d) — train: %d (%.1f%% attack) | val: %d (%.1f%%) | test: %d (%.1f%%)",
        seed, len(train_idx), 100 * y[train_idx].mean(),
        len(val_idx), 100 * y[val_idx].mean(), len(test_idx), 100 * y[test_idx].mean(),
    )
    return train_masks, val_masks, test_masks


def compute_class_weights(data: HeteroData, train_masks: Dict[tuple, torch.Tensor]) -> torch.Tensor:
    n_pos, n_total = 0, 0
    for triple, mask in train_masks.items():
        y = data[triple].y.cpu()
        n_pos += int(y[mask].sum())
        n_total += int(mask.sum())
    n_neg = n_total - n_pos
    pos_weight = torch.tensor(n_neg / (n_pos + 1e-6), dtype=torch.float)
    log.info("pos_weight (attack class): %.2f", pos_weight.item())
    return pos_weight


def principal_disjoint_split(
    data: HeteroData,
    train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42,
) -> Tuple[Dict[tuple, torch.Tensor], Dict[tuple, torch.Tensor], Dict[tuple, torch.Tensor]]:
    """
    Entity-disjoint split, generalized from the earlier single-node-type
    version to this schema's THREE principal-side node types (User, Role,
    UnresolvedPrincipal — collectively "who initiated the action"): every
    edge belonging to a given (node_type, key) identity is placed entirely
    in one split, so no identity is seen across train/val/test. This is
    the standard way to evaluate an INDUCTIVE model (GraphSAGE) — does it
    generalise to principals it never trained on.

    ⚠ SAME CAVEAT AS BEFORE, NOW SHARPER: this dataset has only 13 distinct
    principal-side identities total (3 User, 9 Role, 1 UnresolvedPrincipal
    sentinel), and of those, only 2 (bert-jan,
    stratus-red-team-ec2-get-password-data-role) contribute any label=1
    edges. A split this coarse has high variance — verified directly:
    with the default seed=42, the val split lands entirely on
    stratus-red-team-get-usr-data-role (26 edges), none of which are
    attack-labelled, so this WILL raise on the first attempt. This
    function raises rather than silently returning a degenerate split —
    try a different seed and report the variance if this split is used
    for a paper claim, or prefer stratified_edge_split.
    """
    principal_types = ("User", "Role", "UnresolvedPrincipal")
    identities = sorted({
        (ntype, key)
        for t in data.edge_types if t[0] in principal_types
        for key in getattr(data[t[0]], "key", [])
    })
    if not identities:
        raise ValueError("No principal-side identities found on this data object.")

    rng = np.random.RandomState(seed)
    shuffled = list(identities)
    rng.shuffle(shuffled)

    n_train = max(1, int(len(shuffled) * train_ratio))
    n_val   = max(1, int(len(shuffled) * val_ratio))
    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train:n_train + n_val])
    test_ids  = set(shuffled[n_train + n_val:])

    node_idx_by_type = {
        ntype: {key: i for i, key in enumerate(getattr(data[ntype], "key", []))}
        for ntype in principal_types if ntype in data.node_types
    }
    idx_to_id = {
        ntype: {i: key for key, i in node_idx_by_type[ntype].items()}
        for ntype in node_idx_by_type
    }

    def _project(id_set):
        masks = {}
        for t in data.edge_types:
            src_type = t[0]
            n_local = data[t].y.shape[0]
            if src_type not in principal_types:
                masks[t] = torch.zeros(n_local, dtype=torch.bool)
                continue
            src_indices = data[t].edge_index[0].cpu().numpy()
            mask = torch.zeros(n_local, dtype=torch.bool)
            for i, node_i in enumerate(src_indices):
                key = idx_to_id[src_type].get(int(node_i))
                if (src_type, key) in id_set:
                    mask[i] = True
            masks[t] = mask
        return masks

    train_masks, val_masks, test_masks = _project(train_ids), _project(val_ids), _project(test_ids)

    for name, masks in [("train", train_masks), ("val", val_masks), ("test", test_masks)]:
        y_full = global_labels(data)
        flat = flatten_mask_dict(data, masks).cpu().numpy()
        n_attack = int(y_full[flat].sum())
        log.info("Principal-disjoint split (seed=%d) — %s: %d edges, %d attack",
                  seed, name, int(flat.sum()), n_attack)
        if n_attack == 0:
            raise ValueError(
                f"principal_disjoint_split(seed={seed}) produced a '{name}' split with "
                f"ZERO attack edges — only 2 of this dataset's identities contribute "
                f"attack-labelled edges, so this split is degenerate for this seed. "
                f"Try a different seed and report the variance, or use stratified_edge_split."
            )

    return train_masks, val_masks, test_masks
