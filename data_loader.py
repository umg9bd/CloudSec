"""
data_loader.py
==============
Loads the Neo4j Invictus-AWS graph (see neo4j_graph_builder.py) and converts
it into a PyTorch Geometric HeteroData object.

Node types  : Principal, Target
Edge type   : (Principal)-[INVOKED]->(Target)   — one edge per source CSV row

IMPORTANT — feature provenance
────────────────────────────────────────────────────────────────────────────
Every tensor built here comes directly from properties written by
neo4j_graph_builder.py, which in turn are deterministic functions of
(log_id, source_node, target_node, edge_type). See that file's module
docstring for the full methodological rationale. In particular:

  * `Principal.is_known_attacker_identity` is fetched for reporting/
    descriptive statistics ONLY (see `meta["attacker_identity_by_arn"]`).
    It is deliberately NOT concatenated into `data["principal"].x`,
    because it would let the model use post-hoc incident-response
    knowledge (which identities turned out to be compromised) as an
    input feature for the very task of detecting compromise — a
    circularity a real-time detector would not have. This exclusion is
    enforced in `_principal_features` below; do not add it back without
    updating the paper's methodology section.
  * No timestamp/hour/session-order feature exists anywhere in this file.
    The dataset has no verified event ordering (see neo4j_graph_builder.py
    docstring, point 2), so none is fabricated.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

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

# ── Feature schema (see class docstring for provenance of every column) ──────
PRINCIPAL_NUM_COLS = ["out_degree", "unique_targets", "unique_actions"]
PRINCIPAL_CAT_COLS = ["principal_type"]

TARGET_NUM_COLS = ["in_degree", "unique_principals", "resolved"]
TARGET_CAT_COLS = ["resource_type", "service"]

EDGE_NUM_COLS = ["is_read_only", "is_privilege_sensitive", "action_global_frequency_log"]
EDGE_CAT_COLS = ["edge_type"]


class CloudTrailGraphLoader:
    """Loads the Invictus-AWS Neo4j graph and converts it to HeteroData."""

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        device: str = "cpu",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.device = torch.device(device)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.principal_scaler = StandardScaler()
        self.target_scaler    = StandardScaler()
        self.edge_scaler      = StandardScaler()
        log.info("Connected to Neo4j at %s", uri)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> Tuple[HeteroData, dict]:
        principals, targets = self._fetch_nodes()
        edges_df             = self._fetch_edges()

        p_idx = {arn: i for i, arn in enumerate(principals["arn"])}
        t_idx = {val: i for i, val in enumerate(targets["value"])}

        data = HeteroData()
        data["principal"].x = self._principal_features(principals)
        data["target"].x    = self._target_features(targets)

        log.info(
            "Nodes — principals: %d | targets: %d",
            data["principal"].x.shape[0], data["target"].x.shape[0],
        )

        edge_index, edge_attr, y, log_ids = self._build_edges(edges_df, p_idx, t_idx)

        data["principal", "invoked", "target"].edge_index = edge_index
        data["principal", "invoked", "target"].edge_attr  = edge_attr
        data["principal", "invoked", "target"].y          = y
        # log_id is kept as metadata for traceability back to the source CSV
        # row — NOT part of edge_attr, and NOT used as a temporal signal
        # (see module docstring).
        data["principal", "invoked", "target"].log_id     = log_ids

        log.info(
            "Edges — total: %d | attack: %d (%.2f%%)",
            y.shape[0], y.sum().item(), 100.0 * y.float().mean().item(),
        )

        attacker_identity_by_arn = dict(zip(principals["arn"], principals["is_known_attacker_identity"]))

        meta = {
            "n_principal":   data["principal"].x.shape[0],
            "n_target":      data["target"].x.shape[0],
            "n_edges":       y.shape[0],
            "n_attack":      int(y.sum()),
            "edge_feat_dim": edge_attr.shape[1],
            "p_idx": p_idx, "t_idx": t_idx,
            "label_encoders": self.label_encoders,
            # Descriptive metadata only — see module docstring for why this
            # must not be turned into a training feature.
            "attacker_identity_by_arn": attacker_identity_by_arn,
        }
        self.driver.close()
        return data.to(self.device), meta

    # ── Neo4j queries ─────────────────────────────────────────────────────────

    def _fetch_nodes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        log.info("Fetching Principal nodes …")
        with self.driver.session() as s:
            principals = pd.DataFrame(s.run(
                """
                MATCH (p:Principal)
                RETURN
                    p.arn                       AS arn,
                    p.name                      AS name,
                    p.principal_type            AS principal_type,
                    p.out_degree                AS out_degree,
                    p.unique_targets            AS unique_targets,
                    p.unique_actions            AS unique_actions,
                    p.is_known_attacker_identity AS is_known_attacker_identity
                """
            ).data())
            targets = pd.DataFrame(s.run(
                """
                MATCH (t:Target)
                RETURN
                    t.value              AS value,
                    t.resource_type      AS resource_type,
                    t.service            AS service,
                    t.resolved           AS resolved,
                    t.in_degree          AS in_degree,
                    t.unique_principals  AS unique_principals
                """
            ).data())
        log.info("  principals: %d | targets: %d", len(principals), len(targets))
        return principals, targets

    def _fetch_edges(self) -> pd.DataFrame:
        log.info("Fetching INVOKED edges …")
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (p:Principal)-[r:INVOKED]->(t:Target)
                RETURN
                    p.arn                     AS src_arn,
                    t.value                   AS dst_value,
                    r.log_id                  AS log_id,
                    r.edge_type               AS edge_type,
                    r.is_read_only            AS is_read_only,
                    r.is_privilege_sensitive  AS is_privilege_sensitive,
                    r.action_global_frequency AS action_global_frequency,
                    r.is_attack               AS is_attack
                """
            ).data()
        df = pd.DataFrame(rows)
        log.info("  edges fetched: %d", len(df))
        return df

    # ── Feature builders ──────────────────────────────────────────────────────

    def _principal_features(self, df: pd.DataFrame) -> torch.Tensor:
        df = df.copy()
        pt_enc = LabelEncoder().fit(df["principal_type"])
        self.label_encoders["principal_type"] = pt_enc

        num = df[PRINCIPAL_NUM_COLS].astype(float).values
        num = self.principal_scaler.fit_transform(num)
        cat = pt_enc.transform(df["principal_type"]).astype(float).reshape(-1, 1)

        # NOTE: `is_known_attacker_identity` is intentionally NOT included —
        # see module docstring.
        feats = np.concatenate([num, cat], axis=1)
        return torch.tensor(feats, dtype=torch.float)

    def _target_features(self, df: pd.DataFrame) -> torch.Tensor:
        df = df.copy()
        rt_enc = LabelEncoder().fit(df["resource_type"])
        sv_enc = LabelEncoder().fit(df["service"])
        self.label_encoders["resource_type"] = rt_enc
        self.label_encoders["service"]       = sv_enc

        num = df[TARGET_NUM_COLS].astype(float).values
        num = self.target_scaler.fit_transform(num)
        cat = np.stack([
            rt_enc.transform(df["resource_type"]).astype(float),
            sv_enc.transform(df["service"]).astype(float),
        ], axis=1)

        feats = np.concatenate([num, cat], axis=1)
        return torch.tensor(feats, dtype=torch.float)

    def _build_edges(
        self, df: pd.DataFrame, p_idx: dict, t_idx: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        df = df.copy()
        df["src_idx"] = df["src_arn"].map(p_idx)
        df["dst_idx"] = df["dst_value"].map(t_idx)
        n_before = len(df)
        df = df.dropna(subset=["src_idx", "dst_idx"])
        if len(df) != n_before:
            log.warning("Dropped %d edges with unmapped endpoints", n_before - len(df))
        df["src_idx"] = df["src_idx"].astype(int)
        df["dst_idx"] = df["dst_idx"].astype(int)

        # action_global_frequency is a raw count (can be large, e.g. common
        # Get/List calls) — log1p keeps it on a comparable scale to the
        # binary features before standardisation, without discarding the
        # magnitude information the way a hard cap would.
        df["action_global_frequency_log"] = np.log1p(df["action_global_frequency"].astype(float))

        edge_enc = LabelEncoder().fit(df["edge_type"])
        self.label_encoders["edge_type"] = edge_enc
        cat = edge_enc.transform(df["edge_type"]).astype(float).reshape(-1, 1)

        num = df[EDGE_NUM_COLS].astype(float).values
        num = self.edge_scaler.fit_transform(num)

        edge_feats = np.concatenate([num, cat], axis=1)

        edge_index = torch.tensor(
            np.stack([df["src_idx"].values, df["dst_idx"].values], axis=0), dtype=torch.long
        )
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        y         = torch.tensor(df["is_attack"].astype(int).values, dtype=torch.long)
        log_ids   = torch.tensor(df["log_id"].astype(int).values, dtype=torch.long)

        log.info("Edge feature matrix shape: %s", tuple(edge_attr.shape))
        return edge_index, edge_attr, y, log_ids


# ══════════════════════════════════════════════════════════════════════════
# SPLIT STRATEGIES
#
# The original implementation split edges by row position and called it
# "time-aware", treating log_id as a timestamp proxy. There is no
# documentation (from the dataset's GitHub repository, invictus-ir/
# aws_dataset, or from AWS itself) establishing that log_id — or CloudTrail
# records in general — are chronologically ordered; AWS explicitly states
# the opposite for CloudTrail log files ("events don't appear in any
# specific order", AWS CloudTrail User Guide). That split has been removed.
#
# Two reproducible alternatives are provided instead, with different
# trade-offs that should both be reported if this is written up:
# ══════════════════════════════════════════════════════════════════════════

def stratified_edge_split(
    data: HeteroData,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random split stratified on the label, at the edge (event) level.

    This is the DEFAULT split. It makes no ordering assumption, is fully
    reproducible given `seed`, and preserves the ~4.7% attack-event rate
    in each split. It is a TRANSDUCTIVE evaluation: the same 14 principals
    appear across train/val/test, so it does not by itself demonstrate
    generalisation to unseen identities — see `principal_disjoint_split`
    for that, and its limitations given how few principals this dataset
    has.
    """
    y = data["principal", "invoked", "target"].y.cpu().numpy()
    n = len(y)
    idx = np.arange(n)

    train_idx, rest_idx = train_test_split(
        idx, train_size=train_ratio, random_state=seed, stratify=y
    )
    rel_val = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=rel_val, random_state=seed, stratify=y[rest_idx]
    )

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True

    log.info(
        "Stratified split (seed=%d) — train: %d (%.1f%% attack) | val: %d (%.1f%%) | test: %d (%.1f%%)",
        seed, train_mask.sum(), 100 * y[train_idx].mean(),
        val_mask.sum(), 100 * y[val_idx].mean(),
        test_mask.sum(), 100 * y[test_idx].mean(),
    )
    return train_mask, val_mask, test_mask


def principal_disjoint_split(
    data: HeteroData,
    meta: dict,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entity-disjoint split: every edge belonging to a given principal is
    placed entirely in one split, so no principal is seen in more than one
    of train/val/test. This is the standard way to evaluate an INDUCTIVE
    model such as GraphSAGE — it tests whether the model generalises to
    principals it never trained on, not just to new edges of known
    principals.

    ⚠ KNOWN LIMITATION FOR THIS DATASET: there are only 14 distinct
    principals in total, and of those, only 2 (`bert-jan`,
    `stratus-red-team-ec2-get-password-data-role`) contribute any label=1
    edges — together accounting for all 136 attack-labelled events. A
    principal-level split this coarse has high variance: depending on the
    seed, one split can end up with zero attack principals. This function
    will raise if that happens rather than silently returning a degenerate
    split; callers should treat entity-disjoint results on this dataset as
    indicative, not a robust generalisation claim, and should report
    results across multiple seeds if used.
    """
    edge_index = data["principal", "invoked", "target"].edge_index
    y = data["principal", "invoked", "target"].y.cpu().numpy()
    src = edge_index[0].cpu().numpy()

    principal_ids = np.unique(src)
    rng = np.random.RandomState(seed)
    rng.shuffle(principal_ids)

    n_train = max(1, int(len(principal_ids) * train_ratio))
    n_val   = max(1, int(len(principal_ids) * val_ratio))
    train_p = set(principal_ids[:n_train])
    val_p   = set(principal_ids[n_train:n_train + n_val])
    test_p  = set(principal_ids[n_train + n_val:])

    train_mask = torch.tensor([p in train_p for p in src])
    val_mask   = torch.tensor([p in val_p for p in src])
    test_mask  = torch.tensor([p in test_p for p in src])

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n_attack = int(y[mask.numpy()].sum())
        log.info("Principal-disjoint split (seed=%d) — %s: %d edges, %d attack",
                  seed, name, int(mask.sum()), n_attack)
        if n_attack == 0:
            raise ValueError(
                f"principal_disjoint_split(seed={seed}) produced a '{name}' split with "
                f"ZERO attack edges — this dataset only has 2 attack-contributing "
                f"principals, so this split is degenerate for this seed. Try a "
                f"different seed and report the variance, or use stratified_edge_split."
            )

    return train_mask, val_mask, test_mask


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency positive class weight for the ~4.7% attack rate."""
    n_total = labels.shape[0]
    n_pos   = labels.sum().float()
    n_neg   = n_total - n_pos
    pos_weight = n_neg / (n_pos + 1e-6)
    log.info("pos_weight (attack class): %.2f", pos_weight.item())
    return pos_weight
