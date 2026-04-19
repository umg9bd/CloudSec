"""
data_loader.py
==============
Pulls the Neo4j CloudTrail graph and converts it into a PyTorch Geometric
HeteroData object ready for GraphSAGE / GAT training.

Node types  : Principal, AWSService
Edge types  : (Principal)-[INVOKED]->(AWSService)
              (Principal)-[ORIGINATED_FROM]->(IPAddress)   [optional]

All INVOKED edge features are preserved and used.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from neo4j import GraphDatabase
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Connection defaults (override via constructor) ────────────────────────────
DEFAULT_URI  = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASS = "test1234"

# ── Categorical columns that need label-encoding ─────────────────────────────
CAT_EDGE_COLS = ["edge_type", "event_source", "event_source_category", "aws_region", "attack_technique"]

# ── Numerical edge columns (already float-compatible) ────────────────────────
NUM_EDGE_COLS = [
    "hour_normalized",
    "action_encoded",
    "is_error",
    "is_read_only",
    "privilege_score",
    "is_attack_user",
]


class CloudTrailGraphLoader:
    """
    Loads the CloudTrail Neo4j graph and converts it to HeteroData.

    Why HeteroData?
    ---------------
    The graph has semantically distinct node types (Principal vs AWSService)
    with different feature spaces.  HeteroData lets PyG treat them
    independently in message-passing layers, avoiding feature-space confusion
    and enabling type-specific transformations — critical for accurate
    anomaly detection in heterogeneous cloud environments.
    """

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        device: str = "cpu",
    ):
        self.driver  = GraphDatabase.driver(uri, auth=(user, password))
        self.device  = torch.device(device)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        log.info("Connected to Neo4j at %s", uri)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> Tuple[HeteroData, dict]:
        """
        Returns
        -------
        data      : HeteroData  — ready for PyG models
        meta      : dict        — node/edge counts, label distribution, encoders
        """
        principals, services = self._fetch_nodes()
        edges_df             = self._fetch_edges()

        p_idx  = {arn: i for i, arn in enumerate(principals["arn"])}
        s_idx  = {name: i for i, name in enumerate(services["name"])}

        data   = HeteroData()

        # ── Node features ─────────────────────────────────────────────────────
        data["principal"].x  = self._principal_features(principals)
        data["awsservice"].x = self._service_features(services)

        log.info(
            "Nodes  — principals: %d | services: %d",
            data["principal"].x.shape[0],
            data["awsservice"].x.shape[0],
        )

        # ── Edge index + features ─────────────────────────────────────────────
        edge_index, edge_attr, labels, session_labels = self._build_edges(
            edges_df, p_idx, s_idx
        )

        data["principal", "invoked", "awsservice"].edge_index = edge_index
        data["principal", "invoked", "awsservice"].edge_attr  = edge_attr
        data["principal", "invoked", "awsservice"].y          = labels
        data["principal", "invoked", "awsservice"].y_session  = session_labels

        log.info(
            "Edges  — total: %d | attack: %d (%.1f%%)",
            labels.shape[0],
            labels.sum().item(),
            100.0 * labels.float().mean().item(),
        )

        meta = {
            "n_principal":     data["principal"].x.shape[0],
            "n_service":       data["awsservice"].x.shape[0],
            "n_edges":         labels.shape[0],
            "n_attack":        int(labels.sum()),
            "edge_feat_dim":   edge_attr.shape[1],
            "p_idx":           p_idx,
            "s_idx":           s_idx,
            "label_encoders":  self.label_encoders,
            "scaler":          self.scaler,
        }
        self.driver.close()
        return data.to(self.device), meta

    # ── Neo4j queries ─────────────────────────────────────────────────────────

    def _fetch_nodes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        log.info("Fetching Principal nodes …")
        with self.driver.session() as s:
            principals = pd.DataFrame(
                s.run(
                    """
                    MATCH (p:Principal)
                    RETURN
                        p.arn           AS arn,
                        p.principal_type AS principal_type,
                        p.actor_class   AS actor_class,
                        p.is_attacker   AS is_attacker,
                        p.is_service    AS is_service
                    """
                ).data()
            )
            services = pd.DataFrame(
                s.run(
                    """
                    MATCH (s:AWSService)
                    RETURN s.name AS name, s.category AS category
                    """
                ).data()
            )
        log.info("  principals: %d | services: %d", len(principals), len(services))
        return principals, services

    def _fetch_edges(self) -> pd.DataFrame:
        log.info("Fetching INVOKED edges …")
        with self.driver.session() as s:
            rows = s.run(
                """
                MATCH (p:Principal)-[r:INVOKED]->(srv:AWSService)
                RETURN
                    p.arn                   AS src_arn,
                    srv.name                AS dst_name,
                    r.edge_type             AS edge_type,
                    r.event_source          AS event_source,
                    r.aws_region            AS aws_region,
                    r.source_ip             AS source_ip,
                    r.hour_normalized       AS hour_normalized,
                    r.action_encoded        AS action_encoded,
                    r.is_error              AS is_error,
                    r.is_read_only          AS is_read_only,
                    r.privilege_score       AS privilege_score,
                    r.is_attack_user        AS is_attack_user,
                    r.event_source_category AS event_source_category,
                    r.label                 AS label,
                    r.session_label         AS session_label,
                    r.attack_technique      AS attack_technique,
                    r.count                 AS count,
                    r.attack_count          AS attack_count
                """
            ).data()
        df = pd.DataFrame(rows)
        log.info("  edges fetched: %d", len(df))
        return df

    # ── Feature builders ──────────────────────────────────────────────────────

    def _principal_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Encode principal nodes.
        Features: principal_type (encoded), actor_class (encoded),
                  is_attacker (0/1), is_service (0/1)
        """
        df = df.copy().fillna("unknown")
        pt_enc = LabelEncoder().fit(df["principal_type"])
        ac_enc = LabelEncoder().fit(df["actor_class"])
        self.label_encoders["principal_type"] = pt_enc
        self.label_encoders["actor_class"]    = ac_enc

        feats = np.stack([
            pt_enc.transform(df["principal_type"]).astype(float),
            ac_enc.transform(df["actor_class"]).astype(float),
            df["is_attacker"].astype(float).values,
            df["is_service"].astype(float).values,
        ], axis=1)
        return torch.tensor(feats, dtype=torch.float)

    def _service_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Encode AWSService nodes.
        Features: category label-encoded.
        """
        df = df.copy().fillna("unknown")
        cat_enc = LabelEncoder().fit(df["category"])
        self.label_encoders["service_category"] = cat_enc
        feats = cat_enc.transform(df["category"]).astype(float).reshape(-1, 1)
        return torch.tensor(feats, dtype=torch.float)

    def _build_edges(
        self,
        df: pd.DataFrame,
        p_idx: dict,
        s_idx: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        edge_index    : [2, E]
        edge_attr     : [E, F]  — all numerical + encoded categorical features
        labels        : [E]     — binary 0/1
        session_labels: [E]
        """
        df = df.copy()

        # ── Map ARNs / names to integer indices ───────────────────────────────
        df["src_idx"] = df["src_arn"].map(p_idx)
        df["dst_idx"] = df["dst_name"].map(s_idx)
        df = df.dropna(subset=["src_idx", "dst_idx"])
        df["src_idx"] = df["src_idx"].astype(int)
        df["dst_idx"] = df["dst_idx"].astype(int)

        # ── Fill NaN in categorical columns ───────────────────────────────────
        for col in CAT_EDGE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)

        # ── Encode categorical edge features ──────────────────────────────────
        cat_arrays: List[np.ndarray] = []
        for col in CAT_EDGE_COLS:
            if col not in df.columns:
                continue
            enc = LabelEncoder().fit(df[col])
            self.label_encoders[f"edge_{col}"] = enc
            arr = enc.transform(df[col]).astype(float)
            cat_arrays.append(arr)

        # ── Numerical edge features ───────────────────────────────────────────
        for col in NUM_EDGE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df[NUM_EDGE_COLS] = df[NUM_EDGE_COLS].fillna(0.0).astype(float)

        # Add count / attack_count as extra numerical signals
        df["count_norm"]        = np.log1p(df.get("count", 1).fillna(1))
        df["attack_count_norm"] = np.log1p(df.get("attack_count", 0).fillna(0))
        num_cols_extended = NUM_EDGE_COLS + ["count_norm", "attack_count_norm"]
        num_array = df[num_cols_extended].values.astype(float)

        # ── Standardise numerical features ───────────────────────────────────
        num_array = self.scaler.fit_transform(num_array)

        # ── Concatenate all edge features ─────────────────────────────────────
        if cat_arrays:
            cat_matrix = np.stack(cat_arrays, axis=1)
            edge_feats = np.concatenate([num_array, cat_matrix], axis=1)
        else:
            edge_feats = num_array

        # ── Build tensors ─────────────────────────────────────────────────────
        edge_index = torch.tensor(
            np.stack([df["src_idx"].values, df["dst_idx"].values], axis=0),
            dtype=torch.long,
        )
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        labels    = torch.tensor(df["label"].fillna(0).astype(int).values, dtype=torch.long)
        sess_lbl  = torch.tensor(df["session_label"].fillna(0).astype(int).values, dtype=torch.long)

        log.info("Edge feature matrix shape: %s", tuple(edge_attr.shape))
        return edge_index, edge_attr, labels, sess_lbl


# ── Time-aware train/val/test split ──────────────────────────────────────────

def time_aware_split(
    data: HeteroData,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edges by position (proxy for time ordering in the CSV).
    This avoids data leakage: the model never sees future events during training.

    Returns train_mask, val_mask, test_mask  (all BoolTensors of shape [E])
    """
    n  = data["principal", "invoked", "awsservice"].y.shape[0]
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    idx = torch.arange(n)
    train_mask = idx < n_train
    val_mask   = (idx >= n_train) & (idx < n_train + n_val)
    test_mask  = idx >= n_train + n_val

    log.info(
        "Split — train: %d | val: %d | test: %d",
        train_mask.sum(), val_mask.sum(), test_mask.sum()
    )
    return train_mask, val_mask, test_mask


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle the heavy class imbalance
    typical of security datasets (< 5 % attack events).
    """
    n_total   = labels.shape[0]
    n_positive = labels.sum().float()
    n_negative = n_total - n_positive
    # weight for class 1 (attack) = n_neg / n_pos
    pos_weight = n_negative / (n_positive + 1e-6)
    log.info("pos_weight (attack class): %.2f", pos_weight.item())
    return pos_weight
