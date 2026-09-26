"""
gnn_scorer.py
=============
Per-event attack probability from a trained heterogeneous GNN (HGT, GraphSAGE
or GAT), straight from structural rows -- no Neo4j.

    scorer = GNNScorer("checkpoints/best_HGT_wrapped.pt")
    probs = scorer.score(structural_df)   # DataFrame[log_id, gnn_prob]

The graph is built by offline_graph.OfflineGraphLoader (tensor-identical to the
Neo4j loader) with the checkpoint's training-fitted scalers/encoders applied
transform-only, and the model is built by evaluate_on_real.build_model_from_args
-- the same construction evaluate_session_level.py uses. So a score here is the
score batch evaluation would give the same events in the same graph.

Events whose (source type, relation, target type) never occurred in training
have no trained weights to score them; they get gnn_prob = NaN rather than a
guess, and the caller decides the fallback (pipeline.py leaves the ensemble to
the sequence branch for those).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from data_loader import scored_edge_types
from evaluate_on_real import build_model_from_args
from offline_graph import OfflineGraphLoader


class GNNScorer:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_args" not in ckpt or "fit_artifacts" not in ckpt:
            raise ValueError(f"{ckpt_path} is not an inference-ready checkpoint (need model_args + "
                             f"fit_artifacts: train with graph_construction/train.py, which writes "
                             f"best_<model>_wrapped.pt)")
        self.model_args = ckpt["model_args"]
        self.model_type = self.model_args.get("model_type", "sage")
        self.fit_artifacts = ckpt["fit_artifacts"]
        self.trained_triples = {tuple(t) for t in self.model_args["edge_types"]}
        self.add_reverse_edges = any(str(t[1]).startswith("REV_") for t in self.trained_triples)
        self.device = device
        self.model = build_model_from_args(self.model_type, self.model_args)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()

    @torch.no_grad()
    def score(self, structural_df: pd.DataFrame) -> pd.DataFrame:
        """gnn_prob per structural row (log_id), NaN for rows in an untrained triple."""
        out = pd.DataFrame({"log_id": structural_df["log_id"].astype(str), "gnn_prob": np.nan})
        if structural_df.empty:
            return out
        df = structural_df.copy()
        if "label" not in df.columns:
            df["label"] = 0  # unlabeled live events: the label only feeds `y`, never a feature
        data, _ = OfflineGraphLoader(df, device=self.device, fit_artifacts=self.fit_artifacts,
                                     model_node_types=set(self.model_args["node_feat_dims"]),
                                     add_reverse_edges=self.add_reverse_edges).load()
        for t in list(scored_edge_types(data)):
            if t not in self.trained_triples:
                del data[t]
        triples = scored_edge_types(data)
        if not triples:
            return out
        probs = torch.sigmoid(self.model(data)).cpu().numpy()
        log_ids = [lid for t in triples for lid in data[t].log_id]
        assert len(log_ids) == len(probs), f"{len(log_ids)} log_ids vs {len(probs)} probabilities"
        return out.drop(columns="gnn_prob").merge(
            pd.DataFrame({"log_id": log_ids, "gnn_prob": probs}), on="log_id", how="left")
