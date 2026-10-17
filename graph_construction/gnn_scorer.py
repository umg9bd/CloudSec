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

    def build_graph(self, structural_df: pd.DataFrame):
        """(data, log_ids) exactly as score() scores them: the window's graph with the
        checkpoint's fitted scalers, untrained triples removed, and log_ids in the model's
        output order. (None, []) when nothing in the window can be scored."""
        if structural_df.empty:
            return None, []
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
            return None, []
        return data, [lid for t in triples for lid in data[t].log_id]

    @torch.no_grad()
    def score(self, structural_df: pd.DataFrame) -> pd.DataFrame:
        """gnn_prob per structural row (log_id), NaN for rows in an untrained triple."""
        out = pd.DataFrame({"log_id": structural_df["log_id"].astype(str), "gnn_prob": np.nan})
        data, log_ids = self.build_graph(structural_df)
        if data is None:
            return out
        probs = torch.sigmoid(self.model(data)).cpu().numpy()
        assert len(log_ids) == len(probs), f"{len(log_ids)} log_ids vs {len(probs)} probabilities"
        return out.drop(columns="gnn_prob").merge(
            pd.DataFrame({"log_id": log_ids, "gnn_prob": probs}), on="log_id", how="left")


class EnsembleGNNScorer:
    """Seed ensemble: averages per-edge gnn_prob across several GNNScorer
    checkpoints trained with different seeds. A single HGT checkpoint has high
    run-to-run variance on real data (edge-level ROC-AUC 0.78 +/- 0.11 over 5
    seeds); averaging 3 brings that to ~0.83 +/- 0.018 -- same-or-better mean,
    far more stable (see the seed-ensemble measurement).

    score() returns the mean gnn_prob across checkpoints, NaN only where EVERY
    checkpoint abstains (the edge's triple was untrained in all of them -- they
    share training data, so the abstain pattern is identical). build_graph() and
    .model delegate to the FIRST checkpoint: the per-event explanation is drawn
    from one representative model, while the risk uses the averaged score.
    """

    def __init__(self, ckpt_paths, device: str = "cpu"):
        if not ckpt_paths:
            raise ValueError("EnsembleGNNScorer needs at least one checkpoint")
        self.scorers = [GNNScorer(p, device) for p in ckpt_paths]
        primary = self.scorers[0]
        self.model = primary.model
        self.model_args = primary.model_args
        self.model_type = primary.model_type
        self.fit_artifacts = primary.fit_artifacts
        self.trained_triples = primary.trained_triples
        self.device = device

    def build_graph(self, structural_df: pd.DataFrame):
        return self.scorers[0].build_graph(structural_df)

    def score(self, structural_df: pd.DataFrame) -> pd.DataFrame:
        base = pd.DataFrame({"log_id": structural_df["log_id"].astype(str)})
        cols = []
        for i, s in enumerate(self.scorers):
            p = s.score(structural_df).rename(columns={"gnn_prob": f"_p{i}"})
            p["log_id"] = p["log_id"].astype(str)
            base = base.merge(p, on="log_id", how="left")
            cols.append(f"_p{i}")
        base["gnn_prob"] = base[cols].mean(axis=1, skipna=True)
        return base[["log_id", "gnn_prob"]]
