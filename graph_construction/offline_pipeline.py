"""
offline_pipeline.py
===================
`load_offline(csv_path, device)` -- the Neo4j-free graph loading entry point
used by the GNN-final branch's scripts (run_experiments.py,
compare_hgt_vs_sage_gat_ensemble.py, train_offline_baselines.py,
explainability's runner).

On GNN-final this module carried its own copy of the graph construction,
mirroring an earlier PrivilegePropagationGraphLoader.load(). This branch
already has one Neo4j-free loader, offline_graph.OfflineGraphLoader, verified
tensor-identical to the Neo4j loader and used by train.py --offline-csv and by
the live pipeline. Two loaders would drift (the GNN-final copy predates the
<UNK> edge-type class and reverse edges), so load_offline now delegates to it
and returns the same (data, meta) pair.
"""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from torch_geometric.data import HeteroData

from offline_graph import OfflineGraphLoader


def load_offline(csv_path: str, device: str = "cpu", fit_artifacts: dict = None,
                 add_reverse_edges: bool = False) -> Tuple[HeteroData, dict]:
    """(HeteroData, meta) for a structural CSV, exactly as
    PrivilegePropagationGraphLoader.load() would give for a Neo4j graph built
    from it. Pass fit_artifacts (from a wrapped checkpoint) to apply
    training-time scalers/encoders transform-only when scoring another graph."""
    return OfflineGraphLoader(pd.read_csv(csv_path), device=device, fit_artifacts=fit_artifacts,
                              add_reverse_edges=add_reverse_edges,
                              source_name=os.path.basename(csv_path)).load()
