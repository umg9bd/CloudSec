"""
ablation_groups.py
==================
The feature/relation groups the ablation study removes, and the one hook that removes a graph
group consistently at train AND serve time without touching any source file.

Graph groups (experiment E) act on the tensors OfflineGraphLoader.load() returns -- the single
place both graph_construction/train.py (--offline-csv) and gnn_scorer.GNNScorer build their
graph -- so a model retrained without a group is also scored without it:
  * feature groups: the selected edge_attr / node x columns are set to 0 (no information left);
  * relation groups: the relation's triples are deleted before training, so the model has no
    weights for them and gnn_scorer abstains (NaN) on those events at serve time -- the
    pipeline then scores them with the LSTM alone, exactly as it does any untrained triple.

Temporal groups (experiment F) are plain column lists passed to
temporal-analysis/train_lstm_transformer_v6_3.py --drop (which already supports it); the
pipeline reads the served feature list from the checkpoint, so nothing else changes.
"""
from __future__ import annotations

import contextlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # ablation/ -> privilege-escalation/ -> datasets/ -> root
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "graph_construction"), os.path.join(REPO_ROOT, "temporal-analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import EDGE_ATTR_NUMERIC_COLS, scored_edge_types  # noqa: E402

# edge_attr layout (data_loader.py): EDGE_ATTR_NUMERIC_COLS, then the edge_type one-hot block.
N_EDGE_NUMERIC = len(EDGE_ATTR_NUMERIC_COLS)
N_NODE_NUMERIC = 4  # every NODE_FEATURE_SCHEMA type has 4 numeric columns first (Resource adds 1 categorical)

GRAPH_GROUPS = {
    "privilege":       {"edge_cols": ["privilege_gain", "privilege_gain_defined", "is_privilege_escalation_technique"]},
    "frequency":       {"edge_cols": ["abnormal_path_frequency_rank", "action_global_frequency_rank"]},
    "action_identity": {"edge_onehot": True},
    "node_structure":  {"node_numeric": True},
    "rel_assumes":     {"drop_relations": ["ASSUMES"]},
}

# The LSTM v6.3 trainer always drops these (label-leaking, see its --audit); every F run keeps that.
LSTM_BASE_DROP = ["params_length_normalized", "no_mfa", "mfa_absent"]

TEMPORAL_GROUPS = {
    "timing": ["action_velocity", "session_duration_normalized", "events_per_minute_normalized",
               "time_sin", "time_cos", "is_weekend", "is_off_hours", "log_secs_since_pe"],
    "risk_priors": ["action_risk_prior", "principal_type_prior_risk"],
    "causal_lineage": ["causal_depth_normalized", "lineage_enabling_steps_normalized",
                       "principal_handoff", "pe_write_recent"],
    "permissions": ["policy_statement_count_normalized", "has_wildcard_action", "has_wildcard_resource",
                    "privileged_action_reach", "new_permission_count_log", "permission_expansion_score",
                    "privilege_delta", "target_permission_coverage", "actor_permission_coverage"],
}

_ACTIVE: dict | None = None
_ORIGINAL_LOAD = None


def _apply(data, meta, group: dict):
    if group.get("edge_cols") or group.get("edge_onehot"):
        idx = [EDGE_ATTR_NUMERIC_COLS.index(c) for c in group.get("edge_cols", [])]
        for t in data.edge_types:
            ea = data[t].edge_attr
            if idx:
                ea[:, idx] = 0.0
            if group.get("edge_onehot"):
                ea[:, N_EDGE_NUMERIC:] = 0.0
    if group.get("node_numeric"):
        for ntype in data.node_types:
            data[ntype].x[:, :N_NODE_NUMERIC] = 0.0
    drop = set(group.get("drop_relations", []))
    if drop:
        for t in list(data.edge_types):
            rel = t[1][4:] if t[1].startswith("REV_") else t[1]
            if rel in drop:
                del data[t]
        if meta is not None:
            meta["populated_triples"] = scored_edge_types(data)
            meta["encoder_triples"] = sorted(data.edge_types)
            meta["edge_counts"] = {f"{s}__{r}__{d}": int(data[(s, r, d)].y.shape[0])
                                   for (s, r, d) in meta["populated_triples"]}
    return data, meta


def _patched_load(self):
    data, meta = _ORIGINAL_LOAD(self)
    if _ACTIVE is not None:
        data, meta = _apply(data, meta, _ACTIVE)
    return data, meta


def _install():
    global _ORIGINAL_LOAD
    import offline_graph
    if _ORIGINAL_LOAD is None:
        _ORIGINAL_LOAD = offline_graph.OfflineGraphLoader.load
        offline_graph.OfflineGraphLoader.load = _patched_load


@contextlib.contextmanager
def graph_ablation(name: str | None):
    """Every OfflineGraphLoader.load() inside the block returns the graph without group `name`
    (None = the unmodified graph)."""
    global _ACTIVE
    _install()
    prev, _ACTIVE = _ACTIVE, (GRAPH_GROUPS[name] if name else None)
    try:
        yield
    finally:
        _ACTIVE = prev
