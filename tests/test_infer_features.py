"""
test_infer_features.py
======================
Guards graph_construction/infer.py's streaming path against drifting from the
feature schema the GNN was trained on (data_loader.py), plus its alert output.

  - Feature parity: SubgraphLoader._edge_features / _node_features must build
    the SAME columns, in the SAME order, with the SAME transforms, as
    PrivilegePropagationGraphLoader._edge_features / _node_features do at
    training time -- and the widths the checkpoint's model_args declare
    (edge_feat_dim, node_feat_dims) and its fit_artifacts scalers expect.
    Since commit e8144eb action_global_frequency is rank-normalised within the
    graph (window-invariant, in [0, 1]); a raw count or log1p(count) is the
    train/inference skew that commit removed.
  - _build_heterodata: node types resolved via get_specific_label whatever the
    Neo4j labels() order, edges indexed to the right rows, unmappable edges and
    unlabelled nodes dropped, log_ids carried through.
  - generate_alert / _serialise_blast_report: the alert JSON contract.

The training-side reference is data_loader's own feature code run on the same
synthetic rows (Neo4j bypassed via object.__new__), not a re-implementation.
No Neo4j, no Docker, no network: the driver is a MagicMock.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
import uuid
import warnings
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd
import torch

import infer
from blast_radius import (BlastRadiusReport, CriticalAssetResult, PrivilegeReachabilityResult,
                          PropagationPath, ReachableAssets)
from data_loader import (EDGE_ATTR_NUMERIC_COLS, EDGE_NUM_COLS, NODE_FEATURE_SCHEMA,
                         PrivilegePropagationGraphLoader)
from infer import SubgraphLoader, _serialise_blast_report, generate_alert

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "best_HGT_seed1_wrapped.pt")
CPU = torch.device("cpu")
AGF_RANK_COL = EDGE_ATTR_NUMERIC_COLS.index("action_global_frequency_rank")


def _load_ckpt():
    if not os.path.exists(CKPT):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn InconsistentVersionWarning on unpickle
        return torch.load(CKPT, map_location="cpu", weights_only=False)


CK = _load_ckpt()
needs_ckpt = unittest.skipUnless(CK is not None, f"{CKPT} not present")


def quiet(fn, *a, **k):
    """infer.py's [DIAG] instrumentation prints a lot; keep test output readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def streaming_loader(fit_artifacts):
    return SubgraphLoader(driver=mock.MagicMock(), fit_artifacts=fit_artifacts, device=CPU)


def training_loader(fit_artifacts):
    """A PrivilegePropagationGraphLoader in evaluate-against-checkpoint mode, no Neo4j."""
    ld = object.__new__(PrivilegePropagationGraphLoader)
    ld._fit_artifacts = fit_artifacts
    ld._model_node_types = set(CK["model_args"]["node_feat_dims"]) if CK else None
    ld._add_reverse_edges = False
    ld.label_encoders, ld.node_scalers = {}, {}
    ld.edge_scaler = fit_artifacts["edge_scaler"]
    return ld


# Tiny synthetic Neo4j-shaped edge rows (what _fetch_neighbourhood returns).
# Distinct action_global_frequency counts so its percentile is unambiguous.
EDGE_ROWS = [
    {"src_key": "r1", "src_labels": ["Principal", "Role"], "dst_key": "b1", "dst_labels": ["Resource", "Target"],
     "relation": "LIST", "log_id": "f.csv:0", "edge_type": "ListBuckets", "hop_count": 1,
     "privilege_gain": 0.0, "privilege_gain_defined": False, "abnormal_path_frequency": 0.10,
     "action_global_frequency": 3, "is_privilege_escalation_technique": False, "is_attack": 0},
    {"src_key": "r1", "src_labels": ["Role", "Principal"], "dst_key": "b2", "dst_labels": ["Target", "Resource"],
     "relation": "LIST", "log_id": "f.csv:1", "edge_type": "GetObject", "hop_count": 2,
     "privilege_gain": 0.5, "privilege_gain_defined": True, "abnormal_path_frequency": 0.40,
     "action_global_frequency": 300, "is_privilege_escalation_technique": False, "is_attack": 0},
    {"src_key": "r2", "src_labels": ["Role"], "dst_key": "b1", "dst_labels": ["Resource"],
     "relation": "LIST", "log_id": "f.csv:2", "edge_type": "AttachRolePolicy", "hop_count": 1,
     "privilege_gain": 1.0, "privilege_gain_defined": True, "abnormal_path_frequency": 0.90,
     "action_global_frequency": 30000, "is_privilege_escalation_technique": True, "is_attack": 1},
]

NODE_ROWS = {
    "User": [{"key": "u1", "labels": ["User", "Principal"], "out_degree": 3, "unique_targets": 2,
              "unique_actions": 2, "role_transition_count": 0},
             {"key": "u2", "labels": ["Principal", "User"], "out_degree": 40, "unique_targets": 9,
              "unique_actions": 7, "role_transition_count": 5},
             {"key": "u3", "labels": ["User"], "out_degree": 400, "unique_targets": 30,
              "unique_actions": 20, "role_transition_count": 1}],
    "Resource": [{"key": "b1", "labels": ["Resource"], "in_degree": 5, "unique_principals": 2,
                  "resource_sensitivity": 0.2, "distance_to_sensitive_resource": 1, "resource_type": "unknown"},
                 {"key": "b2", "labels": ["Resource"], "in_degree": 50, "unique_principals": 8,
                  "resource_sensitivity": 0.9, "distance_to_sensitive_resource": None, "resource_type": "unknown"}],
    "Policy": [{"key": "p1", "labels": ["Policy"], "in_degree": 1, "unique_principals": 1,
                "resource_sensitivity": 0.5, "distance_to_sensitive_resource": 2},
               {"key": "p2", "labels": ["Policy"], "in_degree": 7, "unique_principals": 3,
                "resource_sensitivity": 0.7, "distance_to_sensitive_resource": 3}],
}


def training_edge_attr(rows, fit):
    """data_loader's edge_attr for these rows, exactly as load() prepares them."""
    df = pd.DataFrame(rows)
    df["is_read_only"] = df["edge_type"].str.startswith(
        ("Get", "List", "Describe", "Head", "Lookup", "Scan", "Query", "Search", "Check", "Validate")).astype(int)
    df["abnormal_path_frequency_rank"] = PrivilegePropagationGraphLoader._rank_normalize(df["abnormal_path_frequency"])
    df["action_global_frequency_rank"] = PrivilegePropagationGraphLoader._rank_normalize(
        df["action_global_frequency"].astype(float))
    return training_loader(fit)._edge_features(df, fit["label_encoders"]["edge_type"])


def training_node_x(ntype, rows, fit):
    num_cols, cat_cols = NODE_FEATURE_SCHEMA[ntype]
    df = pd.DataFrame([{c: r.get(c) for c in ["key"] + num_cols + cat_cols} for r in rows])
    return training_loader(fit)._node_features(ntype, df)


def no_scaler(fit):
    """The checkpoint's encoders, but no edge scaler: lets the streaming layout be inspected
    even where the checkpoint scaler would reject it."""
    return {**fit, "edge_scaler": None}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Edge features vs. the training schema
# ═══════════════════════════════════════════════════════════════════════════

@needs_ckpt
class TestCheckpointSchemaSelfConsistent(unittest.TestCase):
    """The training reference itself agrees with the checkpoint -- so the failures below are
    in the streaming builder, not in the oracle."""

    def test_training_edge_attr_width_equals_checkpoint_edge_feat_dim(self):
        fit = CK["fit_artifacts"]
        self.assertEqual(fit["edge_scaler"].n_features_in_, len(EDGE_NUM_COLS))
        self.assertEqual(training_edge_attr(EDGE_ROWS, fit).shape[1], CK["model_args"]["edge_feat_dim"])

    def test_training_node_widths_equal_checkpoint_node_feat_dims(self):
        fit = CK["fit_artifacts"]
        for ntype, rows in NODE_ROWS.items():
            with self.subTest(ntype):
                self.assertEqual(training_node_x(ntype, rows, fit).shape[1],
                                 CK["model_args"]["node_feat_dims"][ntype])

    def test_training_action_global_frequency_column_is_rank_in_unit_interval(self):
        col = training_edge_attr(EDGE_ROWS, CK["fit_artifacts"])[:, AGF_RANK_COL]
        np.testing.assert_allclose(col.numpy(), [1 / 3, 2 / 3, 1.0], atol=1e-6)


@needs_ckpt
class TestStreamingEdgeFeatures(unittest.TestCase):

    @unittest.expectedFailure
    def test_streaming_edge_attr_equals_training_edge_attr(self):
        """KNOWN DEFECT: infer.SubgraphLoader._edge_features predates commit e8144eb and the
        one-hot edge_type change. It builds 7 numeric columns
        [hop_count, privilege_gain, privilege_gain_defined, abnormal_path_frequency,
        log1p(action_global_frequency), is_privilege_escalation_technique, is_read_only]
        and passes them to the checkpoint's edge_scaler, which was fit on the 5 columns of
        data_loader.EDGE_NUM_COLS -> sklearn raises "X has 7 features, but StandardScaler is
        expecting 5 features". Training instead builds EDGE_NUM_COLS (scaled) +
        [abnormal_path_frequency_rank, action_global_frequency_rank] (unscaled, within-graph
        percentile) + a one-hot over the 76 edge_type classes = 83 = edge_feat_dim; streaming
        appends ONE ordinal edge_type code (width 8)."""
        fit = CK["fit_artifacts"]
        stream = streaming_loader(fit)._edge_features(EDGE_ROWS)
        torch.testing.assert_close(stream, training_edge_attr(EDGE_ROWS, fit))

    @unittest.expectedFailure
    def test_streaming_edge_attr_width_equals_checkpoint_edge_feat_dim(self):
        """KNOWN DEFECT: with the scaler step bypassed, streaming edge_attr is 8 wide
        (7 numeric + 1 ordinal edge_type code); the checkpoint's model_args['edge_feat_dim']
        is 83 (5 scaled + 2 rank + 76-class one-hot). The HGT edge head would reject it."""
        stream = streaming_loader(no_scaler(CK["fit_artifacts"]))._edge_features(EDGE_ROWS)
        self.assertEqual(stream.shape[1], CK["model_args"]["edge_feat_dim"])

    @unittest.expectedFailure
    def test_streaming_numeric_block_matches_training_column_order(self):
        """KNOWN DEFECT: streaming numeric column order is
        [hop_count, privilege_gain, privilege_gain_defined, abnormal_path_frequency,
        action_global_frequency_log, is_privilege_escalation_technique, is_read_only];
        training (EDGE_ATTR_NUMERIC_COLS) is
        [hop_count, privilege_gain, privilege_gain_defined, is_privilege_escalation_technique,
        is_read_only, abnormal_path_frequency_rank, action_global_frequency_rank].
        Probed per column: vary one input field and see which output column moves."""
        probe = {"hop_count": ("hop_count", [1, 2, 3]),
                 "privilege_gain": ("privilege_gain", [0.0, 0.5, 1.0]),
                 "privilege_gain_defined": ("privilege_gain_defined", [False, True, True]),
                 "is_privilege_escalation_technique": ("is_privilege_escalation_technique", [False, False, True]),
                 "is_read_only": ("edge_type", ["AttachRolePolicy", "GetObject", "ListBuckets"]),
                 "abnormal_path_frequency_rank": ("abnormal_path_frequency", [0.1, 0.4, 0.9]),
                 "action_global_frequency_rank": ("action_global_frequency", [3, 300, 30000])}
        base = [dict(EDGE_ROWS[0], edge_type="AttachRolePolicy", hop_count=1, privilege_gain=0.0,
                     privilege_gain_defined=False, is_privilege_escalation_technique=False,
                     abnormal_path_frequency=0.1, action_global_frequency=3) for _ in range(3)]
        ld = streaming_loader({})  # raw layout: no scaler, no encoder
        ref = ld._edge_features(base)
        positions = {}
        for name, (field, values) in probe.items():
            rows = [dict(r, **{field: v}) for r, v in zip(base, values)]
            moved = (ld._edge_features(rows) != ref).any(dim=0).nonzero().flatten().tolist()
            positions[name] = [m for m in moved if m < len(EDGE_ATTR_NUMERIC_COLS)]
        self.assertEqual(positions, {c: [i] for i, c in enumerate(EDGE_ATTR_NUMERIC_COLS)})

    @unittest.expectedFailure
    def test_action_global_frequency_is_rank_normalised_in_unit_interval(self):
        """KNOWN DEFECT: streaming encodes action_global_frequency as log1p(raw count)
        (infer.py _edge_features: `np.log1p(row.get("action_global_frequency") or 0)`), giving
        [1.386, 5.707, 10.309] for counts [3, 300, 30000]; training (data_loader.load,
        commit e8144eb) uses the within-graph percentile rank -> [1/3, 2/3, 1]."""
        ld = streaming_loader({})
        rows_b = [dict(r, action_global_frequency=1) for r in EDGE_ROWS]
        moved = (ld._edge_features(EDGE_ROWS) != ld._edge_features(rows_b)).any(dim=0)
        cols = ld._edge_features(EDGE_ROWS)[:, moved]
        self.assertEqual(cols.shape[1], 1, "exactly one column should carry action_global_frequency")
        self.assertTrue(bool(((cols >= 0) & (cols <= 1)).all()), f"not in [0,1]: {cols.flatten().tolist()}")
        np.testing.assert_allclose(cols.flatten().numpy(), [1 / 3, 2 / 3, 1.0], atol=1e-6)

    @unittest.expectedFailure
    def test_edge_features_are_window_invariant(self):
        """KNOWN DEFECT: the point of e8144eb is that a 100x larger graph/window (every
        action_global_frequency count x100) yields the same edge features. Streaming's
        log1p(count) shifts by ~log(100) instead."""
        ld = streaming_loader({})
        scaled = [dict(r, action_global_frequency=r["action_global_frequency"] * 100) for r in EDGE_ROWS]
        torch.testing.assert_close(ld._edge_features(EDGE_ROWS), ld._edge_features(scaled))

    @unittest.expectedFailure
    def test_edge_type_is_one_hot_over_fitted_classes(self):
        """KNOWN DEFECT: training one-hot encodes edge_type over the fitted classes (76 here);
        streaming appends a single ordinal LabelEncoder code (e.g. 3.0 for AttachRolePolicy)."""
        enc = CK["fit_artifacts"]["label_encoders"]["edge_type"]
        stream = streaming_loader(no_scaler(CK["fit_artifacts"]))._edge_features(EDGE_ROWS)
        cat = stream[:, len(EDGE_ATTR_NUMERIC_COLS):]
        self.assertEqual(cat.shape[1], len(enc.classes_))
        torch.testing.assert_close(cat.sum(dim=1), torch.ones(len(EDGE_ROWS)))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Node features vs. the training schema
# ═══════════════════════════════════════════════════════════════════════════

@needs_ckpt
class TestStreamingNodeFeatures(unittest.TestCase):

    def test_principal_node_width_equals_checkpoint(self):
        fit = CK["fit_artifacts"]
        for ntype in ("User", "Role", "UnresolvedPrincipal"):
            with self.subTest(ntype):
                x = streaming_loader(fit)._node_features(ntype, NODE_ROWS["User"])
                self.assertEqual(x.shape[1], CK["model_args"]["node_feat_dims"][ntype])

    @unittest.expectedFailure
    def test_principal_node_features_equal_training(self):
        """KNOWN DEFECT: every principal column (out_degree, unique_targets, unique_actions,
        role_transition_count) is in data_loader._COUNT_LIKE_COLS, so training feeds their
        within-graph percentile rank in (0, 1]. The checkpoint has no User scaler, so streaming
        _node_features passes the RAW counts through (e.g. out_degree 400.0 instead of 1.0)."""
        fit = CK["fit_artifacts"]
        torch.testing.assert_close(streaming_loader(fit)._node_features("User", NODE_ROWS["User"]),
                                   training_node_x("User", NODE_ROWS["User"], fit))

    @unittest.expectedFailure
    def test_target_node_features_equal_training(self):
        """KNOWN DEFECT: for Resource/Policy training z-scores only
        [resource_sensitivity, distance_to_sensitive_resource] (checkpoint scaler
        n_features_in_=2) and appends rank(in_degree), rank(unique_principals) AFTER them.
        Streaming feeds all 4 numeric columns, in NODE_FEATURE_SCHEMA order
        [in_degree, unique_principals, resource_sensitivity, distance], into that 2-feature
        scaler -> "X has 4 features, but StandardScaler is expecting 2 features"."""
        fit = CK["fit_artifacts"]
        for ntype in ("Resource", "Policy"):
            with self.subTest(ntype):
                torch.testing.assert_close(streaming_loader(fit)._node_features(ntype, NODE_ROWS[ntype]),
                                           training_node_x(ntype, NODE_ROWS[ntype], fit))

    def test_missing_distance_uses_unreachable_sentinel_like_training(self):
        x = streaming_loader({})._node_features("Resource", NODE_ROWS["Resource"])
        cols = NODE_FEATURE_SCHEMA["Resource"][0]
        self.assertEqual(float(x[1, cols.index("distance_to_sensitive_resource")]),
                         float(infer.UNREACHABLE_DISTANCE_SENTINEL))


# ═══════════════════════════════════════════════════════════════════════════
# 3. _build_heterodata structure (no scalers, so it runs end to end)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildHeteroData(unittest.TestCase):
    TRIPLE = ("Role", "LIST", "Resource")

    def setUp(self):
        self.nodes = [
            {"key": "r1", "labels": ["Principal", "Role"], "out_degree": 2},
            {"key": "r2", "labels": ["Role", "Principal"], "out_degree": 1},
            {"key": "b1", "labels": ["Target", "Resource"], "in_degree": 2},
            {"key": "b2", "labels": ["Resource"], "in_degree": 1},
            {"key": "ghost", "labels": ["Principal"]},  # no specific label -> dropped
        ]
        self.edges = EDGE_ROWS + [dict(EDGE_ROWS[0], dst_key="nowhere", log_id="f.csv:9")]
        self.data = quiet(streaming_loader({})._build_heterodata, self.nodes, self.edges)

    def test_specific_label_resolved_whatever_labels_order(self):
        self.assertEqual(self.data["Role"].key, ["r1", "r2"])
        self.assertEqual(self.data["Resource"].key, ["b1", "b2"])
        self.assertNotIn("Principal", self.data.node_types)

    def test_edges_index_the_right_rows_and_unmappable_edge_dropped(self):
        self.assertEqual(self.data.edge_types, [self.TRIPLE])
        e = self.data[self.TRIPLE]
        self.assertEqual(e.edge_index.tolist(), [[0, 0, 1], [0, 1, 0]])
        self.assertEqual(e.log_id, ["f.csv:0", "f.csv:1", "f.csv:2"])
        self.assertEqual(e.edge_attr.shape[0], 3)

    def test_node_x_follows_node_feature_schema(self):
        for ntype in ("Role", "Resource"):
            num, cat = NODE_FEATURE_SCHEMA[ntype]
            self.assertEqual(self.data[ntype].x.shape, (2, len(num) + len(cat)))

    def test_no_labels_leak_into_inference_graph(self):
        self.assertFalse(hasattr(self.data[self.TRIPLE], "y"))

    def test_empty_neighbourhood_returns_none_without_building(self):
        ld = streaming_loader({})
        ld._fetch_neighbourhood = mock.MagicMock(return_value=([], []))
        ld._build_heterodata = mock.MagicMock()
        self.assertIsNone(quiet(ld.load_neighbourhood, {"r1"}))
        ld._build_heterodata.assert_not_called()

    @needs_ckpt
    @unittest.expectedFailure
    def test_subgraph_with_checkpoint_artifacts_has_model_input_dims(self):
        """KNOWN DEFECT: with the checkpoint's real fit_artifacts, _build_heterodata raises
        before returning (Resource node scaler expects 2 features, gets 4; edge scaler expects
        5, gets 7). Even with those bypassed, edge_attr would be 8 wide vs edge_feat_dim 83.
        See TestStreamingEdgeFeatures / TestStreamingNodeFeatures for the column details."""
        data = quiet(streaming_loader(CK["fit_artifacts"])._build_heterodata, self.nodes, EDGE_ROWS)
        self.assertEqual(data[self.TRIPLE].edge_attr.shape[1], CK["model_args"]["edge_feat_dim"])
        for ntype in data.node_types:
            self.assertEqual(data[ntype].x.shape[1], CK["model_args"]["node_feat_dims"][ntype])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Alert output contract
# ═══════════════════════════════════════════════════════════════════════════

PRINCIPAL = ("User", "arn:aws:iam::111:user/alice")


def blast_report():
    return BlastRadiusReport(
        principal=PRINCIPAL,
        reachable_assets=ReachableAssets(counts={"s3": 2, "iam": 1}, total=3),
        privilege_reachability=PrivilegeReachabilityResult(
            assume_role_chain_depth=2, administrator_reachable=True, pass_role_observed=False,
            cross_account_ids_reachable={"333", "222"}),
        critical_assets=CriticalAssetResult(critical_asset_count=1,
                                            critical_assets=[("Resource", "arn:aws:s3:::secrets")],
                                            exposure_score=0.123456),
        top_paths=[PropagationPath(nodes=[PRINCIPAL, ("Role", "admin")],
                                   edges=[{"edge_type": "AssumeRole"}, {}], impact_score=0.987654)],
        cross_service_count=2, score=0.654321, score_components={"reach": 0.111111, "esc": 0.3},
    )


class TestSerialiseBlastReport(unittest.TestCase):
    def setUp(self):
        self.out = _serialise_blast_report(blast_report())

    def test_is_plain_json(self):
        self.assertEqual(json.loads(json.dumps(self.out)), self.out)  # no default=str needed

    def test_fields(self):
        o = self.out
        self.assertEqual((o["principal_label"], o["principal_key"]), PRINCIPAL)
        self.assertEqual(o["score"], 0.6543)
        self.assertEqual(o["score_components"], {"reach": 0.1111, "esc": 0.3})
        self.assertTrue(o["is_lower_bound"])
        self.assertEqual(o["reachable_assets"], {"total": 3, "by_category": {"s3": 2, "iam": 1}})
        pr = o["privilege_reachability"]
        self.assertEqual(pr["cross_account_ids_reachable"], ["222", "333"])  # set -> sorted list
        self.assertEqual((pr["assume_role_chain_depth"], pr["administrator_reachable"],
                          pr["administrator_reachable_is_proxy"], pr["pass_role_observed"]), (2, True, True, False))
        self.assertEqual(o["critical_assets"], {"count": 1, "exposure_score": 0.1235,
                                                "assets": ["Resource:arn:aws:s3:::secrets"]})
        self.assertEqual(o["cross_service_count"], 2)
        self.assertEqual(o["top_propagation_paths"], [{
            "nodes": ["User:arn:aws:iam::111:user/alice", "Role:admin"],
            "edge_types": ["AssumeRole", "?"], "impact_score": 0.9877}])


class TestGenerateAlert(unittest.TestCase):
    EVENT = {"timestamp": "2026-01-01T00:00:00Z", "principal_arn": PRINCIPAL[1], "principal_type": "IAMUser",
             "event_name": "AttachUserPolicy", "event_source": "iam.amazonaws.com",
             "target_resource": "arn:aws:iam::111:user/bob", "source_ip": "10.0.0.1",
             "aws_region": "us-east-1", "label": 1, "secret": "not exported"}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = os.path.join(self.tmp, "alerts", "nested")  # created on demand
        self.alert = generate_alert(self.EVENT, np.float32(0.91234567), PRINCIPAL, blast_report(), self.dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_written_and_equals_returned_dict(self):
        files = os.listdir(self.dir)
        self.assertEqual(files, [f"alert_{self.alert['alert_id']}.json"])
        with open(os.path.join(self.dir, files[0]), encoding="utf-8") as f:
            self.assertEqual(json.load(f), self.alert)

    def test_schema(self):
        a = self.alert
        self.assertEqual(set(a), {"alert_id", "generated_at", "event", "prediction", "blast_radius"})
        self.assertEqual(uuid.UUID(a["alert_id"]).version, 4)
        self.assertEqual(datetime.fromisoformat(a["generated_at"]).utcoffset().total_seconds(), 0)
        self.assertEqual(a["blast_radius"], _serialise_blast_report(blast_report()))

    def test_event_fields_copied_whitelist_only(self):
        ev = self.alert["event"]
        self.assertEqual(set(ev), {"timestamp", "principal_arn", "principal_type", "event_name", "event_source",
                                   "target_resource", "source_ip", "aws_region", "error_code"})
        for k in ev:
            self.assertEqual(ev[k], self.EVENT.get(k), k)  # error_code absent -> None
        self.assertNotIn("label", ev)

    def test_prediction(self):
        self.assertEqual(self.alert["prediction"], {"probability": round(float(np.float32(0.91234567)), 6),
                                                    "principal_label": "User", "principal_key": PRINCIPAL[1]})
        self.assertIsInstance(self.alert["prediction"]["probability"], float)


if __name__ == "__main__":
    unittest.main()
