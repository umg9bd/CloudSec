"""
test_ensemble_explain.py
========================
Guards ensemble_explain.py and its use in pipeline.py: every alert says WHY it
was flagged, combining the graph model's and the sequence model's explanations.

  * Model shares are exact: the ensemble is linear, so w*p_graph and
    (1-w)*p_sequence split the risk, and they must reproduce pipeline.ensemble_risk.
  * Graph explanations use the exact scored graph: the flagged edge's own
    feature shares, the other events that moved it, and the graph is left
    unchanged afterwards (no requires_grad / grads left on it).
  * Sequence explanations reuse lstm_explain.explain_seq on the scored sequence.
  * The pipeline attaches an explanation and a one-line summary to each alert,
    and an explanation failure never stops an alert.
"""

import math
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

import ensemble_explain as ee
import lstm_explain as lx
import train_lstm_transformer as tlt
from data_loader import EDGE_ATTR_NUMERIC_COLS, scored_edge_types
from explainability import EDGE_FEATURE_NAMES
from pipeline import ensemble_risk

READ = ("User", "READ", "Resource")
WRITE = ("Role", "WRITE", "Resource")
N_COLS = len(EDGE_ATTR_NUMERIC_COLS) + 3


class EdgeModel(nn.Module):
    """logit_i = x_i . w  (+ mixing * sum of every scored edge's row . v)."""

    def __init__(self, mixing=0.0):
        super().__init__()
        self.edge_types = sorted([READ, WRITE])
        g = torch.Generator().manual_seed(7)
        self.w = nn.Parameter(torch.rand(N_COLS, generator=g))
        self.v = nn.Parameter(torch.rand(N_COLS, generator=g))
        self.mixing = mixing

    def forward(self, data):
        triples = [t for t in scored_edge_types(data) if t in self.edge_types]
        total = sum(data[t].edge_attr.sum(0) for t in triples)
        return torch.cat([data[t].edge_attr @ self.w + self.mixing * (total @ self.v) for t in triples])


def graph():
    g = torch.Generator().manual_seed(0)
    data = HeteroData()
    for ntype, n in (("User", 3), ("Role", 2), ("Resource", 3)):
        data[ntype].x = torch.zeros(n, 1)
    for triple, n, prefix in ((READ, 3, "r"), (WRITE, 2, "w")):
        data[triple].edge_index = torch.zeros((2, n), dtype=torch.long)
        data[triple].edge_attr = torch.rand((n, N_COLS), generator=g) + 0.1
        data[triple].y = torch.zeros(n)
        data[triple].log_id = [f"{prefix}{i}" for i in range(n)]
    order = [lid for t in scored_edge_types(data) for lid in data[t].log_id]   # model output order
    return data, order


class TestModelShares(unittest.TestCase):
    def test_shares_split_the_risk_exactly(self):
        for pg, ps, w in ((0.9, 0.2, 0.5), (0.1, 0.8, 0.4), (0.5, 0.5, 0.7)):
            s = ee._model_shares(pg, ps, w)
            self.assertAlmostEqual(s["graph"]["share"] + s["sequence"]["share"], 1.0, places=3)
            self.assertAlmostEqual(s["risk"], float(ensemble_risk(np.array([pg]), np.array([ps]), w)[0]), places=4)

    def test_unscored_graph_leaves_the_sequence_model_alone(self):
        s = ee._model_shares(float("nan"), 0.7, 0.5)
        self.assertFalse(s["graph_scored"])
        self.assertEqual(s["sequence"]["share"], 1.0)
        self.assertEqual(s["graph"]["weight"], 0.0)
        self.assertAlmostEqual(s["risk"], float(ensemble_risk(np.array([np.nan]), np.array([0.7]), 0.5)[0]))


class TestCombine(unittest.TestCase):
    EVENT = {"log_id": "r1", "event_name": "AttachRolePolicy", "username": "alice",
             "timestamp": "2024-01-01", "p_graph": 0.9, "p_sequence": 0.05}

    def test_summary_names_the_driver_and_skips_a_non_driver(self):
        e = ee.combine(self.EVENT, {"top_features": [{"feature": "privilege_gain", "share": 0.6}],
                                    "related_events": []}, None, 0.5, 0.5)
        self.assertEqual(e["driven_by"], "graph")
        self.assertIn("HGT", e["summary"])
        self.assertIn("privilege gained over the role that granted it", e["summary"])
        self.assertEqual(e["graph"]["top_features"][0]["label"], ee.label("privilege_gain"))
        self.assertIn("LSTM 5% (p=0.05): not a driver", e["summary"])

    def test_fast_lane_and_unscored_graph_are_stated(self):
        e = ee.combine(dict(self.EVENT, p_graph=float("nan")), None, None, 0.5, 0.5,
                       fast_lane_reason="CloudTrail trail deleted")
        self.assertTrue(e["summary"].startswith("FAST-LANE rule: CloudTrail trail deleted"))
        self.assertIn("HGT: relation not seen in training, LSTM only", e["summary"])
        self.assertEqual(e["driven_by"], "sequence")


class TestLabels(unittest.TestCase):
    """Every feature either model can report has a plain-English label, so a
    newly added feature cannot silently reach the summary as a raw column name."""

    def test_every_graph_feature_is_labelled(self):
        for name in EDGE_FEATURE_NAMES:
            self.assertIn(name, ee.FEATURE_LABELS, name)

    def test_every_sequence_input_is_labelled(self):
        import feature_engine9 as fe9
        for name in fe9.TEMPORAL_COLS + tlt.PE_CONTEXT_COLS + [lx.DT_FEATURE]:
            self.assertIn(name, ee.FEATURE_LABELS, name)

    def test_event_name_and_unknown_features(self):
        self.assertEqual(ee.label("event_name=AttachRolePolicy"), "the action AttachRolePolicy")
        self.assertEqual(ee.label("brand_new_feature"), "brand_new_feature")


class TestGraphExplanations(unittest.TestCase):
    def test_own_features_match_gradient_times_input(self):
        data, order = graph()
        model = EdgeModel(mixing=0.0)
        out = ee.explain_graph_events(model, data, order, ["r1"], top_k=len(EDGE_FEATURE_NAMES))["r1"]
        x = data[READ].edge_attr[1].detach()
        contrib = (x * model.w.detach()).abs()
        want = {n: float(contrib[c].sum() / contrib.sum())
                for n, c in __import__("explainability")._feature_groups(N_COLS, EDGE_FEATURE_NAMES)}
        for f in out["top_features"]:
            self.assertAlmostEqual(f["share"], want[f["feature"]], places=3, msg=f["feature"])
        self.assertEqual(out["relation"], "READ")
        self.assertEqual(out["related_events"], [])          # no mixing: no other edge matters
        self.assertAlmostEqual(out["own_edge_share"], 1.0, places=4)

    def test_related_events_come_from_message_passing(self):
        data, order = graph()
        structural = pd.DataFrame({"log_id": order, "source_node": "s", "target_node": "t",
                                   "edge_type": [f"E{lid}" for lid in order]})
        out = ee.explain_graph_events(EdgeModel(mixing=1.0), data, order, ["r1"], structural=structural)["r1"]
        related = {r["log_id"] for r in out["related_events"]}
        self.assertTrue(related and "r1" not in related)
        self.assertTrue(all(r["edge_type"] == f"E{r['log_id']}" for r in out["related_events"]))
        self.assertLess(out["own_edge_share"], 1.0)

    def test_every_target_is_explained_at_its_own_position(self):
        data, order = graph()
        model = EdgeModel()
        out = ee.explain_graph_events(model, data, order, order)
        with torch.no_grad():
            logits = model(data)
        for i, lid in enumerate(order):
            self.assertAlmostEqual(out[lid]["logit"], float(logits[i]), places=4, msg=lid)

    def test_graph_is_left_unchanged(self):
        data, order = graph()
        before = {t: data[t].edge_attr.clone() for t in data.edge_types}
        ee.explain_graph_events(EdgeModel(mixing=1.0), data, order, ["r0", "w1"])
        for t in data.edge_types:
            self.assertTrue(torch.equal(data[t].edge_attr, before[t]))
            self.assertFalse(data[t].edge_attr.requires_grad)

    def test_unscored_target_is_skipped(self):
        data, order = graph()
        self.assertEqual(ee.explain_graph_events(EdgeModel(), data, order, ["nope"]), {})


def small_lstm():
    torch.manual_seed(0)
    m = tlt.LSTMTransformerModel(vocab_size=12, n_features=5, secret_ids={3})
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return m.eval()


def small_seq(log_id="x:0"):
    rng = np.random.default_rng(1)
    idxs = rng.integers(1, 12, size=6).astype(np.int64)
    feats = rng.random((6, 5)).astype(np.float32)
    feats[:, -1] = np.log1p(rng.integers(1, 90, size=6)).astype(np.float32)
    feats[0, -1] = 0.0
    idxs, feats, n = tlt.pad_seq(idxs, feats, tlt.SEQ_LEN)
    return tlt.EventSeq(username="u", timestamp="2026-01-01T00:00:00Z", log_id=log_id, event_idxs=idxs,
                        feats=feats, length=n, label=1, last_idx=int(idxs[n - 1]), label_orig=1)


class TestSequenceExplanations(unittest.TestCase):
    def test_uses_lstm_explain_on_the_scored_sequence(self):
        ckpt = {"feature_cols": ["f0", "f1", "f2", "f3"]}
        vocab = {f"E{i}": i for i in range(1, 12)}
        out = ee.explain_sequence_events(small_lstm(), ckpt, vocab, [small_seq()], ["x:0", "missing"],
                                         torch.device("cpu"))
        self.assertEqual(list(out), ["x:0"])
        e = out["x:0"]
        self.assertTrue(e["top_events"] and e["top_features"])
        self.assertLess(e["ig_completeness_error"], 0.1)
        self.assertIn(lx.DT_FEATURE, lx.feature_names(ckpt))


class NoGraph:
    """Stands in for the graph model: scores nothing (every event LSTM-only)."""
    model = None

    def score(self, df):
        return pd.DataFrame({"log_id": df["log_id"].astype(str), "gnn_prob": np.nan})

    def build_graph(self, df):
        return None, []


class TestPipelineAttachesExplanations(unittest.TestCase):
    """Real configured LSTM, real features; the graph model is stubbed because
    the HGT checkpoint is being retrained (graph side is covered above)."""

    @classmethod
    def setUpClass(cls):
        import feature_engine9 as fe9
        import pipeline as P
        cls.P = P
        cls.pipe = P.Pipeline(P.PipelineConfig.load(), write_outputs=False)
        cls.pipe.gnn = NoGraph()
        dev = pd.read_csv("datasets/privilege-escalation/real_dataset_dev.csv", low_memory=False)
        attack = dev[dev["session_label"] == 1]["session_id"].unique()[:2]
        rows = pd.concat([dev[dev["session_id"].isin(attack)].head(300), dev[dev["session_label"] == 0].head(100)])
        # Read back through the pipeline's own input path, as live files are.
        import tempfile, os
        cls.tmp = tempfile.mkdtemp()
        cls.csv = os.path.join(cls.tmp, "slice.csv")
        rows.sort_values("timestamp").to_csv(cls.csv, index=False)
        cls.fe9 = fe9

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def featurized(self, name):
        # Each test is an independent first delivery: empty history, unique file name
        # (log_ids are "<file>:<row>", so re-sending a name would collide with the buffer).
        self.pipe.buffer = pd.DataFrame()
        type(self).n_runs = getattr(type(self), "n_runs", 0) + 1
        return self.pipe.featurize(enumerate(self.fe9.iter_input_rows(self.csv)), f"{self.n_runs}_{name}")

    def run_rows(self):
        return self.pipe.emit(self.pipe.score(self.featurized("t.csv")), "t.csv")

    def test_every_alert_carries_explanations_and_a_summary(self):
        alerts = self.run_rows()
        self.assertTrue(alerts, "expected at least one alert on attack sessions")
        for a in alerts:
            self.assertTrue(a["explanations"], a["principal"])
            self.assertLessEqual(len(a["explanations"]), self.pipe.cfg.explain_top_events)
            for e in a["explanations"]:
                self.assertIn("risk", e["summary"])
                self.assertEqual(e["models"]["sequence"]["share"], 1.0)   # graph stubbed out
                self.assertTrue(e["sequence"]["top_features"])

    def test_explanation_failure_never_blocks_the_alert(self):
        saved = self.pipe.lstm
        try:
            broken = type("Broken", (), {"model": None, "ckpt": {}, "vocab": {}, "device": None})()
            scored = self.pipe.score(self.featurized("t2.csv"))
            self.pipe.lstm = broken
            alerts = self.pipe.emit(scored, "t2.csv")
        finally:
            self.pipe.lstm = saved
        self.assertTrue(alerts)
        self.assertTrue(all(a["explanations"] == [] for a in alerts))

    def test_broken_graph_model_falls_back_to_the_lstm_with_explanations(self):
        """A graph checkpoint that cannot score (e.g. built by a different graph loader)
        must not stop detection: alerts still go out, LSTM-only, and say why."""
        class BrokenGraph(NoGraph):
            def score(self, df):
                raise ValueError("X has 2 features, but StandardScaler is expecting 4 features")

            def build_graph(self, df):
                raise AssertionError("must not be called once scoring failed")

        saved = self.pipe.gnn
        self.pipe.gnn = BrokenGraph()
        try:
            alerts = self.run_rows()
        finally:
            self.pipe.gnn = saved
        self.assertTrue(alerts)
        for a in alerts:
            self.assertTrue(a["explanations"], a["principal"])
            for e in a["explanations"]:
                self.assertIn("HGT: graph model failed on this batch, LSTM only", e["summary"])
                self.assertEqual(e["models"]["sequence"]["share"], 1.0)

    def test_can_be_switched_off(self):
        saved = self.pipe.cfg.explain_top_events
        self.pipe.cfg.explain_top_events = 0
        try:
            alerts = self.run_rows()
        finally:
            self.pipe.cfg.explain_top_events = saved
        self.assertTrue(all(a["explanations"] == [] for a in alerts))


if __name__ == "__main__":
    unittest.main()
