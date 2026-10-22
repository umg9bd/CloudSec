"""
test_gnn_scorer.py
==================
Contract tests for gnn_scorer.GNNScorer and EnsembleGNNScorer -- the graph
branch of the real-time pipeline. No Neo4j and no trained checkpoint: a tiny
HGT is built on a synthetic training graph and written with train.py's own
save_inference_checkpoint, the way tests/test_infer_checkpoint.py does it.

What is protected:

  - input contract: required structural columns; `label` optional and never a
    feature (scores identical whatever the label says)       -> TestInputContract
  - output contract: exactly [log_id, gnn_prob], one row per input row, input
    order preserved, log_id as str, probabilities in [0, 1], NaN only for
    rows in a triple the checkpoint never trained on         -> TestOutputContract
  - EnsembleGNNScorer is the per-row mean of its members, abstaining only
    where every member abstains                              -> TestEnsemble
  - bad checkpoints fail loudly, not with a half-built model -> TestCheckpointErrors
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from evaluate_on_real import build_model_from_args
from gnn_scorer import EnsembleGNNScorer, GNNScorer
from offline_graph import OfflineGraphLoader
from train import save_inference_checkpoint

A = "arn:aws:iam::123456789012"
S = "arn:aws:sts::123456789012"
UNTRAINED = ("User", "LIST", "Service")  # no Service targets in the training graph


def structural(rows):
    """rows: (log_id, source_node, target_node, edge_type, label)."""
    return pd.DataFrame(rows, columns=["log_id", "source_node", "target_node", "edge_type", "label"])


TRAIN = structural([
    ("train.csv:0", f"{A}:user/alice", f"{A}:role/dev", "AssumeRole", 0),
    ("train.csv:1", f"{S}:assumed-role/dev/s1", "arn:aws:s3:::b1", "GetObject", 0),
    ("train.csv:2", f"{A}:user/bob", "arn:aws:s3:::b2", "GetObject", 1),
    ("train.csv:3", f"{A}:user/bob", f"{A}:role/ops", "AssumeRole", 1),
    ("train.csv:4", f"{S}:assumed-role/ops/s2", "arn:aws:s3:::b3", "GetObject", 1),
    ("train.csv:5", f"{A}:user/alice", "arn:aws:s3:::b1", "GetObject", 0),
])

# Scoring window: rows 0,1,3,4 sit in trained triples; row 2 targets a Service node.
WINDOW = structural([
    ("live.csv:0", f"{A}:user/carol", f"{A}:role/dev", "AssumeRole", 0),
    ("live.csv:1", f"{S}:assumed-role/dev/s9", "arn:aws:s3:::b9", "GetObject", 0),
    ("live.csv:2", f"{A}:user/carol", "iam.amazonaws.com", "ListRoles", 0),
    ("live.csv:3", f"{A}:user/dave", "arn:aws:s3:::b8", "GetObject", 0),
    ("live.csv:4", f"{A}:user/dave", f"{A}:role/ops", "AssumeRole", 0),
])
UNTRAINED_ID = "live.csv:2"


class Sandbox(unittest.TestCase):
    """Synthetic checkpoints and their WINDOW scores, built once per module
    (setUpModule) -- every score() call reloads the access-level resolver."""

    @classmethod
    def build(cls):
        cls.tmp = tempfile.mkdtemp()
        loader = OfflineGraphLoader(TRAIN)
        _, cls.meta = loader.load()
        cls.fit = {"edge_scaler": loader.edge_scaler, "node_scalers": loader.node_scalers,
                   "label_encoders": loader.label_encoders}
        args = SimpleNamespace(hidden=8, layers=1, heads=2, dropout=0.0, reverse_edges=False,
                               offline_csv="train.csv", save_dir=cls.tmp)
        torch.manual_seed(0)
        cls.path_a = save_inference_checkpoint("HGT", cls.model(cls.meta["populated_triples"]), cls.meta, loader, args)
        cls.path_b = cls.write_ckpt("b.pt", cls.meta["populated_triples"], seed=1)
        # Same training graph, but one triple dropped: this member abstains where the others score.
        cls.dropped = ("User", "ASSUMES", "Role")
        cls.path_c = cls.write_ckpt("c.pt", [t for t in cls.meta["populated_triples"]
                                             if tuple(t) != cls.dropped], seed=2)
        cls.scorer = GNNScorer(cls.path_a)
        cls.out = cls.scorer.score(WINDOW)
        cls.members = {cls.path_a: cls.out.set_index("log_id")["gnn_prob"]}
        for p in (cls.path_b, cls.path_c):
            cls.members[p] = GNNScorer(p).score(WINDOW).set_index("log_id")["gnn_prob"]

    @classmethod
    def model_args(cls, triples):
        return {"model_type": "hgt", "node_feat_dims": cls.meta["node_feat_dim"], "edge_types": triples,
                "edge_feat_dim": cls.meta["edge_feat_dim"], "hidden_dim": 8, "num_sage_layers": 1,
                "num_gat_layers": 1, "num_hgt_layers": 1, "heads": 2, "dropout": 0.0, "add_reverse_edges": False}

    @classmethod
    def model(cls, triples):
        return build_model_from_args("hgt", cls.model_args(triples))

    @classmethod
    def write_ckpt(cls, name, triples, seed):
        torch.manual_seed(seed)
        path = os.path.join(cls.tmp, name)
        torch.save({"state_dict": cls.model(triples).state_dict(), "model_args": cls.model_args(triples),
                    "fit_artifacts": cls.fit}, path)
        return path


def setUpModule():
    Sandbox.build()


def tearDownModule():
    shutil.rmtree(Sandbox.tmp, ignore_errors=True)


class TestOutputContract(Sandbox):
    def test_columns_and_one_row_per_input_row(self):
        self.assertEqual(list(self.out.columns), ["log_id", "gnn_prob"])
        self.assertEqual(len(self.out), len(WINDOW))
        self.assertEqual(self.out["log_id"].tolist(), WINDOW["log_id"].tolist())

    def test_scores_are_probabilities(self):
        p = self.out["gnn_prob"].dropna()
        self.assertEqual(len(p), len(WINDOW) - 1)
        self.assertTrue(p.between(0, 1).all())

    def test_untrained_triple_abstains_with_nan(self):
        self.assertNotIn(UNTRAINED, self.scorer.trained_triples)
        raw, _ = OfflineGraphLoader(WINDOW, fit_artifacts=self.fit).load()
        self.assertIn(UNTRAINED, raw.edge_types)  # the window really contains it
        nan_ids = self.out.loc[self.out["gnn_prob"].isna(), "log_id"].tolist()
        self.assertEqual(nan_ids, [UNTRAINED_ID])

    def test_row_order_follows_input_not_graph(self):
        rev = WINDOW.iloc[::-1]
        out = self.scorer.score(rev)
        self.assertEqual(out["log_id"].tolist(), rev["log_id"].tolist())
        a = self.out.set_index("log_id")["gnn_prob"]
        b = out.set_index("log_id")["gnn_prob"]
        pd.testing.assert_series_equal(a, b.loc[a.index])  # same graph, same scores

    def test_non_default_index_input(self):
        shifted = WINDOW.set_axis([10, 20, 30, 40, 50])
        out = self.scorer.score(shifted)
        np.testing.assert_allclose(out["gnn_prob"].to_numpy(), self.out["gnn_prob"].to_numpy())

    def test_integer_log_ids_come_back_as_strings(self):
        df = WINDOW.assign(log_id=range(len(WINDOW)))
        out = self.scorer.score(df)
        self.assertEqual(out["log_id"].tolist(), [str(i) for i in range(len(WINDOW))])
        np.testing.assert_allclose(out["gnn_prob"].to_numpy(), self.out["gnn_prob"].to_numpy())

    def test_scoring_is_deterministic(self):
        pd.testing.assert_frame_equal(GNNScorer(self.path_a).score(WINDOW), self.out)

    def test_empty_window(self):
        out = self.scorer.score(WINDOW.iloc[0:0])
        self.assertEqual(list(out.columns), ["log_id", "gnn_prob"])
        self.assertEqual(len(out), 0)
        self.assertEqual(self.scorer.build_graph(WINDOW.iloc[0:0]), (None, []))

    def test_window_with_only_untrained_rows_is_all_nan(self):
        only = WINDOW[WINDOW["log_id"] == UNTRAINED_ID]
        out = self.scorer.score(only)
        self.assertEqual(out["log_id"].tolist(), [UNTRAINED_ID])
        self.assertTrue(out["gnn_prob"].isna().all())

    def test_build_graph_log_ids_match_scored_rows(self):
        data, log_ids = self.scorer.build_graph(WINDOW)
        self.assertEqual(sorted(log_ids), sorted(set(WINDOW["log_id"]) - {UNTRAINED_ID}))
        self.assertNotIn(UNTRAINED, data.edge_types)


class TestInputContract(Sandbox):
    def test_missing_structural_column_is_a_clear_error(self):
        for col in ("source_node", "target_node", "edge_type"):
            with self.subTest(col):
                with self.assertRaises(ValueError) as ctx:
                    self.scorer.score(WINDOW.drop(columns=col))
                self.assertIn(col, str(ctx.exception))

    def test_missing_log_id_raises(self):
        with self.assertRaises(KeyError) as ctx:
            self.scorer.score(WINDOW.drop(columns="log_id"))
        self.assertIn("log_id", str(ctx.exception))

    def test_label_column_is_optional(self):
        pd.testing.assert_frame_equal(self.scorer.score(WINDOW.drop(columns="label")), self.out)

    def test_label_value_never_changes_scores(self):
        """A live event's label is unknown; if it moved the score, inference would
        leak ground truth (via e.g. is_known_attacker_identity)."""
        flipped = WINDOW.assign(label=1)
        pd.testing.assert_frame_equal(self.scorer.score(flipped), self.out)

    def test_caller_frame_not_mutated(self):
        df = WINDOW.drop(columns="label").iloc[:2]
        before = df.copy()
        self.scorer.score(df)
        pd.testing.assert_frame_equal(df, before)


class TestEnsemble(Sandbox):
    def test_ensemble_is_mean_of_members(self):
        ens = EnsembleGNNScorer([self.path_a, self.path_b]).score(WINDOW)
        self.assertEqual(list(ens.columns), ["log_id", "gnn_prob"])
        self.assertEqual(ens["log_id"].tolist(), WINDOW["log_id"].tolist())
        expected = (self.members[self.path_a] + self.members[self.path_b]) / 2
        np.testing.assert_allclose(ens["gnn_prob"].to_numpy(), expected.loc[ens["log_id"]].to_numpy(), rtol=1e-6)
        self.assertTrue(ens["gnn_prob"].dropna().between(0, 1).all())

    def test_members_differ_so_mean_is_meaningful(self):
        a, b = self.members[self.path_a].dropna(), self.members[self.path_b].dropna()
        self.assertFalse(np.allclose(a.to_numpy(), b.loc[a.index].to_numpy()))

    def test_single_member_ensemble_equals_member(self):
        ens = EnsembleGNNScorer([self.path_a]).score(WINDOW)
        np.testing.assert_allclose(ens["gnn_prob"].to_numpy(),
                                   self.members[self.path_a].loc[ens["log_id"]].to_numpy())

    def test_abstains_only_where_every_member_abstains(self):
        ens = EnsembleGNNScorer([self.path_a, self.path_c]).score(WINDOW).set_index("log_id")["gnn_prob"]
        c = self.members[self.path_c]
        abstain_c = set(c[c.isna()].index) - {UNTRAINED_ID}
        self.assertTrue(abstain_c)  # member c really abstains on the dropped triple
        for lid in abstain_c:
            self.assertAlmostEqual(ens[lid], self.members[self.path_a][lid], places=6)
        self.assertTrue(np.isnan(ens[UNTRAINED_ID]))

    def test_delegates_graph_and_metadata_to_first_member(self):
        ens = EnsembleGNNScorer([self.path_c, self.path_a])
        self.assertEqual(ens.trained_triples, ens.scorers[0].trained_triples)
        self.assertIs(ens.model, ens.scorers[0].model)
        self.assertNotIn(self.dropped, ens.trained_triples)

    def test_empty_window(self):
        out = EnsembleGNNScorer([self.path_a, self.path_b]).score(WINDOW.iloc[0:0])
        self.assertEqual(list(out.columns), ["log_id", "gnn_prob"])
        self.assertEqual(len(out), 0)

    def test_no_checkpoints_is_an_error(self):
        with self.assertRaises(ValueError):
            EnsembleGNNScorer([])


class TestCheckpointErrors(Sandbox):
    def test_missing_checkpoint_file(self):
        with self.assertRaises(FileNotFoundError):
            GNNScorer(os.path.join(self.tmp, "does_not_exist.pt"))

    def test_corrupt_checkpoint_file(self):
        path = os.path.join(self.tmp, "corrupt.pt")
        with open(path, "wb") as f:
            f.write(b"not a torch checkpoint")
        with self.assertRaises(Exception):
            GNNScorer(path)

    def test_bare_state_dict_names_the_file(self):
        path = os.path.join(self.tmp, "bare.pt")
        torch.save(self.model(self.meta["populated_triples"]).state_dict(), path)
        with self.assertRaises(ValueError) as ctx:
            GNNScorer(path)
        self.assertIn(path, str(ctx.exception))

    def test_checkpoint_without_fit_artifacts_rejected(self):
        ckpt = torch.load(self.path_a, weights_only=False)
        path = os.path.join(self.tmp, "nofit.pt")
        torch.save({k: v for k, v in ckpt.items() if k != "fit_artifacts"}, path)
        with self.assertRaises(ValueError):
            GNNScorer(path)

    def test_weights_not_matching_args_rejected(self):
        ckpt = torch.load(self.path_a, weights_only=False)
        path = os.path.join(self.tmp, "mismatch.pt")
        torch.save(dict(ckpt, model_args=dict(ckpt["model_args"], hidden_dim=16)), path)
        with self.assertRaises(RuntimeError):
            GNNScorer(path)

    def test_ensemble_fails_if_any_member_is_bad(self):
        with self.assertRaises(FileNotFoundError):
            EnsembleGNNScorer([self.path_a, os.path.join(self.tmp, "missing.pt")])


if __name__ == "__main__":
    unittest.main()
