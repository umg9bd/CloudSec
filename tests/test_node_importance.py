"""
test_node_importance.py
=======================
Guards node_importance.py (from GNN-final: ranks nodes to pick the
"important region" the HGT trains on) and hgt_attention_explainability.py.
Neither had tests.

Regression pinned: node_importance read its four edge signals by position
from a hand-copied column list that disagreed with data_loader's edge_attr
layout, so "abnormal_path_frequency" and "is_privilege_escalation_technique"
were read from action_global_frequency_log and is_read_only. Positions now
come from data_loader.EDGE_ATTR_NUMERIC_COLS.
"""

import unittest

import torch
from torch_geometric.data import HeteroData

import node_importance as ni
from data_loader import EDGE_ATTR_NUMERIC_COLS, NODE_FEATURE_SCHEMA
from hgt_attention_explainability import HGTAttentionExplainer
from model_hgt import HGTAnomalyDetector

TRIPLE = ("User", "READ", "Resource")
N_COLS = len(EDGE_ATTR_NUMERIC_COLS) + 3


def graph_varying(column, n_users=4):
    """Each user has one edge; only `column` differs, increasing with the user index."""
    data = HeteroData()
    data["User"].x = torch.zeros(n_users, len(NODE_FEATURE_SCHEMA["User"][0]))
    data["Resource"].x = torch.zeros(1, len(NODE_FEATURE_SCHEMA["Resource"][0]) + 1)
    data[TRIPLE].edge_index = torch.stack([torch.arange(n_users), torch.zeros(n_users, dtype=torch.long)])
    attr = torch.zeros(n_users, N_COLS)
    attr[:, EDGE_ATTR_NUMERIC_COLS.index(column)] = torch.arange(n_users, dtype=torch.float)
    data[TRIPLE].edge_attr = attr
    data[TRIPLE].y = torch.zeros(n_users)
    return data


def user_scores(data):
    return ni.compute_node_importance(data, [TRIPLE], NODE_FEATURE_SCHEMA)["User"]


class TestSignalColumns(unittest.TestCase):
    def test_signals_read_the_loaders_columns(self):
        self.assertEqual(ni._EDGE_SIGNAL_IDX["abnormal_path_frequency"],
                         EDGE_ATTR_NUMERIC_COLS.index("abnormal_path_frequency_rank"))
        for name in ("hop_count", "privilege_gain", "is_privilege_escalation_technique"):
            self.assertEqual(ni._EDGE_SIGNAL_IDX[name], EDGE_ATTR_NUMERIC_COLS.index(name))

    def test_ranking_signals_move_the_score(self):
        for column in ("hop_count", "privilege_gain", "abnormal_path_frequency_rank",
                       "is_privilege_escalation_technique"):
            s = user_scores(graph_varying(column))
            self.assertTrue(torch.all(s[1:] > s[:-1]), column)

    def test_non_signal_columns_do_not(self):
        for column in ("action_global_frequency_log", "is_read_only", "privilege_gain_defined"):
            s = user_scores(graph_varying(column))
            self.assertTrue(torch.allclose(s, s[0].expand_as(s)), column)


class TestRanking(unittest.TestCase):
    def test_percentile_rank(self):
        self.assertTrue(torch.equal(ni._percentile_rank(torch.tensor([30.0, 10.0, 20.0])),
                                    torch.tensor([1.0, 0.0, 0.5])))
        self.assertTrue(torch.equal(ni._percentile_rank(torch.tensor([7.0])), torch.tensor([0.5])))

    def test_ties_share_a_rank_and_constants_are_neutral(self):
        """Regression: ties used to get distinct ranks in index order, so a
        signal every node shared still ranked nodes by tensor position."""
        self.assertTrue(torch.equal(ni._percentile_rank(torch.zeros(5)), torch.full((5,), 0.5)))
        r = ni._percentile_rank(torch.tensor([1.0, 5.0, 1.0, 9.0]))
        self.assertEqual(float(r[0]), float(r[2]))
        self.assertTrue(torch.allclose(r, torch.tensor([1 / 6, 2 / 3, 1 / 6, 1.0])))

    def test_scores_are_in_unit_range(self):
        s = user_scores(graph_varying("privilege_gain", 6))
        self.assertTrue(torch.all((s >= 0) & (s <= 1)))

    def test_weights_change_the_mix(self):
        data = graph_varying("privilege_gain")
        plain = ni.compute_node_importance(data, [TRIPLE], NODE_FEATURE_SCHEMA)["User"]
        heavy = ni.compute_node_importance(data, [TRIPLE], NODE_FEATURE_SCHEMA,
                                           weights={"privilege_gain": 10.0})["User"]
        self.assertGreater(float(heavy[-1] - heavy[0]), float(plain[-1] - plain[0]))

    def test_selection_takes_the_top_fraction_with_a_floor(self):
        data = graph_varying("privilege_gain", 10)
        picked = ni.select_important_nodes(data, [TRIPLE], NODE_FEATURE_SCHEMA, top_frac=0.2)
        self.assertEqual(sorted(picked["User"].tolist()), [8, 9])
        self.assertEqual(len(picked["Resource"]), 1)        # min_per_type keeps a 1-node type

    def test_untouched_node_types_still_get_scores(self):
        data = graph_varying("hop_count")
        data["Role"].x = torch.zeros(2, len(NODE_FEATURE_SCHEMA["Role"][0]))
        self.assertEqual(user_scores(data).shape[0], 4)
        self.assertEqual(ni.compute_node_importance(data, [TRIPLE], NODE_FEATURE_SCHEMA)["Role"].shape[0], 2)


class TestHGTAttentionExplainer(unittest.TestCase):
    def test_returns_attention_or_none_without_changing_the_model(self):
        data = graph_varying("hop_count")
        data["User"].x = torch.randn(4, 4)
        data["Resource"].x = torch.randn(1, 6)
        data[TRIPLE].edge_attr = torch.randn(4, 7)
        torch.manual_seed(0)
        model = HGTAnomalyDetector(node_feat_dims={"User": 4, "Resource": 6}, edge_types=[TRIPLE],
                                   edge_feat_dim=7, hidden_dim=8, heads=2, num_hgt_layers=1).eval()
        with torch.no_grad():
            before = model(data)
        attn = HGTAttentionExplainer(model).attention_for_layer(data, layer=0)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(data), before))
        self.assertFalse(model.training)
        self.assertTrue(attn is None or isinstance(attn, (tuple, list, dict, torch.Tensor)))


if __name__ == "__main__":
    unittest.main()
