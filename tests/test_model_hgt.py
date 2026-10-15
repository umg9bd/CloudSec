"""
test_model_hgt.py
=================
Contract checks for HGTAnomalyDetector (from the GNN-final branch): output
shape and ordering follow the same flat, scored_edge_types() order the loader's
labels use, the same as GraphSAGE/GAT.
"""
import unittest

import torch
from torch_geometric.data import HeteroData

from model_hgt import HGTAnomalyDetector, build_hgt_from_args


def _toy_data():
    data = HeteroData()
    data["User"].x = torch.randn(5, 4)
    data["Resource"].x = torch.randn(3, 6)
    data["User", "READ", "Resource"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["User", "READ", "Resource"].edge_attr = torch.randn(3, 7)
    data["User", "READ", "Resource"].y = torch.zeros(3)  # scored_edge_types() keys on labels
    return data


def _model(edge_types, hidden_dim=16, heads=2):
    return HGTAnomalyDetector(node_feat_dims={"User": 4, "Resource": 6}, edge_types=edge_types,
                              edge_feat_dim=7, hidden_dim=hidden_dim, heads=heads, num_hgt_layers=1)


class TestHGT(unittest.TestCase):
    def test_forward_shape_matches_edge_count(self):
        self.assertEqual(_model([("User", "READ", "Resource")])(_toy_data()).shape, (3,))

    def test_heads_must_divide_hidden_dim(self):
        with self.assertRaises(ValueError):
            _model([("User", "READ", "Resource")], hidden_dim=15, heads=4)

    def test_missing_triple_at_runtime_is_skipped_not_crashed(self):
        """A triple declared at construction but absent from a batch is skipped, the same contract
        as GraphSAGEAnomalyDetector/GATAnomalyDetector."""
        data = HeteroData()
        data["User"].x = torch.randn(2, 4)
        data["Resource"].x = torch.randn(2, 6)
        data["User", "READ", "Resource"].edge_index = torch.tensor([[0], [0]])
        data["User", "READ", "Resource"].edge_attr = torch.randn(1, 7)
        data["User", "READ", "Resource"].y = torch.zeros(1)
        model = _model([("User", "READ", "Resource"), ("User", "WRITE", "Resource")])
        self.assertEqual(model(data).shape, (1,))

    def test_build_hgt_from_args_uses_defaults(self):
        model = build_hgt_from_args({"node_feat_dims": {"User": 4, "Resource": 6},
                                     "edge_types": [("User", "READ", "Resource")], "edge_feat_dim": 7})
        self.assertEqual(model.encoder.hidden_dim, 128)
        self.assertEqual(model.encoder.heads, 4)


if __name__ == "__main__":
    unittest.main()
