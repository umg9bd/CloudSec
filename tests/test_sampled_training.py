"""
test_sampled_training.py
========================
Guards train.train_model's neighbour-sampling path (train.py --sampling, from
GNN-final): the training forward/backward pass runs on the sampled view, while
validation and test are always measured on the FULL graph -- generalisation
beyond whatever region training was restricted to. Without sampling
(train_view=None) training is full-batch, exactly as before.
"""

import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from model_graphsage import GraphSAGEAnomalyDetector
from train import train_model

TRIPLE = ("User", "READ", "Resource")


def graph(n_edges, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["User"].x = torch.randn(6, 4, generator=g)
    data["Resource"].x = torch.randn(5, 6, generator=g)
    data[TRIPLE].edge_index = torch.stack([torch.arange(n_edges) % 6, torch.arange(n_edges) % 5])
    data[TRIPLE].edge_attr = torch.randn(n_edges, 7, generator=g)
    data[TRIPLE].y = (torch.arange(n_edges) % 3 == 0).long()
    return data


def masks(n, lo, hi):
    m = torch.zeros(n, dtype=torch.bool)
    m[lo:hi] = True
    return {TRIPLE: m}


class TestSampledTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.args = SimpleNamespace(device="cpu", loss="bce", lr=1e-3, epochs=5, save_dir=self.tmp,
                                    split="stratified", seed=0, threshold=0.5, patience=10,
                                    sampling="relation_aware")
        self.full = graph(30)
        self.sampled = graph(12, seed=1)
        torch.manual_seed(0)
        self.model = GraphSAGEAnomalyDetector(node_feat_dims={"User": 4, "Resource": 6},
                                              edge_types=[TRIPLE], edge_feat_dim=7, hidden_dim=8,
                                              num_sage_layers=1, dropout=0.0)
        self.seen = []
        self.model.register_forward_pre_hook(
            lambda mod, inp: self.seen.append((mod.training, inp[0][TRIPLE].edge_index.shape[1])))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_training(self, train_view):
        return train_model("SAGE", self.model, self.full, masks(30, 0, 20), masks(30, 20, 25),
                           masks(30, 25, 30), self.args, torch.tensor(2.0), train_view=train_view)

    def test_training_uses_the_sampled_view_and_evaluation_the_full_graph(self):
        self.run_training((self.sampled, masks(12, 0, 12)))
        training = {n for is_training, n in self.seen if is_training}
        evaluation = {n for is_training, n in self.seen if not is_training}
        self.assertEqual(training, {12})
        self.assertEqual(evaluation, {30})

    def test_without_sampling_training_is_full_batch(self):
        self.args.sampling = "none"
        self.run_training(None)
        self.assertEqual({n for is_training, n in self.seen if is_training}, {30})

    def test_returns_test_metrics_on_the_full_graph(self):
        metrics = self.run_training((self.sampled, masks(12, 0, 12)))
        self.assertIn("f1", metrics)
        self.assertIn("aupr", metrics)


if __name__ == "__main__":
    unittest.main()
