"""
test_ensemble.py
===================
Structural tests for EnsembleModel — checks the coverage-fallback logic
(the part of model_ensemble.py's design most worth a regression test: an
edge only one component covers should still get a sensible logit, not
NaN or a crash) using two toy models, without needing real
GraphSAGE/HGT weights.

From the GNN-final branch; converted from pytest functions to unittest (the
project runner's framework) with the assertions unchanged.
"""

import unittest

import torch
from torch_geometric.data import HeteroData

from model_ensemble import EnsembleModel


class _ConstLogitModel(torch.nn.Module):
    """Toy model: returns a fixed logit per edge, only for the triples
    listed in `edge_types` — mirrors the "only covers what it was
    constructed with, intersected with what data.edge_types has" contract
    every real model in this repo follows."""

    def __init__(self, edge_types, value):
        super().__init__()
        self.edge_types = edge_types
        self.value = value
        self._p = torch.nn.Parameter(torch.tensor(0.0))  # so nn.Module has a param

    def forward(self, data):
        chunks = []
        for t in sorted(data.edge_types):
            if t not in self.edge_types:
                continue
            n = data[t].edge_index.shape[1]
            chunks.append(torch.full((n,), self.value) + self._p * 0)
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0)


def _toy_data():
    data = HeteroData()
    data["A"].x = torch.zeros(2, 1)
    data["B"].x = torch.zeros(2, 1)
    data["A", "R1", "B"].edge_index = torch.tensor([[0, 1], [0, 1]])
    data["A", "R2", "B"].edge_index = torch.tensor([[0], [1]])
    # Labels mark a triple as scored (data_loader.scored_edge_types); reverse
    # message-passing triples carry none. Values are irrelevant here.
    data["A", "R1", "B"].y = torch.zeros(2)
    data["A", "R2", "B"].y = torch.zeros(1)
    return data


class TestEnsembleCoverage(unittest.TestCase):
    def test_full_coverage_blend(self):
        m1 = _ConstLogitModel([("A", "R1", "B"), ("A", "R2", "B")], value=2.0)
        m2 = _ConstLogitModel([("A", "R1", "B"), ("A", "R2", "B")], value=0.0)
        out = EnsembleModel([("m1", m1, 0.5), ("m2", m2, 0.5)])(_toy_data())
        self.assertTrue(torch.allclose(out, torch.full((3,), 1.0), atol=1e-5))

    def test_partial_coverage_falls_back_to_covering_component(self):
        m1 = _ConstLogitModel([("A", "R1", "B"), ("A", "R2", "B")], value=2.0)  # covers everything
        m2 = _ConstLogitModel([("A", "R1", "B")], value=0.0)                    # covers only R1
        out = EnsembleModel([("m1", m1, 0.5), ("m2", m2, 0.5)])(_toy_data())
        # R1 (2 edges): blended 0.5*2 + 0.5*0 = 1.0. R2 (1 edge): only m1
        # covers it, so full weight goes to m1 -> 2.0, not divided by the
        # missing component's weight.
        self.assertTrue(torch.allclose(out[:2], torch.full((2,), 1.0), atol=1e-5))
        self.assertTrue(torch.allclose(out[2:], torch.full((1,), 2.0), atol=1e-5))

    def test_single_component_ensemble_is_a_passthrough(self):
        m1 = _ConstLogitModel([("A", "R1", "B"), ("A", "R2", "B")], value=3.0)
        out = EnsembleModel([("only", m1, 1.0)])(_toy_data())
        self.assertTrue(torch.allclose(out, torch.full((3,), 3.0), atol=1e-5))

    def test_empty_components_raises(self):
        with self.assertRaises(ValueError):
            EnsembleModel([])


if __name__ == "__main__":
    unittest.main()
