"""
test_model_hgt.py
====================
Structural tests for HGTAnomalyDetector — shape/contract checks only, no
numerical-correctness claims (this environment has no working torch
install to run them against real numbers; see model_hgt.py's caveat).
Run these for real in Colab / any environment with torch +
torch_geometric installed before trusting the model.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData  # noqa: E402
from model_hgt import HGTAnomalyDetector, build_hgt_from_args  # noqa: E402


def _toy_data():
    data = HeteroData()
    data["User"].x = torch.randn(5, 4)
    data["Resource"].x = torch.randn(3, 6)
    data["User", "READ", "Resource"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data["User", "READ", "Resource"].edge_attr = torch.randn(3, 7)
    return data


def test_forward_shape_matches_edge_count():
    data = _toy_data()
    model = HGTAnomalyDetector(
        node_feat_dims={"User": 4, "Resource": 6},
        edge_types=[("User", "READ", "Resource")],
        edge_feat_dim=7,
        hidden_dim=16,
        heads=2,
        num_hgt_layers=1,
    )
    out = model(data)
    assert out.shape == (3,)


def test_heads_must_divide_hidden_dim():
    with pytest.raises(ValueError):
        HGTAnomalyDetector(
            node_feat_dims={"User": 4, "Resource": 6},
            edge_types=[("User", "READ", "Resource")],
            edge_feat_dim=7,
            hidden_dim=15,
            heads=4,
        )


def test_missing_triple_at_runtime_is_skipped_not_crashed():
    """A triple declared at construction but absent from a given batch
    (e.g. a mini-batch that happened to sample none of it) should be
    skipped, same contract as GraphSAGEAnomalyDetector/GATAnomalyDetector."""
    data = HeteroData()
    data["User"].x = torch.randn(2, 4)
    data["Resource"].x = torch.randn(2, 6)
    data["User", "READ", "Resource"].edge_index = torch.tensor([[0], [0]])
    data["User", "READ", "Resource"].edge_attr = torch.randn(1, 7)
    model = HGTAnomalyDetector(
        node_feat_dims={"User": 4, "Resource": 6},
        edge_types=[("User", "READ", "Resource"), ("User", "WRITE", "Resource")],
        edge_feat_dim=7,
        hidden_dim=16,
        heads=2,
        num_hgt_layers=1,
    )
    out = model(data)
    assert out.shape == (1,)


def test_build_hgt_from_args_uses_defaults():
    args = {
        "node_feat_dims": {"User": 4, "Resource": 6},
        "edge_types": [("User", "READ", "Resource")],
        "edge_feat_dim": 7,
    }
    model = build_hgt_from_args(args)
    assert model.encoder.hidden_dim == 128  # documented default
    assert model.encoder.heads == 4
