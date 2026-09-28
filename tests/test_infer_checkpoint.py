"""
test_infer_checkpoint.py
========================
Guards infer.load_model_from_checkpoint, which used to build a GraphSAGE model
whatever the checkpoint held. It now dispatches on model_args["model_type"]:
sage (or GNN-final's "graphsage"), gat, hgt, and model_ensemble ensembles.

The round trip that matters: a checkpoint written by train.py's
save_inference_checkpoint must load back into the same architecture with the
same weights, and so produce the same logits.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from evaluate_on_real import build_model_from_args
from infer import load_model_from_checkpoint
from model_ensemble import EnsembleModel
from model_gat import GATAnomalyDetector
from model_graphsage import GraphSAGEAnomalyDetector
from model_hgt import HGTAnomalyDetector
from train import save_inference_checkpoint

TRIPLE = ("User", "READ", "Resource")
META = {"node_feat_dim": {"User": 4, "Resource": 6}, "populated_triples": [TRIPLE], "edge_feat_dim": 7}
ARGS = SimpleNamespace(hidden=16, layers=1, heads=2, dropout=0.0, reverse_edges=False, offline_csv="x.csv")
LOADER = SimpleNamespace(edge_scaler="scaler", node_scalers={"User": "s"}, label_encoders={"edge_type": "e"})
CPU = torch.device("cpu")


def toy_data():
    torch.manual_seed(0)
    data = HeteroData()
    data["User"].x = torch.randn(5, 4)
    data["Resource"].x = torch.randn(3, 6)
    data[TRIPLE].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
    data[TRIPLE].edge_attr = torch.randn(4, 7)
    data[TRIPLE].y = torch.zeros(4)
    return data


class CheckpointSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.args = SimpleNamespace(**vars(ARGS), save_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def trained(self, name):
        torch.manual_seed(1)
        kind = {"GraphSAGE": "sage", "GAT": "gat", "HGT": "hgt"}[name]
        model = build_model_from_args(kind, {
            "node_feat_dims": META["node_feat_dim"], "edge_types": META["populated_triples"],
            "edge_feat_dim": 7, "hidden_dim": 16, "dropout": 0.0, "num_sage_layers": 1,
            "num_gat_layers": 1, "num_hgt_layers": 1, "heads": 2})
        path = save_inference_checkpoint(name, model, META, LOADER, self.args)
        return model.eval(), path


class TestRoundTrip(CheckpointSandbox):
    def test_every_trained_architecture_loads_back_identically(self):
        data = toy_data()
        for name, cls in (("GraphSAGE", GraphSAGEAnomalyDetector), ("GAT", GATAnomalyDetector),
                          ("HGT", HGTAnomalyDetector)):
            with self.subTest(name):
                original, path = self.trained(name)
                loaded, fit = load_model_from_checkpoint(path, CPU)
                self.assertIsInstance(loaded, cls)
                self.assertFalse(loaded.training)
                self.assertEqual(fit["edge_scaler"], "scaler")
                with torch.no_grad():
                    self.assertTrue(torch.allclose(original(data), loaded(data)))

    def test_graphsage_alias_and_missing_type_load_graphsage(self):
        _, path = self.trained("GraphSAGE")
        ckpt = torch.load(path, weights_only=False)
        for model_type in ("graphsage", None):
            args = dict(ckpt["model_args"])
            if model_type is None:
                args.pop("model_type")
            else:
                args["model_type"] = model_type
            torch.save(dict(ckpt, model_args=args), path)
            model, _ = load_model_from_checkpoint(path, CPU)
            self.assertIsInstance(model, GraphSAGEAnomalyDetector, model_type)


class TestEnsembleAndErrors(CheckpointSandbox):
    def test_ensemble_checkpoint(self):
        sage, _ = self.trained("GraphSAGE")
        gat, _ = self.trained("GAT")
        sub = {"node_feat_dims": META["node_feat_dim"], "edge_types": [TRIPLE], "edge_feat_dim": 7,
               "hidden_dim": 16, "num_sage_layers": 1, "num_gat_layers": 1, "heads": 2}
        path = os.path.join(self.tmp, "ensemble.pt")
        torch.save({"model_args": {"model_type": "ensemble", "ensemble": {"components": [
            {"name": "sage", "model_type": "graphsage", "weight": 0.5, "state_dict": sage.state_dict(),
             "model_args": sub},
            {"name": "gat", "model_type": "gat", "weight": 0.5, "state_dict": gat.state_dict(),
             "model_args": sub}]}}, "state_dict": {}}, path)
        model, _ = load_model_from_checkpoint(path, CPU)
        self.assertIsInstance(model, EnsembleModel)
        data = toy_data()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(data), 0.5 * sage(data) + 0.5 * gat(data), atol=1e-5))

    def test_unknown_model_type_is_an_error(self):
        _, path = self.trained("GraphSAGE")
        ckpt = torch.load(path, weights_only=False)
        torch.save(dict(ckpt, model_args=dict(ckpt["model_args"], model_type="gcn")), path)
        with self.assertRaises(ValueError):
            load_model_from_checkpoint(path, CPU)

    def test_bare_state_dict_is_an_error(self):
        model, _ = self.trained("GraphSAGE")
        path = os.path.join(self.tmp, "bare.pt")
        torch.save(model.state_dict(), path)
        with self.assertRaises(ValueError):
            load_model_from_checkpoint(path, CPU)


if __name__ == "__main__":
    unittest.main()
