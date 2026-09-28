"""
test_loader_metrics_feeder.py
=============================
Three small, previously untested pieces:

  * offline_pipeline.load_offline -- GNN-final's loading entry point, now a
    shim over offline_graph.OfflineGraphLoader. It must give exactly the graph
    that loader gives, so there is one graph construction, not two drifting ones.
  * utils.evaluate's AUPR (ported from GNN-final) must equal sklearn's
    average precision on the same flattened labels and probabilities.
  * feed_incoming.feed -- the demo's data feeder: every event delivered once,
    in order, in files of batch_size, with no half-written file ever visible.
"""

import csv
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import HeteroData

import feed_incoming
from offline_graph import OfflineGraphLoader
from offline_pipeline import load_offline
from utils import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRUCTURAL = os.path.join(ROOT, "datasets", "privilege-escalation", "cloudtrail_structural.csv")


class TestOfflinePipelineShim(unittest.TestCase):
    def test_load_offline_equals_offline_graph_loader(self):
        if not os.path.exists(STRUCTURAL):
            self.skipTest("structural CSV not present")
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "slice.csv")
            pd.read_csv(STRUCTURAL, nrows=600).to_csv(path, index=False)
            a, meta_a = load_offline(path)
            b, meta_b = OfflineGraphLoader(pd.read_csv(path), source_name="slice.csv").load()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(sorted(a.edge_types), sorted(b.edge_types))
        self.assertEqual(meta_a["populated_triples"], meta_b["populated_triples"])
        for nt in a.node_types:
            self.assertTrue(torch.equal(a[nt].x, b[nt].x), nt)
        for t in a.edge_types:
            self.assertTrue(torch.equal(a[t].edge_index, b[t].edge_index), t)
            self.assertTrue(torch.equal(a[t].edge_attr, b[t].edge_attr), t)
            self.assertEqual(list(a[t].log_id), list(b[t].log_id), t)


class ConstantModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, data):
        return self.logits


class TestAupr(unittest.TestCase):
    def test_aupr_matches_sklearn(self):
        triple = ("User", "READ", "Resource")
        data = HeteroData()
        data["User"].x = torch.zeros(1, 1)
        data["Resource"].x = torch.zeros(1, 1)
        y = torch.tensor([0, 1, 0, 1, 1, 0, 0, 0])
        data[triple].edge_index = torch.zeros((2, 8), dtype=torch.long)
        data[triple].y = y
        logits = torch.tensor([-2.0, 1.5, 0.3, -0.1, 2.0, -1.0, 0.8, -3.0])
        m = evaluate(ConstantModel(logits), data, {triple: torch.ones(8, dtype=torch.bool)})
        self.assertAlmostEqual(m["aupr"], average_precision_score(y.numpy(), torch.sigmoid(logits).numpy()))

    def test_aupr_is_zero_when_only_one_class_present(self):
        triple = ("User", "READ", "Resource")
        data = HeteroData()
        data["User"].x = torch.zeros(1, 1)
        data["Resource"].x = torch.zeros(1, 1)
        data[triple].edge_index = torch.zeros((2, 3), dtype=torch.long)
        data[triple].y = torch.zeros(3, dtype=torch.long)
        m = evaluate(ConstantModel(torch.zeros(3)), data, {triple: torch.ones(3, dtype=torch.bool)})
        self.assertEqual(m["aupr"], 0.0)


class TestFeeder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dataset = os.path.join(self.tmp, "events.csv")
        with open(self.dataset, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "event_name"])
            for i in range(23):
                w.writerow([f"2024-01-01T10:{i:02d}:00Z", f"E{i}"])
        self.incoming = os.path.join(self.tmp, "incoming")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def delivered(self):
        rows = []
        for name in sorted(os.listdir(self.incoming)):
            with open(os.path.join(self.incoming, name), newline="", encoding="utf-8") as f:
                rows.append([r["event_name"] for r in csv.DictReader(f)])
        return rows

    def test_every_event_once_in_order_in_batches(self):
        feed_incoming.feed(self.dataset, self.incoming, batch_size=10, interval=0)
        batches = self.delivered()
        self.assertEqual([len(b) for b in batches], [10, 10, 3])
        self.assertEqual([e for b in batches for e in b], [f"E{i}" for i in range(23)])
        self.assertEqual(sorted(os.listdir(self.incoming))[0], "events_batch0001.csv")

    def test_start_and_limit(self):
        feed_incoming.feed(self.dataset, self.incoming, batch_size=4, interval=0, start=5, limit=6)
        self.assertEqual([e for b in self.delivered() for e in b], [f"E{i}" for i in range(5, 11)])

    def test_nothing_is_left_half_written(self):
        feed_incoming.feed(self.dataset, self.incoming, batch_size=10, interval=0)
        staging = os.path.join(self.tmp, ".incoming_staging")
        self.assertEqual(os.listdir(staging), [])


if __name__ == "__main__":
    unittest.main()
