"""
test_ingestion_gate.py
======================
The benign/malicious ingestion gate (ingestion_gate.py): log validation and the
routing of benign vs malicious events into the stores.

No torch, Neo4j or checkpoint here -- the gate is built with object.__new__ and a
stub graph updater, so these cover validation, routing and serialisation, not the
detector itself (test_pipeline.py covers scoring). Fully executable in the fast tier.
"""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "graph_construction"), os.path.join(ROOT, "temporal-analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ingestion_gate as ig


class _StubUpdater:
    """Stands in for IncrementalGraphUpdater: records events, reports 'pending' (no Neo4j)."""
    def __init__(self):
        self.applied = []

    def apply_event(self, event):
        self.applied.append(event)
        return SimpleNamespace(relation="READ", sync_status=SimpleNamespace(value="pending"))


def _gate(tmp):
    g = object.__new__(ig.IngestionGate)            # bypass __init__ (which loads the model)
    g.store_dir = tmp
    g.templates_dir = os.path.join(tmp, "revised_templates")
    os.makedirs(g.templates_dir, exist_ok=True)
    g.graph_edges_file = os.path.join(tmp, "benign_graph_edges.jsonl")
    g.lstm_history_file = os.path.join(tmp, "lstm_history.jsonl")
    g.updater = _StubUpdater()
    g._stage2, g._stage2_tried = None, True         # _stage2_service() returns None, no import attempt
    return g


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row(**kw):
    base = dict(log_id="f.csv:0", source_node="arn:aws:iam::1:user/u",
                target_node="arn:aws:secretsmanager:us-east-1:1:secret:x", edge_type="DescribeSecret",
                username="u", principal_arn="arn:aws:iam::1:user/u", timestamp="2026-01-01T00:00:00Z",
                event_name="DescribeSecret", risk_score=0.4, p_graph=float("nan"), p_sequence=0.1,
                fast_lane=False)
    base.update(kw)
    return SimpleNamespace(**base)


class ValidateTests(unittest.TestCase):
    def test_accepts_wellformed(self):
        self.assertEqual(
            ig.IngestionGate.validate({"event_name": "X", "timestamp": "t", "principal_arn": "a"}), (True, ""))

    def test_rejects_non_dict(self):
        self.assertFalse(ig.IngestionGate.validate(["x"])[0])

    def test_rejects_missing_event(self):
        ok, why = ig.IngestionGate.validate({"timestamp": "t", "username": "u"})
        self.assertFalse(ok)
        self.assertIn("event_name", why)

    def test_rejects_missing_timestamp(self):
        ok, why = ig.IngestionGate.validate({"event_name": "X", "username": "u"})
        self.assertFalse(ok)
        self.assertIn("timestamp", why)

    def test_rejects_missing_principal(self):
        ok, why = ig.IngestionGate.validate({"event_name": "X", "timestamp": "t"})
        self.assertFalse(ok)
        self.assertIn("principal", why)


class BenignStoreTests(unittest.TestCase):
    def test_store_benign_writes_graph_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = _gate(tmp)
            status = g._store_benign(_row())
            self.assertEqual(status, "pending")
            self.assertEqual(len(g.updater.applied), 1)
            self.assertEqual(g.updater.applied[0].label, 0)            # stored as clean baseline
            with open(g.graph_edges_file, encoding="utf-8") as f:
                edge = json.loads(f.read().strip())
            self.assertEqual(edge["edge_type"], "DescribeSecret")
            self.assertEqual(edge["neo4j"], "pending")
            with open(g.lstm_history_file, encoding="utf-8") as f:
                hist = json.loads(f.read().strip())
            self.assertEqual(hist["label"], 0)
            self.assertEqual(hist["event_name"], "DescribeSecret")


class MaliciousTemplateTests(unittest.TestCase):
    def test_no_template_when_stage2_unavailable_but_record_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = _gate(tmp)
            self.assertIsNone(g._revised_template(_row(fast_lane=True)))
            payload = _read_json(g._store_malicious_template(_row(fast_lane=True), None))
            self.assertEqual(payload["classification"], "MALICIOUS")
            self.assertEqual(payload["revised_security_template"]["status"], "UNAVAILABLE")

    def test_template_carries_stage2_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = _gate(tmp)
            resp = {"status": "HARDENED", "revised_prompt": "...",
                    "security_gaps": [{"control_id": "IAM-X"}],
                    "added_controls": [{"control_id": "IAM-X", "clause": "c", "verification_expectation": "v"}]}
            payload = _read_json(g._store_malicious_template(_row(), resp))
            self.assertEqual(payload["revised_security_template"]["status"], "HARDENED")
            self.assertEqual(payload["revised_security_template"]["added_controls"][0]["control_id"], "IAM-X")
            self.assertIsNone(payload["p_graph"])                      # NaN -> null


if __name__ == "__main__":
    unittest.main()
