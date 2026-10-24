"""Regression tests for dry-run explainability and blast-radius enrichment."""
from pathlib import Path
import sys
from types import SimpleNamespace
import threading

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "graph_construction") not in sys.path:
    sys.path.insert(0, str(ROOT / "graph_construction"))

import agent_orchestrator as ao  # noqa: E402
from agent_orchestrator import DryRunScorer, plan_blockers  # noqa: E402


PRINCIPAL = "arn:aws:sts::123456789012:assumed-role/DevRole/session-1"


def _raw_ec2_event():
    return {
        "eventTime": "2026-10-24T08:00:00Z",
        "eventSource": "ec2.amazonaws.com",
        "eventName": "RunInstances",
        "awsRegion": "us-east-1",
        "userIdentity": {
            "type": "AssumedRole",
            "arn": PRINCIPAL,
            "userName": "DevRole",
        },
        "requestParameters": {"instanceType": "t3.micro"},
    }


class _FakePipeline:
    def __init__(self, fast_lane=False):
        self.fast_lane = fast_lane
        self._scored_inputs = {}

    def reload_state(self):
        pass

    def featurize(self, rows, name):
        return pd.DataFrame([{"fixture": True}])

    def score(self, new):
        target_node = ("cloudtrail.amazonaws.com" if self.fast_lane else
                       "arn:aws:ec2:us-east-1:123456789012:instance/i-1234567890abcdef0")
        event_name = "StopLogging" if self.fast_lane else "RunInstances"
        edge_type = event_name
        graph_rows = pd.DataFrame([
            {
                "log_id": "history:0", "label": 0,
                "source_node": "arn:aws:iam::123456789012:user/alice",
                "target_node": "arn:aws:iam::123456789012:role/DevRole",
                "edge_type": "AssumeRole",
            },
            {
                "log_id": "dryrun_ec2:0", "label": 0,
                "source_node": PRINCIPAL,
                "target_node": target_node,
                "edge_type": edge_type,
            },
        ])
        self._scored_inputs = {"graph_rows": graph_rows, "seqs": []}
        return pd.DataFrame([{
            "log_id": "dryrun_ec2:0", "event_name": event_name, "target_node": graph_rows.iloc[1]["target_node"],
            "source_node": PRINCIPAL, "principal_arn": PRINCIPAL, "username": "DevRole",
            "timestamp": pd.Timestamp("2026-10-24T08:00:00Z"), "edge_type": edge_type,
            "p_graph": 0.75 if not self.fast_lane else 0.05,
            "p_sequence": 0.80 if not self.fast_lane else 0.10,
            "risk": 0.775 if not self.fast_lane else 0.075,
            "risk_score": 7.75 if not self.fast_lane else 0.75,
            "alert": not self.fast_lane, "fast_lane": self.fast_lane,
        }])

    def explain(self, flagged):
        return {
            "dryrun_ec2:0": {
                "summary": "risk 7.75/10: HGT and LSTM evidence",
                "models": {"graph": {"probability": 0.75}, "sequence": {"probability": 0.80}},
                "graph": {"top_features": [{"feature": "principal_handoff", "label": "identity handoff", "share": 0.7}]},
                "sequence": {"top_events": [{"event_name": "AssumeRole", "minutes_before": 1.0, "effect": 0.3}]},
            }
        }


def test_flagged_ec2_assessment_keeps_full_explanation_and_uses_score_graph_window():
    scorer = DryRunScorer.__new__(DryRunScorer)
    scorer.p = _FakePipeline()
    scorer.cfg = SimpleNamespace(alert_threshold=0.5, weight_graph=0.5)
    scorer._lock = threading.Lock()

    assessment = scorer.assess([_raw_ec2_event()], "dryrun_ec2")
    event = assessment["events"][0]

    # Model values and verdict are unchanged; enrichment is additional metadata only.
    assert event["p_graph"] == 0.75
    assert event["p_sequence"] == 0.80
    assert event["risk"] == 0.775
    assert event["risk_score"] == 7.75
    assert event["alert"] is True
    assert event["fast_lane"] is False

    assert event["explanation"]["graph"]["top_features"][0]["feature"] == "principal_handoff"
    assert event["why"] == "risk 7.75/10: HGT and LSTM evidence"
    assert event["blast_radius"]["status"] == "ok"
    assert event["blast_radius"]["origin"]["label"] == "Role"
    assert event["blast_radius"]["reachable_count"] >= 1
    assert any("i-1234567890abcdef0" in node["key"] for node in event["blast_radius"]["snapshot"]["nodes"])


def test_unresolvable_dry_run_origin_is_unavailable_not_zero():
    scorer = DryRunScorer.__new__(DryRunScorer)
    p = _FakePipeline()
    original_score = p.score

    def unresolved_score(new):
        scored = original_score(new)
        scored.loc[:, "source_node"] = "arn:aws:iam::123456789012:user/not-in-graph-window"
        scored.loc[:, "principal_arn"] = "arn:aws:iam::123456789012:user/not-in-graph-window"
        scored.loc[:, "username"] = "not-in-graph-window"
        return scored

    p.score = unresolved_score
    scorer.p = p
    scorer.cfg = SimpleNamespace(alert_threshold=0.5, weight_graph=0.5)
    scorer._lock = threading.Lock()

    event = scorer.assess([_raw_ec2_event()], "dryrun_ec2")["events"][0]
    assert event["blast_radius"]["status"] == "origin_unresolved"
    assert event["blast_radius"]["reachable_count"] is None
    assert "could not be resolved" in event["blast_radius"]["message"]
    assert event["alert"] is True


def test_fast_lane_event_gets_blast_radius_and_added_metadata_does_not_change_gate_blockers():
    scorer = DryRunScorer.__new__(DryRunScorer)
    scorer.p = _FakePipeline(fast_lane=True)
    scorer.cfg = SimpleNamespace(alert_threshold=0.5, weight_graph=0.5)
    scorer._lock = threading.Lock()
    assessment = scorer.assess([_raw_ec2_event()], "dryrun_ec2")
    event = assessment["events"][0]

    assert event["fast_lane"] is True
    assert event["alert"] is False
    assert event["risk_score"] == 0.75  # below threshold, still flagged by the existing fast-lane rule
    assert event["fast_lane_reason"] == "CloudTrail logging disabled"
    assert event["blast_radius"]["status"] == "ok"

    stage2 = {"status": "ALREADY_ADEQUATE", "clarification_questions": []}
    stage3 = {"stage3_contract": {"status": "GENERATED"}, "downstream_lstm_records": {"Records": [{}]}}
    with_metadata = plan_blockers(stage2, stage3, assessment)
    legacy_event = {k: v for k, v in event.items() if k not in {"explanation", "blast_radius"}}
    without_metadata = plan_blockers(stage2, stage3, {**assessment, "events": [legacy_event]})
    assert with_metadata == without_metadata
    assert any("fast-lane rule" in blocker for blocker in with_metadata)


def test_simulated_execute_still_writes_only_the_approved_plan_to_staging(tmp_path, monkeypatch):
    principal = "arn:aws:iam::111122223333:user/agentic-operator"
    planned = ao.materialize_events([_raw_ec2_event()], principal)
    rnd = ao.Round(
        request="test only", request_id="test-request", approved_records=planned,
        fingerprint=ao.plan_fingerprint(planned),
    )
    gate = ao.AgentGate.__new__(ao.AgentGate)
    gate.principal_arn = principal

    incoming = tmp_path / "incoming"
    staging = tmp_path / ".incoming_staging"
    monkeypatch.setattr(ao, "PIPELINE_INCOMING", incoming)
    monkeypatch.setattr(ao, "PIPELINE_STAGING", staging)
    monkeypatch.setattr(ao, "AUDIT_LOG", tmp_path / "output" / "agent_audit.jsonl")

    result = gate.execute(rnd)
    written = incoming / result["file"]
    assert rnd.approved is True
    assert result["events"] == 1
    assert written.exists()
    payload = __import__("json").loads(written.read_text(encoding="utf-8"))
    assert len(payload["Records"]) == 1
    assert payload["Records"][0]["eventName"] == "RunInstances"
    assert payload["Records"][0]["userIdentity"]["arn"] == principal
    assert not (tmp_path / "output" / "risk_scores.csv").exists()
