"""The bridge that delivers SAFE prompts to Stage 2 and runs the downstream chain.

These run under base Python (no model): they either build a ScreeningResponse
directly to exercise delivery, or rely on the fail-closed path (no classifier ->
REVIEW_REQUIRED) to prove a non-SAFE prompt is never delivered.
"""
import json
import sys

import pytest

from prompt_injection_gate.config import GateConfig
from prompt_injection_gate.pipeline_bridge import PipelineBridge
from prompt_injection_gate.schemas import Decision, ScreeningResponse


def _bridge(tmp_path, **kw):
    cfg = GateConfig(classifier_dir="/none", stage2_incoming_dir=str(tmp_path), **kw)
    return PipelineBridge(cfg)


def _safe_resp(prompt="Create a private S3 bucket for audit logs.", rid="pig-test-0001"):
    return ScreeningResponse(
        decision=Decision.SAFE, handoff_allowed=True, safe_prompt_for_next_stage=prompt,
        threat_categories=[], explanation="clean", rules_version="x", classifier_version="y",
        request_id=rid)


# ---- the delivery rule: SAFE is written, nothing else is ---------------------

def test_safe_response_delivered_to_incoming(tmp_path):
    path = _bridge(tmp_path).forward(_safe_resp("Make an S3 bucket, private."))
    assert path is not None and path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "request_id": "pig-test-0001",
        "screening_status": "SAFE",
        "original_prompt": "Make an S3 bucket, private.",
    }
    assert path.name == "pig-test-0001.json"      # Stage 2 keys its output on this stem


@pytest.mark.parametrize("decision", [Decision.MALICIOUS, Decision.REVIEW_REQUIRED])
def test_non_safe_is_never_delivered(tmp_path, decision):
    resp = ScreeningResponse(
        decision=decision, handoff_allowed=False, safe_prompt_for_next_stage=None,
        threat_categories=["jailbreak"], explanation="held", rules_version="x", classifier_version="y",
        request_id="pig-test-0002")
    assert _bridge(tmp_path).forward(resp) is None
    assert list(tmp_path.glob("*.json")) == []


def test_delivery_can_be_disabled(tmp_path):
    assert _bridge(tmp_path, deliver_on_safe=False).forward(_safe_resp()) is None
    assert list(tmp_path.glob("*.json")) == []


def test_write_is_atomic_no_tmp_left(tmp_path):
    _bridge(tmp_path).forward(_safe_resp())
    assert list(tmp_path.glob("*.tmp")) == []


def test_request_id_is_sanitised_into_filename(tmp_path):
    resp = _safe_resp(rid="weird/../id with spaces")
    path = _bridge(tmp_path).forward(resp)
    assert path is not None and path.name.endswith(".json")
    assert "/" not in path.name and "\\" not in path.name and " " not in path.name


# ---- through the service (no checkpoint: fail closed, so nothing is delivered)

def test_screen_and_forward_holds_without_a_model(tmp_path):
    res = _bridge(tmp_path).screen_and_forward("Create a private S3 bucket for logs.")
    assert res["decision"] == "REVIEW_REQUIRED"      # fail closed
    assert res["forwarded"] is False
    assert list(tmp_path.glob("*.json")) == []


def test_end_to_end_stops_when_blocked(tmp_path):
    out = _bridge(tmp_path).run_end_to_end("Create a private S3 bucket for logs.")
    assert out["stage1"]["forwarded"] is False
    assert out["stage2"] is None and out["stage3"] is None


# ---- cross-stage contract: Stage 2's own validator accepts what we write -----

def test_delivered_request_matches_stage2_schema(tmp_path):
    cfg = GateConfig(classifier_dir="/none", stage2_incoming_dir=str(tmp_path))
    if cfg.stage2_root not in sys.path:
        sys.path.insert(0, cfg.stage2_root)
    try:
        from security_hardening.schema_validation import validate_request
    except Exception:
        pytest.skip("Stage 2 (security_hardening) not importable in this environment")
    path = PipelineBridge(cfg).forward(_safe_resp("Create a private S3 bucket; block public access."))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert validate_request(data) == []
