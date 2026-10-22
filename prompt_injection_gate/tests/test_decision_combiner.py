"""Decision combiner: fail-closed policy, no SAFE on failure or disagreement (spec 6D)."""
from prompt_injection_gate.classifier_interface import ClassifierResult
from prompt_injection_gate.config import GateConfig
from prompt_injection_gate.decision_combiner import combine
from prompt_injection_gate.rule_engine import Finding
from prompt_injection_gate.schemas import Decision

CFG = GateConfig()


def clf(status="ok", verdict=Decision.SAFE, flagged=None):
    return ClassifierResult(status, verdict, 0.1 if verdict == Decision.SAFE else 0.9,
                            flagged_categories=flagged or [])


def rule(rid="R", sev="medium", disp="review", defensive=False, cat="jailbreak"):
    return Finding(rid, cat, sev, disp, "ev", "user_prompt", defensive)


def test_clean_is_safe():
    out = combine([], [], clf(verdict=Decision.SAFE), CFG)
    assert out.decision == Decision.SAFE and out.threat_categories == []


def test_validation_failure_is_review_never_safe():
    assert combine(["prompt_too_large"], [], clf(verdict=Decision.SAFE), CFG).decision == Decision.REVIEW_REQUIRED


def test_classifier_unavailable_fails_closed():
    out = combine([], [], clf(status="unavailable", verdict=Decision.REVIEW_REQUIRED), CFG)
    assert out.decision == Decision.REVIEW_REQUIRED


def test_classifier_error_fails_closed():
    out = combine([], [], clf(status="error", verdict=Decision.REVIEW_REQUIRED), CFG)
    assert out.decision == Decision.REVIEW_REQUIRED


def test_block_rule_forces_malicious_even_if_classifier_safe():
    # spec 6D(2): a high-severity/block rule must not be ignored because the model says SAFE
    out = combine([], [rule("PIG-FAKE-001", "high", "block")], clf(verdict=Decision.SAFE), CFG)
    assert out.decision == Decision.MALICIOUS


def test_block_rule_without_classifier_still_malicious():
    out = combine([], [rule("PIG-FAKE-001", "high", "block")],
                  clf(status="unavailable", verdict=Decision.REVIEW_REQUIRED), CFG)
    assert out.decision == Decision.MALICIOUS


def test_classifier_malicious_forces_malicious_even_if_no_rule():
    # spec 6D(1): a confident malicious classification is never flipped to SAFE by empty rules
    out = combine([], [], clf(verdict=Decision.MALICIOUS, flagged=["jailbreak"]), CFG)
    assert out.decision == Decision.MALICIOUS and "jailbreak" in out.threat_categories


def test_high_rule_vs_classifier_safe_is_disagreement_review():
    # spec 6D(3): serious disagreement never auto-allows
    out = combine([], [rule("PIG-OBF-002", "high", "review")], clf(verdict=Decision.SAFE), CFG)
    assert out.decision == Decision.REVIEW_REQUIRED


def test_medium_review_rule_blocks_safe():
    # spec 6D(6): lack of signature is not proof of safety; a positive medium signal => review
    out = combine([], [rule("PIG-OBF-001", "medium", "review")], clf(verdict=Decision.SAFE), CFG)
    assert out.decision == Decision.REVIEW_REQUIRED


def test_defensive_block_rule_not_treated_as_block():
    # a block-disposition rule that is only defensive framing must not hard-block
    out = combine([], [rule("PIG-FAKE-001", "high", "block", defensive=True)], clf(verdict=Decision.SAFE), CFG)
    assert out.decision != Decision.MALICIOUS
