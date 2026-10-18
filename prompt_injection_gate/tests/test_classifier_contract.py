"""Classifier interface returns a validated structured result (spec 6C)."""
from prompt_injection_gate.classifier_interface import Classifier, ClassifierResult
from prompt_injection_gate.config import GateConfig
from prompt_injection_gate.schemas import Decision
from conftest import HAS_CLASSIFIER, requires_classifier


def test_missing_checkpoint_is_unavailable_not_safe():
    c = Classifier(GateConfig(classifier_dir="/definitely/not/here"))
    assert not c.available
    r = c.classify("anything")
    assert isinstance(r, ClassifierResult)
    assert r.status == "unavailable"
    assert r.verdict == Decision.REVIEW_REQUIRED          # never SAFE when unavailable
    assert r.attack_score is None


def test_result_is_structured_object_not_text():
    r = Classifier(GateConfig(classifier_dir="/none")).classify("x")
    for attr in ("status", "verdict", "attack_score", "category_scores", "model_version", "error"):
        assert hasattr(r, attr)


@requires_classifier
def test_trained_checkpoint_scores_and_validates():
    c = Classifier(GateConfig())
    assert c.available
    r = c.classify("Create an S3 bucket for the public website assets.")
    assert r.status == "ok"
    assert 0.0 <= r.attack_score <= 1.0
    assert set(r.category_scores) and all(0.0 <= v <= 1.0 for v in r.category_scores.values())
    assert r.verdict in (Decision.SAFE, Decision.MALICIOUS, Decision.REVIEW_REQUIRED)
    assert r.model_version and r.model_version != "unavailable"
