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


def _stub_classifier(cfg, logits, cat_thr):
    """A Classifier wired to a fake model returning fixed logits (no checkpoint needed)."""
    torch = __import__("pytest").importorskip("torch")

    class Tok:
        cls_token_id, sep_token_id = 1, 2
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [5, 6, 7]}

    class Model:
        def __call__(self, input_ids, attention_mask):
            return type("Out", (), {"logits": torch.tensor([logits])})()

    c = Classifier(cfg)
    c._loaded, c._tok, c._model = True, Tok(), Model()
    c._meta = {"categories": ["jailbreak", "defense_evasion_logging"], "max_len": 16,
               "malicious_threshold": 0.98, "safe_threshold": 0.5, "category_thresholds": cat_thr,
               "model_version": "stub"}
    return c


def test_unreliable_category_heads_do_not_name_threats():
    """A head that fires but scored below the dev-F1 bar must not label the prompt: v1's
    heads (dev F1 ~0.05) put 'would disable security logging' on a plain bucket request."""
    cat_thr = {"jailbreak": {"threshold": 0.1, "calibrated": True, "dev_f1": 0.76},
               "defense_evasion_logging": {"threshold": 0.1, "calibrated": True, "dev_f1": 0.05}}
    r = _stub_classifier(GateConfig(category_min_dev_f1=0.5), [2.0, 2.0, 2.0], cat_thr).classify("x")
    assert r.status == "ok" and r.verdict == Decision.REVIEW_REQUIRED
    assert r.flagged_categories == ["jailbreak"]
    assert set(r.category_scores) == {"jailbreak", "defense_evasion_logging"}   # scores still reported
    r = _stub_classifier(GateConfig(category_min_dev_f1=0.0), [2.0, 2.0, 2.0], cat_thr).classify("x")
    assert r.flagged_categories == ["jailbreak", "defense_evasion_logging"]
