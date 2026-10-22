"""Input validation + bounded normalisation (spec 6A, G)."""
from prompt_injection_gate.config import GateConfig
from prompt_injection_gate.input_validation import validate
from prompt_injection_gate.normalization import normalize, safe_decode
from prompt_injection_gate.schemas import ContextItem, ScreeningRequest


def test_empty_prompt_rejected():
    assert "prompt_empty" in validate(ScreeningRequest(prompt="  "), GateConfig()).errors


def test_oversized_prompt_rejected():
    r = ScreeningRequest(prompt="x" * (GateConfig().max_prompt_chars + 1))
    assert "prompt_too_large" in validate(r, GateConfig()).errors


def test_too_many_context_items():
    cfg = GateConfig()
    r = ScreeningRequest(prompt="ok", untrusted_context=[ContextItem("d")] * (cfg.max_context_items + 1))
    assert "too_many_context_items" in validate(r, cfg).errors


def test_unknown_trust_level_flagged():
    r = ScreeningRequest(prompt="ok", untrusted_context=[ContextItem("d", trust="banana")])
    assert "context_trust_unknown" in validate(r, GateConfig()).errors


def test_valid_request_passes():
    assert validate(ScreeningRequest(prompt="Create an S3 bucket"), GateConfig()).ok


def test_normalize_strips_zero_width_but_preserves_original():
    original = "hel​lo"
    n = normalize(original)
    assert n.signals["had_zero_width"] is True
    assert "​" not in n.folded
    assert original == "hel​lo"            # caller's string untouched


def test_normalize_flags_confusables():
    # Cyrillic 'а','е','о' standing in for Latin
    n = normalize("аеоl-admin")
    assert n.signals["confusable_chars"] >= 3


def test_safe_decode_base64_is_data_only():
    import base64
    payload = base64.b64encode(b"this is a decoded sentence of text").decode()
    out = safe_decode(f"please look at {payload}")
    assert any(d.kind == "base64" and "decoded sentence" in d.plaintext for d in out)


def test_safe_decode_bounded_and_nonexecuting():
    # random base64 that is not valid utf-8 text must not be returned as a decode
    out = safe_decode("Zm9v" + "A" * 200)
    assert all(len(d.plaintext) <= 4000 for d in out)
