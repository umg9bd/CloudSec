"""Shared fixtures for the Stage-1 gate tests."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.dirname(HERE)
ROOT = os.path.dirname(GATE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prompt_injection_gate.config import GateConfig          # noqa: E402
from prompt_injection_gate.screening_service import ScreeningService  # noqa: E402


def _trained_dir():
    for name in ("distilbert-mbert-pi-v1", "deberta-v3-small-pi-v1"):
        d = os.path.join(GATE, "artifacts", name)
        if os.path.isfile(os.path.join(d, "gate_model.json")) and os.path.isfile(os.path.join(d, "model.safetensors")):
            return d
    return None


CLASSIFIER_DIR = _trained_dir()


def _classifier_loads():
    """True only if the checkpoint exists AND actually loads in this interpreter
    (transformers/torch present). Under the base interpreter without the module
    venv this is False, so model tests skip cleanly instead of failing."""
    if CLASSIFIER_DIR is None:
        return False
    from prompt_injection_gate.classifier_interface import Classifier
    return Classifier(GateConfig(classifier_dir=CLASSIFIER_DIR)).available


HAS_CLASSIFIER = _classifier_loads()
requires_classifier = pytest.mark.skipif(
    not HAS_CLASSIFIER,
    reason="trained classifier not loadable here; run tests under prompt_injection_gate/.venv")


@pytest.fixture
def cfg():
    # default config, but force the classifier dir to the real checkpoint when present,
    # else to a path that does not exist (exercises the fail-closed branch).
    return GateConfig(classifier_dir=CLASSIFIER_DIR or os.path.join(GATE, "artifacts", "__none__"))


@pytest.fixture
def service(cfg):
    return ScreeningService(cfg)
