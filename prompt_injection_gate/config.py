"""
config.py
=========
All tunables for the Stage-1 gate in one place, overridable by environment
variable so nothing operational is hard-coded (spec sections 6C, 6D, 11).

No secrets live here. The classifier is a local fine-tuned checkpoint selected
by path; there is no API key. Decision thresholds are NOT set here as magic
numbers -- they are loaded from the trained checkpoint's gate_model.json, which
derived them from the held-out dev slice. This file only sets that checkpoint's
location and the structural limits.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


def _int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _path(name, default):
    return os.environ.get(name, default)


def _default_classifier_dir():
    """The checkpoint checkpoint.json names (the one fetch_checkpoint.py installs)."""
    try:
        with open(os.path.join(HERE, "checkpoint.json"), encoding="utf-8") as f:
            return os.path.join(HERE, "artifacts", json.load(f)["dir"])
    except (OSError, ValueError, KeyError):
        return os.path.join(HERE, "artifacts", "distilbert-mbert-pi-v1")


@dataclass
class GateConfig:
    # --- input limits (input_validation.py) ---
    max_prompt_chars: int = _int("PIGATE_MAX_PROMPT_CHARS", 20_000)
    max_context_items: int = _int("PIGATE_MAX_CONTEXT_ITEMS", 32)
    max_context_item_chars: int = _int("PIGATE_MAX_CONTEXT_ITEM_CHARS", 50_000)
    max_history_items: int = _int("PIGATE_MAX_HISTORY_ITEMS", 50)
    max_total_chars: int = _int("PIGATE_MAX_TOTAL_CHARS", 200_000)

    # --- rules catalogue (rule_engine.py) ---
    rules_catalogue_path: str = _path("PIGATE_RULES_CATALOGUE",
                                      os.path.join(HERE, "rules_catalogue.yaml"))

    # --- classifier (classifier_interface.py) ---
    # Directory holding the fine-tuned checkpoint + gate_model.json. Empty / missing
    # => the classifier is UNAVAILABLE and the combiner fails closed (never SAFE).
    classifier_dir: str = _path("PIGATE_CLASSIFIER_DIR", _default_classifier_dir())
    classifier_timeout_s: float = float(os.environ.get("PIGATE_CLASSIFIER_TIMEOUT_S", "20"))
    # A category head names a threat (in threat_categories and the explanation) only if
    # its dev F1 in gate_model.json reached this. v1's heads mostly scored 0.03-0.10 with
    # thresholds near 0.1, so they "flagged" nine categories on a plain bucket request.
    # Heads below the bar still report scores; they just don't label the prompt.
    category_min_dev_f1: float = float(os.environ.get("PIGATE_CATEGORY_MIN_DEV_F1", "0.5"))

    # --- decision combiner (decision_combiner.py) ---
    # Rule severities at/above this block outright; "medium" routes to review.
    rule_block_severity: str = _path("PIGATE_RULE_BLOCK_SEVERITY", "high")

    # --- session memory (session_store.py) ---
    session_memory: bool = os.environ.get("PIGATE_SESSION_MEMORY", "1").lower() not in ("0", "false", "no")
    session_max_turns: int = _int("PIGATE_SESSION_MAX_TURNS", 10)
    session_ttl_s: int = _int("PIGATE_SESSION_TTL_S", 1800)
    session_max_sessions: int = _int("PIGATE_SESSION_MAX_SESSIONS", 1000)

    # --- logging (screening_service.py) ---
    # OFF by default: raw prompts are never logged unless explicitly enabled.
    log_raw_input: bool = os.environ.get("PIGATE_LOG_RAW_INPUT", "").lower() in ("1", "true", "yes")
    audit_log_path: str = _path("PIGATE_AUDIT_LOG", "")   # "" => no file sink, caller handles records

    # --- downstream handoff (pipeline_bridge.py) ---
    # Stage 2 (prompt hardening) watches `incoming/`; a SAFE prompt is delivered
    # there as a request file, a non-SAFE prompt is never written. Stage 3
    # (log-generation) turns Stage 2's hardened prompt into predicted CloudTrail
    # events. Defaults point at the sibling modules in this repo.
    deliver_on_safe: bool = os.environ.get("PIGATE_DELIVER_ON_SAFE", "1").lower() not in ("0", "false", "no")
    stage2_root: str = _path("PIGATE_STAGE2_ROOT",
                             os.path.join(os.path.dirname(HERE), "prompt_hardening_realtime"))
    stage2_incoming_dir: str = _path("PIGATE_STAGE2_INCOMING",
                                     os.path.join(os.path.dirname(HERE), "prompt_hardening_realtime", "incoming"))
    stage2_output_dir: str = _path("PIGATE_STAGE2_OUTPUT",
                                   os.path.join(os.path.dirname(HERE), "prompt_hardening_realtime", "output"))
    stage3_dir: str = _path("PIGATE_STAGE3_DIR", os.path.join(os.path.dirname(HERE), "log-generation"))
    # The status written on a delivered request. Only ever written on a SAFE
    # decision, so it is always the "approved" status the next stage expects.
    handoff_screening_status: str = _path("PIGATE_HANDOFF_STATUS", "SAFE")

    # severity rank used across the module
    severity_rank: dict = field(default_factory=lambda: {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4})


def load_config() -> GateConfig:
    return GateConfig()
