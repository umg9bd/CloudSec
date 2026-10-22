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
    classifier_dir: str = _path("PIGATE_CLASSIFIER_DIR",
                                os.path.join(HERE, "artifacts", "distilbert-mbert-pi-v1"))
    classifier_timeout_s: float = float(os.environ.get("PIGATE_CLASSIFIER_TIMEOUT_S", "20"))

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

    # severity rank used across the module
    severity_rank: dict = field(default_factory=lambda: {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4})


def load_config() -> GateConfig:
    return GateConfig()
