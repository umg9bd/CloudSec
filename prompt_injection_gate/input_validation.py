"""
input_validation.py
===================
Component A: validate the request and reject malformed / oversized input before
any detection runs (spec section 6A). Validation failures are a controlled
non-SAFE outcome, never an exception that could be swallowed into SAFE.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import GateConfig
from .schemas import ScreeningRequest, TRUST_LEVELS


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]           # reason codes, e.g. "prompt_too_large"


def validate(req: ScreeningRequest, cfg: GateConfig) -> ValidationResult:
    errs: list[str] = []

    if not isinstance(req.prompt, str):
        errs.append("prompt_not_a_string")
        return ValidationResult(False, errs)          # nothing else is checkable
    if req.prompt.strip() == "":
        errs.append("prompt_empty")
    if len(req.prompt) > cfg.max_prompt_chars:
        errs.append("prompt_too_large")

    if not isinstance(req.conversation_history, list):
        errs.append("history_not_a_list")
    else:
        if len(req.conversation_history) > cfg.max_history_items:
            errs.append("too_many_history_items")
        if any(not isinstance(h, str) for h in req.conversation_history):
            errs.append("history_item_not_a_string")

    if not isinstance(req.untrusted_context, list):
        errs.append("context_not_a_list")
    else:
        if len(req.untrusted_context) > cfg.max_context_items:
            errs.append("too_many_context_items")
        for c in req.untrusted_context:
            if not isinstance(c.content, str):
                errs.append("context_item_not_a_string")
            elif len(c.content) > cfg.max_context_item_chars:
                errs.append("context_item_too_large")
            if c.trust not in TRUST_LEVELS:
                errs.append("context_trust_unknown")

    total = len(req.prompt) + sum(len(h) for h in req.conversation_history if isinstance(h, str)) \
        + sum(len(c.content) for c in req.untrusted_context if isinstance(c.content, str))
    if total > cfg.max_total_chars:
        errs.append("total_input_too_large")

    return ValidationResult(not errs, errs)
