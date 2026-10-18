"""
schemas.py
==========
Request / response contracts and the controlled vocabularies for the Stage-1 gate.

Pure standard-library dataclasses -- no pydantic, no third-party runtime dependency,
so the contract itself loads even when the model stack is absent. Validation that
can reject input lives in input_validation.py; here we define shapes, enums and the
SAFE-only invariants that ScreeningResponse enforces in __post_init__.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    SAFE = "SAFE"
    MALICIOUS = "MALICIOUS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


# Threat categories the gate reports. Kept in sync with the model's category
# outputs (training/train_eval_transformer.py CATEGORIES) and the rules catalogue.
THREAT_CATEGORIES = (
    "prompt_injection_direct",
    "prompt_injection_indirect",
    "jailbreak",
    "system_prompt_extraction",
    "credential_harvesting",
    "exfiltration",
    "privilege_escalation",
    "defense_evasion_logging",
    "data_destruction",
    "obfuscation",
)

# Trust level of a supplied context item. Instruction-like text inside anything
# that is not `user_prompt` is the indirect-injection signal.
TRUST_LEVELS = ("user_prompt", "retrieved_content", "tool_output", "uploaded_document", "other")


@dataclass
class ContextItem:
    """One piece of untrusted context accompanying the prompt (optional)."""
    content: str
    source: str = "other"          # free-text provenance, e.g. "ticket-482", "https://..."
    trust: str = "other"           # one of TRUST_LEVELS

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "source": self.source, "trust": self.trust}


@dataclass
class ScreeningRequest:
    prompt: str
    request_id: str | None = None
    conversation_history: list[str] = field(default_factory=list)
    untrusted_context: list[ContextItem] = field(default_factory=list)
    session_id: str | None = None          # if set, the gate remembers this session's turns

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ScreeningRequest":
        ctx = []
        for c in d.get("untrusted_context") or []:
            if isinstance(c, ContextItem):
                ctx.append(c)
            elif isinstance(c, dict):
                ctx.append(ContextItem(content=c.get("content", ""),
                                       source=c.get("source", "other"),
                                       trust=c.get("trust", "other")))
            else:
                ctx.append(ContextItem(content=str(c)))
        return ScreeningRequest(
            prompt=d.get("prompt", ""),
            request_id=d.get("request_id"),
            conversation_history=list(d.get("conversation_history") or []),
            untrusted_context=ctx,
            session_id=d.get("session_id"),
        )


@dataclass
class ScreeningResponse:
    decision: Decision
    handoff_allowed: bool
    safe_prompt_for_next_stage: str | None
    threat_categories: list[str]
    explanation: str
    rules_version: str
    classifier_version: str
    request_id: str | None = None
    # Non-contractual audit detail (never includes the raw prompt):
    reason_codes: list[str] = field(default_factory=list)
    triggered_rule_ids: list[str] = field(default_factory=list)
    classifier_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float | None = None
    error_status: str | None = None

    def __post_init__(self):
        """Enforce the mandatory SAFE-only invariants (spec section 8) in code."""
        if not isinstance(self.decision, Decision):
            self.decision = Decision(self.decision)
        is_safe = self.decision == Decision.SAFE
        # handoff_allowed <=> decision == SAFE
        if self.handoff_allowed != is_safe:
            raise ValueError("handoff_allowed must be true iff decision == SAFE")
        # safe_prompt present only for SAFE; must equal the original prompt (never a rewrite)
        if is_safe:
            if self.safe_prompt_for_next_stage is None:
                raise ValueError("SAFE result must carry safe_prompt_for_next_stage")
        else:
            if self.safe_prompt_for_next_stage is not None:
                raise ValueError("non-SAFE result must have safe_prompt_for_next_stage = None")

    def to_contract(self) -> dict[str, Any]:
        """The downstream-facing JSON (spec section 8) -- audit fields omitted."""
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
            "handoff_allowed": self.handoff_allowed,
            "safe_prompt_for_next_stage": self.safe_prompt_for_next_stage,
            "threat_categories": list(self.threat_categories),
            "explanation": self.explanation,
            "rules_version": self.rules_version,
            "classifier_version": self.classifier_version,
        }

    def to_audit(self) -> dict[str, Any]:
        """Full record for the audit log (no raw prompt, no secrets)."""
        return {
            **self.to_contract(),
            "reason_codes": list(self.reason_codes),
            "triggered_rule_ids": list(self.triggered_rule_ids),
            "classifier_scores": {k: round(v, 4) for k, v in self.classifier_scores.items()},
            "latency_ms": self.latency_ms,
            "error_status": self.error_status,
        }
