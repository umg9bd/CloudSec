"""
explanation.py
==============
Component: the user-facing explanation for a blocked or withheld prompt (spec 7).

Deterministic and category-based. No second LLM call, no raw prompt, no secret
values, no thresholds or matching internals that would help an attacker tune an
evasion. Produces one or two plain sentences from the decision + categories.
"""
from __future__ import annotations

from .schemas import Decision

# Broad, non-revealing phrasing per category.
_CATEGORY_PHRASE = {
    "prompt_injection_direct": "attempts to override the agent's instructions",
    "prompt_injection_indirect": "contains instructions hidden in supplied content",
    "jailbreak": "tries to bypass the agent's safety constraints",
    "system_prompt_extraction": "tries to reveal the agent's hidden instructions",
    "credential_harvesting": "involves collecting credentials or secret values",
    "exfiltration": "would send sensitive data to an external destination",
    "privilege_escalation": "would grant the agent more access than the task needs",
    "defense_evasion_logging": "would disable security logging or controls",
    "data_destruction": "would destroy data without a clear task reason",
    "obfuscation": "uses hidden or encoded characters to disguise its intent",
}


def _join(phrases: list[str]) -> str:
    phrases = phrases[:2]                                  # keep it short
    if not phrases:
        return "could not be safely interpreted"
    if len(phrases) == 1:
        return phrases[0]
    return phrases[0] + " and " + phrases[1]


def render(decision: Decision, categories: list[str], has_untrusted_context: bool) -> str:
    phrases = [_CATEGORY_PHRASE[c] for c in categories if c in _CATEGORY_PHRASE]

    if decision == Decision.MALICIOUS:
        body = _join(phrases)
        return (f"This request {body}. "
                "It was blocked before reaching the cloud-security planning stage.")

    if decision == Decision.REVIEW_REQUIRED:
        if phrases:
            body = _join(phrases)
            return (f"This request {body}, but the purpose and authorisation are unclear. "
                    "Please clarify the intended task. The prompt has not been forwarded.")
        if has_untrusted_context:
            return ("The supplied content could not be reliably screened. "
                    "Please confirm the task and the source of any attached material. "
                    "The prompt has not been forwarded.")
        return ("The request could not be reliably classified. "
                "Please clarify the intended operation. The prompt has not been forwarded.")

    # SAFE -- a short, neutral note (spec's benign example)
    return "No malicious prompt indicators were identified by the required screening checks."
