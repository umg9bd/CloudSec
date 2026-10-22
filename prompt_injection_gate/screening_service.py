"""
screening_service.py
====================
The Stage-1 entry point: orchestrates validation -> normalisation -> rules ->
classifier -> combiner -> explanation, and builds the SAFE-only handoff response
(spec 6, 8, 11).

Hard guarantees enforced here, regardless of anything the input claims:
  * the verdict is COMPUTED; it is never read from a field in the prompt;
  * `safe_prompt_for_next_stage` is populated only for a SAFE decision and is
    byte-for-byte the ORIGINAL prompt (never a normalised/decoded rewrite);
  * any internal failure yields REVIEW_REQUIRED, never SAFE (fail closed);
  * the audit record carries no raw prompt unless raw logging is explicitly on.

`screen_prompt(...)` is the simple functional API for Stage 2 to call. The
original pipeline is not imported or modified anywhere in this module.
"""
from __future__ import annotations

import hashlib
import time
import uuid

from .classifier_interface import Classifier
from .config import GateConfig, load_config
from .decision_combiner import combine
from .explanation import render
from .input_validation import validate
from .normalization import normalize, safe_decode
from .rule_engine import RuleEngine
from .schemas import ContextItem, Decision, ScreeningRequest, ScreeningResponse
from .session_store import SessionStore


class ScreeningService:
    def __init__(self, cfg: GateConfig | None = None, sessions: SessionStore | None = None):
        self.cfg = cfg or load_config()
        self._rules: RuleEngine | None = None
        self._rules_error: str | None = None
        try:
            self._rules = RuleEngine(self.cfg)
        except Exception as e:                          # invalid/missing catalogue => fail closed
            self._rules_error = f"rules_load_failed:{type(e).__name__}"
        self.classifier = Classifier(self.cfg)
        self.sessions = sessions or SessionStore(
            max_turns=self.cfg.session_max_turns, ttl_s=self.cfg.session_ttl_s,
            max_sessions=self.cfg.session_max_sessions, enabled=self.cfg.session_memory)

    @property
    def rules_version(self) -> str:
        return self._rules.version if self._rules else "unavailable"

    def _run_rules(self, text: str, channel: str):
        norm = normalize(text)
        decoded = safe_decode(text)
        if not self._rules:
            return norm, []
        return norm, self._rules.evaluate(norm, decoded, channel)

    def screen(self, req: ScreeningRequest) -> ScreeningResponse:
        t0 = time.perf_counter()
        rid = req.request_id or f"pig-{uuid.uuid4().hex[:12]}"

        # 1. validate (controlled non-SAFE on failure)
        v = validate(req, self.cfg)

        # if the rules catalogue failed to load, that is itself a fail-closed condition
        val_errors = list(v.errors)
        if self._rules_error:
            val_errors.append(self._rules_error)

        findings = []
        clf_result = None
        if not val_errors:
            # 2. rules over the prompt and each untrusted context item
            _, prompt_findings = self._run_rules(req.prompt, "user_prompt")
            findings.extend(prompt_findings)
            for c in req.untrusted_context:
                _, ctx_findings = self._run_rules(c.content, c.trust or "other")
                findings.extend(ctx_findings)

            # 3. classifier over the prompt, with context = stored session turns
            #    (if a session_id is given) plus any history the caller passed.
            stored = self.sessions.history(req.session_id)
            effective_history = stored + list(req.conversation_history)
            clf_text = req.prompt
            if effective_history:
                clf_text = "\n---\n".join(effective_history[-6:] + [req.prompt])
            clf_result = self.classifier.classify(clf_text)
        else:
            from .classifier_interface import ClassifierResult
            clf_result = ClassifierResult("unavailable", Decision.REVIEW_REQUIRED, None,
                                          error="skipped_validation_failed")

        # 4. combine (fail-closed policy)
        out = combine(val_errors, findings, clf_result, self.cfg)

        # 5. explanation (deterministic, no secrets)
        expl = render(out.decision, out.threat_categories, bool(req.untrusted_context))

        # 6. build the response, enforcing the SAFE-only invariants in schema
        is_safe = out.decision == Decision.SAFE
        scores = {}
        if clf_result and clf_result.attack_score is not None:
            scores = {"attack": clf_result.attack_score, **clf_result.category_scores}
        resp = ScreeningResponse(
            decision=out.decision,
            handoff_allowed=is_safe,
            safe_prompt_for_next_stage=req.prompt if is_safe else None,   # original, unaltered
            threat_categories=out.threat_categories,
            explanation=expl,
            rules_version=self.rules_version,
            classifier_version=self.classifier.version,
            request_id=rid,
            reason_codes=out.reason_codes,
            triggered_rule_ids=out.triggered_rule_ids,
            classifier_scores=scores,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            error_status=(clf_result.error if clf_result and clf_result.status != "ok" else None),
        )
        # remember this turn so the next turn in the session sees it (crescendo detection).
        # Only well-formed prompts are remembered; validation failures are not stored.
        if req.session_id and not val_errors:
            self.sessions.append(req.session_id, req.prompt)
        return resp

    def forget_session(self, session_id: str) -> None:
        """Clear a session's remembered turns (e.g. a 'reset conversation' action)."""
        self.sessions.forget(session_id)

    def audit_record(self, req: ScreeningRequest, resp: ScreeningResponse) -> dict:
        """Privacy-conscious audit metadata (spec 11). No raw prompt by default."""
        rec = resp.to_audit()
        rec["prompt_sha256"] = hashlib.sha256(req.prompt.encode("utf-8")).hexdigest()
        rec["prompt_chars"] = len(req.prompt)
        rec["n_context_items"] = len(req.untrusted_context)
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.cfg.log_raw_input:                      # opt-in only
            rec["raw_prompt"] = req.prompt
        return rec


_DEFAULT: ScreeningService | None = None


def screen_prompt(prompt: str,
                  request_id: str | None = None,
                  conversation_history: list[str] | None = None,
                  untrusted_context: list | None = None,
                  session_id: str | None = None,
                  service: ScreeningService | None = None) -> dict:
    """Functional API for Stage 2. Returns the handoff contract dict (spec 8)."""
    global _DEFAULT
    svc = service
    if svc is None:
        if _DEFAULT is None:
            _DEFAULT = ScreeningService()
        svc = _DEFAULT
    ctx = []
    for c in untrusted_context or []:
        if isinstance(c, ContextItem):
            ctx.append(c)
        elif isinstance(c, dict):
            ctx.append(ContextItem(c.get("content", ""), c.get("source", "other"), c.get("trust", "other")))
    req = ScreeningRequest(prompt=prompt, request_id=request_id,
                           conversation_history=list(conversation_history or []), untrusted_context=ctx,
                           session_id=session_id)
    return svc.screen(req).to_contract()
