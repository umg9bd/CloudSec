"""
prompt_hardener.py
==================
Produces the hardened prompt and the complete HardeningResponse for one request.

WHY THIS EXISTS
---------------
This is the only path from a request to a hardened prompt, so every guarantee lives here:

  * SCREENING GATE FIRST. Without an approved upstream screening status nothing is analysed,
    quoted or echoed: the result is SCREENING_PENDING / SCREENING_NOT_APPROVED /
    INTEGRATION_ERROR with revised_prompt = null. This module never decides a prompt is safe.
  * THE REQUEST IS PRESERVED VERBATIM. The revised prompt is the original text, unchanged,
    followed by one delimited block. Nothing in the user's objective or constraints is
    rewritten, reordered or removed.
  * NOTHING IS INVENTED. Clauses come from the catalogue; the only request-specific text is a
    subject phrase built from resource kinds the analyser actually found ("the S3 bucket").
    Missing context becomes an open item or a stated assumption, never a made-up value.
  * CONFLICTS ARE NOT OVERRIDDEN. A request that conflicts with a control returns
    SECURITY_CONFLICT with an alternative; only a recommended-control conflict that a human
    reviewer accepted (known_constraints.accepted_conflicts) proceeds, with compensating clauses.
  * OUT OF SCOPE IS REPORTED, NOT HANDLED. Lambda, KMS, Secrets Manager and any service outside
    the pipeline vocabulary produce OUT_OF_SCOPE (policy.behaviour.out_of_scope) - no pretending.
  * IT DOES NOT CLAIM TO ELIMINATE RISK. The appended block says so.

Precedence once screening is approved: OUT_OF_SCOPE > SECURITY_CONFLICT > NEEDS_CLARIFICATION >
HARDENED / ALREADY_ADEQUATE.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .catalogue import Catalogue, load_catalogue
from .config import Policy, ScopeConfig, load_policy, load_scope
from .control_selector import ConflictFinding, ControlMatch, Selection, select_controls
from .models import HardeningRequest, HardeningResponse, Status, TaskProfile, canonical_sha256
from .provenance import freshness_report, source_reference_dict
from .task_analyzer import TaskAnalyzer

BLOCK_BEGIN = "--- SECURITY HARDENING (Stage 2, catalogue {version}) ---"
BLOCK_END = "--- END SECURITY HARDENING ---"
LIMITS_NOTE = ("These requirements reduce, but do not eliminate, the risk of insecure configuration. "
               "Report any requirement you could not meet and any unresolved security risk explicitly.")

_KIND_PHRASE = {
    "bucket": "the S3 bucket", "iam_role": "the IAM role", "iam_policy": "the IAM policy", "iam_user": "the IAM user",
    "iam_group": "the IAM group", "access_key": "the access key", "trust_policy": "the trust policy",
    "security_group": "the security group", "ingress_rule": "the inbound rules", "vpc_network": "the VPC network resources",
    "ec2_instance": "the EC2 instance", "load_balancer": "the load balancer", "db_instance": "the database instance",
    "parameter": "the parameter", "trail": "the trail",
}
_KIND_SERVICE = {"bucket": "s3", "iam_role": "iam", "iam_policy": "iam", "iam_user": "iam", "iam_group": "iam",
                 "access_key": "iam", "trust_policy": "iam", "security_group": "ec2", "ingress_rule": "ec2",
                 "vpc_network": "ec2", "ec2_instance": "ec2", "load_balancer": "elasticloadbalancing",
                 "db_instance": "rds", "parameter": "ssm", "trail": "cloudtrail"}
_FALLBACK = {"identity_and_permissions": "the permissions in this request",
             "storage_and_cloud_resources": "the storage resources in this request",
             "network_exposure": "the network resources in this request",
             "application_and_api": "the code in this request", "database_and_sensitive_data": "the code in this request",
             "development_and_observability": "this change"}


def subject_phrase(control: Any, profile: TaskProfile) -> str:
    """Name what the clause applies to, using only resource kinds the analyser found in the prompt."""
    affected = set(control.affected_services_or_components)
    found = set(profile.facts["resource_kinds"])
    if "security_group" in found:
        found.discard("ingress_rule")          # a rule belongs to its group; naming both reads as noise
    kinds = [k for k in sorted(found) if k in _KIND_PHRASE and _KIND_SERVICE[k] in affected]
    if kinds:
        phrases = [_KIND_PHRASE[k] for k in kinds[:3]]
        return phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    app = profile.facts["app_context"]
    if affected & {"application", "api"}:
        if "api_endpoint" in app:
            return "the endpoint"
        if "web_app" in app:
            return "the web application"
    return _FALLBACK.get(control.control_category, "this request")


def _clause(control: Any, profile: TaskProfile) -> str:
    return re.sub(r"\s+", " ", control.required_prompt_clause.replace("{subject}", subject_phrase(control, profile))).strip()


def render_revised_prompt(original: str, clauses: Sequence[Tuple[str, str]], assumptions: Sequence[str],
                          open_items: Sequence[str], verifications: Sequence[Tuple[str, str]], version: str) -> str:
    """Original request verbatim, then one delimited block. `clauses`/`verifications` are (control_id, text)."""
    lines = ["", "", BLOCK_BEGIN.format(version=version),
             "The request above is unchanged. Apply the following in addition; they do not change its objective.", "",
             "Security requirements:"]
    lines += [f"{i}. [{cid}] {text}" for i, (cid, text) in enumerate(clauses, 1)]
    if assumptions:
        lines += ["", "Assumptions (not verified - confirm or correct):"] + [f"- {a}" for a in assumptions]
    if open_items:
        lines += ["", "Open items (not specified in the request - do not guess; ask or leave as a parameter):"] + [f"- {q}" for q in open_items]
    lines += ["", "Verification expectations:"] + [f"- [{cid}] {text}" for cid, text in verifications]
    lines += ["", LIMITS_NOTE, BLOCK_END]
    return original + "\n".join(lines)  # `lines` starts with two empty items => exactly "\n\n" before the block


def split_revised_prompt(revised: str) -> Tuple[str, Optional[str]]:
    """(original request, hardening block) - the inverse of render_revised_prompt, for tests and downstream tools."""
    marker = revised.find("\n\n--- SECURITY HARDENING (Stage 2, catalogue ")
    if marker == -1:
        return revised, None
    return revised[:marker], revised[marker + 2:]


class SecurityHardener:
    def __init__(self, catalogue: Optional[Catalogue] = None, policy: Optional[Policy] = None,
                 scope: Optional[ScopeConfig] = None, analyzer: Optional[TaskAnalyzer] = None):
        self.scope = scope or load_scope()
        self.policy = policy or load_policy()
        self.catalogue = catalogue or load_catalogue(scope=self.scope)
        self.analyzer = analyzer or TaskAnalyzer(self.catalogue, self.scope)
        self.config_sha256 = canonical_sha256({"lexicon": self.analyzer.lex.sha256, "scope": self.scope, "policy": self.policy})
        unknown = sorted(set(self.policy.enforcement_overrides) - {c.control_id for c in self.catalogue.controls})
        if unknown:
            raise ValueError(f"policy.enforcement_overrides names unknown control ids: {unknown}")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _base(self, request_id: str, status: Status, original: Optional[str], **kw: Any) -> HardeningResponse:
        cat = self.catalogue
        return HardeningResponse(request_id=request_id, status=status, original_prompt=original, revised_prompt=kw.pop("revised", None),
                                 source_catalogue_version=cat.version, catalogue_sha256=cat.sha256,
                                 catalogue_freshness={**freshness_report(cat), "analysis_config_sha256": self.config_sha256}, **kw)

    def error(self, request_id: str, code: str, message: str, path: Optional[str] = None) -> HardeningResponse:
        err = {"code": code, "message": message}
        if path:
            err["path"] = path
        return self._base(request_id, Status.INTEGRATION_ERROR, None, errors=[err], requires_user_review=True)

    def _gate(self, request: HardeningRequest) -> Optional[HardeningResponse]:
        disposition = self.policy.screening_disposition(request.screening_status)
        if disposition == "approved":
            return None
        shown = repr(request.screening_status)[:80]
        if disposition == "missing":
            return self.error(request.request_id, "SCREENING_STATUS_MISSING",
                              "screening_status is missing; upstream screening must approve the prompt before hardening.", "screening_status")
        status = Status.SCREENING_PENDING if disposition == "pending" else Status.SCREENING_NOT_APPROVED
        why = ("has not finished" if disposition == "pending" else "did not approve this prompt")
        return self._base(request.request_id, status, None, requires_user_review=True,
                          warnings=[f"Upstream screening {why} (screening_status={shown}); no hardening was performed and no prompt text was processed."])

    @staticmethod
    def _source_refs(control: Any) -> List[Dict[str, Any]]:
        return [source_reference_dict(control)]

    def _added(self, matches: Sequence[ControlMatch], profile: TaskProfile) -> List[Dict[str, Any]]:
        return [{"control_id": m.control.control_id, "control_name": m.control.control_name, "reason": m.reason,
                 "source_references": self._source_refs(m.control),
                 "verification_expectation": re.sub(r"\s+", " ", m.control.verification_expectation).strip(),
                 "clause": _clause(m.control, profile), "enforcement": m.enforcement,
                 "severity": m.control.severity_or_priority} for m in matches]

    def _conflicts(self, found: Sequence[ConflictFinding]) -> List[Dict[str, Any]]:
        return [{"control_id": f.control.control_id, "request_excerpt": f.request_excerpt,
                 "why": re.sub(r"\s+", " ", f.why), "proposed_alternative": re.sub(r"\s+", " ", f.alternative),
                 "enforcement": f.enforcement, "can_be_accepted_by_reviewer": f.enforcement != "mandatory",
                 "source_references": self._source_refs(f.control)} for f in found]

    # ── main entry ───────────────────────────────────────────────────────────
    def harden(self, request: HardeningRequest) -> HardeningResponse:
        gated = self._gate(request)
        if gated:
            return gated
        rid, prompt, policy, cat = request.request_id, request.original_prompt, self.policy, self.catalogue
        if request.source_catalogue_version and request.source_catalogue_version != cat.version:
            return self.error(rid, "CATALOGUE_VERSION_UNAVAILABLE",
                              f"catalogue version {request.source_catalogue_version!r} was requested but {cat.version!r} is loaded.",
                              "source_catalogue_version")
        if len(prompt) > policy.max_prompt_chars:
            return self.error(rid, "PROMPT_TOO_LONG", f"prompt exceeds policy.limits.max_prompt_chars ({policy.max_prompt_chars}).", "original_prompt")

        profile = self.analyzer.analyze(prompt, task_metadata=request.task_metadata, known_constraints=request.known_constraints)
        findings = profile.services
        scope_note = None
        warnings: List[str] = []

        # 1) scope --------------------------------------------------------------------------------------------
        blocked = sorted(set(findings.excluded) | set(findings.out_of_scope))
        if blocked:
            parts = []
            if findings.excluded:
                parts.append("excluded from this module by project decision: " + ", ".join(findings.excluded))
            if findings.out_of_scope:
                parts.append("outside the services the pipeline supports: " + ", ".join(findings.out_of_scope))
            scope_note = "; ".join(parts)
            if policy.out_of_scope == "block":
                approved = ", ".join(sorted(self.scope.approved))
                return self._base(rid, Status.OUT_OF_SCOPE, prompt, requires_user_review=True,
                                  scope_findings=findings.to_scope_findings(scope_note),
                                  clarification_questions=[{
                                      "question": f"This request depends on services that are {scope_note}. Should it be re-submitted without them "
                                                  f"(approved services: {approved}), or handed off to a reviewer who covers them?",
                                      "why_it_matters": "This module only writes security requirements for services the existing pipeline and dataset support; "
                                                        "it will not guess at others.", "blocking": True, "control_id": None}])
            warnings.append(f"Request touches services that are {scope_note}; no requirements were written for them.")

        # 2) unclear intent ---------------------------------------------------------------------------------------
        if profile.intent_unclear:
            qs = [("Which resource or application component is this about (for example a specific S3 bucket, IAM role, security group or endpoint)?",
                   "The request talks about permissions or access but names nothing to secure, so no specific requirement can be chosen."),
                  ("Which principal (user, role or service) needs the access, and which actions does it need?",
                   "Least-privilege scoping needs the who and the what; guessing would either over-grant or break the task."),
                  ("Is any of the data involved sensitive, or must access stay limited to named principals?",
                   "It decides whether exposure beyond named principals is acceptable.")]
            return self._base(rid, Status.NEEDS_CLARIFICATION, prompt, requires_user_review=True,
                              scope_findings=findings.to_scope_findings(scope_note),
                              clarification_questions=[{"question": q, "why_it_matters": w, "blocking": True, "control_id": None} for q, w in qs])

        # 3) selection -----------------------------------------------------------------------------------------------
        sel: Selection = select_controls(profile, cat, policy, request.known_constraints)
        warnings = sel.warnings + warnings
        gaps_summary = [{"control_id": m.control.control_id, "severity": m.control.severity_or_priority,
                         "description": f"{m.control.control_name} - not stated in the request"} for m in sel.gaps]
        deferred = [{"control_id": c.control_id, "reason": why} for c, why in sel.deferred]

        # 4) conflicts -------------------------------------------------------------------------------------------------
        blocking = [c for c in sel.conflicts if not c.accepted]
        if blocking:
            return self._base(rid, Status.SECURITY_CONFLICT, prompt, requires_user_review=True, security_gaps=gaps_summary,
                              conflicts=self._conflicts(blocking), deferred_controls=deferred, warnings=warnings,
                              scope_findings=findings.to_scope_findings(scope_note))

        # 5) blocking questions ---------------------------------------------------------------------------------------------
        if any(q["blocking"] for q in sel.questions):
            return self._base(rid, Status.NEEDS_CLARIFICATION, prompt, requires_user_review=True, security_gaps=gaps_summary,
                              clarification_questions=sel.questions, warnings=warnings, scope_findings=findings.to_scope_findings(scope_note))

        assumptions = list(sel.assumptions)
        for key in sorted(profile.inferred):
            fact, value = key.split(":", 1)
            if fact == "input_flows":
                assumptions.append(f"Treated some task input as untrusted input reaching {value.replace('untrusted_input_to_', '')} "
                                   "(inferred from the task wording; confirm the exact input and sink).")
        for finding in sel.conflicts:
            if finding.accepted:
                assumptions.append(f"A reviewer accepted the conflict '{finding.trigger_id}' on {finding.control.control_id}; "
                                   f"apply the compensating requirement below and flag the exception for review.")

        # 6) adequate ------------------------------------------------------------------------------------------------------
        satisfied = [{"control_id": c.control_id, "evidence": ev, "verification_expectation": re.sub(r"\s+", " ", c.verification_expectation).strip()}
                     for c, ev in sel.satisfied]
        if not sel.gaps:
            if sel.applicable_count == 0 and not sel.deferred:
                warnings.append("No catalogue control applied: nothing in this request touched an in-scope service or an application-security surface.")
            return self._base(rid, Status.ALREADY_ADEQUATE, prompt, revised=prompt, satisfied_controls=satisfied, deferred_controls=deferred,
                              assumptions=assumptions, warnings=warnings, scope_findings=findings.to_scope_findings(scope_note),
                              requires_user_review=bool(sel.deferred))

        # 7) hardened ---------------------------------------------------------------------------------------------------------
        added = self._added(sel.gaps, profile)
        open_items = [q["question"] for q in sel.questions]
        revised = render_revised_prompt(prompt, [(a["control_id"], a["clause"]) for a in added], assumptions, open_items,
                                        [(a["control_id"], a["verification_expectation"]) for a in added], cat.version)
        severity_of = {a["control_id"]: a["severity"] for a in added}
        high_open = any(severity_of.get(q["control_id"]) in ("high", "critical") for q in sel.questions)
        review = bool(high_open or any(c.accepted for c in sel.conflicts) or any("source is currently unavailable" in w for w in warnings))
        return self._base(rid, Status.HARDENED, prompt, revised=revised, security_gaps=gaps_summary, added_controls=added,
                          satisfied_controls=satisfied, deferred_controls=deferred, assumptions=assumptions,
                          clarification_questions=sel.questions, warnings=warnings, requires_user_review=review,
                          scope_findings=findings.to_scope_findings(scope_note))
