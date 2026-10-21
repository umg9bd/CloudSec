"""
control_selector.py
===================
Decides, for one analysed task, which catalogue controls apply, which the prompt
already satisfies, which requests conflict with a control, and what must be asked.

WHY THIS EXISTS
---------------
Selection is data-driven: each control's applicability, "already satisfied" and conflict
rules are conditions in the catalogue evaluated over the TaskProfile (conditions.py), so
retuning a control is a catalogue edit, not a code change. This module adds the
judgement around them:

  * a control is a GAP only if it applies and the prompt does not already state it
    (known_constraints.already_enforced_control_ids counts as stated);
  * a conflict trigger fires on what the prompt REQUESTS, independent of whether the
    control would otherwise apply - and a trigger's `enforcement` decides whether a human
    reviewer may accept it (recommended) or it must be changed (mandatory);
  * the number of clauses added is capped (policy.max_added_controls) so the revised
    prompt stays usable; mandatory controls are never displaced by recommended ones, and
    every control left out is reported as deferred - never dropped silently;
  * unresolved context becomes a targeted question, or - when the catalogue gives a safe
    default - a stated assumption, but only for controls that are actually added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .catalogue import Catalogue
from .conditions import evaluate_condition, referenced_values
from .config import Policy
from .models import SEVERITY_ORDER, Control, TaskProfile
from .provenance import control_usability


@dataclass
class ControlMatch:
    control: Control
    enforcement: str
    reason: str
    accepted_exception: bool = False


@dataclass
class ConflictFinding:
    control: Control
    trigger_id: str
    enforcement: str
    request_excerpt: str
    why: str
    alternative: str
    accepted: bool


@dataclass
class Selection:
    gaps: List[ControlMatch] = field(default_factory=list)                 # controls to add (after the cap)
    satisfied: List[Tuple[Control, str]] = field(default_factory=list)     # (control, evidence)
    deferred: List[Tuple[Control, str]] = field(default_factory=list)      # (control, reason)
    conflicts: List[ConflictFinding] = field(default_factory=list)         # unaccepted AND accepted (see .accepted)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    applicable_count: int = 0


def explain_condition(cond: Mapping[str, Any], facts: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The (fact, value) pairs that make a TRUE condition true - used to cite evidence in a reason."""
    if "all" in cond:
        return [p for c in cond["all"] for p in explain_condition(c, facts)]
    if "any" in cond:
        return [p for c in cond["any"] if evaluate_condition(c, facts) for p in explain_condition(c, facts)]
    if "not" in cond:
        return []
    have = set(facts.get(cond["fact"], ()))
    if "any_of" in cond:
        return [(cond["fact"], v) for v in sorted(have & set(cond["any_of"]))]
    if "all_of" in cond:
        return [(cond["fact"], v) for v in cond["all_of"]]
    if cond.get("present") is True:
        return [(cond["fact"], v) for v in sorted(have)]
    return []


def _reason(control: Control, pairs: List[Tuple[str, str]], profile: TaskProfile) -> str:
    snippets: List[str] = []
    for fact, value in pairs:
        for snip in profile.evidence.get(fact, {}).get(value, []):
            snip = snip if len(snip) <= 130 else snip[:127] + "…"
            if snip and snip not in snippets:
                snippets.append(snip)
    detected = ", ".join(f"{f}={v}" for f, v in dict.fromkeys(pairs))
    if snippets:
        quoted = "; ".join(f"“{s}”" for s in snippets[:2])
        return f"Applies because the request says: {quoted} [{detected}]"
    return f"Applies because the task implies [{detected}]"


def select_controls(profile: TaskProfile, catalogue: Catalogue, policy: Policy,
                    known_constraints: Optional[Mapping[str, Any]] = None) -> Selection:
    constraints = dict(known_constraints or {})
    enforced_elsewhere = set(constraints.get("already_enforced_control_ids") or [])
    accepted_ids = set(constraints.get("accepted_conflicts") or [])
    facts = profile.facts
    result = Selection()
    candidates: List[Tuple[int, ControlMatch]] = []
    satisfied_ids: set = set()

    for index, control in enumerate(catalogue.controls):
        if control.active_status != "active":
            continue
        effective = policy.enforcement_overrides.get(control.control_id, control.enforcement)
        applicable = evaluate_condition(control.applicability_conditions, facts)
        triggers = [t for t in control.conflict_triggers
                    if evaluate_condition(t["when"], facts) and not (t.get("unless") and evaluate_condition(t["unless"], facts))]
        if not applicable and not triggers:
            continue
        usable, warnings = control_usability(control, policy)
        result.warnings.extend(warnings)
        if not usable:
            if warnings:
                result.deferred.append((control, warnings[0].split(": ", 1)[1]))
            continue

        pairs = explain_condition(control.applicability_conditions, facts) if applicable else []
        accepted_exception = False
        for trig in triggers:
            enforcement = trig.get("enforcement") or effective
            values = sorted(referenced_values(trig["when"], "requested_configurations"))
            excerpt = "; ".join(profile.snippets("requested_configurations", values, limit=2)) or "(see request)"
            accepted = control.control_id in accepted_ids and enforcement != "mandatory"
            result.conflicts.append(ConflictFinding(control, trig["id"], enforcement, excerpt, trig["why"].strip(),
                                                    trig["alternative"].strip(), accepted))
            accepted_exception = accepted_exception or accepted
            if accepted:
                result.warnings.append(f"{control.control_id}: conflict '{trig['id']}' was accepted by a reviewer "
                                       f"(known_constraints.accepted_conflicts); compensating requirements are still added.")

        if not applicable and not accepted_exception:
            continue
        result.applicable_count += 1
        if control.control_id in enforced_elsewhere and not accepted_exception:
            result.satisfied.append((control, "enforced elsewhere (known_constraints.already_enforced_control_ids)"))
            satisfied_ids.add(control.control_id)
            continue
        if control.satisfied_by and evaluate_condition(control.satisfied_by, facts) and not accepted_exception:
            evidence = "; ".join(profile.snippets("stated_requirements", [v for _, v in explain_condition(control.satisfied_by, facts)], limit=2))
            result.satisfied.append((control, evidence or "stated in the request"))
            satisfied_ids.add(control.control_id)
            continue
        reason = _reason(control, pairs, profile) if pairs else "Applies because a reviewer accepted a conflicting request; this compensating requirement remains."
        candidates.append((index, ControlMatch(control, effective, reason, accepted_exception)))

    # priority: mandatory first, then severity, then catalogue order; cap to the policy limit
    candidates.sort(key=lambda ic: (0 if ic[1].enforcement == "mandatory" else 1, SEVERITY_ORDER[ic[1].control.severity_or_priority], ic[0]))
    keep = [m for _, m in candidates[:policy.max_added_controls]]
    for _, m in candidates[policy.max_added_controls:]:
        result.deferred.append((m.control, f"not added: lower priority than the {policy.max_added_controls}-control limit (policy.limits.max_added_controls)"))
    result.gaps = keep

    # unresolved context for the controls that are actually added
    seen_q: set = set()
    for match in keep:
        for ctx in match.control.required_context:
            if evaluate_condition(ctx["provided_when"], facts):
                continue
            if ctx.get("default_assumption"):
                text = ctx["default_assumption"].strip()
                if text not in result.assumptions:
                    result.assumptions.append(text)
            elif ctx["question"] not in seen_q:
                seen_q.add(ctx["question"])
                result.questions.append({"question": ctx["question"].strip(), "why_it_matters": ctx["why_it_matters"].strip(),
                                         "blocking": bool(ctx.get("blocking", False)), "control_id": match.control.control_id})
    result.questions.sort(key=lambda q: (not q["blocking"], SEVERITY_ORDER[catalogue.get(q["control_id"]).severity_or_priority]))
    result.questions = result.questions[:policy.max_clarification_questions]
    return result

