"""
decision_combiner.py
====================
Component D: fuse the rules engine and the classifier into one decision, with a
policy that fails closed (spec 6D). Neither signal alone can authorise a prompt.

Policy (SAFE requires ALL to hold):
  * input validation passed;
  * the classifier ran successfully (status "ok");
  * the classifier verdict is SAFE (attack score below the dev-derived safe cut-off);
  * no rule with disposition "block" fired;
  * no high/critical-severity rule fired.
Otherwise:
  * MALICIOUS when there is sufficient evidence -- a block-disposition rule fired,
    or the classifier is confidently MALICIOUS, or both a review-rule and the
    classifier agree on risk;
  * REVIEW_REQUIRED for everything else, including every failure mode
    (validation error, classifier unavailable/error, rule/model disagreement).

The combiner reports the decision, the union of threat categories, the triggered
rule ids, reason codes and the scores -- but never the raw prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .classifier_interface import ClassifierResult
from .config import GateConfig
from .rule_engine import Finding
from .schemas import Decision


@dataclass
class CombinerOutput:
    decision: Decision
    threat_categories: list[str]
    reason_codes: list[str]
    triggered_rule_ids: list[str] = field(default_factory=list)


def combine(validation_errors: list[str],
            findings: list[Finding],
            clf: ClassifierResult,
            cfg: GateConfig) -> CombinerOutput:
    rank = cfg.severity_rank
    block_rank = rank.get(cfg.rule_block_severity, 3)
    reasons: list[str] = []
    cats: set[str] = set()

    # 1. input validation is a hard gate -> review (not a silent allow)
    if validation_errors:
        return CombinerOutput(Decision.REVIEW_REQUIRED, [],
                              ["input_validation_failed"] + [f"val:{e}" for e in validation_errors])

    rule_ids = [f.rule_id for f in findings]
    # a block-disposition finding is only honoured when it is NOT merely defensive framing
    blocking = [f for f in findings if f.disposition == "block" and not f.defensive_context]
    high = [f for f in findings if rank.get(f.severity, 0) >= block_rank and not f.defensive_context]
    review_rules = [f for f in findings if f not in blocking]
    for f in findings:
        cats.add(f.category)

    # 2. classifier must have run; unavailable/error => fail closed to review
    if clf.status != "ok":
        reasons.append(f"classifier_{clf.status}")
        if clf.error:
            reasons.append(f"clf:{clf.error}")
        if blocking:
            reasons.append("rule_block_without_classifier")
            return CombinerOutput(Decision.MALICIOUS, sorted(cats),
                                  reasons + ["ruleblock:" + ",".join(f.rule_id for f in blocking)],
                                  rule_ids)
        return CombinerOutput(Decision.REVIEW_REQUIRED, sorted(cats),
                              reasons + ["fail_closed_no_classifier"], rule_ids)

    cats.update(clf.flagged_categories)

    # 3. a structural block rule fired -> MALICIOUS
    if blocking:
        reasons.append("rule_block:" + ",".join(f.rule_id for f in blocking))
        return CombinerOutput(Decision.MALICIOUS, sorted(cats), reasons, rule_ids)

    # 4. classifier is confidently malicious -> MALICIOUS
    if clf.verdict == Decision.MALICIOUS:
        reasons.append("classifier_malicious")
        if high:
            reasons.append("rule_high_agrees:" + ",".join(f.rule_id for f in high))
        return CombinerOutput(Decision.MALICIOUS, sorted(cats), reasons, rule_ids)

    # 5. high-severity rule + classifier not clearly safe -> MALICIOUS;
    #    high-severity rule alone (classifier says safe) -> disagreement -> review
    if high:
        if clf.verdict == Decision.SAFE:
            reasons.append("disagreement_rule_high_vs_classifier_safe")
            return CombinerOutput(Decision.REVIEW_REQUIRED, sorted(cats), reasons, rule_ids)
        reasons.append("rule_high_and_classifier_uncertain:" + ",".join(f.rule_id for f in high))
        return CombinerOutput(Decision.MALICIOUS, sorted(cats), reasons, rule_ids)

    # 6. classifier uncertain -> review (optionally noting review-rule agreement)
    if clf.verdict == Decision.REVIEW_REQUIRED:
        reasons.append("classifier_uncertain")
        if review_rules:
            reasons.append("review_rules:" + ",".join(f.rule_id for f in review_rules))
        return CombinerOutput(Decision.REVIEW_REQUIRED, sorted(cats), reasons, rule_ids)

    # 7. classifier SAFE, but a medium review-rule fired -> review, not SAFE
    #    (a lack of a known signature is not proof of safety; a positive signal blocks SAFE)
    if review_rules:
        reasons.append("classifier_safe_but_review_rule:" + ",".join(f.rule_id for f in review_rules))
        return CombinerOutput(Decision.REVIEW_REQUIRED, sorted(cats), reasons, rule_ids)

    # 8. all clear
    reasons.append("classifier_safe_no_rules")
    return CombinerOutput(Decision.SAFE, [], reasons, [])
