"""
Deterministic Planner: Operation selector, dependency resolver, and parameter checker.

Resolves supported actions strictly from the validated catalogue, attaches security hardening
actions triggered by Stage 2 constraints, and resolves dependencies deterministically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .loader import get_catalogue
    from .schemas import (
        ActionMappingStatus,
        ConditionalEvent,
        ParsedIntent,
        PredictionCoverage,
        ProposedAction,
        RequirementType,
        SecurityConstraint,
        WorkflowStatus,
    )
except (ImportError, ValueError):
    from loader import get_catalogue
    from schemas import (
        ActionMappingStatus,
        ConditionalEvent,
        ParsedIntent,
        PredictionCoverage,
        ProposedAction,
        RequirementType,
        SecurityConstraint,
        WorkflowStatus,
    )

logger = logging.getLogger(__name__)


class PlanningResult:
    def __init__(
        self,
        status: WorkflowStatus,
        proposed_actions: List[ProposedAction],
        conditional_events: List[ConditionalEvent],
        unmodelled_effects: List[str],
        prediction_coverage: PredictionCoverage,
        missing_parameters: List[str],
        clarification_questions: List[str],
        assumptions: List[str],
    ):
        self.status = status
        self.proposed_actions = proposed_actions
        self.conditional_events = conditional_events
        self.unmodelled_effects = unmodelled_effects
        self.prediction_coverage = prediction_coverage
        self.missing_parameters = missing_parameters
        self.clarification_questions = clarification_questions
        self.assumptions = assumptions


class DeterministicPlanner:
    def __init__(self):
        self.catalogue = get_catalogue()

    def plan_workflow(
        self,
        intent: ParsedIntent,
        security_constraints: List[SecurityConstraint],
        original_prompt: str,
    ) -> PlanningResult:
        task_category = intent.task_category
        missing_params = list(intent.missing_parameters)
        questions = list(intent.clarification_questions)
        assumptions: List[str] = []
        unmodelled_effects: List[str] = []
        conditional_events: List[ConditionalEvent] = []

        # 1. Unsupported task handling
        if task_category == "unsupported" or not self.catalogue.is_task_supported(task_category):
            return PlanningResult(
                status=WorkflowStatus.UNSUPPORTED,
                proposed_actions=[],
                conditional_events=[],
                unmodelled_effects=[f"Task '{task_category}' is outside the validated catalogue scope."],
                prediction_coverage=PredictionCoverage(
                    scope="validated_catalogue",
                    complete_within_scope=False,
                    limitations=[f"No validated mapping exists for task category '{task_category}'."]
                ),
                missing_parameters=[],
                clarification_questions=["This request involves AWS operations that are not yet in the validated catalogue."],
                assumptions=[]
            )

        # 2. Check for partial tasks (e.g. user asked for supported task + unsupported secondary task like Macie)
        orig_lower = original_prompt.lower()
        if "macie" in orig_lower:
            unmodelled_effects.append("Amazon Macie sensitive data discovery job is requested but outside validated catalogue scope.")
        if "neptune" in orig_lower:
            unmodelled_effects.append("Amazon Neptune graph cluster provisioning is requested but outside validated catalogue scope.")

        # 3. Check for ambiguous IAM role policy attachment (populates conditional_events)
        if task_category == "iam_create_role" and ("read" in orig_lower or "policy" in orig_lower or "access" in orig_lower):
            if "policyArn" not in intent.provided_parameters:
                conditional_events.append(ConditionalEvent(
                    prediction_id="cond-iam-attach",
                    action_id="act-iam-attach-role-policy",
                    eventSource="iam.amazonaws.com",
                    eventName="AttachRolePolicy",
                    condition="Triggered if user specifies an existing AWS managed policy ARN (e.g. arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess).",
                    evidence="AWS IAM API Reference: AttachRolePolicy attaches a managed policy to the specified IAM role.",
                    potential_parameters={
                        "roleName": intent.provided_parameters.get("roleName", "UNKNOWN"),
                        "policyArn": "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
                    }
                ))
                conditional_events.append(ConditionalEvent(
                    prediction_id="cond-iam-put",
                    action_id="act-iam-put-role-policy",
                    eventSource="iam.amazonaws.com",
                    eventName="PutRolePolicy",
                    condition="Triggered if user provides an inline policy document rather than an existing managed policy ARN.",
                    evidence="AWS IAM API Reference: PutRolePolicy adds or updates an inline policy document that is embedded in the specified IAM role.",
                    potential_parameters={
                        "roleName": intent.provided_parameters.get("roleName", "UNKNOWN"),
                        "policyName": "CustomInlinePolicy"
                    }
                ))
                return PlanningResult(
                    status=WorkflowStatus.NEEDS_CLARIFICATION,
                    proposed_actions=[],
                    conditional_events=conditional_events,
                    unmodelled_effects=unmodelled_effects,
                    prediction_coverage=PredictionCoverage(
                        scope="validated_catalogue",
                        complete_within_scope=False,
                        limitations=["Policy attachment method ambiguous; conditional events enumerated."]
                    ),
                    missing_parameters=["policyArn or inline policy document"],
                    clarification_questions=[
                        "Should permissions be granted via managed policy ARN (AttachRolePolicy) or inline policy document (PutRolePolicy)?"
                    ],
                    assumptions=assumptions
                )

        # 4. Check for clarification needs or conflicts already flagged in intent
        if intent.needs_clarification:
            return PlanningResult(
                status=WorkflowStatus.NEEDS_CLARIFICATION,
                proposed_actions=[],
                conditional_events=[],
                unmodelled_effects=unmodelled_effects,
                prediction_coverage=PredictionCoverage(
                    scope="validated_catalogue",
                    complete_within_scope=False,
                    limitations=["Ambiguities, conflicts, or missing parameters prevent deterministic planning."]
                ),
                missing_parameters=missing_params,
                clarification_questions=questions,
                assumptions=assumptions
            )

        # 5. Select base actions from catalogue for the task category
        task_def = self.catalogue.get_task_def(task_category)
        if not task_def:
            return PlanningResult(
                status=WorkflowStatus.UNSUPPORTED,
                proposed_actions=[],
                conditional_events=[],
                unmodelled_effects=unmodelled_effects,
                prediction_coverage=PredictionCoverage(
                    scope="validated_catalogue",
                    complete_within_scope=False,
                    limitations=["Task definition missing from catalogue."]
                ),
                missing_parameters=[],
                clarification_questions=[],
                assumptions=[]
            )

        selected_action_ids: List[str] = list(task_def.get("primary_actions", []))
        action_rationales: Dict[str, str] = {
            act_id: "Primary action required to fulfill user's functional request"
            for act_id in selected_action_ids
        }

        # 6. Evaluate security constraints and add hardening actions in preferred order
        hardening_candidates = []
        for constraint in security_constraints:
            for act_id in constraint.triggered_action_ids:
                act = self.catalogue.get_action(act_id)
                if act and act["service"] == task_def["service"]:
                    if act_id not in hardening_candidates:
                        hardening_candidates.append(act_id)
                        action_rationales[act_id] = f"Triggered by security constraint: {constraint.description}"

        preferred_hardening = task_def.get("security_hardening_actions", [])
        hardening_candidates.sort(key=lambda x: preferred_hardening.index(x) if x in preferred_hardening else 999)

        for act_id in hardening_candidates:
            if act_id not in selected_action_ids:
                selected_action_ids.append(act_id)

        # 7. Dependency resolution: Topological sort
        ordered_action_ids: List[str] = self._topological_sort(selected_action_ids)

        # 8. Parameter validation for every planned action
        proposed_actions: List[ProposedAction] = []
        for step_idx, act_id in enumerate(ordered_action_ids, start=1):
            act_def = self.catalogue.get_action(act_id)
            if not act_def:
                continue

            # Check required parameters
            req_params = act_def.get("required_parameters", [])
            for rp in req_params:
                if rp == "keyId" and "act-kms-create-key" in ordered_action_ids:
                    continue
                if rp == "groupId" and "act-ec2-create-security-group" in ordered_action_ids:
                    continue
                if rp not in intent.provided_parameters:
                    missing_params.append(rp)
                    questions.append(f"Action '{act_def['api_operation']}' requires missing parameter '{rp}'.")

            if missing_params:
                return PlanningResult(
                    status=WorkflowStatus.NEEDS_CLARIFICATION,
                    proposed_actions=[],
                    conditional_events=[],
                    unmodelled_effects=unmodelled_effects,
                    prediction_coverage=PredictionCoverage(
                        scope="validated_catalogue",
                        complete_within_scope=False,
                        limitations=[f"Action {act_id} missing required parameter(s): {', '.join(missing_params)}"]
                    ),
                    missing_parameters=missing_params,
                    clarification_questions=questions,
                    assumptions=assumptions
                )

            depends_on_ids = [d["action_id"] for d in act_def.get("depends_on", []) if d["action_id"] in ordered_action_ids]
            req_type = (
                RequirementType.REQUIRED
                if act_id in task_def.get("primary_actions", [])
                else RequirementType.CONDITIONAL
            )

            proposed_actions.append(ProposedAction(
                step=step_idx,
                action_id=act_id,
                service=act_def["service"],
                operation=act_def["api_operation"],
                iam_permission=act_def["iam_permission"],
                status=ActionMappingStatus.SUPPORTED,
                mapping_rule_id=act_def["cloudtrail_mapping"]["mapping_rule_id"],
                requirement_type=req_type,
                depends_on=depends_on_ids,
                rationale=action_rationales.get(act_id, "")
            ))

        # Check if partial workflow due to unmodelled effects
        status = WorkflowStatus.PARTIAL if unmodelled_effects else WorkflowStatus.GENERATED
        complete = len(unmodelled_effects) == 0

        limitations = []
        if unmodelled_effects:
            limitations.append("Workflow contains unmodelled secondary tasks outside validated catalogue.")

        return PlanningResult(
            status=status,
            proposed_actions=proposed_actions,
            conditional_events=conditional_events,
            unmodelled_effects=unmodelled_effects,
            prediction_coverage=PredictionCoverage(
                scope="validated_catalogue",
                complete_within_scope=complete,
                limitations=limitations
            ),
            missing_parameters=[],
            clarification_questions=[],
            assumptions=assumptions
        )

    def _topological_sort(self, action_ids: List[str]) -> List[str]:
        """
        Sort actions so that dependencies come before actions that depend on them,
        preserving the relative order specified in action_ids.
        """
        action_set = set(action_ids)
        graph: Dict[str, Set[str]] = {aid: set() for aid in action_ids}
        in_degree: Dict[str, int] = {aid: 0 for aid in action_ids}

        for aid in action_ids:
            act = self.catalogue.get_action(aid)
            if not act:
                continue
            for dep in act.get("depends_on", []):
                dep_id = dep["action_id"]
                if dep_id in action_set:
                    graph[dep_id].add(aid)
                    in_degree[aid] += 1

        queue = [aid for aid in action_ids if in_degree[aid] == 0]
        sorted_list = []

        while queue:
            curr = queue.pop(0)
            sorted_list.append(curr)
            # Visit neighbors in the order they appear in action_ids
            neighbors = sorted(graph[curr], key=lambda x: action_ids.index(x))
            for neighbor in neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        for aid in action_ids:
            if aid not in sorted_list:
                sorted_list.append(aid)

        return sorted_list
