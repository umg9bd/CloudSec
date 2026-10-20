"""
Deterministic CloudTrail Event Enumerator: Exhaustive event-by-event generator.

Generates explicit predicted CloudTrail event records with official documentation provenance,
parameter tracing, dependency linkages, and explicit unknown_fields without hallucinating
runtime identifiers.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from .loader import get_catalogue
    from .schemas import (
        PredictedCloudTrailEvent,
        ProposedAction,
        RequirementType,
    )
except (ImportError, ValueError):
    from loader import get_catalogue
    from schemas import (
        PredictedCloudTrailEvent,
        ProposedAction,
        RequirementType,
    )

logger = logging.getLogger(__name__)


class EventEnumerator:
    def __init__(self):
        self.catalogue = get_catalogue()

    def enumerate_events(
        self,
        proposed_actions: List[ProposedAction],
        provided_parameters: Dict[str, Any],
        parameter_provenance: Dict[str, str],
    ) -> List[PredictedCloudTrailEvent]:
        events: List[PredictedCloudTrailEvent] = []
        action_to_pred_id: Dict[str, str] = {}

        for idx, action in enumerate(proposed_actions, start=1):
            act_def = self.catalogue.get_action(action.action_id)
            if not act_def:
                continue

            ct_map = act_def["cloudtrail_mapping"]
            pred_id = f"pred-{idx:04d}"
            action_to_pred_id[action.action_id] = pred_id

            # Resolve request parameters using template and provided values
            req_params, req_prov = self._resolve_request_parameters(
                ct_map["request_parameter_template"],
                provided_parameters,
                parameter_provenance,
                action.action_id,
            )

            # Map dependencies to prediction IDs
            depends_on_preds = [
                action_to_pred_id[dep_id]
                for dep_id in action.depends_on
                if dep_id in action_to_pred_id
            ]

            event = PredictedCloudTrailEvent(
                prediction_id=pred_id,
                step=action.step,
                action_id=action.action_id,
                service=action.service,
                api_operation=action.operation,
                record_type="PREDICTED_CLOUDTRAIL_EVENT",
                eventSource=ct_map["eventSource"],
                eventName=ct_map["eventName"],
                requestParameters=req_params,
                parameter_provenance=req_prov,
                depends_on_predictions=depends_on_preds,
                requirement_type=action.requirement_type,
                mapping_rule_id=ct_map["mapping_rule_id"],
                source_refs=ct_map["source_urls"],
                fixture_ref=ct_map.get("fixture_ref"),
                unknown_fields=list(ct_map.get("unknown_fields", [
                    "eventID", "requestID", "eventTime", "responseElements",
                    "recipientAccountId", "awsRegion", "sourceIPAddress", "userAgent"
                ])),
                support_status=ct_map.get("validation_status", "VERIFIED"),
                recording_scope_assumption=ct_map.get(
                    "recording_scope_assumption",
                    "Management events with read/write type Write enabled on CloudTrail trail"
                ),
            )
            events.append(event)

        return events

    def _resolve_request_parameters(
        self,
        template: Dict[str, Any],
        provided_parameters: Dict[str, Any],
        parameter_provenance: Dict[str, str],
        action_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        resolved: Dict[str, Any] = {}
        provenance: Dict[str, str] = {}

        def _substitute(value: Any) -> Tuple[Any, Optional[str]]:
            if isinstance(value, str):
                if value.startswith("{") and value.endswith("}"):
                    var_name = value[1:-1]
                    if var_name in provided_parameters:
                        return provided_parameters[var_name], parameter_provenance.get(var_name, "USER_PROMPT")
                    # If this is a step-output reference (e.g. keyId from act-kms-create-key)
                    if var_name == "keyId":
                        return "<PROPOSED_RESOURCE_ID_FROM_STEP_1>", "DERIVED_FROM_PREDECESSOR_STEP"
                    if var_name == "groupId":
                        return "<PROPOSED_RESOURCE_ID_FROM_STEP_1>", "DERIVED_FROM_PREDECESSOR_STEP"
                    if var_name == "sseAlgorithm":
                        return "AES256", "SECURITY_CONSTRAINT_DEFAULT"
                    return value, None
                return value, None
            elif isinstance(value, dict):
                sub_dict = {}
                for k, v in value.items():
                    sub_val, _ = _substitute(v)
                    sub_dict[k] = sub_val
                return sub_dict, None
            elif isinstance(value, list):
                return [_substitute(item)[0] for item in value], None
            return value, None

        for k, v in template.items():
            val, prov = _substitute(v)
            resolved[k] = val
            if prov:
                provenance[k] = prov
            elif k in parameter_provenance:
                provenance[k] = parameter_provenance[k]
            else:
                provenance[k] = "ACTION_TEMPLATE"

        return resolved, provenance
