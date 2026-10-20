"""
Strict Pydantic v2 schemas for Stage 3 Intent Extraction and Event Prediction.

All models adhere to the zero-hallucination policy and the 16-field Output Contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class WorkflowStatus(str, Enum):
    GENERATED = "GENERATED"
    PARTIAL = "PARTIAL"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


class RequirementType(str, Enum):
    REQUIRED = "REQUIRED"
    CONDITIONAL = "CONDITIONAL"
    OPTIONAL = "OPTIONAL"


class ActionMappingStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class ParameterProvenanceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    parameter_name: str
    value: Any
    source: Literal["USER_PROMPT", "SECURITY_CONSTRAINT", "DEFAULT", "UNKNOWN"]
    rule_ref: Optional[str] = None


class ParsedIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_category: str = Field(
        ...,
        description="Canonical task category recognized by the catalogue, or 'unsupported'."
    )
    desired_outcome: str = Field(
        ...,
        description="High-level goal extracted from the prompt."
    )
    explicit_requirements: List[str] = Field(
        default_factory=list,
        description="Explicit functional requirements from the prompt."
    )
    provided_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known parameter values extracted directly from the prompt or verified state."
    )
    parameter_provenance: Dict[str, str] = Field(
        default_factory=dict,
        description="Provenance for each parameter (USER_PROMPT, SECURITY_CONSTRAINT, DEFAULT)."
    )
    missing_parameters: List[str] = Field(
        default_factory=list,
        description="Required parameters that were not specified in the input."
    )
    ambiguities: List[str] = Field(
        default_factory=list,
        description="Ambiguities or alternative implementation paths identified."
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if missing parameters or ambiguities prevent deterministic planning."
    )
    clarification_questions: List[str] = Field(
        default_factory=list,
        description="Specific clarification questions to ask the user."
    )


class SecurityConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    description: str
    triggered_action_ids: List[str] = Field(default_factory=list)
    source: str = "STAGE_2_HARDENING"


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    action_id: str
    service: str
    operation: str
    iam_permission: str
    status: ActionMappingStatus
    mapping_rule_id: str
    requirement_type: RequirementType = RequirementType.REQUIRED
    depends_on: List[str] = Field(default_factory=list)
    rationale: str = ""


class PredictedCloudTrailEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    step: int
    action_id: str
    service: str
    api_operation: str
    record_type: Literal["PREDICTED_CLOUDTRAIL_EVENT"] = "PREDICTED_CLOUDTRAIL_EVENT"
    eventSource: str
    eventName: str
    requestParameters: Dict[str, Any] = Field(default_factory=dict)
    parameter_provenance: Dict[str, str] = Field(default_factory=dict)
    depends_on_predictions: List[str] = Field(default_factory=list)
    requirement_type: RequirementType = RequirementType.REQUIRED
    mapping_rule_id: str
    source_refs: List[str] = Field(default_factory=list)
    fixture_ref: Optional[str] = None
    unknown_fields: List[str] = Field(
        default_factory=lambda: [
            "eventID",
            "requestID",
            "eventTime",
            "responseElements",
            "recipientAccountId",
            "awsRegion",
            "sourceIPAddress",
            "userAgent",
        ]
    )
    support_status: str = "VERIFIED"
    recording_scope_assumption: str = (
        "Management events with read/write type Write enabled on CloudTrail trail"
    )


class ConditionalEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    action_id: str
    eventSource: str
    eventName: str
    condition: str
    evidence: str
    potential_parameters: Dict[str, Any] = Field(default_factory=dict)
    unknown_fields: List[str] = Field(
        default_factory=lambda: [
            "eventID",
            "requestID",
            "eventTime",
            "responseElements",
            "recipientAccountId",
            "awsRegion",
            "sourceIPAddress",
            "userAgent",
        ]
    )


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str  # Task, ProposedAction, Service, Principal, Resource, Policy, PredictedEvent
    label: str
    state: Literal["PROPOSED", "EXISTING", "PREDICTED"] = "PROPOSED"
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation: str  # PROPOSES, INVOKES, TARGETS, MODIFIES_POLICY, DEPENDS_ON, MAPS_TO_EVENT, GRANTS_ACCESS
    state: Literal["PROPOSED", "OBSERVED"] = "PROPOSED"
    properties: Dict[str, Any] = Field(default_factory=dict)


class ActionDependencyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class PredictionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["validated_catalogue"] = "validated_catalogue"
    complete_within_scope: bool
    limitations: List[str] = Field(default_factory=list)


class Stage3Output(BaseModel):
    """
    The mandatory 16-field output contract for Stage 3.
    """
    model_config = ConfigDict(extra="forbid")

    # 1. status
    status: WorkflowStatus
    # 2. request_id
    request_id: str
    # 3. parsed_intent
    parsed_intent: ParsedIntent
    # 4. security_constraints
    security_constraints: List[SecurityConstraint] = Field(default_factory=list)
    # 5. proposed_actions
    proposed_actions: List[ProposedAction] = Field(default_factory=list)
    # 6. expected_direct_events
    expected_direct_events: List[PredictedCloudTrailEvent] = Field(default_factory=list)
    # 7. conditional_events
    conditional_events: List[ConditionalEvent] = Field(default_factory=list)
    # 8. action_dependency_graph
    action_dependency_graph: ActionDependencyGraph
    # 9. unmodelled_effects
    unmodelled_effects: List[str] = Field(default_factory=list)
    # 10. prediction_coverage
    prediction_coverage: PredictionCoverage
    # 11. missing_parameters
    missing_parameters: List[str] = Field(default_factory=list)
    # 12. clarification_questions
    clarification_questions: List[str] = Field(default_factory=list)
    # 13. assumptions
    assumptions: List[str] = Field(default_factory=list)
    # 14. source_provenance
    source_provenance: List[str] = Field(default_factory=list)
    # 15. catalogue_version
    catalogue_version: str
    # 16. intent_model_version
    intent_model_version: str
