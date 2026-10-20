"""
Stage 3 Engine: Main entrypoint for intent extraction, action planning, and event prediction.

Ensures end-to-end execution adheres to the zero-hallucination policy and returns the validated
16-field Output Contract.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

try:
    from .enumerator import EventEnumerator
    from .extractor import ExtractorException, HybridIntentExtractor, LLMProvider
    from .graph_builder import ActionGraphBuilder
    from .loader import get_catalogue
    from .planner import DeterministicPlanner
    from .schemas import (
        ActionDependencyGraph,
        ParsedIntent,
        PredictionCoverage,
        Stage3Output,
        WorkflowStatus,
    )
except (ImportError, ValueError):
    from enumerator import EventEnumerator
    from extractor import ExtractorException, HybridIntentExtractor, LLMProvider
    from graph_builder import ActionGraphBuilder
    from loader import get_catalogue
    from planner import DeterministicPlanner
    from schemas import (
        ActionDependencyGraph,
        ParsedIntent,
        PredictionCoverage,
        Stage3Output,
        WorkflowStatus,
    )

logger = logging.getLogger(__name__)


class Stage3Engine:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.catalogue = get_catalogue()
        self.extractor = HybridIntentExtractor(llm_provider=llm_provider)
        self.planner = DeterministicPlanner()
        self.enumerator = EventEnumerator()
        self.graph_builder = ActionGraphBuilder()

    def predict(
        self,
        original_prompt: str,
        hardened_prompt: str,
        request_id: Optional[str] = None,
    ) -> Stage3Output:
        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        # 1. Intent Extraction (with exception handling for malformed JSON, timeouts, etc.)
        try:
            intent, security_constraints, model_version = self.extractor.extract(
                original_prompt, hardened_prompt
            )
        except ExtractorException as ee:
            return Stage3Output(
                status=ee.status,
                request_id=req_id,
                parsed_intent=ParsedIntent(
                    task_category="unsupported",
                    desired_outcome=original_prompt,
                    explicit_requirements=[],
                    provided_parameters={},
                    parameter_provenance={},
                    missing_parameters=[],
                    ambiguities=[str(ee)],
                    needs_clarification=False,
                    clarification_questions=[]
                ),
                security_constraints=[],
                proposed_actions=[],
                expected_direct_events=[],
                conditional_events=[],
                action_dependency_graph=ActionDependencyGraph(nodes=[], edges=[]),
                unmodelled_effects=[str(ee)],
                prediction_coverage=PredictionCoverage(
                    scope="validated_catalogue",
                    complete_within_scope=False,
                    limitations=[f"Intent extraction failed: {str(ee)}"]
                ),
                missing_parameters=[],
                clarification_questions=["Unable to parse request. Please verify prompt formatting."],
                assumptions=[],
                source_provenance=[],
                catalogue_version=self.catalogue.version,
                intent_model_version="extraction-failure"
            )

        # 2. Action Planning & Dependency Resolution
        planning_result = self.planner.plan_workflow(
            intent=intent,
            security_constraints=security_constraints,
            original_prompt=original_prompt,
        )

        # 3. CloudTrail Event Enumeration
        predicted_events = []
        if planning_result.proposed_actions:
            predicted_events = self.enumerator.enumerate_events(
                proposed_actions=planning_result.proposed_actions,
                provided_parameters=intent.provided_parameters,
                parameter_provenance=intent.parameter_provenance,
            )

        # 4. Action Dependency Graph Construction
        graph = self.graph_builder.build_graph(
            request_id=req_id,
            intent=intent,
            proposed_actions=planning_result.proposed_actions,
            predicted_events=predicted_events,
        )

        # 5. Compile source provenance URLs
        source_provenance = []
        for ev in predicted_events:
            source_provenance.extend(ev.source_refs)
        source_provenance = sorted(list(set(source_provenance)))

        # 6. Build and validate 16-field Stage3Output contract
        output = Stage3Output(
            status=planning_result.status,
            request_id=req_id,
            parsed_intent=intent,
            security_constraints=security_constraints,
            proposed_actions=planning_result.proposed_actions,
            expected_direct_events=predicted_events,
            conditional_events=planning_result.conditional_events,
            action_dependency_graph=graph,
            unmodelled_effects=planning_result.unmodelled_effects,
            prediction_coverage=planning_result.prediction_coverage,
            missing_parameters=planning_result.missing_parameters,
            clarification_questions=planning_result.clarification_questions,
            assumptions=planning_result.assumptions,
            source_provenance=source_provenance,
            catalogue_version=self.catalogue.version,
            intent_model_version=model_version,
        )

        return output


def predict_workflow(
    original_prompt: str,
    hardened_prompt: str,
    request_id: Optional[str] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> Stage3Output:
    """Convenience functional interface for Stage 3 engine."""
    engine = Stage3Engine(llm_provider=llm_provider)
    return engine.predict(original_prompt, hardened_prompt, request_id)
