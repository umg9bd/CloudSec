"""
Stage 3: Evidence-Grounded Hybrid Intent Extraction and Exact Predicted CloudTrail Event Generation.

This module provides a deterministic, zero-hallucination pipeline that translates user requests
and upstream security constraints into validated AWS action plans, exact predicted CloudTrail-style
events with official documentation provenance, and an action dependency graph for downstream
privilege-escalation and risk analysis (Stage 4).
"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from .schemas import (
        Stage3Output,
        WorkflowStatus,
        ParsedIntent,
        SecurityConstraint,
        ProposedAction,
        PredictedCloudTrailEvent,
        ConditionalEvent,
        ActionDependencyGraph,
        PredictionCoverage,
    )
    from .engine import Stage3Engine, predict_workflow
except (ImportError, ValueError):
    from schemas import (
        Stage3Output,
        WorkflowStatus,
        ParsedIntent,
        SecurityConstraint,
        ProposedAction,
        PredictedCloudTrailEvent,
        ConditionalEvent,
        ActionDependencyGraph,
        PredictionCoverage,
    )
    from engine import Stage3Engine, predict_workflow

__version__ = "1.0.0"

__all__ = [
    "Stage3Engine",
    "predict_workflow",
    "Stage3Output",
    "WorkflowStatus",
    "ParsedIntent",
    "SecurityConstraint",
    "ProposedAction",
    "PredictedCloudTrailEvent",
    "ConditionalEvent",
    "ActionDependencyGraph",
    "PredictionCoverage",
]
