"""
Stage-1 prompt-injection screening gate.

An isolated, additive module: it reads nothing from the existing pipeline and
changes none of it. Downstream stages call `screen_prompt` (see screening_service)
and forward a prompt only when the result's `handoff_allowed` is True.
"""
from .schemas import (  # noqa: F401
    Decision,
    ScreeningRequest,
    ScreeningResponse,
    ContextItem,
)
from .screening_service import ScreeningService, screen_prompt  # noqa: F401

__all__ = [
    "Decision",
    "ScreeningRequest",
    "ScreeningResponse",
    "ContextItem",
    "ScreeningService",
    "screen_prompt",
]
__version__ = "0.1.0"
