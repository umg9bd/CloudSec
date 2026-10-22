"""
integration.py
==============
The boundary between this module and the teams that own the stages around it.

    upstream screening  -->  HardeningService.handle(request dict)  -->  downstream planning

WHY THIS EXISTS
---------------
Neither neighbour exists in this repository yet (they belong to teammates), so the contract is
the integration: a JSON request validated against schemas/hardening_request.schema.json and a JSON
response validated against schemas/hardening_response.schema.json. This adapter guarantees:

  * it NEVER raises to the caller: every failure becomes an INTEGRATION_ERROR response;
  * it FAILS CLOSED: a malformed request, an unusable catalogue, an unexpected exception or a
    response that violates its own schema all yield revised_prompt = null;
  * a missing or unapproved screening status is an explicit result (see prompt_hardener), never a
    silent pass-through;
  * the request path does no network I/O and uses no AWS credentials.

Downstream planning should call `to_planner_payload(response)`: it returns None unless the status is
HARDENED or ALREADY_ADEQUATE, so a conflict, clarification, scope or screening result can never be
planned from by accident.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .catalogue import CatalogueError
from .config import ConfigError
from .models import PROCEED_STATUSES, HardeningRequest, HardeningResponse, Status
from .prompt_hardener import SecurityHardener
from .schema_validation import validate_request, validate_response


def error_response(request_id: str, code: str, message: str, path: Optional[str] = None,
                   extra: Optional[List[Dict[str, str]]] = None, hardener: Optional[SecurityHardener] = None) -> Dict[str, Any]:
    """A schema-valid INTEGRATION_ERROR dict (works even when no hardener/catalogue could be built)."""
    errors = [{"code": code, "message": message, **({"path": path} if path else {})}] + list(extra or [])
    if hardener is not None:
        return hardener.error(request_id, code, message, path).to_dict() | {"errors": errors}
    return HardeningResponse(request_id=request_id, status=Status.INTEGRATION_ERROR, original_prompt=None, revised_prompt=None,
                             errors=errors, requires_user_review=True).to_dict()


class HardeningService:
    """Stateless request handler. Build once (it loads and validates the catalogue), call `handle` per request."""

    def __init__(self, hardener: Optional[SecurityHardener] = None):
        self._hardener: Optional[SecurityHardener] = None
        self._init_problems: List[str] = []
        try:
            self._hardener = hardener or SecurityHardener()
        except (CatalogueError, ConfigError, ValueError, OSError) as exc:
            self._init_problems = list(getattr(exc, "problems", [str(exc)]))

    @property
    def ready(self) -> bool:
        return self._hardener is not None

    def handle(self, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, Mapping):
            return error_response("unknown", "INVALID_REQUEST", "request must be a JSON object", hardener=self._hardener)
        rid = raw.get("request_id") if isinstance(raw.get("request_id"), str) and raw.get("request_id") else "unknown"
        status = raw.get("screening_status")
        if not isinstance(status, str) or not status.strip():
            return error_response(rid, "SCREENING_STATUS_MISSING",
                                  "screening_status is missing; upstream screening must approve the prompt before hardening.",
                                  "screening_status", hardener=self._hardener)
        problems = validate_request(dict(raw))
        if problems:
            return error_response(rid, "INVALID_REQUEST", f"request does not match hardening_request.schema.json ({len(problems)} problem(s))",
                                  extra=[{"code": "SCHEMA", "message": p} for p in problems[:5]], hardener=self._hardener)
        if self._hardener is None:
            return error_response(rid, "CATALOGUE_UNAVAILABLE", "the control catalogue or configuration could not be loaded; failing closed",
                                  extra=[{"code": "CATALOGUE", "message": p} for p in self._init_problems[:3]])
        try:
            response = self._hardener.harden(HardeningRequest.from_dict(raw)).to_dict()
        except Exception as exc:  # noqa: BLE001 - the contract is "never raise": report the class only, never prompt text
            return error_response(rid, "INTERNAL_ERROR", f"hardening failed with {type(exc).__name__}; failing closed", hardener=self._hardener)
        problems = validate_response(response)
        if problems:
            return error_response(rid, "RESPONSE_SCHEMA_VIOLATION", "internal response failed its own schema; failing closed",
                                  extra=[{"code": "SCHEMA", "message": p} for p in problems[:5]], hardener=self._hardener)
        return response


_default: Optional[HardeningService] = None


def harden_request(raw: Any, service: Optional[HardeningService] = None) -> Dict[str, Any]:
    """Convenience: handle one request dict with a lazily built default service."""
    global _default
    if service is None:
        if _default is None:
            _default = HardeningService()
        service = _default
    return service.handle(raw)


def to_planner_payload(response: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """What downstream planning may consume. None unless the response is HARDENED or ALREADY_ADEQUATE."""
    if response.get("status") not in {s.value for s in PROCEED_STATUSES} or not response.get("revised_prompt"):
        return None
    return {
        "request_id": response["request_id"],
        "status": response["status"],
        "prompt": response["revised_prompt"],
        "control_ids": [c["control_id"] for c in response.get("added_controls", [])],
        "verification": [{"control_id": c["control_id"], "expectation": c["verification_expectation"]}
                         for c in response.get("added_controls", []) + response.get("satisfied_controls", [])
                         if c.get("verification_expectation")],
        "catalogue_version": response["source_catalogue_version"],
        "catalogue_sha256": response.get("catalogue_sha256"),
        "requires_user_review": response["requires_user_review"],
    }
