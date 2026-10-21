"""Shared test fixtures for the standalone security_hardening package.

The upstream CloudSec pipeline and its original datasets are intentionally not copied into
this deliverable. ROOT therefore points to this package's project root; tests which require
those upstream assets skip with an explicit reason in test_security_hardening_integrity.py.
"""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

ROOT = str(Path(__file__).resolve().parents[1])
PKG = Path(ROOT) / "security_hardening"
CATALOGUE_PATH = PKG / "catalogue" / "security_controls.yaml"

P_S3_PHOTOS = "Create a new S3 bucket to store uploaded user photos."
P_SECURITY_GROUP = "Create a security group for an EC2 application server."
P_ROLE_BUCKET = "Create an IAM role that can upload files to an existing S3 bucket."
P_FLASK_SQL = (
    "Build a Flask API endpoint that accepts a username and file upload, stores records in a SQL database, "
    "and returns the requested user's record. Add logging and use a new dependency for validation."
)
P_ADEQUATE = (
    "Build an authenticated Flask API endpoint for profile records. Validate all input using an allow-list, "
    "use parameterized SQL queries, enforce object-level authorization for the authenticated user, "
    "use a least-privilege database role with only the required profile-record permissions, "
    "cap request and response sizes and request rates with timeouts, return generic error messages without stack traces, "
    "log access-control failures, and never log passwords, tokens, secrets or personal data."
)
P_AMBIGUOUS = "Change access permissions so the task works."
P_ADMIN = "Attach AdministratorAccess to the application role so it can do everything."
P_SSH_WORLD = "Open SSH port 22 to the entire internet from 0.0.0.0/0."
P_OUT_OF_SCOPE = "Create a Lambda function that stores secrets in KMS and reads from DynamoDB."
P_S3_PUBLIC = "Create an S3 bucket with public read and write access."
P_S3_ENCRYPT = "Create a new S3 bucket for customer documents."
P_S3_EXISTING = "Grant the application role access to read from an existing S3 bucket."


def catalogue_data() -> dict:
    return yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8"))


def control_in(data: dict, control_id: str) -> dict:
    return next(c for c in data["controls"] if c["control_id"] == control_id)


def catalogue(data: Optional[Mapping[str, Any]] = None):
    from security_hardening.catalogue import parse_catalogue, load_catalogue
    if data is None:
        return load_catalogue()
    return parse_catalogue(copy.deepcopy(dict(data)))


def hardener_for(data: Optional[Mapping[str, Any]] = None, policy=None):
    from security_hardening.config import load_policy, load_scope
    from security_hardening.prompt_hardener import SecurityHardener
    cat = catalogue(data)
    return SecurityHardener(catalogue=cat, policy=policy or load_policy(), scope=load_scope())


def service(data: Optional[Mapping[str, Any]] = None, policy=None):
    from security_hardening.integration import HardeningService
    return HardeningService(hardener_for(data, policy=policy) if data is not None or policy is not None else None)


def request(prompt: str, rid: str = "test-request", status: Optional[str] = "SAFE", **kwargs) -> dict:
    request = {
        "request_id": rid,
        "screening_status": status,
        "original_prompt": prompt,
        "conversation_context": {},
        "task_metadata": {},
        "known_constraints": {},
        "source_catalogue_version": None,
    }
    request.update(kwargs)
    return request


def harden(prompt: str, **kwargs) -> dict:
    return service().handle(request(prompt, **kwargs))


def ids(controls) -> list[str]:
    return [c.get("control_id") for c in controls]
