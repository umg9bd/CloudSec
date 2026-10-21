"""
config.py
=========
Loads and validates config/policy.yaml and config/scope.yaml.

WHY THIS EXISTS
---------------
Policy (which screening statuses are approved, how many clauses to add, what to do
when a source is unavailable) and scope (which AWS services this module may write
service-specific controls for) are project decisions. Keeping them in YAML beside the
catalogue means they change without a code change, and loading them here means a typo
fails loudly at start-up instead of changing behaviour silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ConfigError(ValueError):
    """A configuration file is missing, malformed or inconsistent."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    # Public loaders accept strings as well as Path objects; normalise at the boundary.
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at top level")
    return data


# ── policy ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Policy:
    approved_screening_statuses: FrozenSet[str]
    pending_screening_statuses: FrozenSet[str]
    max_prompt_chars: int = 20000
    max_added_controls: int = 8
    max_clarification_questions: int = 5
    out_of_scope: str = "block"
    on_unavailable_source: str = "apply_with_warning"
    require_snapshot_verified_provenance: bool = False
    enforcement_overrides: Dict[str, str] = field(default_factory=dict)

    def screening_disposition(self, status: Optional[str]) -> str:
        """'approved' | 'pending' | 'not_approved' | 'missing'. Exact, case-sensitive match on purpose."""
        if not isinstance(status, str) or not status.strip():
            return "missing"
        if status in self.approved_screening_statuses:
            return "approved"
        if status in self.pending_screening_statuses:
            return "pending"
        return "not_approved"


def load_policy(path: Optional[Path] = None) -> Policy:
    path = path or CONFIG_DIR / "policy.yaml"
    raw = _load_yaml(path)
    screening = raw.get("screening") or {}
    limits = raw.get("limits") or {}
    behaviour = raw.get("behaviour") or {}
    approved = screening.get("approved_statuses")
    pending = screening.get("pending_statuses", [])
    if not approved or not all(isinstance(s, str) and s for s in approved):
        raise ConfigError("policy.screening.approved_statuses must be a non-empty list of strings")
    if set(approved) & set(pending):
        raise ConfigError("a screening status cannot be both approved and pending")
    if behaviour.get("out_of_scope", "block") not in ("block", "warn"):
        raise ConfigError("policy.behaviour.out_of_scope must be 'block' or 'warn'")
    if behaviour.get("on_unavailable_source", "apply_with_warning") not in ("apply_with_warning", "skip"):
        raise ConfigError("policy.behaviour.on_unavailable_source must be 'apply_with_warning' or 'skip'")
    overrides = raw.get("enforcement_overrides") or {}
    if any(v not in ("mandatory", "recommended") for v in overrides.values()):
        raise ConfigError("policy.enforcement_overrides values must be 'mandatory' or 'recommended'")
    for key in ("max_prompt_chars", "max_added_controls", "max_clarification_questions"):
        if key in limits and (not isinstance(limits[key], int) or limits[key] < 1):
            raise ConfigError(f"policy.limits.{key} must be a positive integer")
    return Policy(
        approved_screening_statuses=frozenset(approved),
        pending_screening_statuses=frozenset(pending),
        max_prompt_chars=limits.get("max_prompt_chars", 20000),
        max_added_controls=limits.get("max_added_controls", 8),
        max_clarification_questions=limits.get("max_clarification_questions", 5),
        out_of_scope=behaviour.get("out_of_scope", "block"),
        on_unavailable_source=behaviour.get("on_unavailable_source", "apply_with_warning"),
        require_snapshot_verified_provenance=bool(behaviour.get("require_snapshot_verified_provenance", False)),
        enforcement_overrides=dict(overrides),
    )


# ── scope ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScopeConfig:
    scope_version: str
    approved: Dict[str, Dict[str, Any]]
    recognised_without_controls: FrozenSet[str]
    excluded: Dict[str, Dict[str, Any]]

    def classify(self, service: str) -> str:
        """'approved' | 'recognised' | 'excluded' | 'out_of_scope' for a short service key (e.g. 's3')."""
        if service in self.excluded:
            return "excluded"
        if service in self.approved:
            return "approved"
        if service in self.recognised_without_controls:
            return "recognised"
        return "out_of_scope"

    @property
    def vocabulary_services(self) -> FrozenSet[str]:
        return frozenset(self.approved) | self.recognised_without_controls | frozenset(self.excluded)


def load_scope(path: Optional[Path] = None) -> ScopeConfig:
    path = path or CONFIG_DIR / "scope.yaml"
    raw = _load_yaml(path)
    approved = raw.get("approved_services") or {}
    excluded = raw.get("excluded_by_project_decision") or {}
    recognised = raw.get("recognised_without_service_controls") or []
    if not approved:
        raise ConfigError("scope.approved_services must not be empty")
    overlap = (set(approved) & set(excluded)) | (set(approved) & set(recognised)) | (set(excluded) & set(recognised))
    if overlap:
        raise ConfigError(f"services listed in more than one scope section: {sorted(overlap)}")
    for key, entry in {**approved, **excluded}.items():
        if not str(entry.get("event_source", "")).endswith(".amazonaws.com"):
            raise ConfigError(f"scope entry {key!r} needs an event_source ending in .amazonaws.com")
    return ScopeConfig(
        scope_version=str(raw.get("scope_version", "")),
        approved=dict(approved),
        recognised_without_controls=frozenset(recognised),
        excluded=dict(excluded),
    )
