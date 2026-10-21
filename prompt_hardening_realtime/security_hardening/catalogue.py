"""
catalogue.py
============
Loads, validates and diffs the versioned security-control catalogue.

WHY THIS EXISTS
---------------
The catalogue is data (catalogue/security_controls.yaml) so controls can change without
a code release. Data that drives security advice must fail LOUDLY when it is wrong, so
loading does far more than parse YAML:

  * structural validation against schemas/control_catalogue.schema.json;
  * every fact name and value in every condition must exist in models.FACT_VOCABULARY
    (a typo cannot silently turn a control off);
  * every "stated requirement" tag a control relies on must have cue patterns, and every
    pattern must compile;
  * control ids are unique and no control's version is newer than the catalogue's;
  * scope: affected services must be approved in config/scope.yaml, and no control text
    may mention Lambda, KMS or Secrets Manager (excluded by project decision);
  * provenance: every control is traceable (see provenance.py).

A catalogue that fails any check is rejected as a whole (CatalogueError lists every
problem). The adapter then reports INTEGRATION_ERROR instead of hardening with a
half-valid catalogue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Pattern, Tuple

import yaml

from .conditions import referenced_values, validate_condition
from .config import ScopeConfig, load_scope
from .models import FACT_VOCABULARY, Control, canonical_sha256
from .provenance import provenance_problems
from .schema_validation import validate_catalogue_structure

CATALOGUE_PATH = Path(__file__).resolve().parent / "catalogue" / "security_controls.yaml"
GENERIC_COMPONENTS = frozenset({"application", "api", "database", "iac"})
_EXCLUDED_TEXT = re.compile(r"\bKMS\b|secrets\s*manager|secretsmanager|\bAWS Lambda\b|\bLambda function|lambda\.amazonaws", re.I)


class CatalogueError(ValueError):
    """The catalogue is malformed; `problems` lists everything that is wrong."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        shown = "; ".join(self.problems[:8]) + (f" (+{len(self.problems) - 8} more)" if len(self.problems) > 8 else "")
        super().__init__(f"invalid control catalogue: {shown}")


def _semver(text: str) -> Tuple[int, int, int]:
    major, minor, patch = (int(p) for p in text.split("."))
    return major, minor, patch


@dataclass(frozen=True)
class Catalogue:
    catalogue_id: str
    catalogue_version: str
    schema_version: str
    updated_at: str
    scope_version: str
    freshness: Dict[str, Any]
    changelog: Tuple[Dict[str, Any], ...]
    cues: Dict[str, Tuple[Pattern[str], ...]]
    controls: Tuple[Control, ...]
    sha256: str
    path: Optional[Path] = field(default=None, compare=False)

    def active(self) -> List[Control]:
        return [c for c in self.controls if c.active_status == "active"]

    def get(self, control_id: str) -> Optional[Control]:
        return next((c for c in self.controls if c.control_id == control_id), None)

    @property
    def version(self) -> str:
        return self.catalogue_version


def validate_catalogue_data(data: Mapping[str, Any], scope: ScopeConfig) -> List[str]:
    """Every structural and semantic problem in a loaded catalogue document (empty list = valid)."""
    problems = list(validate_catalogue_structure(data))
    if problems:  # semantic checks assume the structure is sound
        return problems

    cue_tags = set(data["requirement_cues"])
    declared_tags = FACT_VOCABULARY["stated_requirements"]
    for tag, patterns in data["requirement_cues"].items():
        if tag not in declared_tags:
            problems.append(f"requirement_cues: unknown tag {tag!r} (not in FACT_VOCABULARY['stated_requirements'])")
        for pattern in patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                problems.append(f"requirement_cues.{tag}: bad regex {pattern!r}: {exc}")

    catalogue_version = _semver(data["catalogue_version"])
    approved = set(scope.approved)
    seen: set = set()
    for control in data["controls"]:
        cid = control["control_id"]
        if cid in seen:
            problems.append(f"{cid}: duplicate control_id")
        seen.add(cid)
        if _semver(control["catalogue_version"]) > catalogue_version:
            problems.append(f"{cid}: control catalogue_version {control['catalogue_version']} is newer than the catalogue")
        problems.extend(f"{cid}.applicability_conditions: {p}" for p in validate_condition(control["applicability_conditions"]))
        conditions = [("satisfied_by", control.get("satisfied_by"))]
        for trig in control.get("conflict_triggers", []):
            conditions += [(f"conflict_triggers[{trig['id']}].when", trig["when"]), (f"conflict_triggers[{trig['id']}].unless", trig.get("unless"))]
        for ctx in control.get("required_context", []):
            conditions.append((f"required_context[{ctx['key']}].provided_when", ctx["provided_when"]))
        for label, cond in conditions:
            if cond is None:
                continue
            issues = validate_condition(cond)
            problems.extend(f"{cid}.{label}: {p}" for p in issues)
            if not issues:
                for tag in referenced_values(cond, "stated_requirements"):
                    if tag not in cue_tags:
                        problems.append(f"{cid}.{label}: stated requirement {tag!r} has no cue patterns, so it could never match")
        trigger_ids = [t["id"] for t in control.get("conflict_triggers", [])]
        if len(trigger_ids) != len(set(trigger_ids)):
            problems.append(f"{cid}: duplicate conflict trigger ids")
        for component in control["affected_services_or_components"]:
            if component not in GENERIC_COMPONENTS and component not in approved:
                excluded = " (excluded by project decision)" if component in scope.excluded else ""
                problems.append(f"{cid}: affected service {component!r} is not in the approved scope{excluded}")
        for field_name in ("control_name", "control_description", "required_prompt_clause", "verification_expectation"):
            if _EXCLUDED_TEXT.search(control[field_name]):
                problems.append(f"{cid}.{field_name}: mentions a service excluded from this module (Lambda, KMS or Secrets Manager)")
        problems.extend(provenance_problems(Control.from_dict(control)))
    return problems


def parse_catalogue(data: Mapping[str, Any], scope: Optional[ScopeConfig] = None, path: Optional[Path] = None) -> Catalogue:
    scope = scope or load_scope()
    problems = validate_catalogue_data(data, scope)
    if problems:
        raise CatalogueError(problems)
    cues = {tag: tuple(re.compile(p, re.IGNORECASE) for p in patterns) for tag, patterns in data["requirement_cues"].items()}
    return Catalogue(
        catalogue_id=data["catalogue_id"], catalogue_version=data["catalogue_version"],
        schema_version=data["schema_version"], updated_at=data["updated_at"],
        scope_version=str(data.get("scope_version", "")), freshness=dict(data["freshness"]),
        changelog=tuple(data.get("changelog", ())), cues=cues,
        controls=tuple(Control.from_dict(c) for c in data["controls"]),
        sha256=canonical_sha256(data), path=path,
    )


def load_catalogue(path: Optional[Path] = None, scope: Optional[ScopeConfig] = None) -> Catalogue:
    path = Path(path) if path else CATALOGUE_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogueError([f"cannot read {path}: {exc}"]) from exc
    if not isinstance(data, dict):
        raise CatalogueError([f"{path}: top level must be a mapping"])
    return parse_catalogue(data, scope, path)


def diff_catalogues(old: Catalogue, new: Catalogue) -> Dict[str, Any]:
    """Which controls were added, removed, changed or (de)activated between two catalogue versions."""
    old_by, new_by = {c.control_id: c for c in old.controls}, {c.control_id: c for c in new.controls}
    changed = sorted(i for i in old_by.keys() & new_by.keys() if old_by[i].fingerprint() != new_by[i].fingerprint())
    return {
        "from_version": old.catalogue_version, "to_version": new.catalogue_version,
        "added": sorted(new_by.keys() - old_by.keys()),
        "removed": sorted(old_by.keys() - new_by.keys()),
        "changed": changed,
        "activated": sorted(i for i in changed if old_by[i].active_status != "active" and new_by[i].active_status == "active"),
        "deactivated": sorted(i for i in changed if old_by[i].active_status == "active" and new_by[i].active_status != "active"),
    }
