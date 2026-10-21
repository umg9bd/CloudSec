"""
models.py
=========
The typed contracts of the prompt-hardening module: statuses, the fact vocabulary
that links the task analyser to the control catalogue, and the request / response /
control dataclasses.

WHY THIS EXISTS
---------------
Three pieces have to agree without importing each other's internals: the task
analyser (which emits facts about a prompt), the control catalogue (whose
applicability rules read those facts) and the response schema (what downstream
stages consume). `FACT_VOCABULARY` is the one place that names every fact and every
value, so a typo in the YAML ("least_privilage") fails when the catalogue is LOADED
instead of silently never matching.

Standard library only. Dicts crossing the module boundary are validated against the
JSON Schemas in schemas/ by schema_validation.py; these dataclasses are the internal
representation.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set

SCHEMA_VERSION = "1"


class Status(str, enum.Enum):
    """Response status. Only HARDENED and ALREADY_ADEQUATE carry a revised_prompt."""
    HARDENED = "HARDENED"
    ALREADY_ADEQUATE = "ALREADY_ADEQUATE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    SECURITY_CONFLICT = "SECURITY_CONFLICT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"                    # depends on a service outside the approved scope
    SCREENING_PENDING = "SCREENING_PENDING"          # upstream screening not finished
    SCREENING_NOT_APPROVED = "SCREENING_NOT_APPROVED"  # upstream did not approve
    INTEGRATION_ERROR = "INTEGRATION_ERROR"          # malformed/missing input, unusable catalogue, internal failure


PROCEED_STATUSES: FrozenSet[Status] = frozenset({Status.HARDENED, Status.ALREADY_ADEQUATE})

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Every fact the analyser can emit and the catalogue can test, with its legal values.
FACT_VOCABULARY: Dict[str, FrozenSet[str]] = {
    # approved services detected in the prompt (see config/scope.yaml)
    "services": frozenset({"iam", "sts", "s3", "ec2", "ssm", "cloudtrail", "rds", "elasticloadbalancing"}),
    "resource_kinds": frozenset({
        "bucket", "object", "iam_role", "iam_policy", "iam_user", "iam_group", "access_key", "trust_policy",
        "security_group", "ingress_rule", "vpc_network", "ec2_instance", "load_balancer", "db_instance",
        "parameter", "trail",
    }),
    "action_classes": frozenset({"create", "modify", "grant", "expose", "delete", "read", "write", "query", "disable"}),
    # Object-scoped actions prevent a role being created to access an existing bucket from being
    # mistaken for a request to create or reconfigure that bucket.
    "resource_actions": frozenset({"create_bucket", "modify_bucket", "grant_bucket_access", "create_role"}),
    "exposure": frozenset({"public_internet", "public_principal", "admin_port_open", "db_port_open",
                           "unauthenticated", "cross_account"}),
    "permission_shapes": frozenset({"wildcard_action", "wildcard_resource", "admin_policy", "passrole",
                                    "iam_write", "trust_wildcard", "policy_authoring"}),
    "sensitive_data": frozenset({"credentials", "personal_data", "financial", "health", "confidential"}),
    # source -> sink relationships, not keywords: "untrusted input reaches a SQL query", etc.
    "input_flows": frozenset({"untrusted_input_present", "untrusted_input_to_sql", "untrusted_input_to_shell",
                              "untrusted_input_to_path", "untrusted_input_to_url_fetch", "untrusted_input_to_html"}),
    "app_context": frozenset({"api_endpoint", "web_app", "authentication", "file_upload", "code_task",
                              "database_access", "dependency_change", "iac", "user_data_endpoint", "logging_task"}),
    "high_impact": frozenset({"destructive", "production_target", "irreversible", "bulk_change",
                              "security_control_change"}),
    "credential_handling": frozenset({"create_access_key", "long_term_credentials", "use_root", "handles_secrets"}),
    # insecure configurations the prompt ASKS FOR (negation-aware): these drive conflict detection
    "requested_configurations": frozenset({
        "attach_admin_policy", "wildcard_permissions", "wildcard_trust_principal", "public_bucket_access",
        "public_bucket_write", "disable_public_access_block", "open_admin_port_to_world", "open_db_port_to_world",
        "open_all_ports_to_world", "disable_audit_logging", "embed_credentials", "use_root_credentials",
        "string_built_sql", "shell_with_user_input", "disable_authentication", "disable_encryption",
    }),
    # security requirements the prompt ALREADY states (cue regexes live in the catalogue)
    "stated_requirements": frozenset({
        "least_privilege", "scoped_resources", "no_wildcards", "restricted_trust", "block_public_access",
        "private_only_access", "encryption_at_rest", "encryption_in_transit", "versioning_or_backup",
        "deletion_protection", "audit_logging", "replacement_trail", "temporary_credentials", "mfa_required",
        "no_hardcoded_secrets", "parameterized_queries", "safe_command_execution", "input_validation",
        "output_encoding", "authn_required", "object_level_authz", "ssrf_protection", "path_restriction",
        "rate_limiting", "secure_error_handling", "no_sensitive_logging", "dependency_scanning",
        "security_tests", "rollback_plan", "change_review", "restricted_cidr", "least_privilege_db",
    }),
    "languages": frozenset({"python", "javascript", "typescript", "java", "go", "csharp", "ruby", "php",
                            "bash", "sql"}),
}


def _plain(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / sets into JSON-ready values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(_plain(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    return obj


def canonical_sha256(obj: Any) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no whitespace) of any plain structure."""
    blob = json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── catalogue entry ──────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    source_name: str
    source_url: str
    source_version_or_section: str
    retrieved_at: Optional[str]
    content_hash: Optional[str]
    provenance_status: str
    derived: bool = False
    supporting_references: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Control:
    control_id: str
    control_category: str
    control_name: str
    control_description: str
    enforcement: str
    applicability_conditions: Dict[str, Any]
    affected_services_or_components: List[str]
    required_prompt_clause: str
    verification_expectation: str
    severity_or_priority: str
    source_name: str
    source_url: str
    source_version_or_section: str
    retrieved_at: Optional[str]
    content_hash: Optional[str]
    provenance_status: str
    derived: bool
    supporting_references: List[Dict[str, str]]
    catalogue_version: str
    active_status: str
    satisfied_by: Optional[Dict[str, Any]] = None
    conflict_triggers: List[Dict[str, Any]] = field(default_factory=list)
    required_context: List[Dict[str, Any]] = field(default_factory=list)
    source_status: str = "available"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Control":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def source_reference(self) -> SourceReference:
        return SourceReference(
            source_name=self.source_name, source_url=self.source_url,
            source_version_or_section=self.source_version_or_section, retrieved_at=self.retrieved_at,
            content_hash=self.content_hash, provenance_status=self.provenance_status, derived=self.derived,
            supporting_references=list(self.supporting_references),
        )

    def fingerprint(self) -> str:
        """Hash of the whole definition: detects a changed control between catalogue versions."""
        return canonical_sha256(dataclasses.asdict(self))


# ── request / response ───────────────────────────────────────────────────────

@dataclass
class HardeningRequest:
    request_id: str
    screening_status: Optional[str]
    original_prompt: str
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    task_metadata: Dict[str, Any] = field(default_factory=dict)
    known_constraints: Dict[str, Any] = field(default_factory=dict)
    source_catalogue_version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HardeningRequest":
        return cls(
            request_id=str(data["request_id"]), screening_status=data.get("screening_status"),
            original_prompt=str(data["original_prompt"]),
            conversation_context=dict(data.get("conversation_context") or {}),
            task_metadata=dict(data.get("task_metadata") or {}),
            known_constraints=dict(data.get("known_constraints") or {}),
            source_catalogue_version=data.get("source_catalogue_version"),
        )


@dataclass
class ServiceFindings:
    approved: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    recognised: List[str] = field(default_factory=list)

    def to_scope_findings(self, note: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "approved_services": sorted(self.approved), "excluded_services": sorted(self.excluded),
            "out_of_scope_services": sorted(self.out_of_scope), "recognised_services": sorted(self.recognised),
        }
        if note:
            out["note"] = note
        return out


@dataclass
class TaskProfile:
    """What the analyser understood about a prompt. `facts` always has every key of FACT_VOCABULARY."""
    facts: Dict[str, Set[str]]
    evidence: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # fact -> value -> verbatim snippets
    entities: Dict[str, List[str]] = field(default_factory=dict)             # literal names/CIDRs/ports in the prompt
    services: ServiceFindings = field(default_factory=ServiceFindings)
    intent_unclear: bool = False
    inferred: Set[str] = field(default_factory=set)                          # "fact:value" pairs inferred, not stated
    notes: List[str] = field(default_factory=list)                           # non-sensitive analyser caveats; never echo literal secret values

    def has(self, fact: str, *values: str) -> bool:
        have = self.facts.get(fact, set())
        return bool(have.intersection(values)) if values else bool(have)

    def snippets(self, fact: str, values: Optional[List[str]] = None, limit: int = 2) -> List[str]:
        found: List[str] = []
        for value, texts in self.evidence.get(fact, {}).items():
            if values is None or value in values:
                for text in texts:
                    if text not in found:
                        found.append(text)
        return found[:limit]


def empty_facts() -> Dict[str, Set[str]]:
    return {name: set() for name in FACT_VOCABULARY}


@dataclass
class HardeningResponse:
    request_id: str
    status: Status
    original_prompt: Optional[str]
    revised_prompt: Optional[str]
    security_gaps: List[Dict[str, Any]] = field(default_factory=list)
    added_controls: List[Dict[str, Any]] = field(default_factory=list)
    satisfied_controls: List[Dict[str, Any]] = field(default_factory=list)
    deferred_controls: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    clarification_questions: List[Dict[str, Any]] = field(default_factory=list)
    scope_findings: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    source_catalogue_version: Optional[str] = None
    catalogue_sha256: Optional[str] = None
    catalogue_freshness: Optional[Dict[str, Any]] = None
    requires_user_review: bool = False
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        out = _plain(self)
        for optional in ("scope_findings", "catalogue_freshness"):
            if out.get(optional) is None and optional == "scope_findings":
                out.pop(optional)
        return out
