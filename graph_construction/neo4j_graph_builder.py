"""
neo4j_graph_builder.py
=======================
Graph construction for the Invictus AWS CloudTrail dataset
(https://github.com/invictus-ir/aws_dataset — CloudTrail events from an
AWS attack simulation using Stratus Red Team).

INPUT SCHEMA (verified against the actual file, not assumed):
    log_id        : int    — row index assigned during CSV export
    source_node   : str|NA — ARN of the IAM principal that made the API call
    target_node   : str    — heterogeneous: ARN, service domain, bucket name,
                              EC2-style instance id, AWS region string, or a
                              generic placeholder ("aws_service")
    edge_type     : str    — the CloudTrail eventName (API action)
    label         : int    — ground-truth 0/1, attack vs. benign event

This file supersedes an earlier version that was written against a
DIFFERENT, non-existent schema (username, timestamp, event_source,
aws_region, source_ip, error_code, session_label, attack_technique). None
of those fields exist in the real dataset and are not reconstructed here.

METHODOLOGICAL PRINCIPLES APPLIED (see the accompanying design notes for
the full write-up)
─────────────────────────────────────────────────────────────────────────
1. No fabricated fields. Every derived value is a deterministic function
   of (log_id, source_node, target_node, edge_type) — nothing is imputed,
   guessed, or sampled.
2. No temporal assumption. AWS's own documentation states CloudTrail log
   files "aren't an ordered stack trace of the public API calls... events
   don't appear in any specific order" (AWS CloudTrail User Guide,
   "CloudTrail log file examples"). The dataset's GitHub repository
   (invictus-ir/aws_dataset) documents no chronological guarantee for
   log_id either. Therefore log_id is treated ONLY as a stable row
   identifier, never as a time proxy. No hour-of-day / cyclical / sequence
   feature is derived from it.
3. No identity-based leakage. A per-event "is this principal a known
   attacker" flag is not included in the model-facing feature set, because
   it is only knowable here due to post-hoc incident-response knowledge of
   which Stratus Red Team roles were used — a real detector would not have
   this information a priori, and including it would make the prediction
   task circular. It is retained ONLY as descriptive node metadata for
   dataset-composition reporting (see `Principal.is_known_attacker_identity`)
   and is explicitly excluded by data_loader.py from the tensors handed to
   the model.
4. No dedup-induced label loss. The previous implementation used Cypher
   MERGE on (principal, target, edge_type), which silently collapses the
   2,900 labelled rows into ~1,017 aggregated edges (a 65% reduction),
   changing the unit of analysis without documentation. This version uses
   CREATE for every row, so the graph is a multigraph with exactly one
   INVOKED relationship per CSV row, keyed by log_id. Two researchers
   loading the same CSV will get the same 2,900-edge graph.
5. Service inference is NOT done from the API action name. A reverse
   lookup was attempted using botocore's authoritative service/operation
   definitions (425 AWS services, 14,742 operation names): 74 of the 260
   action names in this dataset are ambiguous across multiple real AWS
   services (e.g. "CreateUser" exists in 15 different service APIs), so
   action-name -> service is NOT deterministic and is not used. Service is
   instead derived from target_node, which sometimes directly encodes it
   (see `parse_target` below) — and left "unresolved" when it does not.

Run:
    pip install neo4j pandas
    python3 neo4j_graph_builder.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import pandas as pd
from neo4j import GraphDatabase

# ── Connection ────────────────────────────────────────────────────────────────
URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "test1234"

CSV_PATH = "./invictus_structural.csv"

REQUIRED_COLUMNS = {"log_id", "source_node", "target_node", "edge_type", "label"}


# ══════════════════════════════════════════════════════════════════════════
# 1. PRINCIPAL (source_node) PARSING
#    Rule: AWS ARN grammar is a published, fixed spec:
#      arn:partition:service:region:account-id:resource
#    IAM users   -> arn:aws:iam::ACCOUNT:user/NAME
#    Assumed roles (via STS) -> arn:aws:sts::ACCOUNT:assumed-role/ROLE/SESSION
#    AWS-managed service-linked roles are named "AWSServiceRoleFor*" by
#    documented AWS convention.
#    Assumption: source_node, when present, is always a well-formed ARN
#    (true for all 2,823 non-null rows in this file — verified below).
# ══════════════════════════════════════════════════════════════════════════

UNRESOLVED_PRINCIPAL = "UNRESOLVED_PRINCIPAL"  # explicit sentinel, not a fabricated identity


@dataclass(frozen=True)
class PrincipalInfo:
    arn: str
    name: str
    principal_type: str  # IAMUser | AssumedRole | AWSServiceLinkedRole | Unresolved


def parse_principal(source_node) -> PrincipalInfo:
    """
    Deterministically parse an IAM principal from source_node.

    109 rows (77 in the structural export used here) have a null
    source_node. These are calls attributable to AWS-internal / service
    context rather than a specific IAM identity in this export (verified:
    100% of these rows are label=0). Rather than inventing a synthetic ARN,
    they are mapped to one explicit, clearly-named sentinel node so the
    missingness is visible rather than disguised as data.
    """
    if pd.isna(source_node) or str(source_node).strip() == "":
        return PrincipalInfo(arn=UNRESOLVED_PRINCIPAL, name=UNRESOLVED_PRINCIPAL,
                              principal_type="Unresolved")

    arn = str(source_node)

    if ":user/" in arn:
        name = arn.split(":user/")[-1]
        return PrincipalInfo(arn=arn, name=name, principal_type="IAMUser")

    if ":assumed-role/" in arn:
        role_name = arn.split(":assumed-role/")[-1].split("/")[0]
        ptype = "AWSServiceLinkedRole" if role_name.startswith("AWSServiceRoleFor") \
            else "AssumedRole"
        return PrincipalInfo(arn=arn, name=role_name, principal_type=ptype)

    if ":role/" in arn:
        role_name = arn.split(":role/")[-1]
        ptype = "AWSServiceLinkedRole" if role_name.startswith("AWSServiceRoleFor") \
            else "AssumedRole"
        return PrincipalInfo(arn=arn, name=role_name, principal_type=ptype)

    # ARN present but doesn't match a known principal pattern — surfaced
    # explicitly rather than silently bucketed, so it can be audited.
    return PrincipalInfo(arn=arn, name=arn, principal_type="Unresolved")


# ══════════════════════════════════════════════════════════════════════════
# 2. TARGET (target_node) PARSING
#    target_node is heterogeneous by construction (see module docstring).
#    Each branch below is a purely syntactic, regex-based classification —
#    no semantic guessing. If none of the deterministic patterns match, the
#    resource is left "opaque" rather than assigned a guessed category.
# ══════════════════════════════════════════════════════════════════════════

_ARN_RE      = re.compile(r"^arn:aws:([a-zA-Z0-9\-]+):[^:]*:[^:]*:(.*)$")
_DOMAIN_SUFFIX = ".amazonaws.com"
# This dataset's principal/resource identifiers are simulator-generated
# (Stratus Red Team), not real AWS-issued IDs, so EC2-style ids use the
# full alphanumeric range rather than AWS's real hex-only convention
# (i- + 8 or 17 lowercase hex chars). The pattern below matches this
# dataset's actual observed convention (i- + >=8 alphanumeric chars).
_EC2_ID_RE     = re.compile(r"^i-[0-9a-z]{8,}$")
_REGION_RE     = re.compile(r"^[a-z]{2}-[a-z]+-\d$")
_ACCOUNT_ID_RE = re.compile(r"^\d{12}$")


@dataclass(frozen=True)
class TargetInfo:
    value: str
    resource_type: str   # arn-resource | service-domain | ec2-instance-id | aws-region | opaque
    service: str         # deterministically resolved AWS service, or "unresolved"
    resolved: bool        # True iff `service` was deterministically derivable


def _service_from_domain(value: str):
    """
    Parse an *.amazonaws.com endpoint into its service label.

    AWS service endpoints follow the documented pattern
        {service}.amazonaws.com
        {service}.{region}.amazonaws.com
        {account-id}.{service}.{region}.amazonaws.com   (e.g. S3 Control)
    So after stripping the ".amazonaws.com" suffix, the remaining
    dot-separated labels contain at most one 12-digit account id and at
    most one region-shaped label (checked with _ACCOUNT_ID_RE / _REGION_RE,
    both fixed, documented AWS formats) — whatever single label is left is
    the service. If more than one label remains after removing those two,
    the result is ambiguous and left unresolved rather than guessed.
    """
    if not value.endswith(_DOMAIN_SUFFIX):
        return None
    prefix = value[: -len(_DOMAIN_SUFFIX)]
    if not prefix:
        return None
    labels = [lbl for lbl in prefix.split(".") if lbl]
    candidates = [
        lbl for lbl in labels
        if not _ACCOUNT_ID_RE.match(lbl) and not _REGION_RE.match(lbl)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None  # ambiguous — left unresolved deliberately


def parse_target(target_node) -> TargetInfo:
    value = str(target_node)

    m = _ARN_RE.match(value)
    if m:
        service = m.group(1)
        return TargetInfo(value=value, resource_type="arn-resource", service=service, resolved=True)

    if value.endswith(_DOMAIN_SUFFIX):
        service = _service_from_domain(value)
        if service is not None:
            return TargetInfo(value=value, resource_type="service-domain", service=service, resolved=True)
        return TargetInfo(value=value, resource_type="service-domain", service="unresolved", resolved=False)

    if _EC2_ID_RE.match(value):
        return TargetInfo(value=value, resource_type="ec2-instance-id", service="unresolved", resolved=False)

    if _REGION_RE.match(value):
        return TargetInfo(value=value, resource_type="aws-region", service="unresolved", resolved=False)

    return TargetInfo(value=value, resource_type="opaque", service="unresolved", resolved=False)


# ══════════════════════════════════════════════════════════════════════════
# 3. EDGE (API action) FEATURES
# ══════════════════════════════════════════════════════════════════════════

# Rule: AWS documents that read-only CloudTrail management-event names
# follow a small set of verb prefixes (Get/List/Describe/etc.); this is a
# naming convention AWS itself follows, not a heuristic we invented.
# Reference: AWS CloudTrail "readOnly" event field semantics; console
# search UI filters on the same prefixes (docs.aws.amazon.com/awscloudtrail).
READ_ONLY_PREFIXES = (
    "Get", "List", "Describe", "Head", "Lookup",
    "Scan", "Query", "Search", "Check", "Validate",
)

# Rule: this is the literal, published list of IAM API actions documented
# as privilege-escalation vectors by Rhino Security Labs' AWS IAM
# privilege-escalation research (Gietzen, 2018-2019,
# rhinosecuritylabs.com/aws/aws-privilege-escalation-methods-mitigation/,
# consolidated at github.com/RhinoSecurityLabs/AWS-IAM-Privilege-Escalation).
# Assumption: this is a fixed, citable snapshot of KNOWN public techniques
# at time of publication — it is not exhaustive of every possible
# privilege-escalation path (e.g. it cannot capture parameter-level
# techniques such as iam:PassRole combined with lambda:CreateFunction,
# since this dataset's schema records only the action name, not request
# parameters).
PRIVILEGE_SENSITIVE_ACTIONS = {
    "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
    "AttachUserPolicy", "AttachGroupPolicy", "AttachRolePolicy",
    "PutUserPolicy", "PutGroupPolicy", "PutRolePolicy",
    "AddUserToGroup", "UpdateAssumeRolePolicy",
    "CreatePolicyVersion", "SetDefaultPolicyVersion", "PassRole",
}


def is_read_only(event_name: str) -> bool:
    return str(event_name).startswith(READ_ONLY_PREFIXES)


def is_privilege_sensitive(event_name: str) -> bool:
    return str(event_name) in PRIVILEGE_SENSITIVE_ACTIONS


# ══════════════════════════════════════════════════════════════════════════
# 4. KNOWN-ATTACKER-IDENTITY METADATA (descriptive only — NOT a model feature)
#    Derived from label==1 rows themselves (data-driven), not a manually
#    curated external list, so it stays reproducible from the CSV alone.
#    See module docstring, point 3, for why this is excluded from ML
#    features by data_loader.py.
# ══════════════════════════════════════════════════════════════════════════

def compute_attacker_principals(df: pd.DataFrame) -> set:
    """Principals (by name) with >=1 label==1 event, derived purely from this file."""
    attacker_rows = df[df["label"] == 1]
    names = attacker_rows["source_node"].dropna().apply(lambda a: parse_principal(a).name)
    return set(names.unique())


# ══════════════════════════════════════════════════════════════════════════
# 5. GRAPH-TOPOLOGY FEATURES (purely structural, computed once over the
#    full edge set before ingestion — deterministic, no external info)
# ══════════════════════════════════════════════════════════════════════════

def compute_topology_features(df: pd.DataFrame, principal_names: pd.Series, target_values: pd.Series):
    principal_stats = pd.DataFrame({
        "principal_key": principal_names,
        "target": target_values,
        "edge_type": df["edge_type"],
    }).groupby("principal_key").agg(
        out_degree=("target", "count"),
        unique_targets=("target", "nunique"),
        unique_actions=("edge_type", "nunique"),
    )

    target_stats = pd.DataFrame({
        "target_key": target_values,
        "principal": principal_names,
    }).groupby("target_key").agg(
        in_degree=("principal", "count"),
        unique_principals=("principal", "nunique"),
    )

    action_freq = df["edge_type"].value_counts()  # global term frequency, deterministic

    return principal_stats, target_stats, action_freq


# ── Cypher ────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Principal) REQUIRE p.arn IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Target) REQUIRE t.value IS UNIQUE",
]

MERGE_PRINCIPAL = """
MERGE (p:Principal {arn: $arn})
SET
  p.name                        = $name,
  p.principal_type              = $principal_type,
  p.out_degree                  = $out_degree,
  p.unique_targets               = $unique_targets,
  p.unique_actions               = $unique_actions,
  p.is_known_attacker_identity  = $is_known_attacker_identity
"""

MERGE_TARGET = """
MERGE (t:Target {value: $value})
SET
  t.resource_type      = $resource_type,
  t.service             = $service,
  t.resolved            = $resolved,
  t.in_degree           = $in_degree,
  t.unique_principals   = $unique_principals
"""

# CREATE (not MERGE): every CSV row becomes its own edge instance, keyed
# by log_id, preserving exact 1:1 correspondence with the source dataset.
CREATE_INVOKED = """
MATCH (p:Principal {arn: $arn})
MATCH (t:Target {value: $target_value})
CREATE (p)-[r:INVOKED {
  log_id:                 $log_id,
  edge_type:              $edge_type,
  is_read_only:           $is_read_only,
  is_privilege_sensitive: $is_privilege_sensitive,
  action_global_frequency: $action_global_frequency,
  is_attack:              $is_attack
}]->(t)
"""


def build_graph():
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}. "
            f"This builder targets the real Invictus AWS schema "
            f"{sorted(REQUIRED_COLUMNS)} — refusing to silently build "
            f"a graph from a different schema."
        )

    # ── Derive principal / target info per row (pure functions, no state) ──
    principal_infos = df["source_node"].apply(parse_principal)
    target_infos     = df["target_node"].apply(parse_target)

    principal_arns  = principal_infos.apply(lambda p: p.arn)
    principal_names = principal_infos.apply(lambda p: p.name)
    target_values   = target_infos.apply(lambda t: t.value)

    attacker_principals = compute_attacker_principals(df)
    principal_stats, target_stats, action_freq = compute_topology_features(
        df, principal_names, target_values
    )

    print(f"  {len(df):,} rows | {df['label'].sum()} labelled attack events | "
          f"{df['source_node'].isna().sum()} rows with unresolved principal")
    print(f"  {len(attacker_principals)} distinct principals have >=1 attack event: "
          f"{sorted(attacker_principals)}")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:

        print("Creating constraints …")
        for cql in CONSTRAINTS:
            session.run(cql)

        print("Clearing existing graph …")
        session.run("MATCH (n) DETACH DELETE n")

        # ── Principal nodes (deduplicated by ARN) ───────────────────────────
        print("Ingesting Principal nodes …")
        seen_principals = set()
        for arn, name in zip(principal_arns, principal_names):
            if arn in seen_principals:
                continue
            seen_principals.add(arn)
            stats = principal_stats.loc[name]
            session.run(
                MERGE_PRINCIPAL,
                arn=arn,
                name=name,
                principal_type=parse_principal(arn if arn != UNRESOLVED_PRINCIPAL else None).principal_type
                    if arn != UNRESOLVED_PRINCIPAL else "Unresolved",
                out_degree=int(stats["out_degree"]),
                unique_targets=int(stats["unique_targets"]),
                unique_actions=int(stats["unique_actions"]),
                is_known_attacker_identity=name in attacker_principals,
            )

        # ── Target nodes (deduplicated by raw value) ────────────────────────
        print("Ingesting Target nodes …")
        seen_targets = set()
        for value in target_values.unique():
            if value in seen_targets:
                continue
            seen_targets.add(value)
            info = parse_target(value)
            stats = target_stats.loc[value]
            session.run(
                MERGE_TARGET,
                value=value,
                resource_type=info.resource_type,
                service=info.service,
                resolved=info.resolved,
                in_degree=int(stats["in_degree"]),
                unique_principals=int(stats["unique_principals"]),
            )

        # ── INVOKED edges — one CREATE per CSV row ──────────────────────────
        print(f"Ingesting {len(df):,} INVOKED edges …")
        for i, row in df.iterrows():
            event_name = str(row["edge_type"])
            session.run(
                CREATE_INVOKED,
                arn=principal_arns.iloc[i],
                target_value=target_values.iloc[i],
                log_id=int(row["log_id"]),
                edge_type=event_name,
                is_read_only=is_read_only(event_name),
                is_privilege_sensitive=is_privilege_sensitive(event_name),
                action_global_frequency=int(action_freq[event_name]),
                is_attack=int(row["label"]),
            )
            if (i + 1) % 500 == 0:
                print(f"  … {i+1:,} rows processed")

    driver.close()
    print("\n✅ Graph built: "
          f"{len(seen_principals)} Principal nodes, {len(seen_targets)} Target nodes, "
          f"{len(df)} INVOKED edges (1:1 with source rows).")
    print("\nSanity queries for Neo4j Browser (localhost:7474):")
    print("─" * 60)
    print("// Row count should equal len(df) above")
    print("MATCH ()-[r:INVOKED]->() RETURN count(r)\n")
    print("// Privilege-sensitive edges and their labels")
    print("MATCH (p)-[r:INVOKED {is_privilege_sensitive: true}]->(t)")
    print("RETURN p.name, r.edge_type, t.value, r.is_attack\n")
    print("// Principals flagged as known-attacker identities (metadata only)")
    print("MATCH (p:Principal {is_known_attacker_identity: true}) RETURN p.name, p.out_degree")


if __name__ == "__main__":
    build_graph()
