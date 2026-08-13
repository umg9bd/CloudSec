"""
neo4j_graph_builder.py  (v3 — Privilege Propagation Graph)
============================================================
Builds a heterogeneous Privilege Propagation Graph from CloudTrail-derived
structural CSVs (log_id, source_node, target_node, edge_type, label) —
originally the static Invictus AWS dataset, now the Feature Engine's
cloudtrail_structural.csv (feature_engine9.py) — superseding the earlier
bipartite Principal->Target design (v2).

log_id TYPE CONTRACT
─────────────────────────────────────────────────────────────────────────
log_id is a unique, opaque, STRING event identifier — e.g. the Feature
Engine's own "<source_file>:<row_index>" scheme (feature_engine9.py's
process_batch_file, "synthetic_cloudtrail.csv:0", ...). It is never
parsed, never assumed numeric, and never coerced with int() anywhere in
this file (or in privilege_features.py, which stores whatever value it is
given without inspecting it — see that module's docstring). It is used
purely as a lookup/join key and as a Neo4j edge property (Neo4j natively
supports string property values, so no schema change was needed there).
Code that previously needed a NOTION of "which row came first" (there is
exactly one such place — see blast_radius.py's edges_on_shortest_path)
now derives that from actual insertion/stream order instead of from
log_id's value, since log_id itself carries no ordering guarantee (true
even under the old integer Invictus IDs — AWS CloudTrail log files are
not delivered in a guaranteed order either).

SCHEMA
──────
NODE TYPES (all also carry a secondary supertype label for convenient
generic queries: User/Role/UnresolvedPrincipal -> also :Principal;
Service/Resource/Policy -> also :Target):

  (:User          :Principal {key, out_degree, unique_targets, unique_actions,
                               role_transition_count, is_known_attacker_identity})
      IAM users (source_node ARNs containing ":user/").

  (:Role          :Principal {key, out_degree, unique_targets, unique_actions,
                               role_transition_count, is_known_attacker_identity})
      CANONICAL role identity, keyed by ROLE NAME rather than ARN. This is
      the key structural contribution of this redesign: AWS records the
      same role under two different ARN namespaces (see privilege_features.py
      module docstring), and unifying them by name is what makes multi-hop
      privilege chains visible at all in this dataset.

  (:Service       :Target {key, in_degree, unique_principals, resource_sensitivity})
      *.amazonaws.com control-plane endpoints (deterministically parsed).

  (:Resource      :Target {key, resource_type, in_degree, unique_principals,
                            resource_sensitivity, distance_to_sensitive_resource})
      Everything else the graph acts on: EC2 instances, instance types,
      regions, opaque bucket names, etc.

  (:Policy        :Target {key, resource_sensitivity})
      IAM policy ARNs (":policy/" in target_node). Verified sparse in this
      dataset (only 2 such targets) but modeled as its own type since it
      is unambiguously identifiable.

  (:UnresolvedPrincipal :Principal {key})
      Sentinel for the 77 rows with null source_node (all label=0) —
      unchanged rationale from v2.

EDGES — one CREATE per CSV row (still a multigraph; the v2 fix against
MERGE-based edge deduplication — which silently collapsed 2,900 rows into
1,017 edges — still applies and is preserved here).

  Relation TYPE is resolved deterministically from edge_type via
  privilege_features.resolve_relation_type: ASSUMES (STS role-assumption
  actions — the edge that drives Role-canonicalization) or one of AWS's
  own access-level categories (LIST / READ / WRITE / TAGGING /
  PERMISSIONS_MANAGEMENT), or UNKNOWN_ACTION when unresolvable. See
  privilege_features.py's module docstring for full sourcing and the
  verified 92% coverage / 86% unambiguous numbers.

  Edge properties carry: log_id, edge_type (verbatim action name),
  relation, access_level, is_privilege_escalation_technique (the narrower,
  Rhino-cited technique list — kept distinct from the broader access_level
  category; see privilege_features.py), hop_count, privilege_gain (only
  meaningful for hop_count==2 edges), abnormal_path_frequency (structural
  rarity only — never label-derived, see privilege_features.py),
  action_global_frequency, is_attack (ground truth, renamed from the
  CSV's `label` for the same reason as v2).

A CYPHER IMPLEMENTATION NOTE
─────────────────────────────────────────────────────────────────────────
Vanilla Cypher does not allow a parameterized node label or relationship
type (`MERGE (n:$label ...)` is not valid Cypher) — only the APOC plugin's
`apoc.create.relationship`/`apoc.merge.node` procedures support that. To
keep this pipeline runnable on a stock Neo4j instance (no assumption that
APOC is installed), node-label and relationship-type dispatch is done in
Python: a small dict of near-identical static Cypher templates, selected
by the row's resolved type before `session.run(...)`. This is more
verbose than a single parameterized query but has zero extra Neo4j
plugin dependencies.

WHAT THIS VERSION DOES NOT DO (see privilege_features.py for full detail)
─────────────────────────────────────────────────────────────────────────
- No :Session nodes are populated (schema documented, not fabricated —
  see privilege_features.SessionNodeBuilder).
- No session_duration / temporal_gap_between_hops properties exist
  anywhere (no timestamp column in this dataset).
- Multi-hop chains top out at 2 hops empirically on this data (verified);
  the extraction code is general and not artificially capped.

Run:
    pip install neo4j pandas networkx policy_sentry
    python3 neo4j_graph_builder.py
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass

import pandas as pd
from neo4j import GraphDatabase

import privilege_features as pf

# ── Connection ────────────────────────────────────────────────────────────────
URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "test1234"

CSV_PATH = "./cloudtrail_structural.csv"

REQUIRED_COLUMNS = {"log_id", "source_node", "target_node", "edge_type", "label"}
UNRESOLVED_PRINCIPAL = "UNRESOLVED_PRINCIPAL"


# ══════════════════════════════════════════════════════════════════════════
# Principal / Target string parsing (unchanged rules from v2 — reproduced
# here rather than imported, since privilege_features.py deliberately has
# no dependency on this exact CSV's raw-string conventions; it only
# consumes the already-parsed GraphNodeKey / access-level outputs)
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PrincipalInfo:
    arn: str
    name: str
    principal_type: str  # IAMUser | AssumedRole | AWSServiceLinkedRole | Unresolved


def parse_principal(source_node) -> PrincipalInfo:
    if pd.isna(source_node) or str(source_node).strip() == "":
        return PrincipalInfo(arn=UNRESOLVED_PRINCIPAL, name=UNRESOLVED_PRINCIPAL, principal_type="Unresolved")
    arn = str(source_node)
    if ":user/" in arn:
        return PrincipalInfo(arn=arn, name=arn.split(":user/")[-1], principal_type="IAMUser")
    if ":assumed-role/" in arn:
        role_name = arn.split(":assumed-role/")[-1].split("/")[0]
        ptype = "AWSServiceLinkedRole" if role_name.startswith("AWSServiceRoleFor") else "AssumedRole"
        return PrincipalInfo(arn=arn, name=role_name, principal_type=ptype)
    if ":role/" in arn:
        role_name = arn.split(":role/")[-1]
        ptype = "AWSServiceLinkedRole" if role_name.startswith("AWSServiceRoleFor") else "AssumedRole"
        return PrincipalInfo(arn=arn, name=role_name, principal_type=ptype)
    return PrincipalInfo(arn=arn, name=arn, principal_type="Unresolved")


_ARN_RE        = re.compile(r"^arn:aws:([a-zA-Z0-9\-]+):[^:]*:[^:]*:(.*)$")
_DOMAIN_SUFFIX = ".amazonaws.com"
_EC2_ID_RE     = re.compile(r"^i-[0-9a-z]{8,}$")
_EC2_TYPE_RE   = re.compile(r"^[a-z][0-9a-z]*\.[a-z0-9]+$")  # e.g. p2.xlarge, t2.micro
_REGION_RE     = re.compile(r"^[a-z]{2}-[a-z]+-\d$")
_ACCOUNT_ID_RE = re.compile(r"^\d{12}$")


@dataclass(frozen=True)
class TargetInfo:
    value: str
    resource_type: str
    service: str
    resolved: bool


def _service_from_domain(value: str):
    if not value.endswith(_DOMAIN_SUFFIX):
        return None
    prefix = value[: -len(_DOMAIN_SUFFIX)]
    if not prefix:
        return None
    labels = [lbl for lbl in prefix.split(".") if lbl]
    candidates = [lbl for lbl in labels if not _ACCOUNT_ID_RE.match(lbl) and not _REGION_RE.match(lbl)]
    return candidates[0] if len(candidates) == 1 else None


def parse_target(target_node) -> TargetInfo:
    value = str(target_node)
    m = _ARN_RE.match(value)
    if m:
        return TargetInfo(value=value, resource_type="arn-resource", service=m.group(1), resolved=True)
    if value.endswith(_DOMAIN_SUFFIX):
        service = _service_from_domain(value)
        return TargetInfo(value=value, resource_type="service-domain",
                           service=service or "unresolved", resolved=service is not None)
    if _EC2_ID_RE.match(value):
        return TargetInfo(value=value, resource_type="ec2-instance-id", service="unresolved", resolved=False)
    if _EC2_TYPE_RE.match(value):
        return TargetInfo(value=value, resource_type="ec2-instance-type", service="unresolved", resolved=False)
    if _REGION_RE.match(value):
        return TargetInfo(value=value, resource_type="aws-region", service="unresolved", resolved=False)
    return TargetInfo(value=value, resource_type="opaque", service="unresolved", resolved=False)


READ_ONLY_PREFIXES = ("Get", "List", "Describe", "Head", "Lookup", "Scan", "Query", "Search", "Check", "Validate")


def is_read_only(event_name: str) -> bool:
    return str(event_name).startswith(READ_ONLY_PREFIXES)


# Rhino Security Labs' published IAM privilege-escalation TECHNIQUE list —
# narrower and more specific than the broad "Permissions management"
# access-level category from policy_sentry (kept as a separate property;
# see module docstring).
PRIVILEGE_ESCALATION_TECHNIQUES = {
    "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
    "AttachUserPolicy", "AttachGroupPolicy", "AttachRolePolicy",
    "PutUserPolicy", "PutGroupPolicy", "PutRolePolicy",
    "AddUserToGroup", "UpdateAssumeRolePolicy",
    "CreatePolicyVersion", "SetDefaultPolicyVersion", "PassRole",
}


def compute_attacker_principals(df: pd.DataFrame) -> set:
    attacker_rows = df[df["label"] == 1]
    names = attacker_rows["source_node"].dropna().apply(lambda a: parse_principal(a).name)
    return set(names.unique())


# ══════════════════════════════════════════════════════════════════════════
# Cypher templates — node label / relationship type dispatched in Python
# (see module docstring for why this avoids requiring APOC)
# ══════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:User) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Role) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:UnresolvedPrincipal) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Service) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Resource) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Policy) REQUIRE n.key IS UNIQUE",
]

_NODE_MERGE_TEMPLATES = {
    "User": "MERGE (n:User:Principal {key: $key}) SET n += $props",
    "Role": "MERGE (n:Role:Principal {key: $key}) SET n += $props",
    "UnresolvedPrincipal": "MERGE (n:UnresolvedPrincipal:Principal {key: $key}) SET n += $props",
    "Service": "MERGE (n:Service:Target {key: $key}) SET n += $props",
    "Resource": "MERGE (n:Resource:Target {key: $key}) SET n += $props",
    "Policy": "MERGE (n:Policy:Target {key: $key}) SET n += $props",
}

_RELATION_TYPES = ["ASSUMES", "LIST", "READ", "WRITE", "TAGGING", "PERMISSIONS_MANAGEMENT", "UNKNOWN_ACTION"]

_EDGE_CREATE_TEMPLATES = {
    rel: f"""
        MATCH (src {{key: $src_key}}) MATCH (dst {{key: $dst_key}})
        CREATE (src)-[r:{rel} {{
            log_id: $log_id, edge_type: $edge_type, relation: $relation,
            access_level: $access_level, is_privilege_escalation_technique: $is_priv_esc,
            hop_count: $hop_count, privilege_gain: $privilege_gain,
            privilege_gain_defined: $privilege_gain_defined,
            abnormal_path_frequency: $abnormal_path_frequency,
            action_global_frequency: $action_global_frequency,
            is_attack: $is_attack
        }}]->(dst)
    """
    for rel in _RELATION_TYPES
}


def merge_node(session, node_key: pf.GraphNodeKey, props: dict):
    template = _NODE_MERGE_TEMPLATES[node_key.label]
    session.run(template, key=node_key.key, props=props)


def create_edge(session, relation: str, src_key: str, dst_key: str, **props):
    # NOTE: unrelated to the log_id migration — pre-existing bug found
    # during end-to-end verification. `relation` selects which Cypher
    # template to use, but the template also references it as the
    # $relation query parameter (`relation: $relation` — see
    # _EDGE_CREATE_TEMPLATES above); it must be forwarded here rather
    # than left for the caller to supply (the caller supplying it caused
    # a "multiple values for argument 'relation'" conflict against this
    # function's own positional `relation` parameter).
    session.run(_EDGE_CREATE_TEMPLATES[relation], src_key=src_key, dst_key=dst_key, relation=relation, **props)


# ══════════════════════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════════════════════

def build_graph():
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}")

    resolver = pf.ActionAccessLevelResolver()
    print(f"Action access-level resolver: {resolver.source}")

    principal_infos = df["source_node"].apply(parse_principal)
    target_infos     = df["target_node"].apply(parse_target)

    src_keys = [
        pf.node_key_for_principal(arn, info.principal_type, info.name)
        for arn, info in zip(df["source_node"], principal_infos)
    ]

    # Bare role/user names appearing as targets (not full ARNs) need to be
    # reconciled against names already resolved on the principal side --
    # see node_key_for_target's docstring for why this matters.
    known_role_names = {
        info.name for info in principal_infos
        if info.principal_type in ("AssumedRole", "AWSServiceLinkedRole")
    }
    known_user_names = {
        info.name for info in principal_infos if info.principal_type == "IAMUser"
    }
    dst_keys = [
        pf.node_key_for_target(t.value, t.resource_type, t.service,
                                known_role_names, known_user_names)
        for t in target_infos
    ]

    attacker_principals = compute_attacker_principals(df)
    action_freq = df["edge_type"].value_counts()

    # Build the structural graph once (shared source of truth for every
    # topology-derived feature — see privilege_features.py).
    # log_id is kept exactly as read from the CSV — a unique opaque STRING
    # identifier under the Feature Engine schema (e.g.
    # "synthetic_cloudtrail.csv:0"), not an integer. PrivilegePropagationGraph
    # never assumed int here (it just stores whatever it's given — see its
    # module/class docstrings), so no int() coercion was ever required by
    # that class; it was only ever done here, and is now removed.
    rows_for_graph = [
        {"log_id": lid, "source_key": sk, "target_key": dk, "edge_type": et, "label": int(lbl)}
        for lid, sk, dk, et, lbl in zip(df["log_id"], src_keys, dst_keys, df["edge_type"], df["label"])
    ]
    ppg = pf.PrivilegePropagationGraph(resolver).build_from_rows(rows_for_graph)
    edge_features = ppg.compute_all_edge_features().set_index("log_id")

    # Node-level topology stats (out/in-degree, fan-out/fan-in, role
    # transitions) computed once over the shared graph.
    node_out_degree      = collections.Counter()
    node_in_degree       = collections.Counter()
    node_unique_targets  = collections.defaultdict(set)
    node_unique_sources  = collections.defaultdict(set)
    node_unique_actions  = collections.defaultdict(set)
    for u, v, d in ppg.graph.edges(data=True):
        node_out_degree[u] += 1
        node_in_degree[v] += 1
        node_unique_targets[u].add(v)
        node_unique_sources[v].add(u)
        node_unique_actions[u].add(d["edge_type"])

    target_info_by_value = {t.value: t for t in target_infos}
    sensitivity_lookup = {}
    for n in ppg.graph.nodes:
        label, key = n
        if label in ("Service", "Resource", "Policy"):
            matching = target_info_by_value.get(key)
            svc = matching.service if matching else "unresolved"
            rtype = matching.resource_type if matching else "opaque"
            sensitivity_lookup[n] = pf.resource_sensitivity_score(svc, rtype)
        else:
            sensitivity_lookup[n] = -1  # principals aren't scored for sensitivity

    print(f"  {len(df):,} rows | {df['label'].sum()} labelled attack | "
          f"{ppg.graph.number_of_nodes()} nodes | {ppg.graph.number_of_edges()} edges")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        print("Creating constraints …")
        for cql in CONSTRAINTS:
            session.run(cql)
        print("Clearing existing graph …")
        session.run("MATCH (n) DETACH DELETE n")

        print("Ingesting nodes …")
        seen_nodes = set()
        for n in ppg.graph.nodes:
            if n in seen_nodes:
                continue
            seen_nodes.add(n)
            label, key = n
            props = {
                "out_degree": node_out_degree.get(n, 0),
                "in_degree": node_in_degree.get(n, 0),
                "unique_targets": len(node_unique_targets.get(n, set())),
                "unique_principals": len(node_unique_sources.get(n, set())),
                "unique_actions": len(node_unique_actions.get(n, set())),
                "role_transition_count": ppg.role_transition_count(n),
                "resource_sensitivity": sensitivity_lookup.get(n, -1),
                "distance_to_sensitive_resource": ppg.distance_to_sensitive_resource(n, sensitivity_lookup),
            }
            if label in ("User", "Role", "UnresolvedPrincipal"):
                props["is_known_attacker_identity"] = key in attacker_principals
            merge_node(session, pf.GraphNodeKey(label, key), props)

        print(f"Ingesting {len(df):,} typed edges …")
        for i, row in df.iterrows():
            src, dst = src_keys[i], dst_keys[i]
            relation = pf.resolve_relation_type(str(row["edge_type"]), resolver)
            # log_id is a unique opaque string (Feature Engine schema) — the
            # edge_features DataFrame is indexed by that same string (see
            # rows_for_graph above), so this is a plain, un-coerced lookup.
            feats = edge_features.loc[row["log_id"]]
            create_edge(
                session, relation,
                src_key=src.key, dst_key=dst.key,
                log_id=str(row["log_id"]), edge_type=str(row["edge_type"]),
                access_level=resolver.access_level(str(row["edge_type"])),
                is_priv_esc=str(row["edge_type"]) in PRIVILEGE_ESCALATION_TECHNIQUES,
                hop_count=int(feats["hop_count"]),
                privilege_gain=float(feats["privilege_gain"]),
                privilege_gain_defined=bool(feats["privilege_gain_defined"]),
                abnormal_path_frequency=float(feats["abnormal_path_frequency"]),
                action_global_frequency=int(action_freq[row["edge_type"]]),
                is_attack=int(row["label"]),
            )
            if (i + 1) % 500 == 0:
                print(f"  … {i+1:,} rows processed")

    driver.close()
    print(f"\n✅ Privilege Propagation Graph built: {len(seen_nodes)} nodes, {len(df)} typed edges.")
    print("\nSanity queries for Neo4j Browser (localhost:7474):")
    print("─" * 60)
    print("// The one verified real attack chain in this dataset")
    print("MATCH (u:User {key:'bert-jan'})-[a:ASSUMES]->(r:Role)-[x:READ]->(res:Resource)")
    print("WHERE x.is_attack = 1 RETURN u, a, r, x, res LIMIT 10\n")
    print("// All 2-hop (role-mediated) edges")
    print("MATCH (p)-[r]->(t) WHERE r.hop_count = 2 RETURN p.key, type(r), r.edge_type, t.key, r.is_attack\n")
    print("// Rarest structural patterns (abnormal_path_frequency, label-blind)")
    print("MATCH (p)-[r]->(t) RETURN p.key, type(r), t.key, r.abnormal_path_frequency")
    print("ORDER BY r.abnormal_path_frequency DESC LIMIT 20")


if __name__ == "__main__":
    build_graph()
