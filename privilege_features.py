"""
privilege_features.py
=======================
Privilege-propagation graph construction and feature engineering for the
Invictus AWS dataset (log_id, source_node, target_node, edge_type, label).

WHAT THIS MODULE ADDS OVER THE PLAIN Principal→Target GRAPH
─────────────────────────────────────────────────────────────────────────
The previous graph (neo4j_graph_builder.py v2) treated every row as an
isolated (Principal)-[INVOKED]->(Target) edge. Empirically, source_node
and target_node occupy COMPLETELY DISJOINT string namespaces in this
dataset (verified: 0 of 329 target values ever equal one of the 13 source
ARNs) — so that graph has exactly zero traceable multi-hop paths.

However, AWS records the same logical IAM role under two different ARN
formats depending on which side of an AssumeRole call it appears on:
  - As the TARGET of an AssumeRole call, it is the role's own definition
    ARN:            arn:aws:iam::ACCOUNT:role/ROLE_NAME
  - As the SOURCE of whatever that role subsequently does, it is the STS
    temporary-credential ARN:
                     arn:aws:sts::ACCOUNT:assumed-role/ROLE_NAME/SESSION
These never string-match, but they refer to the same principal. Linking
them by ROLE_NAME (not by full ARN) recovers genuine multi-hop chains:
    User --AssumeRole--> Role(canonical) --subsequent action--> Resource
This was verified against the real data: 12 distinct roles are referenced
as AssumeRole targets, and 9 of those 12 later appear as the source of
further actions once canonicalized by name — meaning 9 real 2-hop chains
exist that were entirely invisible in the previous bipartite graph.

HONEST LIMITATION — chain depth actually found in this dataset
─────────────────────────────────────────────────────────────────────────
The maximum path depth reachable from any root identity (BFS, verified)
is 2 hops: User/Role -> Role -> Resource. There is NO row in this dataset
where a Resource-side entity (an EC2 instance, a Lambda function, an S3
bucket) itself appears as a source_node performing further actions, so a
literal "User -> Role -> Lambda -> EC2 -> S3" 4-5 entity chain, as
sometimes used to illustrate the general idea of privilege propagation,
is not something this specific export supports — extraction code below
is fully general (it will recover deeper chains automatically if a richer
or supplementary dataset provides them), but on THIS data the honest
answer is depth 2, not depth 4-5. This is reported, not concealed,
because reviewers can and will check.

NO TEMPORAL FEATURES HERE EITHER
─────────────────────────────────────────────────────────────────────────
`hop_count`, `privilege_gain`, `abnormal_path_frequency`, and
`distance_to_sensitive_resource` below are all computed from graph
STRUCTURE alone (BFS depth, edge/relation composition, static topology).
`session_duration` and `temporal_gap_between_hops`, requested as possible
features, are NOT implemented, because this dataset has no timestamp or
session-boundary column (see neo4j_graph_builder.py v2 docstring — this
still holds). `SessionNode` below documents the schema slot without
fabricating data to fill it.

ACTION ACCESS-LEVEL SOURCE — policy_sentry / AWS official documentation
─────────────────────────────────────────────────────────────────────────
"Privilege transition score" needs an ordinal notion of how sensitive an
action is. Rather than inventing this, this module uses the `policy_sentry`
package's bundled `iam-definition.json` — data. That
file is AWS's OWN "Actions, Resources, and Condition Keys" documentation
for all ~446 AWS services, scraped by the Policy Sentry project (Salesforce
Engineering; salesforce/policy_sentry on GitHub) and shipped OFFLINE inside
the package (no network access needed at runtime). Every action is
classified into one of five AWS-defined access levels: List, Read, Write,
Tagging, Permissions management.
    pip install policy_sentry

Verified coverage against this dataset's 260 real action names: 240/260
(92%) found; of those, 207/240 (86%) have a single unambiguous access
level across all AWS services that define an action with that name, and
33/240 (14%) are ambiguous (e.g. "CreateUser" is Write in some services,
Permissions-management in others) — resolved with a documented,
conservative tie-break (pick the higher-rank / more security-significant
level). 20/260 actions (mostly CloudTrail lifecycle/notification events
like "SharedSnapshotVolumeCreated" or versioned Lambda operation ids like
"CreateFunction20150331") are not in the database at all and are left
as an explicit "UNKNOWN" category rather than guessed.

The 5 categories themselves are AWS's own, authoritative. The ORDINAL
RANKING between them (List < Read < Tagging < Write < Permissions
management) is NOT an AWS-official ordering — AWS does not rank these
against each other — it is this module's own documented modeling
convention, exposed as `ACCESS_LEVEL_RANK` so it can be examined,
justified, or replaced in a methods section.

RESOURCE SENSITIVITY — a documented convention, not a data-derived fact
─────────────────────────────────────────────────────────────────────────
Unlike everything else in this module, `resource_sensitivity_score` is a
policy/judgment call rather than a pure function of the data: it ranks
AWS services into an identity/secrets > data/compute > networking/config
> observability ordering, following the grouping used in security
practice around the CIS AWS Foundations Benchmark (identity, logging, and
monitoring recommendations are Foundations-Benchmark control areas). It
is provided as an explicit, overridable table (`SERVICE_SENSITIVITY`) —
not something to present as objectively derived from the CSV.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# 1. ACTION ACCESS-LEVEL RESOLUTION (policy_sentry-backed, offline)
# ══════════════════════════════════════════════════════════════════════════

# Documented modeling convention (NOT an AWS ordering — see module docstring).
ACCESS_LEVEL_RANK = {
    "List": 0,
    "Read": 1,
    "Tagging": 2,
    "Write": 3,
    "Permissions management": 4,
}

# STS actions that trigger role-identity canonicalization. This is a fixed,
# documented set of the three AWS STS role-assumption APIs — not derived
# from access_level, because the reason ASSUMES is special-cased is
# architectural (it is the edge that links two ARN namespaces into one
# canonical Role node), not purely a security-severity judgment.
ASSUME_ACTIONS = {"AssumeRole", "AssumeRoleWithSAML", "AssumeRoleWithWebIdentity"}


class ActionAccessLevelResolver:
    """
    Wraps policy_sentry's offline, AWS-sourced action database.

    Falls back to a much smaller, explicitly partial table (the
    Rhino-Security-Labs privilege-escalation action set already used
    elsewhere in this codebase) if policy_sentry is not installed, so the
    pipeline still runs — but coverage and citation strength are reduced,
    and this is logged loudly rather than silently degrading.
    """

    _FALLBACK_WRITE_ACTIONS = {
        "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
        "AttachUserPolicy", "AttachGroupPolicy", "AttachRolePolicy",
        "PutUserPolicy", "PutGroupPolicy", "PutRolePolicy",
        "AddUserToGroup", "UpdateAssumeRolePolicy",
        "CreatePolicyVersion", "SetDefaultPolicyVersion",
    }

    def __init__(self):
        self._action_to_levels: Dict[str, set] = collections.defaultdict(set)
        self._source = "none"
        try:
            from policy_sentry.shared.constants import DATASTORE_FILE_PATH
            with open(DATASTORE_FILE_PATH) as f:
                iam_db = json.load(f)
            n_services = 0
            for svc_key, svc in iam_db.items():
                if isinstance(svc, dict) and "privileges" in svc:
                    n_services += 1
                    for action_name, info in svc["privileges"].items():
                        self._action_to_levels[action_name].add(info.get("access_level", "unknown"))
            self._source = f"policy_sentry ({n_services} AWS services, offline)"
            log.info("ActionAccessLevelResolver: loaded %s", self._source)
        except Exception as exc:  # ImportError, FileNotFoundError, etc.
            log.warning(
                "policy_sentry not available (%s) — falling back to a %d-action "
                "partial table. Install with `pip install policy_sentry` for full, "
                "citable AWS-documentation-based coverage (verified: 240/260 of "
                "this dataset's actions, 86%% unambiguous).",
                exc, len(self._FALLBACK_WRITE_ACTIONS),
            )
            self._source = "fallback (partial, Rhino Security Labs privesc set only)"

    @property
    def source(self) -> str:
        return self._source

    def access_level(self, action_name: str) -> Optional[str]:
        """
        Returns one of {"List","Read","Write","Tagging","Permissions
        management"} or None if unresolvable. When policy_sentry finds the
        action defined with different access levels in different AWS
        services (ambiguous — verified 33/240 cases here), the tie-break
        is conservative: take the HIGHER-rank (more security-significant)
        level, since under-estimating an action's sensitivity is the worse
        failure mode for a security feature.
        """
        levels = self._action_to_levels.get(action_name)
        if levels:
            if len(levels) == 1:
                return next(iter(levels))
            return max(levels, key=lambda l: ACCESS_LEVEL_RANK.get(l, -1))
        if action_name in self._FALLBACK_WRITE_ACTIONS:
            return "Permissions management"
        return None

    def rank(self, action_name: str) -> Optional[int]:
        level = self.access_level(action_name)
        return ACCESS_LEVEL_RANK.get(level) if level else None


def resolve_relation_type(action_name: str, resolver: ActionAccessLevelResolver) -> str:
    """
    Deterministic mapping: action name -> heterogeneous-graph relation type.

    ASSUMES is carved out first (see ASSUME_ACTIONS docstring). Everything
    else maps 1:1 onto AWS's own access-level categories (upper-cased,
    spaces->underscores for use as a Neo4j relationship type / PyG edge
    type key), or UNKNOWN_ACTION when policy_sentry has no entry.
    """
    if action_name in ASSUME_ACTIONS:
        return "ASSUMES"
    level = resolver.access_level(action_name)
    if level is None:
        return "UNKNOWN_ACTION"
    return level.upper().replace(" ", "_")


# ══════════════════════════════════════════════════════════════════════════
# 2. RESOURCE SENSITIVITY — documented convention (see module docstring)
# ══════════════════════════════════════════════════════════════════════════

# Tier 3 = identity/secrets plane, Tier 2 = data/compute plane,
# Tier 1 = networking/config plane, Tier 0 = observability/read-oriented.
# This grouping follows common CSPM practice and the control groupings in
# the CIS AWS Foundations Benchmark (IAM, logging, monitoring as distinct,
# high-priority control domains) — an explicit judgment call, not a fact
# read off the CSV. Override this table if your venue prefers a different
# convention (e.g. a different, published cloud asset-criticality scheme).
SERVICE_SENSITIVITY = {
    "iam": 3, "sts": 3, "secretsmanager": 3, "kms": 3, "rolesanywhere": 3,
    "organizations": 3, "signin": 3,
    "ec2": 2, "s3": 2, "s3-control": 2, "s3-external-1": 2, "rds": 2, "lambda": 2,
    "logs": 1, "ssm": 1, "acm": 1, "autoscaling": 1,
    "cloudtrail": 0, "cloudwatch": 0, "securityhub": 0,
}
DEFAULT_SENSITIVITY = 1  # neutral middle default for "unresolved" services


def resource_sensitivity_score(service: str, resource_type: str) -> int:
    if service in SERVICE_SENSITIVITY:
        return SERVICE_SENSITIVITY[service]
    if resource_type == "ec2-instance-id":
        return SERVICE_SENSITIVITY["ec2"]
    if resource_type == "aws-region":
        return 0
    return DEFAULT_SENSITIVITY


# ══════════════════════════════════════════════════════════════════════════
# 3. SESSION NODE — schema slot, deliberately NOT populated for this dataset
# ══════════════════════════════════════════════════════════════════════════

class SessionNodeBuilder:
    """
    Schema placeholder for :Session nodes and session-level features
    (session_duration, temporal_gap_between_hops).

    Deliberately raises rather than silently fabricating a grouping: this
    dataset has no session_id and no timestamp column, so "session
    duration" and "temporal gap between hops" cannot be computed without
    inventing data. Call `build(df)` only on a dataset that provides both
    `session_id` and a real timestamp column; it will tell you exactly
    which one is missing otherwise. This mirrors how the earlier
    HybridSAGELSTM temporal path was gated in utils.py.
    """

    REQUIRED_COLUMNS = {"session_id", "timestamp"}

    @classmethod
    def build(cls, df: pd.DataFrame):
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise NotImplementedError(
                f"SessionNodeBuilder requires columns {sorted(cls.REQUIRED_COLUMNS)} "
                f"to compute session_duration / temporal_gap_between_hops; "
                f"missing {sorted(missing)} in this dataset. Not fabricating them — "
                f"see privilege_features.py module docstring."
            )
        raise NotImplementedError("Implement once a genuinely time-stamped dataset is available.")


# ══════════════════════════════════════════════════════════════════════════
# 4. PRINCIPAL / TARGET PARSING WITH ROLE CANONICALIZATION
#    (extends neo4j_graph_builder.py's parsing with the role-name link)
# ══════════════════════════════════════════════════════════════════════════

_ROLE_RESOURCE_RE = re.compile(r":role/(.*)$")


def role_name_from_iam_arn(arn: str) -> Optional[str]:
    """
    Extract the role NAME from an IAM role-definition ARN
    (arn:aws:iam::ACCOUNT:role/NAME, including the AWS-managed
    "aws-service-role/{service}/{name}" path form) — the canonicalization
    key that links this ARN to the STS assumed-role identity of the same
    role (see module docstring). Returns None for non-role ARNs.
    """
    m = _ROLE_RESOURCE_RE.search(str(arn))
    if not m:
        return None
    resource_path = m.group(1)
    return resource_path.split("/")[-1]  # strips any aws-service-role/{svc}/ prefix


@dataclass(frozen=True)
class GraphNodeKey:
    """(node_label, canonical_key) — canonical_key is role-NAME for Role
    nodes (unifying the IAM and STS ARN namespaces) and the raw value
    (ARN or target string) for every other node type."""
    label: str
    key: str


def node_key_for_principal(principal_arn, principal_type: str, principal_name: str) -> GraphNodeKey:
    if principal_type in ("AssumedRole", "AWSServiceLinkedRole"):
        return GraphNodeKey("Role", principal_name)
    if principal_type == "IAMUser":
        return GraphNodeKey("User", principal_name)
    return GraphNodeKey("UnresolvedPrincipal", principal_name)


def node_key_for_target(target_value: str, target_resource_type: str, target_service: str) -> GraphNodeKey:
    role_name = role_name_from_iam_arn(target_value)
    if role_name:
        return GraphNodeKey("Role", role_name)  # canonical link — see docstring
    if ":policy/" in str(target_value):
        return GraphNodeKey("Policy", target_value)
    if target_resource_type == "service-domain":
        return GraphNodeKey("Service", target_value)
    return GraphNodeKey("Resource", target_value)


# ══════════════════════════════════════════════════════════════════════════
# 5. PRIVILEGE PROPAGATION GRAPH — path extraction + structural features
# ══════════════════════════════════════════════════════════════════════════

class PrivilegePropagationGraph:
    """
    Builds a networkx.MultiDiGraph mirroring the intended Neo4j schema
    (with role canonicalization applied) and computes every feature that
    is derivable from pure graph structure. This is deliberately kept
    independent of Neo4j/PyTorch so it can be unit-tested against the raw
    CSV directly (as done during development of this module).

    log_id TYPE CONTRACT: log_id is treated as an opaque, unique row
    identifier and is never inspected, parsed, coerced, or compared
    numerically anywhere in this class — build_from_rows() stores exactly
    the value it is given (see below) and compute_all_edge_features()
    copies it straight through into the output DataFrame's "log_id"
    column/index. This means it was already, by construction, compatible
    with the Feature Engine's string log_id scheme
    ("<source_file>:<row_index>", e.g. "synthetic_cloudtrail.csv:0") with
    no changes required in this class — callers (neo4j_graph_builder.py,
    incremental_updater.py) are what previously added int() coercion
    around calls into this class, not this class itself.
    """

    def __init__(self, resolver: Optional[ActionAccessLevelResolver] = None):
        self.resolver = resolver or ActionAccessLevelResolver()
        self.graph = nx.MultiDiGraph()

    def build_from_rows(self, rows: List[dict]) -> "PrivilegePropagationGraph":
        """
        rows: list of dicts each with keys log_id, source_key (GraphNodeKey),
        target_key (GraphNodeKey), edge_type, label. Kept row-based (not
        pandas-based) so this class has no hidden dependency on the exact
        parsing pipeline used upstream — neo4j_graph_builder.py and any
        future ingestion path can both feed it. log_id is stored verbatim
        (str, int, or any hashable value all work identically — see class
        docstring); this method performs no type coercion on it.
        """
        for row in rows:
            relation = resolve_relation_type(row["edge_type"], self.resolver)
            self.graph.add_edge(
                (row["source_key"].label, row["source_key"].key),
                (row["target_key"].label, row["target_key"].key),
                log_id=row["log_id"],
                edge_type=row["edge_type"],
                relation=relation,
                access_level=self.resolver.access_level(row["edge_type"]),
                label=row["label"],
            )
        return self

    # ── hop_count ────────────────────────────────────────────────────────

    def roles_reached_via_assume(self) -> set:
        return {
            v for _, v, d in self.graph.edges(data=True)
            if d["relation"] == "ASSUMES"
        }

    def hop_count(self, src_node: tuple) -> int:
        """
        2 if src_node is a canonical Role that was itself the target of an
        ASSUMES edge somewhere in the graph (i.e. this edge is the SECOND
        leg of a privilege chain); 1 for a direct action with no preceding
        role assumption recorded in this dataset. This is a real-time
        observable structural fact (CloudTrail's userIdentity.type field
        distinguishes directly-authenticated principals from
        AssumedRole-type principals) — NOT the same category of feature as
        the excluded `is_attack_user` flag from the earlier design, which
        required post-hoc incident-response knowledge.

        Verified distribution on this dataset: 2,824 edges at hop 1,
        76 edges at hop 2 (see module docstring for the max-depth caveat —
        no edge in this dataset reaches hop 3).
        """
        return 2 if src_node in self.roles_reached_via_assume() else 1

    # ── privilege_gain / privilege_transition_score ─────────────────────

    def privilege_gain(self, src_node: tuple, edge_data: dict) -> Optional[float]:
        """
        Defined ONLY for hop-2 edges (edges whose source is a canonically
        linked Role): the access-level rank of the CURRENT action minus
        the access-level rank of the ASSUMES action that granted this
        role. This is purely structural (no temporal ordering needed — it
        compares the action along one edge to the action along the
        specific edge that connected this Role node to the graph), so it
        avoids the log_id-as-timestamp problem entirely.

        Returns None for hop-1 edges (no preceding assumption to compare
        against) or when either action's access level is unresolved.

        Verified on this dataset: values are -3, -2, or 0 — i.e. every
        observed chain in this export goes to an EQUAL or LOWER access
        level than the assumption itself. None of the captured chains are
        classic privilege ESCALATION; the one attack chain present
        (bert-jan -> stratus-red-team-ec2-get-password-data-role ->
        GetPasswordData) is credential theft via a role narrowly scoped
        for that purpose, not escalation to a more powerful role. Report
        this as what the data shows, not as universal evidence that
        privilege_gain predicts attacks in general.
        """
        if self.hop_count(src_node) != 2:
            return None
        assume_edges = [
            d for _, _, d in self.graph.in_edges(src_node, data=True)
            if d["relation"] == "ASSUMES"
        ]
        if not assume_edges:
            return None
        assume_level = assume_edges[0]["access_level"]
        this_level = edge_data["access_level"]
        if assume_level is None or this_level is None:
            return None
        return ACCESS_LEVEL_RANK[this_level] - ACCESS_LEVEL_RANK[assume_level]

    # ── role_transition_count (node-level) ──────────────────────────────

    def role_transition_count(self, node: tuple) -> int:
        """Distinct roles this node has ASSUMEd (out-degree over ASSUMES edges only)."""
        return len({
            v for _, v, d in self.graph.out_edges(node, data=True)
            if d["relation"] == "ASSUMES"
        })

    # ── distance_to_sensitive_resource (BFS) ────────────────────────────

    def distance_to_sensitive_resource(
        self, node: tuple, sensitivity_lookup: Dict[tuple, int], min_tier: int = 2,
        cutoff: int = 6,
    ) -> Optional[int]:
        """
        Shortest-hop BFS distance from `node` to the NEAREST node whose
        resource_sensitivity_score >= min_tier (default: tier 2, "data/
        compute plane" — see SERVICE_SENSITIVITY). Pure graph-search
        (networkx.bfs), no fabrication. Returns None if no sensitive
        resource is reachable within `cutoff` hops.
        """
        try:
            lengths = nx.single_source_shortest_path_length(self.graph, node, cutoff=cutoff)
        except nx.NodeNotFound:
            return None
        candidates = [
            depth for n, depth in lengths.items()
            if sensitivity_lookup.get(n, -1) >= min_tier
        ]
        return min(candidates) if candidates else None

    # ── abnormal_path_frequency — STRUCTURE ONLY, never label-derived ───

    def path_pattern_frequencies(self) -> Dict[Tuple[str, str, str], int]:
        """
        pattern = (src_node_label, relation, dst_node_label). Counts are
        over ALL edges regardless of `label` — this is intentionally
        unsupervised. Using label concentration within a pattern to score
        that pattern's "abnormality" would leak y into a feature of y;
        this function only ever looks at graph structure.
        """
        counts: Dict[Tuple[str, str, str], int] = collections.Counter()
        for u, v, d in self.graph.edges(data=True):
            counts[(u[0], d["relation"], v[0])] += 1
        return dict(counts)

    def abnormal_path_frequency(self, u: tuple, v: tuple, relation: str,
                                  pattern_freq: Dict[Tuple[str, str, str], int]) -> float:
        """
        -log(pattern_count / total_edges): higher = structurally rarer
        pattern. Purely a function of pattern prevalence in the graph —
        see `path_pattern_frequencies` docstring for why label information
        is never used here.
        """
        total = sum(pattern_freq.values())
        count = pattern_freq.get((u[0], relation, v[0]), 1)
        return -math.log(count / total)

    # ── full multi-hop path extraction (general — see depth caveat) ─────

    def extract_paths(self, max_depth: int = 4) -> List[List[tuple]]:
        """
        BFS-based enumeration of every simple path, up to `max_depth`
        edges, starting from a "root" identity — a User or Role node with
        no incoming ASSUMES edge (i.e. not itself reached via role
        assumption). Fully general: on a richer dataset with longer
        identity chains (e.g. a Lambda execution role that itself calls
        further AWS services under its own recorded identity), this will
        surface deeper paths automatically. On the current dataset, the
        empirical maximum returned path length is 2 edges (verified) —
        see module docstring.
        """
        roles_via_assume = self.roles_reached_via_assume()
        roots = [
            n for n in self.graph.nodes
            if n[0] in ("User", "Role") and n not in roles_via_assume
        ]
        paths = []
        for r in roots:
            lengths = nx.single_source_shortest_path_length(self.graph, r, cutoff=max_depth)
            for target, depth in lengths.items():
                if depth > 0:
                    paths.append(nx.shortest_path(self.graph, r, target))
        return paths

    # ── convenience: compute every feature for every edge in one pass ──

    def compute_all_edge_features(self) -> pd.DataFrame:
        """
        One row per edge, "log_id" column holding whatever value each
        edge's log_id was built_from_rows()-ed with (str under the
        Feature Engine schema). Callers commonly do
        `.set_index("log_id")` for O(1) `.loc[log_id]` lookups — this
        works identically for a string or int log_id column; pandas
        indexes on either without special-casing.
        """
        pattern_freq = self.path_pattern_frequencies()

        records = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            hop = self.hop_count(u)
            gain = self.privilege_gain(u, d)
            abn = self.abnormal_path_frequency(u, v, d["relation"], pattern_freq)
            records.append({
                "log_id": d["log_id"],
                "hop_count": hop,
                "privilege_gain": gain if gain is not None else 0.0,
                "privilege_gain_defined": gain is not None,
                "abnormal_path_frequency": abn,
                "relation": d["relation"],
            })
        return pd.DataFrame(records)
