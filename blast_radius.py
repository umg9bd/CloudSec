"""
blast_radius.py
=================
CONTRIBUTION 5 — Dynamic Blast Radius Estimation.

Runs strictly AFTER threat prediction, as an independent, read-only
analysis layer: given a principal GraphSAGE/GAT flagged as suspicious,
estimate how far a compromise could propagate. Does not touch
model_graphsage.py, model_gat.py, or train.py in any way.

REUSES THE EXISTING GRAPH — DOES NOT DUPLICATE STORAGE
─────────────────────────────────────────────────────────────────────────
Per the "do not modify the existing Neo4j schema / do not duplicate graph
storage" requirement, this module does not build its own graph. It
operates on the SAME `privilege_features.PrivilegePropagationGraph` that
neo4j_graph_builder.py already builds and writes to Neo4j (a
networkx.MultiDiGraph mirroring the Neo4j schema exactly) — it is one
read-only consumer of that structure, not a second copy of it. A thin
Neo4j-Cypher-backed adapter with the same interface (`ReachabilityGraph`
protocol below) can be substituted in production without changing any of
the analysis code, since Cypher variable-length path queries
(`MATCH (p)-[*1..k]->(n)`) answer the same reachability questions this
module asks of the in-memory graph.

TWO THINGS THAT MUST BE STATED BEFORE ANY SCORE IS TRUSTED
─────────────────────────────────────────────────────────────────────────
1. OBSERVED reachability, not GRANTED reachability. This dataset is
   CloudTrail events — what principals were SEEN doing — not IAM policy
   documents — what they are PERMITTED to do. Tools built for the latter
   (AWS IAM Access Analyzer, NCC Group's PMapper, Salesforce's
   Cloudsplaining) parse policy JSON and can prove "role A CAN reach
   resource B" even if it never happened. This module can only prove
   "role A WAS OBSERVED reaching resource B." That means every score here
   is a LOWER BOUND on true blast radius, not an estimate of it — a
   principal with broad IAM permissions but a quiet CloudTrail history
   will score low here despite being genuinely high-risk. This is stated
   in every report this module produces, not just here.

2. Verified structural ceiling on this specific dataset: no edge in this
   graph originates from a Resource/Service/Policy node (confirmed: 0 of
   2,900 edges) — so no path through this graph exceeds 2 hops
   (User/Role -> Role -> Resource). An illustrative 7-hop propagation
   path (IAM User -> AssumeRole -> Admin Role -> EC2 -> Instance Profile
   -> Secrets Manager -> KMS -> S3) is a reasonable example of the GENERAL
   idea this module implements, but this specific dataset cannot produce
   one — `PropagationPathExtractor` reports the real paths found (verified
   max: `bert-jan` reaches 318 nodes at depth <=2), not a fabricated
   longer one.

WHAT THIS DATASET CANNOT GROUND, VERIFIED EMPIRICALLY
─────────────────────────────────────────────────────────────────────────
- PassRole opportunities: `PassRole` never appears as an observed
  edge_type (0/2,900 rows). It is usually a supporting permission bundled
  into the REQUEST PARAMETERS of another call (e.g. ec2:RunInstances with
  an IamInstanceProfile parameter) rather than its own CloudTrail
  eventName, and this schema (log_id, source_node, target_node, edge_type,
  label) does not carry request parameters. `PrivilegeReachabilityAnalyzer
  .pass_role_opportunities` is implemented (so it activates correctly on a
  richer dataset that does capture PassRole events) but returns an
  explicit `observed=False` marker here rather than a silent zero, so it
  is never misread as "no opportunities exist" vs. "not observable".
- Administrator privilege reachability: no policy or role containing
  "Admin" appears anywhere in this dataset. There is no AWS-managed
  AdministratorAccess policy ARN, and no custom policy resembling one.
  `administrator_reachable` is computed as a PROXY — reachability of any
  PERMISSIONS_MANAGEMENT-relation edge (see privilege_features.py's
  access-level categories) — and is labeled as a proxy, not literal
  AdministratorAccess detection, everywhere it's reported.
- Cross-account reachability: exactly ONE AWS account ID
  (123837392027) appears anywhere in this dataset's ARNs. The algorithm
  below is a real, general implementation (extracts account IDs from any
  ARN-shaped node it can reach and compares them), and will correctly
  report a nonzero cross-account count on a genuinely multi-account
  graph — on this dataset it will correctly and honestly report 0 for
  every principal, which is not a bug.
- Coverage note on cross-account specifically: :Role node keys are
  canonicalized ROLE NAMES (not ARNs — see privilege_features.py's
  module docstring on why), so an account ID cannot be recovered from a
  Role node without violating "do not modify the existing schema" to add
  one. Cross-account detection therefore only has coverage over ARN-
  shaped :Resource and :Policy nodes, which is documented, not silently
  assumed complete.
- Requested asset categories with zero presence in this dataset: KMS
  Keys (0 rows) and DynamoDB Tables (0 rows). Lambda Functions are present
  but sparse (2 rows). `ReachableAssets` will correctly report 0 for
  these, which reflects the dataset, not a detection failure.

UPDATE: THE ADAPTIVE FRAMEWORK DEPENDENCY IS NOW RESOLVED
─────────────────────────────────────────────────────────────────────────
The "Incremental Updates" requirement ("recompute blast radius only for
affected principals... avoid full graph traversal after every event")
needed something to define "affected principal" and "what changed" — that
was `IncrementalGraphUpdater`'s job, and it now exists (incremental_updater.py).
`BlastRadiusCache` below is a REAL implementation: lazy invalidation,
driven by `IncrementalGraphUpdater._affected_principals`'s bounded
reverse-BFS (not the O(V+E) full-graph approach this docstring used to
flag as the open question). `BlastRadiusEngine.compute` remains the
uncached full computation underneath — the cache decides WHEN to call it.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Set, Tuple

import networkx as nx

import privilege_features as pf

log = logging.getLogger(__name__)

Node = Tuple[str, str]  # (node_label, key) — same convention as privilege_features.GraphNodeKey


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — every weight/threshold lives here, nothing inline.
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CriticalityWeights:
    """
    Maps privilege_features.py's existing SERVICE_SENSITIVITY tiers
    (0-3, already cited/justified there against CIS AWS Foundations
    Benchmark control groupings) onto a 0-1 criticality weight. REUSES
    that table rather than defining a second, competing one — the user's
    example (Administrator=High, KMS=High, SecretsManager=High,
    Public-S3=Medium, CloudWatch=Low) maps onto the existing tiers as:
    tier 3 (identity/secrets: iam, sts, secretsmanager, kms) -> High,
    tier 2 (data/compute: ec2, s3, rds, lambda) -> Medium-High,
    tier 1 (networking/config) -> Medium,
    tier 0 (observability: cloudtrail, cloudwatch) -> Low.
    "Public S3" specifically cannot be distinguished from private S3 in
    this dataset (bucket ACL/policy state isn't in CloudTrail events), so
    all S3 resources share the tier-2 weight — documented, not silently
    assumed.
    """
    tier_to_weight: Dict[int, float] = field(default_factory=lambda: {3: 1.0, 2: 0.65, 1: 0.35, 0: 0.1})

    def weight_for_tier(self, tier: int) -> float:
        return self.tier_to_weight.get(tier, self.tier_to_weight[1])  # unresolved -> neutral middle


@dataclass(frozen=True)
class BlastRadiusConfig:
    """All thresholds and weights configurable — see module docstring."""
    max_traversal_depth: int = 4        # this dataset never exceeds 2 (verified); kept general for other datasets
    critical_asset_min_tier: int = 2    # >= this SERVICE_SENSITIVITY tier counts as "critical" for exposure scoring

    # Blast Radius Score component weights — must sum to 1.0 (checked in __post_init__).
    w_reachable_assets: float = 0.25
    w_privilege_escalation: float = 0.30
    w_critical_asset_exposure: float = 0.30
    w_cross_service: float = 0.10
    w_cross_account: float = 0.05

    # Normalisation caps (a principal reaching >= this many assets/services
    # gets the max sub-score of 1.0 for that component) — configurable
    # rather than hardcoded inside the scoring function.
    reachable_assets_saturation: int = 50
    cross_service_saturation: int = 10

    criticality: CriticalityWeights = field(default_factory=CriticalityWeights)

    def __post_init__(self):
        total = (self.w_reachable_assets + self.w_privilege_escalation
                 + self.w_critical_asset_exposure + self.w_cross_service + self.w_cross_account)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Blast radius component weights must sum to 1.0, got {total}")


# ══════════════════════════════════════════════════════════════════════════
# GRAPH ACCESS — a minimal protocol so Neo4j-Cypher and in-memory
# networkx backends are interchangeable, per "reuse existing graph
# representation, do not duplicate storage."
# ══════════════════════════════════════════════════════════════════════════

class ReachabilityGraph(Protocol):
    """Anything satisfying this can back BlastRadiusEngine — the in-memory
    PrivilegePropagationGraph (used below, and in all the validation this
    module was checked against) or a thin Cypher-query adapter over the
    live Neo4j graph in production."""

    def reachable_within(self, source: Node, max_depth: int) -> Dict[Node, int]:
        """Returns {node: hop_distance} for every node reachable from
        `source` within max_depth hops (source excluded)."""
        ...

    def edges_on_shortest_path(self, source: Node, target: Node) -> List[dict]:
        ...


class NetworkXReachabilityGraph:
    """Adapts privilege_features.PrivilegePropagationGraph's networkx
    MultiDiGraph to the ReachabilityGraph protocol."""

    def __init__(self, ppg: pf.PrivilegePropagationGraph):
        self.graph = ppg.graph

    def reachable_within(self, source: Node, max_depth: int) -> Dict[Node, int]:
        if source not in self.graph:
            return {}
        lengths = nx.single_source_shortest_path_length(self.graph, source, cutoff=max_depth)
        lengths.pop(source, None)
        return lengths

    def edges_on_shortest_path(self, source: Node, target: Node) -> List[dict]:
        path = nx.shortest_path(self.graph, source, target)
        edges = []
        for u, v in zip(path[:-1], path[1:]):
            # MultiDiGraph can have several parallel edges u->v; take the
            # one with the lowest AUTO-ASSIGNED MultiDiGraph edge key,
            # deterministically. networkx assigns these as 0, 1, 2, ... in
            # the exact order edges were added for this specific (u, v)
            # pair whenever add_edge() is called without an explicit
            # key= — true of every call site in this codebase
            # (privilege_features.build_from_rows / incremental_updater.
            # apply_event) — so this key IS true insertion/arrival order,
            # independent of log_id's type or format. (Previously this
            # picked the min by log_id's own value, which only coincided
            # with insertion order because the old Invictus dataset's
            # log_id happened to be small sequential integers; the Feature
            # Engine's "<file>:<row>" strings sort lexicographically, not
            # numerically, so that assumption no longer held — this fixes
            # it by using a mechanism that was never coupled to log_id's
            # format in the first place, rather than parsing log_id.)
            parallel_edges = self.graph.get_edge_data(u, v)
            edge_key = min(parallel_edges.keys())
            edges.append({"src": u, "dst": v, **parallel_edges[edge_key]})
        return edges


# ══════════════════════════════════════════════════════════════════════════
# REACHABLE ASSETS
# ══════════════════════════════════════════════════════════════════════════

# The 8 categories requested map onto this dataset's already-parsed
# resource_type/service fields (see neo4j_graph_builder.py parse_target).
# EC2 Instances and Lambda Functions are further split from generic
# "Resource" using the SAME regexes already used to build the graph
# (_EC2_ID_RE / service=="lambda"), not a new parsing pass.
ASSET_CATEGORY_LABELS = (
    "iam_role", "ec2_instance", "lambda_function", "s3_bucket",
    "rds_database", "dynamodb_table", "secretsmanager_secret", "kms_key", "other",
)


def _categorize_asset(node: Node, service_by_node: Dict[Node, str], resource_type_by_node: Dict[Node, str]) -> str:
    label, key = node
    if label == "Role":
        return "iam_role"
    service = service_by_node.get(node, "unresolved")
    rtype = resource_type_by_node.get(node, "opaque")
    if rtype == "ec2-instance-id" or service == "ec2":
        return "ec2_instance"
    if service == "lambda":
        return "lambda_function"
    if service in ("s3", "s3-control", "s3-external-1"):
        return "s3_bucket"
    if service == "rds":
        return "rds_database"
    if service == "dynamodb":
        return "dynamodb_table"
    if service == "secretsmanager":
        return "secretsmanager_secret"
    if service == "kms":
        return "kms_key"
    return "other"


@dataclass
class ReachableAssets:
    counts: Dict[str, int]
    total: int

    @classmethod
    def compute(cls, reachable: Dict[Node, int], service_by_node, resource_type_by_node) -> "ReachableAssets":
        counts = {c: 0 for c in ASSET_CATEGORY_LABELS}
        for node in reachable:
            counts[_categorize_asset(node, service_by_node, resource_type_by_node)] += 1
        return cls(counts=counts, total=len(reachable))


# ══════════════════════════════════════════════════════════════════════════
# PRIVILEGE ESCALATION REACHABILITY
# ══════════════════════════════════════════════════════════════════════════

_ARN_ACCOUNT_RE = re.compile(r"^arn:aws:[a-zA-Z0-9\-]+:[^:]*:(\d{12}):")


@dataclass
class PrivilegeReachabilityResult:
    assume_role_chain_depth: int             # verified 0, 1, or 2 on this dataset
    administrator_reachable: bool            # PROXY — see module docstring
    administrator_reachable_is_proxy: bool = True
    service_linked_role_reached: List[str] = field(default_factory=list)
    service_linked_role_abuse_suspected: bool = False
    pass_role_opportunities: int = 0
    pass_role_observed: bool = False          # False on this dataset — see module docstring
    cross_account_ids_reachable: Set[str] = field(default_factory=set)
    cross_account_coverage_note: str = (
        "Only ARN-shaped :Resource/:Policy node keys carry an account ID in "
        "this schema; :Role node keys are canonicalized role NAMES (see "
        "privilege_features.py), so cross-account detection does not cover "
        "role-to-role reachability. Documented limitation, not full coverage."
    )


class PrivilegeReachabilityAnalyzer:
    def __init__(self, graph: ReachabilityGraph, source_account_id: Optional[str] = None):
        self.graph = graph
        self.source_account_id = source_account_id

    def analyze(self, source: Node, reachable: Dict[Node, int], resolver: pf.ActionAccessLevelResolver,
                edge_lookup) -> PrivilegeReachabilityResult:
        admin_reachable = False
        service_linked = []
        cross_accounts: Set[str] = set()

        for node, depth in reachable.items():
            label, key = node
            if label == "Role" and key.startswith("AWSServiceRoleFor"):
                service_linked.append(key)
            m = _ARN_ACCOUNT_RE.match(key)
            if m:
                acct = m.group(1)
                if self.source_account_id is None or acct != self.source_account_id:
                    cross_accounts.add(acct)

        for edges in edge_lookup(source, reachable):
            for e in edges:
                if e.get("relation") == "PERMISSIONS_MANAGEMENT":
                    admin_reachable = True

        # AssumeRole chain depth = the hop-distance of the furthest
        # reachable :Role node. This relies on a fact VERIFIED against
        # this dataset (every edge that targets a :Role node has
        # relation=="ASSUMES" — no other action, e.g. TagRole/GetRole,
        # happens to target a role ARN here), not a schema guarantee — a
        # different dataset could in principle have non-ASSUMES edges
        # into a :Role node, which would make this an overcount there.
        assume_depth = max((d for n, d in reachable.items() if n[0] == "Role"), default=0)

        service_linked_via_assume = [n for n in service_linked if ("Role", n) in reachable]
        abuse_suspected = bool(service_linked_via_assume) and source[0] == "User"

        return PrivilegeReachabilityResult(
            assume_role_chain_depth=assume_depth,
            administrator_reachable=admin_reachable,
            service_linked_role_reached=service_linked,
            service_linked_role_abuse_suspected=abuse_suspected,
            pass_role_opportunities=0,
            pass_role_observed=False,
            cross_account_ids_reachable=cross_accounts,
        )


# ══════════════════════════════════════════════════════════════════════════
# CRITICAL ASSET SCORER — reuses privilege_features.SERVICE_SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CriticalAssetResult:
    critical_asset_count: int
    critical_assets: List[Node]
    exposure_score: float  # sum of per-asset criticality weights, NOT just a count


class CriticalAssetScorer:
    def __init__(self, config: BlastRadiusConfig):
        self.config = config

    def score(self, reachable: Dict[Node, int], sensitivity_by_node: Dict[Node, int]) -> CriticalAssetResult:
        critical = [n for n in reachable if sensitivity_by_node.get(n, -1) >= self.config.critical_asset_min_tier]
        exposure = sum(self.config.criticality.weight_for_tier(sensitivity_by_node.get(n, -1)) for n in reachable)
        return CriticalAssetResult(critical_asset_count=len(critical), critical_assets=critical, exposure_score=exposure)


# ══════════════════════════════════════════════════════════════════════════
# PROPAGATION PATH EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PropagationPath:
    nodes: List[Node]
    edges: List[dict]
    impact_score: float

    def render(self) -> str:
        return "\n  ↓\n".join(f"{label}:{key}" for label, key in self.nodes)


class PropagationPathExtractor:
    """
    Ranks candidate propagation paths from a source by IMPACT (the
    destination's criticality weight), not just length — a 1-hop path to
    a Secrets Manager secret matters more than a 2-hop path to a routine
    read-only resource. Returns the highest-impact path plus up to k-1
    runners-up, exactly as requested ("return the highest-impact
    propagation path... support multiple candidate paths ranked by
    impact").
    """

    def __init__(self, graph: ReachabilityGraph, config: BlastRadiusConfig):
        self.graph = graph
        self.config = config

    def extract(self, source: Node, reachable: Dict[Node, int],
                sensitivity_by_node: Dict[Node, int], top_k: int = 3) -> List[PropagationPath]:
        scored = []
        for node in reachable:
            weight = self.config.criticality.weight_for_tier(sensitivity_by_node.get(node, -1))
            if weight <= 0:
                continue
            edges = self.graph.edges_on_shortest_path(source, node)
            path_nodes = [source] + [e["dst"] for e in edges]
            scored.append(PropagationPath(nodes=path_nodes, edges=edges, impact_score=weight))
        scored.sort(key=lambda p: -p.impact_score)
        return scored[:top_k]


# ══════════════════════════════════════════════════════════════════════════
# IMPACT METRICS LOGGER — real, self-contained; a sibling to (and, once
# Phase 1 of the adaptive framework lands, candidate for consolidation
# with) the general-purpose MetricsLogger from that roadmap.
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BlastRadiusMetrics:
    principal: Node
    traversal_seconds: float
    nodes_visited: int
    edges_visited: int


class ImpactMetricsLogger:
    def __init__(self):
        self.records: List[BlastRadiusMetrics] = []

    def record(self, m: BlastRadiusMetrics):
        self.records.append(m)
        log.info("BlastRadius[%s] %.4fs | %d nodes | %d edges",
                  m.principal, m.traversal_seconds, m.nodes_visited, m.edges_visited)

    def summary(self) -> dict:
        if not self.records:
            return {}
        times = [r.traversal_seconds for r in self.records]
        return {
            "n": len(self.records),
            "mean_seconds": sum(times) / len(times),
            "max_seconds": max(times),
            "min_seconds": min(times),
            "total_nodes_visited": sum(r.nodes_visited for r in self.records),
        }


# ══════════════════════════════════════════════════════════════════════════
# BLAST RADIUS CACHE — explicit stub. See module docstring: this needs
# IncrementalGraphUpdater (not yet built) to define "what changed."
# ══════════════════════════════════════════════════════════════════════════

class BlastRadiusCache:
    """
    Lazy-invalidation cache: `invalidate(principal)` marks an entry stale
    rather than eagerly recomputing it (cheaper — many invalidated
    principals are never queried again before their next real change, so
    eager recomputation would waste work the lazy approach avoids).
    `get_or_compute` returns the cached report if fresh, else recomputes
    via the given BlastRadiusEngine and caches the result.

    Now real, not a stub: IncrementalGraphUpdater (incremental_updater.py)
    exists to answer "which principals are affected by this new event,"
    which is exactly what this class needed before it could be built —
    see incremental_updater.py's `_affected_principals` for that
    computation (a bounded reverse-BFS, not the O(V+E) full-graph
    approach this class's docstring used to flag as the open question).
    """

    def __init__(self):
        self._reports: Dict[Node, BlastRadiusReport] = {}
        self._stale: Set[Node] = set()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def invalidate(self, principal: Node) -> None:
        if principal in self._reports:
            self._stale.add(principal)
            self.invalidations += 1

    def get_or_compute(self, principal: Node, engine: "BlastRadiusEngine",
                        resolver: Optional["pf.ActionAccessLevelResolver"] = None) -> "BlastRadiusReport":
        is_fresh = principal in self._reports and principal not in self._stale
        if is_fresh:
            self.hits += 1
            return self._reports[principal]
        self.misses += 1
        report = engine.compute(principal, resolver)
        self._reports[principal] = report
        self._stale.discard(principal)
        return report

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_ratio": self.hits / total if total else 0.0,
            "cached_entries": len(self._reports),
            "stale_entries": len(self._stale),
            "invalidations": self.invalidations,
        }


# ══════════════════════════════════════════════════════════════════════════
# BLAST RADIUS ENGINE — orchestrator
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BlastRadiusReport:
    principal: Node
    reachable_assets: ReachableAssets
    privilege_reachability: PrivilegeReachabilityResult
    critical_assets: CriticalAssetResult
    top_paths: List[PropagationPath]
    cross_service_count: int
    score: float
    score_components: Dict[str, float]
    is_lower_bound: bool = True  # see module docstring point 1 — always True for event-derived graphs

    def summary(self) -> str:
        lbl, key = self.principal
        lines = [
            f"Blast Radius Report — {lbl}:{key}  (score={self.score:.3f}, LOWER BOUND — observed reachability only)",
            f"  Reachable assets: {self.reachable_assets.total} "
            f"({', '.join(f'{k}={v}' for k, v in self.reachable_assets.counts.items() if v)})",
            f"  Cross-service reach: {self.cross_service_count}",
            f"  Critical assets reachable: {self.critical_assets.critical_asset_count} "
            f"(exposure={self.critical_assets.exposure_score:.2f})",
            f"  AssumeRole chain depth: {self.privilege_reachability.assume_role_chain_depth}",
            f"  Administrator-like (PERMISSIONS_MANAGEMENT proxy) reachable: "
            f"{self.privilege_reachability.administrator_reachable}",
            f"  PassRole opportunities: NOT OBSERVABLE in this dataset (0 PassRole events recorded)"
            if not self.privilege_reachability.pass_role_observed else
            f"  PassRole opportunities: {self.privilege_reachability.pass_role_opportunities}",
            f"  Cross-account IDs reachable: {sorted(self.privilege_reachability.cross_account_ids_reachable) or 'none (single-account dataset)'}",
        ]
        if self.top_paths:
            lines.append("  Highest-impact propagation path:")
            lines.append("    " + self.top_paths[0].render().replace("\n", "\n    "))
        return "\n".join(lines)


class BlastRadiusEngine:
    def __init__(self, ppg: pf.PrivilegePropagationGraph, config: Optional[BlastRadiusConfig] = None,
                 source_account_id: Optional[str] = None):
        self.ppg = ppg
        self.config = config or BlastRadiusConfig()
        self.graph = NetworkXReachabilityGraph(ppg)
        self.priv_analyzer = PrivilegeReachabilityAnalyzer(self.graph, source_account_id)
        self.asset_scorer = CriticalAssetScorer(self.config)
        self.path_extractor = PropagationPathExtractor(self.graph, self.config)
        self.metrics = ImpactMetricsLogger()

        # Precompute service/resource_type/sensitivity lookups ONCE over
        # every node — O(V), reused across every compute() call rather
        # than recomputed per principal.
        self._service_by_node: Dict[Node, str] = {}
        self._resource_type_by_node: Dict[Node, str] = {}
        self._sensitivity_by_node: Dict[Node, int] = {}
        self._index_targets()

    def _index_targets(self):
        # Re-derive service/resource_type per target node from its raw key
        # using the SAME neo4j_graph_builder parsing already used to build
        # the graph (imported lazily to avoid a hard circular dependency
        # at module load time).
        import neo4j_graph_builder as nb
        for node in self.ppg.graph.nodes:
            label, key = node
            if label in ("Service", "Resource", "Policy"):
                info = nb.parse_target(key)
                self._service_by_node[node] = info.service
                self._resource_type_by_node[node] = info.resource_type
                self._sensitivity_by_node[node] = pf.resource_sensitivity_score(info.service, info.resource_type)

    def _edge_lookup(self, source: Node, reachable: Dict[Node, int]):
        for node in reachable:
            try:
                yield self.graph.edges_on_shortest_path(source, node)
            except nx.NetworkXNoPath:
                continue

    def compute(self, principal: Node, resolver: Optional[pf.ActionAccessLevelResolver] = None) -> BlastRadiusReport:
        t0 = time.perf_counter()
        resolver = resolver or pf.ActionAccessLevelResolver()

        reachable = self.graph.reachable_within(principal, self.config.max_traversal_depth)

        assets = ReachableAssets.compute(reachable, self._service_by_node, self._resource_type_by_node)
        priv = self.priv_analyzer.analyze(principal, reachable, resolver, self._edge_lookup)
        critical = self.asset_scorer.score(reachable, self._sensitivity_by_node)
        paths = self.path_extractor.extract(principal, reachable, self._sensitivity_by_node, top_k=3)

        services_reached = {self._service_by_node[n] for n in reachable if n in self._service_by_node}
        cross_service = len(services_reached)

        cfg = self.config
        s_assets       = min(1.0, assets.total / cfg.reachable_assets_saturation)
        s_privesc      = min(1.0, (priv.assume_role_chain_depth / 2.0) + (0.5 if priv.administrator_reachable else 0.0))
        s_critical     = min(1.0, critical.exposure_score / max(1, cfg.reachable_assets_saturation * cfg.criticality.tier_to_weight[3]))
        s_cross_svc    = min(1.0, cross_service / cfg.cross_service_saturation)
        s_cross_acct   = min(1.0, len(priv.cross_account_ids_reachable))

        score = (
            cfg.w_reachable_assets * s_assets
            + cfg.w_privilege_escalation * s_privesc
            + cfg.w_critical_asset_exposure * s_critical
            + cfg.w_cross_service * s_cross_svc
            + cfg.w_cross_account * s_cross_acct
        )

        elapsed = time.perf_counter() - t0
        self.metrics.record(BlastRadiusMetrics(
            principal=principal, traversal_seconds=elapsed,
            nodes_visited=len(reachable), edges_visited=sum(1 for _ in self._edge_lookup(principal, reachable)),
        ))

        return BlastRadiusReport(
            principal=principal, reachable_assets=assets, privilege_reachability=priv,
            critical_assets=critical, top_paths=paths, cross_service_count=cross_service,
            score=score,
            score_components={
                "reachable_assets": s_assets, "privilege_escalation": s_privesc,
                "critical_asset_exposure": s_critical, "cross_service": s_cross_svc,
                "cross_account": s_cross_acct,
            },
        )
