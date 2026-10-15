"""
incremental_updater.py
========================
IncrementalGraphUpdater — streaming, O(local-neighborhood) updates to the
Privilege Propagation Graph, replacing full-graph rebuilds for new
CloudTrail events.

ARCHITECTURAL PREMISE: BATCH IS A SPECIAL CASE OF INCREMENTAL
─────────────────────────────────────────────────────────────────────────
`apply_event()` is written so that calling it once per CSV row, in order,
starting from an empty graph, produces the SAME graph neo4j_graph_builder.py
builds in one batch pass. This is verified empirically below (see the
equivalence test), not just claimed. It means there is exactly one code
path for "how does an event affect the graph," used both for the initial
build and for every subsequent streamed event — never two parallel
implementations to keep in sync with each other.

WHAT IS AND ISN'T O(1) HERE — SEE MODULE-LEVEL DESIGN NOTES BELOW
─────────────────────────────────────────────────────────────────────────
- out_degree, in_degree, unique_targets, unique_actions, unique_principals,
  role_transition_count: O(1) amortized (hash-set insert + counter bump),
  touching only the event's two endpoint nodes.
- resource_sensitivity: O(1), computed once at node-creation time from the
  node's own (service, resource_type) — never depends on the rest of the
  graph, so it never needs recomputation.
- distance_to_sensitive_resource: bounded BFS relaxation, O(local
  neighborhood within `cutoff` hops), NOT O(1) — see
  `_propagate_distance_update`'s docstring for the monotonicity argument
  that makes early termination sound.
- action_global_frequency / abnormal_path_frequency: O(1) counter bump,
  but deliberately NOT retroactively applied to already-ingested edges —
  see module docstring in this file's "EVENTUAL CONSISTENCY" section.
- Affected-principal computation for BlastRadiusCache invalidation:
  bounded reverse-BFS, O(local neighborhood within max_traversal_depth-1).

EVENTUAL CONSISTENCY OF GLOBAL/FREQUENCY-DERIVED FEATURES
─────────────────────────────────────────────────────────────────────────
`abnormal_path_frequency` for an edge is computed against the pattern
count AT INGESTION TIME, not the final count after the stream ends — the
same logical edge can get a different value depending on stream position.
This is the same trade-off any online/streaming frequency estimator makes
(you don't rewrite history every time a new observation arrives), not an
oversight. The equivalence test below quantifies exactly how much this
differs from a full recompute, rather than asserting "small" without a
number.

DELETION IS OUT OF SCOPE
─────────────────────────────────────────────────────────────────────────
Only insertion is requested (requirement 3) and CloudTrail is append-only
in reality. Edge removal breaks the monotonic-relaxation argument this
file's distance-propagation algorithm depends on (removing an edge can
only INCREASE distances, requiring a full or partial re-BFS rather than
bounded local relaxation) — a known harder problem in the incremental
shortest-path literature, not implemented here.

EQUIVALENCE TEST RESULTS (test_incremental_updater.py; figures below were
measured against the prior 2,900-row Invictus dataset under its integer
log_id scheme — NOT re-measured against the Feature Engine's
cloudtrail_structural.csv, which has a different row count and different
synthetic data. The MECHANISM described below is what the current code
implements either way; the specific counts/ids are the historical
Invictus run and are kept here as a worked example, not a live claim
about the current dataset.)
─────────────────────────────────────────────────────────────────────────
Streaming all rows through apply_event(), in file/stream ARRIVAL order
(the CSV's row order — see test_incremental_updater.py's setUpClass),
from an empty graph:
- Node/edge counts, out_degree, unique_targets: IDENTICAL to the batch
  pipeline for every single node (2,900/2,900 edges, exact match on the
  Invictus run).
- distance_to_sensitive_resource: IDENTICAL for every node — the bounded
  relaxation algorithm converges to the same final distances regardless
  of arrival order, confirming the monotonicity argument empirically, not
  just in theory.
- hop_count: identical for 2,899/2,900 edges on the Invictus run. The one
  exception (row 199 in arrival order) is not a bug — it's a direct,
  traceable consequence of log_id carrying NO ordering guarantee (true
  under either the old integer scheme or the new opaque-string one — AWS
  CloudTrail log files aren't delivered in a guaranteed order either):
  arrival-position 199 is AWSServiceRoleForAmazonInspector2 performing
  DescribeInstances, but the AssumeRole events that grant that exact role
  arrive LATER (positions 200/201/1009/1010). The batch computation sees
  the whole dataset and knows (with hindsight) that this role was assumed
  at some point; the incremental computation, at the moment it processes
  arrival-position 199, has genuinely not seen that AssumeRole event yet
  — because, causally, it hasn't happened yet from the stream's point of
  view. This is the incremental path being MORE correct for a real
  streaming system, not less — a live detector cannot see future events
  either. (test_incremental_updater.py verifies this causal property by
  comparing STREAM POSITION, not log_id's value — see that file for why.)
- abnormal_path_frequency: differs for 2,899/2,900 edges on the Invictus
  run (expected — see "EVENTUAL CONSISTENCY" above), with a max absolute
  difference of 3.98. That's a substantive amount of drift, not a
  rounding error — worth knowing precisely rather than assuming "probably
  small."
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set

import networkx as nx

import neo4j_graph_builder as nb
import privilege_features as pf

log = logging.getLogger(__name__)

Node = tuple  # (node_label, key) — same convention as privilege_features.GraphNodeKey


# ══════════════════════════════════════════════════════════════════════════
# INPUT CONTRACT
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CloudTrailEvent:
    """One streamed event, matching the CSV schema exactly (log_id,
    source_node, target_node, edge_type, label) so the incremental and
    batch paths consume identical inputs.

    log_id is an opaque, unique STRING identifier (Feature Engine schema:
    "<source_file>:<row_index>", e.g. "synthetic_cloudtrail.csv:0") — it
    is stored and forwarded verbatim (to the in-memory graph's edge
    attributes and to Neo4j) but never parsed, coerced, or compared
    numerically anywhere in this module. See module docstring for how
    "arrival order" is determined WITHOUT relying on log_id's value."""
    log_id: str
    source_node: Optional[str]
    target_node: str
    edge_type: str
    label: int


class SyncStatus(str, Enum):
    """Neo4j write-through outcome for one event — see module docstring,
    concurrency section, on why this is a visible flag rather than a
    silently-assumed guarantee."""
    SYNCED = "synced"
    PENDING = "pending"   # no Neo4j session configured; in-memory only
    FAILED = "failed"     # Neo4j write attempted and raised


@dataclass
class UpdateResult:
    """What changed, for logging/metrics/tests — never silently discarded."""
    log_id: str
    source_key: pf.GraphNodeKey
    target_key: pf.GraphNodeKey
    relation: str
    created_new_source_node: bool
    created_new_target_node: bool
    affected_principals: Set[Node]
    distance_updates: Dict[Node, int]     # nodes whose distance_to_sensitive_resource improved
    sync_status: SyncStatus
    sync_error: Optional[str] = None


@dataclass(frozen=True)
class IncrementalUpdateConfig:
    """
    All thresholds configurable — no magic numbers. `max_traversal_depth`
    and `sensitive_cutoff`/`critical_asset_min_tier` MUST match whatever
    BlastRadiusEngine/BlastRadiusConfig instance this updater is paired
    with (see design notes: using a different radius here would silently
    under- or over-invalidate the cache). Pass the same BlastRadiusConfig
    in rather than letting these drift independently.
    """
    max_traversal_depth: int = 4
    sensitive_cutoff: int = 6
    critical_asset_min_tier: int = 2

    @classmethod
    def from_blast_radius_config(cls, brc) -> "IncrementalUpdateConfig":
        return cls(
            max_traversal_depth=brc.max_traversal_depth,
            critical_asset_min_tier=brc.critical_asset_min_tier,
        )


# ══════════════════════════════════════════════════════════════════════════
# THE UPDATER
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# NEO4J PROPERTY SERIALIZATION (write-through only — see docstring below)
# ══════════════════════════════════════════════════════════════════════════

# unique_targets / unique_actions / unique_principals / role_transitions are
# kept as Python `set`s on the live NetworkX graph, deliberately, for O(1)
# incremental insertion + uniqueness (see _get_or_create_node /
# _touch_source_counters / _touch_target_counters below — UNCHANGED by this
# fix). The Neo4j schema neo4j_graph_builder.py's batch writer establishes
# (see that module's SCHEMA docstring, and its
# `"unique_targets": len(node_unique_targets.get(n, set()))`-style
# construction) has ALWAYS exposed these as INTEGER COUNTS, never as raw
# member lists. Three keep the same property name on both sides (set
# in-memory, count in Neo4j); "role_transitions" (the in-memory set name)
# maps to "role_transition_count" — the only name that property has ever
# had in the Neo4j schema; there is no Neo4j property literally called
# "role_transitions". Converting via this mapping is what makes the fix
# below schema-PRESERVING: it reproduces exactly the property names and
# types the batch pipeline already writes for the same nodes, rather than
# introducing a second, different representation for the same feature.
_SET_MEMBERSHIP_COUNT_PROPS: Dict[str, str] = {
    "unique_targets": "unique_targets",
    "unique_actions": "unique_actions",
    "unique_principals": "unique_principals",
    "role_transitions": "role_transition_count",
}


def _neo4j_safe_props(props: dict) -> dict:
    """
    Returns a NEW dict safe to pass as Neo4j's `$props` bind parameter for
    a `SET n += $props` write. Never mutates `props` — the caller always
    passes a live reference into a NetworkX node's attribute store (see
    `_write_through_neo4j` below), and that in-memory structure must keep
    using `set`s exactly as it does today.

    WHY THIS FUNCTION EXISTS: the Neo4j Python driver cannot serialize a
    raw Python `set` at all — `ValueError: Values of type <class 'set'>
    are not supported`, regardless of what the set contains (verified
    directly against the installed driver's packstream packer).

    WHY IT'S NOT JUST `sorted(v) if isinstance(v, set) else v`: three of
    the four sets above (`unique_targets`, `unique_principals`,
    `role_transitions`) hold `(node_label, key)` tuples, not flat
    primitives (`Node = tuple` — see module top). `sorted()`-ing one of
    those gives a LIST OF TUPLES, which the driver's packstream layer
    actually accepts on the wire (tuples pack like lists) — but Neo4j's
    property-value type system does not: a stored property must be a
    primitive or a flat, homogeneous array of ONE primitive type, never a
    nested/composite structure, so writing a list of tuples would only
    trade today's driver-side error for a Neo4j-side "invalid property
    type" error on the exact same write. It would also silently change
    the schema (a member list instead of the count every batch-built node
    already has under that property name) — which this fix must not do.
    `len()` is therefore the only conversion that is both serializable
    AND schema-correct for these four properties.

    Any OTHER, currently-nonexistent set-valued property (none exist
    today — see the four above — but this keeps the function correct if
    one is ever added) falls back to `sorted()`: safe as long as its
    members are themselves flat, already-orderable primitives, and
    deterministic (`sorted()`, not `list()`, so write-order never affects
    the property's serialized value).
    """
    safe: dict = {}
    for key, value in props.items():
        if not isinstance(value, set):
            safe[key] = value
            continue
        if key in _SET_MEMBERSHIP_COUNT_PROPS:
            safe[_SET_MEMBERSHIP_COUNT_PROPS[key]] = len(value)
        else:
            safe[key] = sorted(value)
    return safe


class IncrementalGraphUpdater:
    """
    Applies one CloudTrailEvent at a time to an existing
    privilege_features.PrivilegePropagationGraph, in place, without ever
    rescanning the full graph. See module docstring for complexity/
    consistency details.
    """

    def __init__(
        self,
        ppg: pf.PrivilegePropagationGraph,
        config: IncrementalUpdateConfig,
        resolver: Optional[pf.ActionAccessLevelResolver] = None,
        blast_radius_cache: Optional["BlastRadiusCacheProtocol"] = None,
        neo4j_session_factory=None,
    ):
        self.ppg = ppg
        self.graph = ppg.graph
        self.config = config
        self.resolver = resolver or ppg.resolver
        self.blast_radius_cache = blast_radius_cache
        self.neo4j_session_factory = neo4j_session_factory
        self._lock = threading.RLock()

        # Global running statistics — O(1) to update, deliberately not
        # retroactively applied to prior edges (see module docstring).
        self._action_freq: Dict[str, int] = {}
        self._pattern_freq: Dict[tuple, int] = {}
        self._total_edges = 0

        # Role-name-keyed identity index (the mechanism that keeps
        # canonical Role nodes correctly merged — see module docstring).
        self._known_nodes: Set[Node] = set(self.graph.nodes)

        # Running name sets for target-side bare-name reconciliation (see
        # node_key_for_target's docstring) — kept incremental rather than
        # recomputed from _known_nodes per event, matching this class's O(1)
        # per-event design.
        self._known_role_names: Set[str] = {k for (l, k) in self._known_nodes if l == "Role"}
        self._known_user_names: Set[str] = {k for (l, k) in self._known_nodes if l == "User"}

    # ── Public API ────────────────────────────────────────────────────────

    def apply_event(self, event: CloudTrailEvent) -> UpdateResult:
        with self._lock:
            return self._apply_event_locked(event)

    def _apply_event_locked(self, event: CloudTrailEvent) -> UpdateResult:
        principal_info = nb.parse_principal(event.source_node)
        source_key = pf.node_key_for_principal(event.source_node, principal_info.principal_type, principal_info.name)

        target_info = nb.parse_target(event.target_node)
        target_key = pf.node_key_for_target(target_info.value, target_info.resource_type, target_info.service,
                                             self._known_role_names, self._known_user_names)

        relation = pf.resolve_relation_type(event.edge_type, self.resolver)

        src_node: Node = (source_key.label, source_key.key)
        dst_node: Node = (target_key.label, target_key.key)

        created_src = self._get_or_create_node(src_node, principal_info=principal_info)
        created_dst = self._get_or_create_node(dst_node, target_info=target_info)

        # ── Mutate O(1) counters (out/in-degree, unique sets) ───────────────
        self._touch_source_counters(src_node, dst_node, event.edge_type, relation)
        self._touch_target_counters(dst_node, src_node)

        # ── Global frequency counters (O(1), ingestion-time snapshot) ──────
        self._action_freq[event.edge_type] = self._action_freq.get(event.edge_type, 0) + 1
        pattern = (src_node[0], relation, dst_node[0])
        self._pattern_freq[pattern] = self._pattern_freq.get(pattern, 0) + 1
        self._total_edges += 1

        hop_count = 2 if src_node in self._roles_reached_via_assume() else 1
        abnormal_freq = -__import__("math").log(self._pattern_freq[pattern] / self._total_edges)

        # ── Add the edge itself ─────────────────────────────────────────────
        self.graph.add_edge(
            src_node, dst_node,
            log_id=event.log_id, edge_type=event.edge_type, relation=relation,
            access_level=self.resolver.access_level(event.edge_type),
            label=event.label, hop_count=hop_count, abnormal_path_frequency=abnormal_freq,
        )

        # ── Bounded distance-to-sensitive-resource propagation ─────────────
        distance_updates = self._propagate_distance_update(dst_node)
        # the new edge also potentially shortens the SOURCE's own distance
        distance_updates.update(self._propagate_distance_update(src_node))

        # ── Bounded reverse-BFS: who is affected for cache invalidation ────
        affected = self._affected_principals(src_node)

        sync_status, sync_error = self._write_through_neo4j(event, src_node, dst_node, relation, hop_count, abnormal_freq)

        if self.blast_radius_cache is not None:
            for p in affected:
                self.blast_radius_cache.invalidate(p)

        return UpdateResult(
            log_id=event.log_id, source_key=source_key, target_key=target_key, relation=relation,
            created_new_source_node=created_src, created_new_target_node=created_dst,
            affected_principals=affected, distance_updates=distance_updates,
            sync_status=sync_status, sync_error=sync_error,
        )

    # ── Node creation ─────────────────────────────────────────────────────

    def _get_or_create_node(self, node: Node, principal_info=None, target_info=None) -> bool:
        """Returns True iff this call created a new node. Get-or-create is
        the mechanism that keeps role canonicalization correct: a second
        event under an already-seen role name merges into the SAME node
        rather than creating a duplicate."""
        if node in self._known_nodes:
            return False
        self._known_nodes.add(node)
        label, key = node
        if label == "Role":
            self._known_role_names.add(key)
        elif label == "User":
            self._known_user_names.add(key)
        attrs = {"out_degree": 0, "in_degree": 0, "unique_targets": set(),
                 "unique_actions": set(), "unique_principals": set(), "role_transitions": set()}
        if label in ("Service", "Resource", "Policy") and target_info is not None:
            # resource_sensitivity is O(1) and permanent — a function only
            # of this node's own service/resource_type, never of the rest
            # of the graph, so it is computed once, here, and never revisited.
            attrs["resource_sensitivity"] = pf.resource_sensitivity_score(target_info.service, target_info.resource_type)
            attrs["distance_to_sensitive_resource"] = (
                0 if attrs["resource_sensitivity"] >= self.config.critical_asset_min_tier else None
            )
        else:
            attrs["distance_to_sensitive_resource"] = None
        self.graph.add_node(node, **attrs)
        return True

    # ── O(1) counter updates ──────────────────────────────────────────────

    def _touch_source_counters(self, src: Node, dst: Node, edge_type: str, relation: str) -> None:
        attrs = self.graph.nodes[src]
        attrs["out_degree"] = attrs.get("out_degree", 0) + 1
        attrs.setdefault("unique_targets", set()).add(dst)
        attrs.setdefault("unique_actions", set()).add(edge_type)
        if relation == "ASSUMES":
            attrs.setdefault("role_transitions", set()).add(dst)

    def _touch_target_counters(self, dst: Node, src: Node) -> None:
        attrs = self.graph.nodes[dst]
        attrs["in_degree"] = attrs.get("in_degree", 0) + 1
        attrs.setdefault("unique_principals", set()).add(src)

    def _roles_reached_via_assume(self) -> Set[Node]:
        # Small, cheap: iterating this graph's ASSUMES edges is bounded by
        # how many role-assumption events exist (verified 51/2900 rows on
        # the real dataset) — not the whole edge set. Cached per-call
        # rather than maintained as running state, since it's already cheap
        # and keeping a running set correct under concurrent access would
        # add lock-scope complexity for negligible gain at this event rate.
        return {v for _, v, d in self.graph.edges(data=True) if d.get("relation") == "ASSUMES"}

    # ── Bounded distance-to-sensitive-resource propagation ─────────────────

    def _propagate_distance_update(self, start: Node) -> Dict[Node, int]:
        """
        Bounded BFS relaxation. SOUNDNESS ARGUMENT: in an unweighted graph
        with insertion-only edges, distances are monotonically
        non-increasing — adding an edge can only shorten a path, never
        lengthen one. So the instant a node's distance fails to improve,
        nothing reachable ONLY through that node can improve either,
        making early termination correct, not just an optimisation. This
        is why the propagation is bounded by the local neighbourhood that
        actually improves, rather than needing a full graph rescan.
        """
        updates: Dict[Node, int] = {}
        start_attrs = self.graph.nodes[start]
        best = self._best_neighbor_distance(start)
        if best is None or best > self.config.sensitive_cutoff:
            return updates
        current = start_attrs.get("distance_to_sensitive_resource")
        if current is not None and current <= best:
            return updates

        queue = deque([(start, best)])
        while queue:
            node, d = queue.popleft()
            if d > self.config.sensitive_cutoff:
                continue
            node_attrs = self.graph.nodes[node]
            existing = node_attrs.get("distance_to_sensitive_resource")
            if existing is not None and existing <= d:
                continue  # no improvement here -> nothing behind it improves either
            node_attrs["distance_to_sensitive_resource"] = d
            updates[node] = d
            for pred in self.graph.predecessors(node):
                queue.append((pred, d + 1))
        return updates

    def _best_neighbor_distance(self, node: Node) -> Optional[int]:
        node_attrs = self.graph.nodes[node]
        if node_attrs.get("resource_sensitivity", -1) >= self.config.critical_asset_min_tier:
            return 0
        best = None
        for _, v, d in self.graph.out_edges(node, data=True):
            v_dist = self.graph.nodes[v].get("distance_to_sensitive_resource")
            if v_dist is not None:
                candidate = v_dist + 1
                best = candidate if best is None else min(best, candidate)
        return best

    # ── Affected-principal computation (bounded reverse-BFS) ────────────────

    def _affected_principals(self, changed_source: Node) -> Set[Node]:
        """
        Principals whose blast radius could have grown because of an edge
        originating at `changed_source`: exactly those that can reach
        `changed_source` within max_traversal_depth-1 reverse hops, plus
        `changed_source` itself if it is a principal. `graph.reverse
        (copy=False)` is an O(1) view (networkx does not copy the graph),
        so this touches only the local reverse-neighbourhood of
        `changed_source`, not the whole graph.
        """
        affected: Set[Node] = set()
        if changed_source[0] in ("User", "Role", "UnresolvedPrincipal"):
            affected.add(changed_source)
        rev = self.graph.reverse(copy=False)
        depth_budget = self.config.max_traversal_depth - 1
        if depth_budget > 0:
            lengths = nx.single_source_shortest_path_length(rev, changed_source, cutoff=depth_budget)
            affected |= {n for n in lengths if n[0] in ("User", "Role", "UnresolvedPrincipal")}
        return affected

    # ── Neo4j write-through (best-effort; see concurrency/consistency notes) ─

    def _write_through_neo4j(self, event, src_node, dst_node, relation, hop_count, abnormal_freq):
        if self.neo4j_session_factory is None:
            return SyncStatus.PENDING, None
        try:
            with self.neo4j_session_factory() as session:
                session.run(
                    nb._NODE_MERGE_TEMPLATES[src_node[0]],
                    key=src_node[1], props=_neo4j_safe_props(self.graph.nodes[src_node]),
                )
                session.run(
                    nb._NODE_MERGE_TEMPLATES[dst_node[0]],
                    key=dst_node[1], props=_neo4j_safe_props(self.graph.nodes[dst_node]),
                )
                session.run(
                    nb._EDGE_CREATE_TEMPLATES[relation],
                    src_key=src_node[1], dst_key=dst_node[1],
                    log_id=event.log_id, edge_type=event.edge_type, relation=relation,
                    access_level=self.resolver.access_level(event.edge_type),
                    is_priv_esc=event.edge_type in nb.PRIVILEGE_ESCALATION_TECHNIQUES,
                    hop_count=hop_count, privilege_gain=0.0, privilege_gain_defined=False,
                    abnormal_path_frequency=abnormal_freq,
                    action_global_frequency=self._action_freq[event.edge_type],
                    is_attack=event.label,
                )
            return SyncStatus.SYNCED, None
        except Exception as exc:
            log.error("Neo4j write-through failed for log_id=%s: %s", event.log_id, exc)
            return SyncStatus.FAILED, str(exc)


class BlastRadiusCacheProtocol:
    """Structural type only (for the constructor type hint above) — see
    blast_radius.py for the real implementation."""
    def invalidate(self, principal: Node) -> None: ...
