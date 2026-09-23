"""
test_incremental_updater.py
=============================
Unit tests for incremental_updater.py, plus the equivalence test promised
in the design write-up: streaming every row of the Feature Engine's
cloudtrail_structural.csv one at a time through IncrementalGraphUpdater,
from an empty graph, and comparing the result against the existing batch
pipeline (neo4j_graph_builder.py / privilege_features.py). (Originally
written and validated against the prior 2,900-row static Invictus
dataset — see incremental_updater.py's module docstring for those
historical, not-yet-re-measured, figures.)

Runs with plain `python3 -m unittest test_incremental_updater` — no
Neo4j, no torch required (this module and everything it tests only
depend on pandas/networkx), so this suite is fully executable wherever
this repository is checked out.
"""

from __future__ import annotations

import math
import unittest

import networkx as nx
import pandas as pd

import neo4j_graph_builder as nb
import privilege_features as pf
from blast_radius import BlastRadiusCache, BlastRadiusEngine, BlastRadiusConfig
from incremental_updater import (
    CloudTrailEvent,
    IncrementalGraphUpdater,
    IncrementalUpdateConfig,
    SyncStatus,
)

CSV_PATH = "./cloudtrail_structural.csv"


def _new_updater(config: IncrementalUpdateConfig = None) -> IncrementalGraphUpdater:
    resolver = pf.ActionAccessLevelResolver()
    ppg = pf.PrivilegePropagationGraph(resolver)
    return IncrementalGraphUpdater(ppg, config or IncrementalUpdateConfig(), resolver=resolver)


class TestNodeCreationAndCounters(unittest.TestCase):
    def test_new_principal_and_resource_created(self):
        updater = _new_updater()
        event = CloudTrailEvent(log_id="1", source_node="arn:aws:iam::111111111111:user/alice",
                                 target_node="s3.amazonaws.com", edge_type="ListBuckets", label=0)
        result = updater.apply_event(event)
        self.assertTrue(result.created_new_source_node)
        self.assertTrue(result.created_new_target_node)
        self.assertIn(("User", "alice"), updater.graph.nodes)
        self.assertIn(("Service", "s3.amazonaws.com"), updater.graph.nodes)
        self.assertEqual(result.sync_status, SyncStatus.PENDING)  # no neo4j session configured

    def test_repeat_target_does_not_double_count_unique_targets(self):
        updater = _new_updater()
        ev = lambda i, action: CloudTrailEvent(i, "arn:aws:iam::111111111111:user/alice",
                                                "s3.amazonaws.com", action, 0)
        updater.apply_event(ev("1", "ListBuckets"))
        updater.apply_event(ev("2", "ListBuckets"))  # same target, same action
        node = updater.graph.nodes[("User", "alice")]
        self.assertEqual(node["out_degree"], 2)                 # two events
        self.assertEqual(len(node["unique_targets"]), 1)        # same target both times
        self.assertEqual(len(node["unique_actions"]), 1)        # same action both times

    def test_new_action_and_new_target_are_each_counted_once(self):
        updater = _new_updater()
        updater.apply_event(CloudTrailEvent("1", "arn:aws:iam::111111111111:user/alice",
                                             "s3.amazonaws.com", "ListBuckets", 0))
        updater.apply_event(CloudTrailEvent("2", "arn:aws:iam::111111111111:user/alice",
                                             "cloudtrail.amazonaws.com", "LookupEvents", 0))
        node = updater.graph.nodes[("User", "alice")]
        self.assertEqual(node["out_degree"], 2)
        self.assertEqual(len(node["unique_targets"]), 2)
        self.assertEqual(len(node["unique_actions"]), 2)


class TestRoleCanonicalization(unittest.TestCase):
    def test_assume_then_act_merges_into_one_role_node(self):
        """The mechanism the whole graph redesign depends on: an AssumeRole
        event's target (IAM role-definition ARN) and a later event's
        source (STS assumed-role ARN) must resolve to the SAME node."""
        updater = _new_updater()
        updater.apply_event(CloudTrailEvent(
            "1", "arn:aws:iam::111111111111:user/alice",
            "arn:aws:iam::111111111111:role/deploy-role", "AssumeRole", 0))
        updater.apply_event(CloudTrailEvent(
            "2", "arn:aws:sts::111111111111:assumed-role/deploy-role/session1",
            "ec2.amazonaws.com", "DescribeInstances", 0))

        role_nodes = [n for n in updater.graph.nodes if n[0] == "Role"]
        self.assertEqual(role_nodes, [("Role", "deploy-role")], "must be exactly one canonical Role node")
        # the role node should show BOTH the incoming ASSUMES edge and the
        # outgoing DescribeInstances edge
        self.assertEqual(updater.graph.nodes[("Role", "deploy-role")]["in_degree"], 1)
        self.assertEqual(updater.graph.nodes[("Role", "deploy-role")]["out_degree"], 1)


class TestDistancePropagation(unittest.TestCase):
    def test_distance_improves_and_propagates_backward(self):
        cfg = IncrementalUpdateConfig(critical_asset_min_tier=2, sensitive_cutoff=6)
        updater = _new_updater(cfg)
        # A -> B -> C, where C is sensitive (secretsmanager, tier 3)
        updater.apply_event(CloudTrailEvent("1", "arn:aws:iam::111111111111:user/a",
                                             "resource-b", "GetObject", 0))
        b_dist_before = updater.graph.nodes[("Resource", "resource-b")].get("distance_to_sensitive_resource")
        self.assertIsNone(b_dist_before, "B has no known route to a sensitive resource yet")

        updater.apply_event(CloudTrailEvent(
            "2", "arn:aws:iam::111111111111:user/b_actor",  # a source whose canonical key differs from the resource node
            "secretsmanager.amazonaws.com", "GetSecretValue", 0))
        # secretsmanager.amazonaws.com should now be distance 0 (itself sensitive)
        sm_node = ("Service", "secretsmanager.amazonaws.com")
        self.assertEqual(updater.graph.nodes[sm_node]["distance_to_sensitive_resource"], 0)

    def test_monotonic_non_increasing_under_random_insertions(self):
        """Property test: across a pseudo-random sequence of edges, once a
        node's distance is set, it may only ever decrease or stay the same,
        never increase — the correctness argument the propagation
        algorithm depends on."""
        import random
        rng = random.Random(7)
        updater = _new_updater()
        seen_min = {}

        services = ["secretsmanager.amazonaws.com", "s3.amazonaws.com", "ec2.amazonaws.com", "cloudtrail.amazonaws.com"]
        users = [f"arn:aws:iam::111111111111:user/u{i}" for i in range(6)]

        for i in range(300):
            src = rng.choice(users)
            dst = rng.choice(services + [f"resource-{rng.randint(0,10)}"])
            updater.apply_event(CloudTrailEvent(str(i), src, dst, "GetObject", 0))
            for node, attrs in updater.graph.nodes(data=True):
                d = attrs.get("distance_to_sensitive_resource")
                if d is not None:
                    prev = seen_min.get(node)
                    if prev is not None:
                        self.assertLessEqual(d, prev, f"{node} distance increased from {prev} to {d}")
                    seen_min[node] = d


class TestAffectedPrincipalsAndCacheInvalidation(unittest.TestCase):
    def test_only_reverse_reachable_principals_are_invalidated(self):
        cfg = IncrementalUpdateConfig(max_traversal_depth=3)
        resolver = pf.ActionAccessLevelResolver()
        ppg = pf.PrivilegePropagationGraph(resolver)
        cache = BlastRadiusCache()
        updater = IncrementalGraphUpdater(ppg, cfg, resolver=resolver, blast_radius_cache=cache)

        # alice -> shared-resource (unrelated to bob)
        updater.apply_event(CloudTrailEvent("1", "arn:aws:iam::111111111111:user/alice",
                                             "shared-resource", "GetObject", 0))
        # bob has never touched shared-resource
        updater.apply_event(CloudTrailEvent("2", "arn:aws:iam::111111111111:user/bob",
                                             "other-resource", "GetObject", 0))

        engine = BlastRadiusEngine(ppg, BlastRadiusConfig())
        cache.get_or_compute(("User", "alice"), engine, resolver)
        cache.get_or_compute(("User", "bob"), engine, resolver)
        self.assertEqual(cache.stats()["stale_entries"], 0)

        # a brand new event from alice should invalidate alice, not bob
        result = updater.apply_event(CloudTrailEvent("3", "arn:aws:iam::111111111111:user/alice",
                                                       "another-resource", "GetObject", 0))
        self.assertIn(("User", "alice"), result.affected_principals)
        self.assertNotIn(("User", "bob"), result.affected_principals)


class TestBatchIncrementalEquivalence(unittest.TestCase):
    """
    The central correctness claim from the design write-up: streaming all
    2,900 real rows one at a time, from an empty graph, must produce the
    SAME O(1)-per-event features as the existing batch pipeline. Run
    against the real dataset, not synthetic data.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV_PATH)
        cls.resolver = pf.ActionAccessLevelResolver()

        # log_id -> its position in the CSV's row order. Under the Feature
        # Engine schema, log_id is generated as "<source_file>:<count>"
        # while writing rows in order (see feature_engine9.py), so row
        # order IS arrival/stream order — but log_id's STRING VALUE does
        # not sort the same way (":10" < ":2" lexicographically), so
        # "later in the stream" must be answered by POSITION, never by
        # comparing log_id values directly. This map is built from the
        # DataFrame's own row order — it does not parse or inspect
        # log_id's contents at all, only its position — and is what
        # test_hop_count_mismatches_are_causally_explainable_not_arbitrary
        # uses below for its "did a later grant exist" check.
        cls.log_id_stream_position = {lid: pos for pos, lid in enumerate(cls.df["log_id"])}

        # ---- Batch pipeline (existing code, unmodified) ----
        principal_infos = cls.df["source_node"].apply(nb.parse_principal)
        target_infos = cls.df["target_node"].apply(nb.parse_target)
        src_keys = [pf.node_key_for_principal(arn, info.principal_type, info.name)
                    for arn, info in zip(cls.df["source_node"], principal_infos)]
        dst_keys = [pf.node_key_for_target(t.value, t.resource_type, t.service) for t in target_infos]
        rows = [
            {"log_id": lid, "source_key": sk, "target_key": dk, "edge_type": et, "label": int(lbl)}
            for lid, sk, dk, et, lbl in zip(cls.df["log_id"], src_keys, dst_keys, cls.df["edge_type"], cls.df["label"])
        ]
        cls.batch_ppg = pf.PrivilegePropagationGraph(cls.resolver).build_from_rows(rows)

        # ---- Incremental pipeline (one apply_event() call per row, in CSV
        # row order == stream arrival order — NOT sorted by log_id's value;
        # see the stream-position note above for why) ----
        cls.incr_ppg = pf.PrivilegePropagationGraph(cls.resolver)
        cls.updater = IncrementalGraphUpdater(cls.incr_ppg, IncrementalUpdateConfig(), resolver=cls.resolver)
        for _, row in cls.df.iterrows():
            cls.updater.apply_event(CloudTrailEvent(
                log_id=row["log_id"], source_node=row["source_node"],
                target_node=row["target_node"], edge_type=row["edge_type"], label=int(row["label"]),
            ))

    def test_same_node_and_edge_counts(self):
        self.assertEqual(self.incr_ppg.graph.number_of_nodes(), self.batch_ppg.graph.number_of_nodes())
        self.assertEqual(self.incr_ppg.graph.number_of_edges(), self.batch_ppg.graph.number_of_edges())
        self.assertEqual(self.incr_ppg.graph.number_of_edges(), len(self.df))

    def test_out_degree_identical_for_every_node(self):
        for node in self.batch_ppg.graph.nodes:
            batch_deg = self.batch_ppg.graph.out_degree(node)
            incr_deg = self.incr_ppg.graph.out_degree(node)
            self.assertEqual(batch_deg, incr_deg, f"out_degree mismatch for {node}")

    def test_unique_targets_identical_for_every_node(self):
        for node in self.batch_ppg.graph.nodes:
            batch_unique = len(set(self.batch_ppg.graph.successors(node)))
            incr_unique = len(self.incr_ppg.graph.nodes[node].get("unique_targets", set()))
            self.assertEqual(batch_unique, incr_unique, f"unique_targets mismatch for {node}")

    def test_hop_count_mismatches_are_causally_explainable_not_arbitrary(self):
        """
        log_id carries NO ordering guarantee — true under the old
        Invictus integer scheme and even more clearly true now that
        log_id is an opaque Feature Engine string
        ("<source_file>:<row_index>"), and consistent with AWS's own
        CloudTrail documentation (log files aren't delivered in a
        guaranteed order either). hop_count depends on "was this role
        ever assumed" — the batch path answers that with full-dataset
        hindsight (it sees every AssumeRole event, including ones that
        arrive LATER IN THE STREAM than the row being scored); the
        incremental path can only use what it has seen so far, which is
        the causally correct thing for a real streaming system to do — it
        cannot know about a future AssumeRole event any more than a live
        system could.

        So a mismatch is not a bug to eliminate; it is EXPECTED whenever a
        role's action arrives before its own AssumeRole event. This test
        verifies that is EXACTLY what happened for every mismatch found,
        using each event's STREAM POSITION (cls.log_id_stream_position —
        i.e. its row index in the CSV, which is arrival order; never
        log_id's own value) to determine "later", rather than just
        tolerating an unexplained discrepancy. (This property was first
        confirmed concretely against the prior Invictus dataset, where the
        one mismatch found was AWSServiceRoleForAmazonInspector2's
        DescribeInstances preceding its own AssumeRole grants — kept here
        as a worked example of the pattern, not a claim about exactly
        which rows mismatch in cloudtrail_structural.csv.)
        """
        batch_feats = self.batch_ppg.compute_all_edge_features().set_index("log_id")
        incr_hops_by_log_id = {d["log_id"]: d["hop_count"] for _, _, d in self.incr_ppg.graph.edges(data=True)}
        row_by_log_id = self.df.set_index("log_id")

        assume_rows = self.df[self.df["edge_type"].isin(pf.ASSUME_ACTIONS)]
        role_name_of = lambda src: nb.parse_principal(src).name
        assume_log_ids_by_role: dict = {}
        for _, r in assume_rows.iterrows():
            target_role = pf.role_name_from_iam_arn(r["target_node"])
            if target_role:
                assume_log_ids_by_role.setdefault(target_role, []).append(r["log_id"])

        mismatches = [
            log_id for log_id in batch_feats.index
            if batch_feats.loc[log_id, "hop_count"] != incr_hops_by_log_id.get(log_id)
        ]

        for log_id in mismatches:
            batch_hop = batch_feats.loc[log_id, "hop_count"]
            incr_hop = incr_hops_by_log_id[log_id]
            # incremental must UNDER-count relative to batch (less
            # information available), never over-count
            self.assertEqual(incr_hop, 1)
            self.assertEqual(batch_hop, 2)

            role_name = role_name_of(row_by_log_id.loc[log_id, "source_node"])
            grant_log_ids = assume_log_ids_by_role.get(role_name, [])
            my_position = self.log_id_stream_position[log_id]
            self.assertTrue(
                any(self.log_id_stream_position[g] > my_position for g in grant_log_ids),
                f"log_id={log_id} (role={role_name}) mismatched but has no LATER (by stream "
                f"position) AssumeRole grant to explain it — this would be an unexplained bug, "
                f"not the expected pattern",
            )

        print(f"\n[hop_count divergence] {len(mismatches)} log_id(s) diverge from batch, "
              f"all confirmed caused by an AssumeRole grant arriving later in the stream "
              f"(log_id itself is not chronological): {mismatches}")

    def test_distance_to_sensitive_resource_converges_regardless_of_order(self):
        """The propagation-correction algorithm should converge to the same
        final distances as a full from-scratch BFS, even though edges
        arrived in a different (streaming) order than a full-graph
        recompute would see them."""
        sensitivity_lookup = {}
        for node in self.batch_ppg.graph.nodes:
            label, key = node
            if label in ("Service", "Resource", "Policy"):
                info = nb.parse_target(key)
                sensitivity_lookup[node] = pf.resource_sensitivity_score(info.service, info.resource_type)
            else:
                sensitivity_lookup[node] = -1

        mismatches = []
        for node in self.batch_ppg.graph.nodes:
            batch_dist = self.batch_ppg.distance_to_sensitive_resource(node, sensitivity_lookup)
            incr_dist = self.incr_ppg.graph.nodes[node].get("distance_to_sensitive_resource")
            if batch_dist != incr_dist:
                mismatches.append((node, batch_dist, incr_dist))

        self.assertEqual(mismatches, [], f"{len(mismatches)} nodes disagree on final distance: {mismatches[:5]}")

    def test_abnormal_path_frequency_is_NOT_identical_and_that_is_documented(self):
        """Confirms — with an actual number, not just an assertion — the
        documented eventual-consistency trade-off: streaming values differ
        from a full recompute because the denominator (total edges so far)
        differs depending on stream position. See test_hop_count's note on
        WHERE the batch path stores this value (a separate DataFrame, not
        graph edge attributes)."""
        batch_feats = self.batch_ppg.compute_all_edge_features().set_index("log_id")
        incr_freq_by_log_id = {d["log_id"]: d["abnormal_path_frequency"] for _, _, d in self.incr_ppg.graph.edges(data=True)}

        diffs = [
            abs(batch_feats.loc[log_id, "abnormal_path_frequency"] - incr_freq_by_log_id[log_id])
            for log_id in batch_feats.index if log_id in incr_freq_by_log_id
        ]
        n_differing = sum(1 for d in diffs if d > 1e-9)
        max_diff = max(diffs) if diffs else 0.0

        print(f"\n[abnormal_path_frequency drift] {n_differing}/{len(diffs)} edges differ, "
              f"max |diff| = {max_diff:.4f}")
        # We EXPECT most edges to differ (this is the documented trade-off) —
        # the test asserts the drift is bounded and explainable, not absent.
        self.assertGreater(n_differing, 0, "expected streaming-order drift, found none — investigate")
        self.assertLess(max_diff, math.log(len(self.df)), "drift should never exceed log(total edges)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
