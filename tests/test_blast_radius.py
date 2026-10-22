"""
test_blast_radius.py
====================
Boundary cases of blast_radius.py (Contribution 5), run on a tiny hand-built
PrivilegePropagationGraph -- no Neo4j, no CSV.

What is protected:

  - a principal with no outgoing edges (or absent from the graph) has ZERO reach
    and a zero score, not an exception or a phantom path        -> TestZeroReach
  - max_traversal_depth really bounds the reach                 -> TestTraversalDepth
  - asset categories / critical-asset tiers / exposure weights  -> TestCriticalAssets
  - the highest-impact propagation path and its rendering       -> TestPropagationPaths
  - adding / removing a privilege edge changes reach and report -> TestPrivilegePathChanges
  - BlastRadiusCache only recomputes after invalidate()         -> TestBlastRadiusCache
  - BlastRadiusConfig rejects invalid weightings                -> TestConfigValidation

Fixture graph (account 111111111111):

    User:alice --AssumeRole--> Role:dev --GetSecretValue--> secretsmanager secret  (tier 3)
                                        --GetObject------> s3 bucket                (tier 2)
                                        --DescribeAlarms-> cloudwatch service       (tier 0)
    User:bob   --GetObject--> other s3 bucket
    User:carol  (isolated: present in the graph, no edges)
"""

import dataclasses
import unittest

import blast_radius as br
import privilege_features as pf

K = pf.GraphNodeKey
ACCT = "111111111111"
SECRET = f"arn:aws:secretsmanager:us-east-1:{ACCT}:secret:db-password"
BUCKET = "arn:aws:s3:::logs-bucket"
OTHER_BUCKET = "arn:aws:s3:::other-bucket"
CLOUDWATCH = "monitoring.cloudwatch.amazonaws.com"

ALICE, BOB, CAROL = ("User", "alice"), ("User", "bob"), ("User", "carol")
DEV = ("Role", "dev")

_RESOLVER = None


def resolver():
    global _RESOLVER  # policy_sentry load is the slow part; share one resolver
    if _RESOLVER is None:
        _RESOLVER = pf.ActionAccessLevelResolver()
    return _RESOLVER


def row(i, src, dst, action):
    return {"log_id": f"t.csv:{i}", "source_key": K(*src), "target_key": K(*dst),
            "edge_type": action, "label": 0}


def base_rows():
    return [
        row(0, ALICE, DEV, "AssumeRole"),
        row(1, DEV, ("Resource", SECRET), "GetSecretValue"),
        row(2, DEV, ("Resource", BUCKET), "GetObject"),
        row(3, DEV, ("Service", CLOUDWATCH), "DescribeAlarms"),
        row(4, BOB, ("Resource", OTHER_BUCKET), "GetObject"),
    ]


def make_ppg(rows=None):
    ppg = pf.PrivilegePropagationGraph(resolver()).build_from_rows(base_rows() if rows is None else rows)
    ppg.graph.add_node(CAROL)
    return ppg


def make_engine(ppg=None, **cfg):
    return br.BlastRadiusEngine(ppg or make_ppg(), br.BlastRadiusConfig(**cfg), source_account_id=ACCT)


def add_edge(ppg, src, dst, action, log_id="t.csv:99"):
    ppg.graph.add_edge(src, dst, log_id=log_id, edge_type=action,
                       relation=pf.resolve_relation_type(action, resolver()),
                       access_level=resolver().access_level(action), label=0)


class TestZeroReach(unittest.TestCase):
    def assert_zero_report(self, report):
        self.assertEqual(report.reachable_assets.total, 0)
        self.assertTrue(all(v == 0 for v in report.reachable_assets.counts.values()))
        self.assertEqual(report.score, 0.0)
        self.assertTrue(all(v == 0.0 for v in report.score_components.values()))
        self.assertEqual(report.top_paths, [])
        self.assertEqual(report.critical_assets.critical_asset_count, 0)
        self.assertEqual(report.privilege_reachability.assume_role_chain_depth, 0)
        self.assertFalse(report.privilege_reachability.administrator_reachable)
        self.assertTrue(report.is_lower_bound)

    def test_isolated_principal_has_zero_reach_and_score(self):
        self.assert_zero_report(make_engine().compute(CAROL, resolver()))

    def test_principal_absent_from_graph_is_zero_not_an_error(self):
        self.assert_zero_report(make_engine().compute(("User", "nobody"), resolver()))

    def test_resource_node_has_no_outgoing_reach(self):
        self.assert_zero_report(make_engine().compute(("Resource", SECRET), resolver()))

    def test_zero_report_summary_still_states_lower_bound(self):
        text = make_engine().compute(CAROL, resolver()).summary()
        self.assertIn("LOWER BOUND", text)
        self.assertIn("Reachable assets: 0", text)
        self.assertNotIn("Highest-impact propagation path", text)

    def test_metrics_recorded_for_zero_reach(self):
        engine = make_engine()
        engine.compute(CAROL, resolver())
        self.assertEqual(engine.metrics.summary()["n"], 1)
        self.assertEqual(engine.metrics.records[0].nodes_visited, 0)


class TestTraversalDepth(unittest.TestCase):
    def test_depth_limits_reach(self):
        for depth, expected in ((0, 0), (1, 1), (2, 4), (4, 4)):
            with self.subTest(depth=depth):
                report = make_engine(max_traversal_depth=depth).compute(ALICE, resolver())
                self.assertEqual(report.reachable_assets.total, expected)

    def test_depth_one_stops_at_the_assumed_role(self):
        report = make_engine(max_traversal_depth=1).compute(ALICE, resolver())
        self.assertEqual(report.reachable_assets.counts["iam_role"], 1)
        self.assertEqual(report.critical_assets.critical_asset_count, 0)
        self.assertEqual([p.nodes[-1] for p in report.top_paths], [DEV])

    def test_reachable_within_reports_hop_distance_without_source(self):
        g = br.NetworkXReachabilityGraph(make_ppg())
        self.assertEqual(g.reachable_within(ALICE, 2), {
            DEV: 1, ("Resource", SECRET): 2, ("Resource", BUCKET): 2, ("Service", CLOUDWATCH): 2})
        self.assertEqual(g.reachable_within(("User", "nobody"), 2), {})

    def test_assume_role_chain_depth(self):
        rows = base_rows() + [row(5, DEV, ("Role", "prod"), "AssumeRole")]
        report = make_engine(make_ppg(rows)).compute(ALICE, resolver())
        self.assertEqual(report.privilege_reachability.assume_role_chain_depth, 2)
        self.assertEqual(report.score_components["privilege_escalation"], 1.0)


class TestCriticalAssets(unittest.TestCase):
    def test_asset_categories(self):
        counts = make_engine().compute(ALICE, resolver()).reachable_assets.counts
        self.assertEqual(set(counts), set(br.ASSET_CATEGORY_LABELS))
        self.assertEqual(counts["iam_role"], 1)
        self.assertEqual(counts["secretsmanager_secret"], 1)
        self.assertEqual(counts["s3_bucket"], 1)
        self.assertEqual(counts["other"], 1)  # cloudwatch service
        self.assertEqual(sum(counts.values()), 4)

    def test_critical_assets_respect_min_tier(self):
        r2 = make_engine(critical_asset_min_tier=2).compute(ALICE, resolver()).critical_assets
        self.assertEqual(set(r2.critical_assets), {("Resource", SECRET), ("Resource", BUCKET)})
        r3 = make_engine(critical_asset_min_tier=3).compute(ALICE, resolver()).critical_assets
        self.assertEqual(r3.critical_assets, [("Resource", SECRET)])

    def test_exposure_is_sum_of_tier_weights_not_a_count(self):
        reach = {"t3": 1, "t2": 1, "t1": 1, "t0": 1}
        res = br.CriticalAssetScorer(br.BlastRadiusConfig()).score(reach, {"t3": 3, "t2": 2, "t1": 1, "t0": 0})
        self.assertEqual(res.critical_asset_count, 2)
        self.assertAlmostEqual(res.exposure_score, 1.0 + 0.65 + 0.35 + 0.1)

    def test_unresolved_tier_falls_back_to_neutral_weight(self):
        w = br.CriticalityWeights()
        self.assertEqual(w.weight_for_tier(-1), w.tier_to_weight[1])
        self.assertEqual(w.weight_for_tier(7), w.tier_to_weight[1])

    def test_custom_criticality_weights_are_used(self):
        cfg = br.BlastRadiusConfig(criticality=br.CriticalityWeights({3: 2.0, 2: 0.0, 1: 0.0, 0: 0.0}))
        res = br.CriticalAssetScorer(cfg).score({"a": 1, "b": 1}, {"a": 3, "b": 2})
        self.assertEqual(res.exposure_score, 2.0)

    def test_cross_account_resource_is_counted(self):
        rows = base_rows() + [row(5, DEV, ("Resource", "arn:aws:s3:us-east-1:222222222222:accesspoint/x"),
                                  "GetObject")]
        report = make_engine(make_ppg(rows)).compute(ALICE, resolver())
        self.assertEqual(report.privilege_reachability.cross_account_ids_reachable, {"222222222222"})
        self.assertEqual(report.score_components["cross_account"], 1.0)
        same = make_engine().compute(ALICE, resolver())
        self.assertEqual(same.privilege_reachability.cross_account_ids_reachable, set())

    def test_service_linked_role_abuse_only_flagged_for_users(self):
        slr = ("Role", "AWSServiceRoleForSupport")
        rows = base_rows() + [row(5, BOB, slr, "AssumeRole"), row(6, DEV, slr, "AssumeRole")]
        engine = make_engine(make_ppg(rows))
        self.assertTrue(engine.compute(BOB, resolver()).privilege_reachability.service_linked_role_abuse_suspected)
        self.assertFalse(engine.compute(DEV, resolver()).privilege_reachability.service_linked_role_abuse_suspected)

    def test_score_stays_in_unit_interval_when_saturated(self):
        rows = [row(i, ALICE, ("Resource", f"arn:aws:iam::{ACCT}:policy/p{i}"), "AttachUserPolicy")
                for i in range(30)]
        rows += [row(100, ALICE, ("Role", "r1"), "AssumeRole"), row(101, ("Role", "r1"), ("Role", "r2"), "AssumeRole"),
                 row(102, ("Role", "r2"), ("Resource", "arn:aws:s3:eu-west-1:999999999999:accesspoint/z"),
                     "GetObject")]
        report = make_engine(make_ppg(rows), reachable_assets_saturation=5).compute(ALICE, resolver())
        self.assertGreater(report.score, 0.5)
        self.assertLessEqual(report.score, 1.0)
        for name, v in report.score_components.items():
            self.assertTrue(0.0 <= v <= 1.0, name)


class TestPropagationPaths(unittest.TestCase):
    def test_highest_impact_path_reaches_the_secret(self):
        top = make_engine().compute(ALICE, resolver()).top_paths
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0].nodes, [ALICE, DEV, ("Resource", SECRET)])
        self.assertEqual([e["relation"] for e in top[0].edges], ["ASSUMES", "READ"])
        self.assertEqual(top[0].impact_score, 1.0)
        self.assertEqual([p.impact_score for p in top], sorted((p.impact_score for p in top), reverse=True))

    def test_top_k_limits_candidates(self):
        engine = make_engine()
        reach = engine.graph.reachable_within(ALICE, 4)
        self.assertEqual(len(engine.path_extractor.extract(ALICE, reach, engine._sensitivity_by_node, top_k=1)), 1)
        self.assertEqual(len(engine.path_extractor.extract(ALICE, reach, engine._sensitivity_by_node, top_k=10)), 4)

    def test_render_and_summary(self):
        report = make_engine().compute(ALICE, resolver())
        self.assertEqual(report.top_paths[0].render(),
                         f"User:alice\n  ↓\nRole:dev\n  ↓\nResource:{SECRET}")
        text = report.summary()
        self.assertIn("Highest-impact propagation path", text)
        self.assertIn(f"Resource:{SECRET}", text)
        self.assertIn("NOT OBSERVABLE", text)  # PassRole never claimed as zero

    def test_parallel_edges_resolve_to_first_inserted(self):
        ppg = make_ppg()
        add_edge(ppg, ALICE, DEV, "GetRole", log_id="a.csv:0")  # second parallel alice->dev edge
        edges = br.NetworkXReachabilityGraph(ppg).edges_on_shortest_path(ALICE, DEV)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["edge_type"], "AssumeRole")


class TestPrivilegePathChanges(unittest.TestCase):
    def test_adding_assume_edge_expands_reach_and_score(self):
        ppg = make_ppg()
        engine = make_engine(ppg)
        before = engine.compute(BOB, resolver())
        add_edge(ppg, BOB, DEV, "AssumeRole")
        after = engine.compute(BOB, resolver())
        self.assertEqual(before.reachable_assets.total, 1)
        self.assertEqual(after.reachable_assets.total, 5)
        self.assertGreater(after.score, before.score)
        self.assertEqual(after.privilege_reachability.assume_role_chain_depth, 1)
        self.assertEqual(after.top_paths[0].nodes[-1], ("Resource", SECRET))

    def test_removing_assume_edge_cuts_reach_to_zero(self):
        ppg = make_ppg()
        engine = make_engine(ppg)
        ppg.graph.remove_edge(ALICE, DEV)
        report = engine.compute(ALICE, resolver())
        self.assertEqual(report.reachable_assets.total, 0)
        self.assertEqual(report.score, 0.0)

    def test_permissions_management_edge_flags_admin_proxy(self):
        rows = base_rows() + [row(5, DEV, ("Role", "admin"), "AttachRolePolicy")]
        report = make_engine(make_ppg(rows)).compute(ALICE, resolver())
        pr = report.privilege_reachability
        self.assertTrue(pr.administrator_reachable)
        self.assertTrue(pr.administrator_reachable_is_proxy)
        self.assertFalse(make_engine().compute(ALICE, resolver()).privilege_reachability.administrator_reachable)

    @unittest.expectedFailure
    def test_admin_proxy_sees_permissions_management_on_a_parallel_edge(self):
        """KNOWN DEFECT: PrivilegeReachabilityAnalyzer.analyze decides
        administrator_reachable only from the edges returned by
        NetworkXReachabilityGraph.edges_on_shortest_path, which keeps ONE edge
        per hop (the lowest MultiDiGraph key = first inserted). If a principal
        first did a READ on a role (GetRole) and later a PERMISSIONS_MANAGEMENT
        action on the same role (AttachRolePolicy), the escalation edge is a
        higher-key parallel edge and is never inspected, so the admin proxy is
        False -- while the same events in the reverse order give True. The
        result depends on event order, and a real escalation is missed."""
        rows = base_rows() + [row(5, BOB, ("Role", "admin"), "GetRole"),
                              row(6, BOB, ("Role", "admin"), "AttachRolePolicy")]
        report = make_engine(make_ppg(rows)).compute(BOB, resolver())
        self.assertTrue(report.privilege_reachability.administrator_reachable)


class TestBlastRadiusCache(unittest.TestCase):
    def test_hit_then_lazy_invalidation_recomputes(self):
        ppg = make_ppg()
        engine, cache = make_engine(ppg), br.BlastRadiusCache()
        first = cache.get_or_compute(BOB, engine, resolver())
        self.assertIs(cache.get_or_compute(BOB, engine, resolver()), first)
        add_edge(ppg, BOB, DEV, "AssumeRole")
        self.assertIs(cache.get_or_compute(BOB, engine, resolver()), first)  # not invalidated: stale by design
        cache.invalidate(BOB)
        self.assertEqual(cache.stats()["stale_entries"], 1)
        fresh = cache.get_or_compute(BOB, engine, resolver())
        self.assertIsNot(fresh, first)
        self.assertEqual(fresh.reachable_assets.total, 5)
        self.assertEqual(cache.stats(), {"hits": 2, "misses": 2, "hit_ratio": 0.5, "cached_entries": 1,
                                         "stale_entries": 0, "invalidations": 1})

    def test_invalidating_uncached_principal_is_a_no_op(self):
        cache = br.BlastRadiusCache()
        cache.invalidate(ALICE)
        self.assertEqual(cache.stats()["invalidations"], 0)
        self.assertEqual(cache.stats()["hit_ratio"], 0.0)

    def test_invalidation_is_per_principal(self):
        engine, cache = make_engine(), br.BlastRadiusCache()
        a = cache.get_or_compute(ALICE, engine, resolver())
        cache.get_or_compute(BOB, engine, resolver())
        cache.invalidate(BOB)
        self.assertIs(cache.get_or_compute(ALICE, engine, resolver()), a)

    @unittest.expectedFailure
    def test_recompute_after_invalidate_scores_a_newly_added_target(self):
        """KNOWN DEFECT: BlastRadiusEngine builds its service / resource_type /
        sensitivity index once, in __init__ (_index_targets). A target node
        added to the graph afterwards (the incremental path the cache exists
        for) is absent from that index, so after invalidate() + recompute a
        newly reached Secrets Manager secret is categorised "other" with no
        sensitivity: critical_asset_count stays 0. Nothing in blast_radius.py
        re-indexes; infer.py calls engine._index_targets() only AFTER
        get_or_compute, so the report for the triggering event is still wrong."""
        ppg = make_ppg()
        engine, cache = make_engine(ppg), br.BlastRadiusCache()
        cache.get_or_compute(BOB, engine, resolver())
        new_secret = ("Resource", f"arn:aws:secretsmanager:us-east-1:{ACCT}:secret:new")
        add_edge(ppg, BOB, new_secret, "GetSecretValue")
        cache.invalidate(BOB)
        report = cache.get_or_compute(BOB, engine, resolver())
        self.assertEqual(report.reachable_assets.counts["secretsmanager_secret"], 1)
        self.assertIn(new_secret, report.critical_assets.critical_assets)


class TestConfigValidation(unittest.TestCase):
    def test_default_weights_sum_to_one(self):
        br.BlastRadiusConfig()

    def test_weights_not_summing_to_one_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            br.BlastRadiusConfig(w_reachable_assets=0.5)
        self.assertIn("sum to 1.0", str(ctx.exception))

    def test_float_rounding_tolerated(self):
        br.BlastRadiusConfig(w_reachable_assets=0.1 + 0.2, w_privilege_escalation=0.25, w_critical_asset_exposure=0.30,
                             w_cross_service=0.10, w_cross_account=0.05)

    def test_config_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            br.BlastRadiusConfig().max_traversal_depth = 9

    @unittest.expectedFailure
    def test_negative_weight_rejected(self):
        """KNOWN DEFECT: __post_init__ only checks that the five component
        weights SUM to 1.0. Negative weights (e.g. 1.25 and -0.25) pass, which
        lets the blast-radius score -- a convex combination of [0,1]
        sub-scores by design -- go below 0 or above 1."""
        with self.assertRaises(ValueError):
            br.BlastRadiusConfig(w_reachable_assets=1.25, w_privilege_escalation=-0.25,
                                 w_critical_asset_exposure=0.0, w_cross_service=0.0, w_cross_account=0.0)


if __name__ == "__main__":
    unittest.main()
