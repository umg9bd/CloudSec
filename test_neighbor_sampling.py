"""
test_neighbor_sampling.py
============================
Unit tests for neighbor_sampling.py — task requirement 13's "add unit
tests for the sampler" and "add tests confirming edge direction and edge
types are preserved", covering both the pure allocation functions (run
with no torch/PyG needed) and the HeteroData-level sampler (skipped if
torch/torch_geometric are not importable, same convention as
test_model_hgt.py / test_ensemble.py already use).
"""

from __future__ import annotations

import pytest

from neighbor_sampling import (
    SamplingConfig,
    decide_sample_size,
    allocate_relation_quota,
    allocate_uniform_quota,
    _largest_remainder_allocate,
)


# ══════════════════════════════════════════════════════════════════════════
# Section 2 (pure, torch-free) — decide_sample_size
# ══════════════════════════════════════════════════════════════════════════

class TestDecideSampleSize:
    def test_degree_at_or_below_threshold_keeps_everything(self):
        assert decide_sample_size(20, max_neighbors=50) == 20
        assert decide_sample_size(50, max_neighbors=50) == 50  # boundary: equal counts as "fits"
        assert decide_sample_size(0, max_neighbors=50) == 0

    def test_degree_above_threshold_is_capped(self):
        assert decide_sample_size(5000, max_neighbors=50) == 50
        assert decide_sample_size(51, max_neighbors=50) == 50

    def test_negative_degree_rejected(self):
        with pytest.raises(ValueError):
            decide_sample_size(-1, max_neighbors=50)

    def test_decision_is_independent_of_anything_but_this_nodes_own_degree(self):
        """Task requirement 4: 'do NOT make the threshold depend on the
        total number of graph edges'. decide_sample_size's signature is
        (degree, max_neighbors) — there is nowhere to even pass a
        whole-graph edge count, so two nodes with the same degree get the
        same decision no matter how large the rest of the graph is."""
        small_graph_node_degree = 30
        huge_graph_node_degree = 30  # same node-local degree, imagine graph has 10M other edges elsewhere
        assert decide_sample_size(small_graph_node_degree, 50) == decide_sample_size(huge_graph_node_degree, 50)


# ══════════════════════════════════════════════════════════════════════════
# Section 2 — allocate_relation_quota (the actual anti-crowding-out logic)
# ══════════════════════════════════════════════════════════════════════════

RARE = ("User", "ASSUMES", "Role")
COMMON = ("User", "READ", "Resource")
MID = ("User", "WRITE", "Resource")


class TestAllocateRelationQuota:
    def test_returns_everything_when_it_fits(self):
        counts = {RARE: 2, COMMON: 10}
        q = allocate_relation_quota(counts, budget=50, num_samples_per_relation=5)
        assert q == counts

    def test_rare_relation_is_protected_from_crowding_out(self):
        """The scenario task requirement 5 describes verbatim: a node
        with 2 ASSUMES edges and 5000 READ edges, budget 50. Under a
        naive proportional-only split, 2/5002 * 50 rounds to 0 — this
        must not happen."""
        counts = {RARE: 2, COMMON: 5000}
        q = allocate_relation_quota(counts, budget=50, num_samples_per_relation=5)
        assert q[RARE] == 2  # floor >= availability, so it keeps everything it has
        assert q[COMMON] <= counts[COMMON]
        assert sum(q.values()) <= 50

    def test_never_exceeds_availability_per_relation(self):
        counts = {RARE: 2, COMMON: 5000, MID: 40}
        q = allocate_relation_quota(counts, budget=50, num_samples_per_relation=5)
        for t, k in q.items():
            assert k <= counts[t]

    def test_never_exceeds_budget(self):
        counts = {RARE: 2, COMMON: 5000, MID: 40}
        for budget in (1, 5, 10, 50, 100):
            q = allocate_relation_quota(counts, budget=budget, num_samples_per_relation=5)
            assert sum(q.values()) <= budget

    def test_remainder_split_favours_larger_relation_in_absolute_terms(self):
        """Rare relations are PROTECTED (floor), not favoured over common
        ones once floors are met — a relation with more availability
        should still end up with more total slots than a smaller one,
        just not zero."""
        counts = {RARE: 2, COMMON: 5000, MID: 40}
        q = allocate_relation_quota(counts, budget=50, num_samples_per_relation=5)
        assert q[COMMON] > q[MID] > 0
        assert q[RARE] < q[MID]

    def test_extreme_case_many_relations_tiny_budget(self):
        """More distinct relations than the budget has room for floors —
        must degrade gracefully (no crash, no over-budget, no negative)."""
        counts = {("T", f"REL{i}", "T"): 3 for i in range(10)}
        q = allocate_relation_quota(counts, budget=4, num_samples_per_relation=2)
        assert sum(q.values()) <= 4
        assert all(k >= 0 for k in q.values())
        assert all(q[t] <= counts[t] for t in counts)

    def test_zero_budget_returns_all_zero(self):
        counts = {RARE: 2, COMMON: 10}
        q = allocate_relation_quota(counts, budget=0, num_samples_per_relation=5)
        assert sum(q.values()) == 0

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError):
            allocate_relation_quota({RARE: 2}, budget=-1, num_samples_per_relation=5)

    def test_zero_floor_behaves_like_pure_proportional_split(self):
        """num_samples_per_relation=0 removes the floor protection entirely
        — included so the floor's effect can be directly ablated against
        (see test comparing to allocate_uniform_quota below)."""
        counts = {RARE: 2, COMMON: 5000}
        q = allocate_relation_quota(counts, budget=50, num_samples_per_relation=0)
        assert q == allocate_uniform_quota(counts, budget=50)


# ══════════════════════════════════════════════════════════════════════════
# relation_aware vs uniform — the actual A/B the "why sampling this way"
# section of the final report is based on
# ══════════════════════════════════════════════════════════════════════════

class TestRelationAwareVsUniformBaseline:
    def test_uniform_baseline_crowds_out_the_rare_relation(self):
        """Demonstrates the failure mode requirement 5 warns against,
        concretely, on the naive baseline — this is what motivates
        relation_aware existing at all, and is the regression guard for
        "why sampling this way" in FINAL_REPORT.md."""
        counts = {RARE: 2, COMMON: 5000}
        q_uniform = allocate_uniform_quota(counts, budget=50)
        assert q_uniform[RARE] == 0

    def test_relation_aware_does_not(self):
        counts = {RARE: 2, COMMON: 5000}
        q_aware = allocate_relation_quota(counts, budget=50, num_samples_per_relation=5)
        assert q_aware[RARE] > 0

    def test_both_strategies_respect_the_same_hard_constraints(self):
        counts = {RARE: 2, COMMON: 5000, MID: 40}
        for q in (
            allocate_uniform_quota(counts, budget=50),
            allocate_relation_quota(counts, budget=50, num_samples_per_relation=5),
        ):
            assert sum(q.values()) <= 50
            assert all(q[t] <= counts[t] for t in counts)


# ══════════════════════════════════════════════════════════════════════════
# _largest_remainder_allocate — the shared apportionment primitive
# ══════════════════════════════════════════════════════════════════════════

class TestLargestRemainderAllocate:
    def test_exact_division(self):
        assert _largest_remainder_allocate({"a": 10, "b": 10}, 10) == {"a": 5, "b": 5}

    def test_total_matches_request_when_available(self):
        weights = {"a": 7, "b": 3, "c": 1}
        result = _largest_remainder_allocate(weights, 8)
        assert sum(result.values()) == 8

    def test_never_exceeds_own_weight(self):
        weights = {"a": 1, "b": 100}
        result = _largest_remainder_allocate(weights, 50)
        assert result["a"] <= 1

    def test_zero_total_is_all_zero(self):
        assert _largest_remainder_allocate({"a": 5, "b": 5}, 0) == {"a": 0, "b": 0}

    def test_deterministic_across_repeated_calls(self):
        weights = {"a": 7, "b": 3, "c": 5, "d": 2}
        r1 = _largest_remainder_allocate(weights, 9)
        r2 = _largest_remainder_allocate(weights, 9)
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════════
# SamplingConfig validation
# ══════════════════════════════════════════════════════════════════════════

class TestSamplingConfig:
    def test_defaults_are_valid(self):
        SamplingConfig()  # must not raise

    @pytest.mark.parametrize("kwargs", [
        {"max_neighbors": 0},
        {"max_neighbors": -5},
        {"num_hops": 0},
        {"num_samples_per_relation": -1},
        {"strategy": "not_a_real_strategy"},
        {"within_relation_selection": "not_a_real_selection"},
    ])
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SamplingConfig(**kwargs)


# ══════════════════════════════════════════════════════════════════════════
# Section 3 (HeteroData-level sampler) — skipped without torch/PyG,
# same convention test_model_hgt.py / test_ensemble.py already use.
# ══════════════════════════════════════════════════════════════════════════

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData  # noqa: E402
from neighbor_sampling import AdaptiveRelationAwareSampler  # noqa: E402


def _build_toy_graph():
    """
    User(2) --ASSUMES--> Role(1) --READ--> Resource(30)
    Role0's OWN incoming edges: 1 ASSUMES edge (rare) — used to test that
    expansion from Role finds User, not Resource (direction test).
    Resource's incoming edges: 30 READ edges from Role0 (common) — used to
    test per-node capping/quota when Resource is NOT the node being
    capped; the crowding scenario is tested at Role/Resource level with a
    second relation added in test_role_with_two_incoming_relations_is_not_crowded_out.
    """
    data = HeteroData()
    data["User"].x = torch.randn(2, 3)
    data["Role"].x = torch.randn(1, 3)
    data["Resource"].x = torch.randn(30, 3)
    data["User"].key = ["u0", "u1"]
    data["Role"].key = ["r0"]
    data["Resource"].key = [f"res{i}" for i in range(30)]

    data["User", "ASSUMES", "Role"].edge_index = torch.tensor([[0], [0]])
    data["User", "ASSUMES", "Role"].edge_attr = torch.randn(1, 4)
    data["User", "ASSUMES", "Role"].y = torch.tensor([0])
    data["User", "ASSUMES", "Role"].log_id = ["e_assume"]

    data["Role", "READ", "Resource"].edge_index = torch.stack(
        [torch.zeros(30, dtype=torch.long), torch.arange(30)]
    )
    data["Role", "READ", "Resource"].edge_attr = torch.randn(30, 4)
    data["Role", "READ", "Resource"].y = torch.zeros(30, dtype=torch.long)
    data["Role", "READ", "Resource"].log_id = [f"e_read{i}" for i in range(30)]
    return data


EDGE_TYPES = [("User", "ASSUMES", "Role"), ("Role", "READ", "Resource")]


class TestAdaptiveRelationAwareSampler:
    def test_expansion_follows_message_passing_direction(self):
        """Seeding from Role must reach User (Role's incoming ASSUMES
        edge) and must NOT reach Resource (Role -> Resource is OUTGOING
        from Role, i.e. Resource depends on Role, not the reverse) —
        this is exactly what forward()'s message passing needs."""
        data = _build_toy_graph()
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=100, num_hops=1, seed=0))
        sub = sampler.sample(data, EDGE_TYPES, seed_nodes={"Role": [0]})
        assert "User" in sub.node_types
        assert "Resource" not in sub.node_types or sub["Resource"].x.shape[0] == 0

    def test_two_hop_expansion_reaches_second_hop_neighbors(self):
        data = _build_toy_graph()
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=100, num_hops=2, seed=0))
        sub = sampler.sample(data, EDGE_TYPES, seed_nodes={"Resource": [5]})
        assert "Role" in sub.node_types
        assert "User" in sub.node_types  # only reachable via the 2nd hop

    def test_one_hop_expansion_does_not_reach_second_hop_neighbors(self):
        data = _build_toy_graph()
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=100, num_hops=1, seed=0))
        sub = sampler.sample(data, EDGE_TYPES, seed_nodes={"Resource": [5]})
        assert "Role" in sub.node_types
        assert "User" not in sub.node_types

    def test_high_degree_node_is_capped_at_max_neighbors(self):
        data = _build_toy_graph()  # Resource(0..29) all feed edges INTO nothing further, but
        # Role0 itself has 30 outgoing READ edges — irrelevant to Role's OWN cap (those are not
        # Role's incoming edges). To test the cap, seed from a node whose incoming degree is large:
        # here, that's not directly Role (only 1 incoming edge) — build a dedicated high-degree case.
        data2 = HeteroData()
        data2["Role"].x = torch.randn(1, 3)
        data2["Resource"].x = torch.randn(500, 3)
        data2["Resource", "REVREAD", "Role"].edge_index = torch.stack(
            [torch.arange(500), torch.zeros(500, dtype=torch.long)]
        )
        data2["Resource", "REVREAD", "Role"].edge_attr = torch.randn(500, 2)
        data2["Resource", "REVREAD", "Role"].log_id = [f"e{i}" for i in range(500)]
        edge_types2 = [("Resource", "REVREAD", "Role")]
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=25, num_hops=1, seed=0))
        sub = sampler.sample(data2, edge_types2, seed_nodes={"Role": [0]})
        assert sub[("Resource", "REVREAD", "Role")].edge_index.shape[1] <= 25

    def test_low_degree_node_keeps_full_neighborhood(self):
        data = _build_toy_graph()
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=100, num_hops=1, seed=0))
        sub = sampler.sample(data, EDGE_TYPES, seed_nodes={"Role": [0]})
        assert sub[("User", "ASSUMES", "Role")].edge_index.shape[1] == 1  # the only edge available

    def test_role_with_two_incoming_relations_is_not_crowded_out(self):
        """The full end-to-end version of TestRelationAwareVsUniformBaseline,
        at the HeteroData level: a Resource with 2 rare TAGGED edges and
        4000 common READ edges, both incoming. relation_aware must keep
        at least one TAGGED edge; uniform must not (regression-style
        assertion on real sampler output, not just the pure function)."""
        data = HeteroData()
        data["Role"].x = torch.randn(1, 3)
        data["User"].x = torch.randn(1, 3)
        data["Resource"].x = torch.randn(1, 3)
        n_read = 4000
        data["Role", "READ", "Resource"].edge_index = torch.stack(
            [torch.zeros(n_read, dtype=torch.long), torch.zeros(n_read, dtype=torch.long)]
        )
        data["Role", "READ", "Resource"].edge_attr = torch.randn(n_read, 2)
        data["Role", "READ", "Resource"].log_id = [f"r{i}" for i in range(n_read)]
        data["User", "TAGGED", "Resource"].edge_index = torch.tensor([[0, 0], [0, 0]])
        data["User", "TAGGED", "Resource"].edge_attr = torch.randn(2, 2)
        data["User", "TAGGED", "Resource"].log_id = ["t0", "t1"]
        edge_types = [("Role", "READ", "Resource"), ("User", "TAGGED", "Resource")]

        aware = AdaptiveRelationAwareSampler(
            SamplingConfig(max_neighbors=50, num_hops=1, num_samples_per_relation=5, strategy="relation_aware", seed=0)
        )
        uniform = AdaptiveRelationAwareSampler(
            SamplingConfig(max_neighbors=50, num_hops=1, strategy="uniform", seed=0)
        )
        sub_aware = aware.sample(data, edge_types, seed_nodes={"Resource": [0]})
        sub_uniform = uniform.sample(data, edge_types, seed_nodes={"Resource": [0]})

        assert ("User", "TAGGED", "Resource") in sub_aware.edge_types
        assert ("User", "TAGGED", "Resource") not in sub_uniform.edge_types

    def test_edge_direction_and_type_are_never_altered(self):
        """Task requirement 12/13: no (src,rel,dst) triple is ever
        flipped or renamed by sampling."""
        data = _build_toy_graph()
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=5, num_hops=2, seed=0))
        sub = sampler.sample(data, EDGE_TYPES, seed_nodes={"Resource": [0, 1, 2]})
        for t in sub.edge_types:
            assert t in EDGE_TYPES  # every triple in the output was a triple in the input, verbatim
            src_type, _, dst_type = t
            assert src_type in sub.node_types
            assert dst_type in sub.node_types
            ei = sub[t].edge_index
            assert ei.shape[0] == 2
            assert int(ei.max()) < sub[src_type].x.shape[0] or int(ei[0].max()) < sub[src_type].x.shape[0]
            assert int(ei[1].max()) < sub[dst_type].x.shape[0]

    def test_sampler_never_touches_y_or_edge_attr(self):
        """Structural leakage guard: build the identical graph with and
        without `.y`/`.edge_attr` at all, and confirm sampling produces
        the SAME node/edge counts either way — proving the sampler's
        decisions cannot be a function of label or edge feature content,
        because they still run identically when that content is absent."""
        data_with_labels = _build_toy_graph()
        data_no_labels = _build_toy_graph()
        for t in EDGE_TYPES:
            del data_no_labels[t].y
            del data_no_labels[t].edge_attr

        cfg = SamplingConfig(max_neighbors=10, num_hops=2, num_samples_per_relation=3, seed=7)
        sub_with = AdaptiveRelationAwareSampler(cfg).sample(data_with_labels, EDGE_TYPES, seed_nodes={"Resource": [0, 1, 2, 3]})
        sub_without = AdaptiveRelationAwareSampler(cfg).sample(data_no_labels, EDGE_TYPES, seed_nodes={"Resource": [0, 1, 2, 3]})

        assert set(sub_with.node_types) == set(sub_without.node_types)
        for nt in sub_with.node_types:
            assert sub_with[nt].x.shape[0] == sub_without[nt].x.shape[0]
        assert set(sub_with.edge_types) == set(sub_without.edge_types)
        for t in sub_with.edge_types:
            assert sub_with[t].edge_index.shape[1] == sub_without[t].edge_index.shape[1]

    def test_same_seed_is_reproducible(self):
        data = _build_toy_graph()
        cfg = SamplingConfig(max_neighbors=3, num_hops=1, seed=123)
        sub1 = AdaptiveRelationAwareSampler(cfg).sample(data, EDGE_TYPES, seed_nodes={"Resource": [0]})
        sub2 = AdaptiveRelationAwareSampler(cfg).sample(data, EDGE_TYPES, seed_nodes={"Resource": [0]})
        for t in sub1.edge_types:
            assert torch.equal(sub1[t].edge_index, sub2[t].edge_index)

    def test_device_placement(self):
        data = _build_toy_graph()
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=10, num_hops=1, seed=0))
        sub = sampler.sample(data, EDGE_TYPES, seed_nodes={"Role": [0]}, device="cpu")
        for nt in sub.node_types:
            assert sub[nt].x.device.type == "cpu"


class TestBuildSampledTrainingView:
    def test_val_and_test_edges_are_never_used_as_seeds(self):
        from neighbor_sampling import build_sampled_training_view

        data = _build_toy_graph()
        n_read = data[("Role", "READ", "Resource")].edge_index.shape[1]
        # Only the FIRST read edge is "train"; everything else is held out.
        train_mask = {
            ("User", "ASSUMES", "Role"): torch.tensor([True]),
            ("Role", "READ", "Resource"): torch.zeros(n_read, dtype=torch.bool),
        }
        train_mask[("Role", "READ", "Resource")][0] = True

        cfg = SamplingConfig(max_neighbors=50, num_hops=1, seed=0)
        sampled, projected = build_sampled_training_view(data, EDGE_TYPES, train_mask, cfg)

        # Only log_id "e_read0" may be marked train in the projected mask.
        read_t = ("Role", "READ", "Resource")
        if read_t in sampled.edge_types:
            kept_log_ids = sampled[read_t].log_id
            for lid, is_train in zip(kept_log_ids, projected[read_t].tolist()):
                if is_train:
                    assert lid == "e_read0"
