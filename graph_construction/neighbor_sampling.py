"""
neighbor_sampling.py
=======================
Adaptive, relation-aware k-hop neighbor sampling for the heterogeneous
Privilege Propagation Graph, added so HGT (and, for the ablation in item
10/11, GraphSAGE/GAT too) can train on a bounded-size neighborhood as the
graph grows, instead of the current full-batch approach (every model file
today does `model(data)` over the ENTIRE HeteroData every step — see
train.py / model_graphsage.py / model_hgt.py; there is no sampling anywhere
in this codebase before this file. `GraphSAGEWithSampling` in
model_graphsage.py is a docstring-only stub with `pass` as its body, and
train_scalable.py's LinkNeighborLoader/HGTLoader usage passes a single flat
fanout list identically to every relation — see that file's module
docstring — which is exactly the "blind uniform sampling that can crowd out
rare relations" this file exists to avoid).

WHY A CUSTOM SAMPLER, NOT JUST PyG's NeighborLoader/HGTLoader
─────────────────────────────────────────────────────────────────────────
PyG's built-in loaders take one fanout LIST (or, at best, one fanout PER
EDGE TYPE decided up front and never adapted) — they do not implement
"give every populated relation type at least a floor of representation,
then split the remainder by availability" (task requirement 5). Nor do
they expose the "if degree <= MAX_NEIGHBORS use every neighbor, else cap
at MAX_NEIGHBORS" decision as a directly unit-testable function of one
node's OWN degree (requirement 4's explicit "must not depend on total
graph edges"). This file implements that decision and that allocation as
plain, dependency-light functions first (`decide_sample_size`,
`allocate_relation_quota` — pure Python/int arithmetic, no torch, no
randomness), and only then a thin HeteroData-walking sampler on top
(`AdaptiveRelationAwareSampler`) that calls them. Splitting it this way is
deliberate: the two pure functions are exactly what task requirement 13
asks to unit-test, and keeping them torch-free means those tests run
anywhere, including environments with no torch/PyG install (like most of
this codebase's own explainability/HGT/ensemble files were originally
written and shipped in — see those files' "honest caveat" docstrings).
train_scalable.py's existing LinkNeighborLoader/HGTLoader mini-batch
infrastructure is NOT removed or replaced by this file — it stays as
useful headroom for a future move to a DataLoader-based training loop.
This file is what actually satisfies requirements 4/5 today, wired into
train.py as an explicit, opt-in preprocessing step (see build_sampled_view
below and train.py's --sampling flag) so the effect of sampling can be
A/B'd against full-neighborhood training under otherwise identical
conditions (task requirement 10/11) — sampling does not silently change
anything when disabled.

NO LABEL LEAKAGE — BY SIGNATURE, NOT BY POLICY (task requirement 5, 8)
─────────────────────────────────────────────────────────────────────────
`decide_sample_size` and `allocate_relation_quota` take only integers
(a degree, a budget, per-relation neighbor COUNTS). Neither function's
signature has anywhere to put a label even if someone wanted to — there
is no `y` / `is_attack` / `label` parameter anywhere in this module, on
any function, including the HeteroData-level sampler. Test
`test_neighbor_sampling.py::test_sampler_never_touches_y_or_edge_attr`
goes further and asserts this structurally: it runs the sampler on a
HeteroData object that has had its `.y` tensors deleted entirely, and
confirms sampling still runs and produces the same node/edge COUNTS as
the identical graph with labels present — i.e. the sampler's decisions
cannot even be a hidden function of `.y`, because `.y` need not exist at
all for it to run.
`RANDOM` (uniform-within-relation, no ranking) is the only selection rule
within a relation's own available edges — the ranking signals in
node_importance.py (hop_count, privilege_gain, abnormal_path_frequency,
is_privilege_escalation_technique — all independently verified in that
file to be structural, not label-derived) are NOT reused as a sampling
priority here. This is a deliberate choice, not an oversight: reusing them
would still not be target leakage (they're not the label), but it would
entangle two different, independently-auditable design decisions —
"which relations get how much budget" (this file) and "which nodes look
important" (node_importance.py, used for HGTLoader's seed selection in
train_scalable.py) — into one, making each harder to ablate on its own.
`strategy="degree_weighted"` (see SamplingConfig) is provided as an
explicit, opt-in alternative that breaks ties by preferring higher-degree
neighbors within a relation (a structural, non-label signal, motivated by
"a well-connected neighbor contributes more useful message-passing
context") — included so the choice of within-relation selection rule is
itself ablatable, per this file's "every modification must have a clear
technical reason and should be validated experimentally" mandate.

CONFIGURATION (task requirement 7)
─────────────────────────────────────────────────────────────────────────
Every numeric knob lives in SamplingConfig, constructed by train.py from
argparse flags (--max_neighbors, --num_hops, --num_samples_per_relation,
--sampling_strategy, --sampling_seed) — nothing in this file hardcodes a
threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class SamplingConfig:
    """All sampling knobs, in one place, with no value baked in anywhere
    else in this file (task requirement 7)."""

    max_neighbors: int = 50
    """MAX_NEIGHBORS — per-node cap on how many incoming neighbors are
    kept, evaluated against THAT NODE's OWN degree (see decide_sample_size)
    — never against len(data[triple].edge_index) or any other whole-graph
    quantity (task requirement 4's explicit "do NOT make the threshold
    depend on the total number of graph edges")."""

    num_hops: int = 2
    """NUM_HOPS — how many rounds of neighbor expansion outward from the
    seed nodes. 2 matches this dataset's verified max structural chain
    depth (privilege_features.py) but is a free parameter, not hardcoded
    to that number in this file."""

    num_samples_per_relation: int = 5
    """NUM_SAMPLES_PER_RELATION — the minimum number of neighbors a
    per-node sampling decision guarantees to EVERY populated relation
    before splitting whatever budget remains proportionally (task
    requirement 5's "avoid a situation where thousands of ordinary
    read/access edges crowd out rare identity/privilege-related
    relationships"). Only a floor, never a ceiling: a relation with more
    room in the proportional split gets more than this."""

    strategy: str = "relation_aware"
    """"relation_aware" (default, implements requirement 5) | "uniform"
    (pools every relation together and samples uniformly at random,
    provided ONLY as the naive baseline requirement 5 explicitly asks to
    compare against — see run_experiments.py ablation D vs the
    "uniform_sampling" variant) | "full" (no cap at all — every neighbor
    kept regardless of degree; the requirement-4 "GPU has more compute but
    not unlimited memory" case is this compared against a real cap)."""

    within_relation_selection: str = "random"
    """"random" (default) | "degree_weighted" (prefers higher-degree
    neighbors within a relation once that relation's quota is smaller than
    its availability — see module docstring)."""

    seed: int = 42
    """RNG seed — sampling is otherwise non-deterministic, which would
    make the ablation in requirement 11 (tune K on validation, report test
    ONCE) unreproducible."""

    min_per_relation_floor: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if self.max_neighbors <= 0:
            raise ValueError(f"max_neighbors must be positive, got {self.max_neighbors}")
        if self.num_hops <= 0:
            raise ValueError(f"num_hops must be positive, got {self.num_hops}")
        if self.num_samples_per_relation < 0:
            raise ValueError(f"num_samples_per_relation must be >= 0, got {self.num_samples_per_relation}")
        if self.strategy not in ("relation_aware", "uniform", "full"):
            raise ValueError(f"Unknown strategy {self.strategy!r}")
        if self.within_relation_selection not in ("random", "degree_weighted"):
            raise ValueError(f"Unknown within_relation_selection {self.within_relation_selection!r}")


# ══════════════════════════════════════════════════════════════════════════
# 2. PURE, TORCH-FREE DECISION FUNCTIONS (task requirement 13's unit-test
#    target; task requirement 4/5's actual logic)
# ══════════════════════════════════════════════════════════════════════════

def decide_sample_size(degree: int, max_neighbors: int) -> int:
    """
    Task requirement 4, verbatim: if a node's degree is <= max_neighbors,
    use every neighbor; otherwise cap at max_neighbors. A pure function of
    THIS node's own degree and the configured threshold — nothing else.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    return degree if degree <= max_neighbors else max_neighbors


def _largest_remainder_allocate(weights: Dict[str, int], total: int) -> Dict[str, int]:
    """
    Apportions an integer `total` across keys proportionally to `weights`
    (Hamilton's / "largest remainder" method), never allocating a key more
    than its own weight (weights are themselves availabilities here, not
    abstract proportions — see callers). Deterministic given `weights` and
    `total` (ties broken by key order, not by any randomness), which is
    what makes this half of the allocator unit-testable without a seeded
    RNG at all.
    """
    keys = [k for k in weights if weights[k] > 0]
    result = {k: 0 for k in weights}
    if not keys or total <= 0:
        return result
    w_sum = sum(weights[k] for k in keys)
    total = min(total, w_sum)  # never promise more than exists in total across all keys

    raw = {k: weights[k] * total / w_sum for k in keys}
    for k in keys:
        result[k] = min(int(raw[k]), weights[k])
    remainder = total - sum(result.values())

    # Largest-fractional-part-first, skipping any key already at its cap;
    # stable order (sorted by (-fraction, key)) so results are reproducible.
    order = sorted(keys, key=lambda k: (-(raw[k] - int(raw[k])), k))
    idx = 0
    guard = 0
    while remainder > 0 and guard < len(order) * 2 + 1:
        k = order[idx % len(order)]
        if result[k] < weights[k]:
            result[k] += 1
            remainder -= 1
        idx += 1
        guard += 1
    return result


def allocate_relation_quota(
    counts: Dict[EdgeTriple, int],
    budget: int,
    num_samples_per_relation: int,
) -> Dict[EdgeTriple, int]:
    """
    Task requirement 5. `counts`: {relation_triple: available_neighbor_count}
    for ONE node being expanded (every populated triple where this node is
    the destination). `budget`: decide_sample_size()'s output for this
    node. Returns {relation_triple: quota}, quota <= counts[triple] always,
    sum(quota.values()) <= budget always.

    Algorithm (max-min-fair "floor, then proportional remainder"):
      1. If the node's total degree already fits the budget, keep
         everything — no relation is ever trimmed unless the node
         actually exceeds MAX_NEIGHBORS (this is the same "don't sample
         what doesn't need sampling" principle as decide_sample_size,
         just applied per-relation instead of per-node).
      2. Otherwise, give every populated relation a floor of
         min(num_samples_per_relation, its own availability) — this is
         the concrete mechanism preventing a high-volume relation (READ)
         from starving a low-volume one (ASSUMES): ASSUMES gets its floor
         BEFORE READ gets anything beyond its own floor.
      3. If floors alone already exceed the budget (an extreme case: many
         distinct relations, a very small MAX_NEIGHBORS), floors are
         themselves apportioned via the same largest-remainder method,
         which — because it allocates proportionally to each relation's
         OWN availability — still gives a systematic, non-arbitrary
         preference to relations with fewer available edges only in the
         sense of not being able to over-allocate past their own count;
         it does NOT specifically privilege rare relations beyond what
         the floor already guarantees. This edge case is exercised
         directly in test_neighbor_sampling.py.
      4. Whatever budget remains after every floor is met is distributed
         across relations proportionally to their REMAINING availability
         (largest-remainder again) — so a relation with many more edges
         than its floor still ends up with proportionally more total
         slots, it just no longer gets ALL of them at the expense of
         rarer relations getting zero.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    counts = {t: c for t, c in counts.items() if c > 0}
    total = sum(counts.values())
    if total <= budget:
        return dict(counts)

    floor_target = {t: min(num_samples_per_relation, c) for t, c in counts.items()}
    floor_sum = sum(floor_target.values())

    if floor_sum > budget:
        # Even the floors don't fit — apportion the floors themselves.
        return _largest_remainder_allocate(floor_target, budget)

    remaining_budget = budget - floor_sum
    remaining_avail = {t: counts[t] - floor_target[t] for t in counts}
    extra = _largest_remainder_allocate(remaining_avail, remaining_budget)

    quota = {t: floor_target[t] + extra.get(t, 0) for t in counts}
    return quota


def allocate_uniform_quota(counts: Dict[EdgeTriple, int], budget: int) -> Dict[EdgeTriple, int]:
    """
    The naive baseline requirement 5 asks to compare against: pool every
    relation together and split the budget proportionally to raw
    availability, with NO floor for rare relations. Implemented via the
    same largest-remainder allocator so the only difference from
    allocate_relation_quota is the floor step — isolating exactly the
    variable the ablation in run_experiments.py is testing.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    counts = {t: c for t, c in counts.items() if c > 0}
    total = sum(counts.values())
    if total <= budget:
        return dict(counts)
    return _largest_remainder_allocate(counts, budget)


# ══════════════════════════════════════════════════════════════════════════
# 3. HeteroData-LEVEL SAMPLER (torch/PyG-dependent — the thin layer on top
#    of section 2's pure functions)
# ══════════════════════════════════════════════════════════════════════════

def _lazy_torch_imports():
    """Deferred import so section 2 above can be imported and unit-tested
    in an environment with no torch/PyG install at all (matches this
    codebase's existing convention — e.g. data_loader.py's own functions
    vs. model_*.py's torch dependency)."""
    import torch
    from torch_geometric.data import HeteroData
    return torch, HeteroData


class AdaptiveRelationAwareSampler:
    """
    k-hop neighbor sampler over a HeteroData graph, applying
    decide_sample_size + allocate_relation_quota (or the uniform/full
    strategy variants) at every node visited during expansion.

    USAGE
    ─────
        sampler = AdaptiveRelationAwareSampler(SamplingConfig(max_neighbors=50, num_hops=2))
        sampled = sampler.sample(data, edge_types, seed_nodes={"User": [0, 3], "Role": [1]})
        # `sampled` is a new HeteroData: same node types, node feature
        # tensors sliced to the visited subset, `key`/log_id carried
        # through, edge_index re-indexed into the subset, edge direction
        # and (src_type, relation, dst_type) semantics UNCHANGED (task
        # requirement 12 — see test_edge_direction_preserved.py).

    EXPANSION DIRECTION
    ─────────────────────────────────────────────────────────────────────
    Every model in this codebase (model_graphsage.py, model_gat.py,
    model_hgt.py) aggregates messages FROM a triple's src type TO its dst
    type — that's what "edge_index[0]=src, edge_index[1]=dst" +
    HeteroConv/manual-per-relation message passing means. So a seed
    node's OWN embedding depends on its INCOMING edges (it is the dst),
    recursively. This sampler therefore expands via each frontier node's
    incoming edges only — outgoing edges from a frontier node are never
    traversed (that would collect nodes that DEPEND ON the frontier node,
    not nodes the frontier node depends on, and is not what message
    passing needs). This mirrors the direction PyG's own NeighborLoader
    samples in relative to edge_index, and is verified directly in
    test_neighbor_sampling.py::test_expansion_follows_message_passing_direction.
    """

    def __init__(self, config: SamplingConfig):
        self.config = config

    def sample(
        self,
        data,
        edge_types: List[EdgeTriple],
        seed_nodes: Dict[str, Sequence[int]],
        device: Optional[str] = None,
    ):
        """
        Returns a new HeteroData induced over the visited node set,
        containing exactly the sampled edges traversed during expansion
        (their union across all `num_hops` rounds — an edge visited at
        hop 1 and again at hop 2 from a different frontier node is kept
        once). `device`: if given, the returned subgraph's tensors are
        moved there (task requirement 6) — sampling ITSELF (index
        selection) always runs on CPU via Python's `random`, since the
        indices involved are small integers and this keeps sampling
        deterministic and inspectable regardless of what device `data`
        currently lives on; only the FINAL subgraph is placed on `device`.
        """
        torch, HeteroData = _lazy_torch_imports()
        import random as _random
        rng = _random.Random(self.config.seed)

        # {ntype: set of global node indices visited so far}
        visited: Dict[str, set] = {nt: set(seed_nodes.get(nt, [])) for nt in data.node_types}
        frontier: Dict[str, set] = {nt: set(seed_nodes.get(nt, [])) for nt in data.node_types}

        # {triple: set of local edge indices (into data[triple].edge_index) sampled}
        sampled_edges: Dict[EdgeTriple, set] = {t: set() for t in edge_types}

        # Precompute, per triple, per destination node: which local edge
        # indices point at it — built once (not per-hop) since the graph
        # itself doesn't change during sampling.
        incoming_by_dst = self._build_incoming_index(data, edge_types)

        for hop in range(self.config.num_hops):
            next_frontier: Dict[str, set] = {nt: set() for nt in data.node_types}
            any_expanded = False
            for ntype, node_ids in frontier.items():
                for node_id in node_ids:
                    per_relation_indices = {
                        t: incoming_by_dst.get(t, {}).get((ntype, node_id), [])
                        for t in edge_types
                        if t[2] == ntype
                    }
                    per_relation_indices = {t: idxs for t, idxs in per_relation_indices.items() if idxs}
                    if not per_relation_indices:
                        continue
                    any_expanded = True
                    chosen = self._choose_edges_for_node(per_relation_indices, data, rng)
                    for t, local_idxs in chosen.items():
                        for li in local_idxs:
                            if li in sampled_edges[t]:
                                continue
                            sampled_edges[t].add(li)
                            src_idx = int(data[t].edge_index[0, li])
                            src_type = t[0]
                            if src_idx not in visited[src_type]:
                                visited[src_type].add(src_idx)
                                next_frontier[src_type].add(src_idx)
            frontier = next_frontier
            if not any_expanded or not any(frontier.values()):
                break

        return self._build_induced_subgraph(data, edge_types, visited, sampled_edges, device)

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _build_incoming_index(data, edge_types: List[EdgeTriple]) -> Dict[EdgeTriple, Dict[Tuple[str, int], List[int]]]:
        """{triple: {(dst_type, dst_node_id): [local_edge_idx, ...]}} — the
        per-destination adjacency lists that decide_sample_size /
        allocate_relation_quota are applied against."""
        index: Dict[EdgeTriple, Dict[Tuple[str, int], List[int]]] = {}
        for t in edge_types:
            if t not in data.edge_types:
                continue
            dst_type = t[2]
            edge_index = data[t].edge_index
            per_dst: Dict[Tuple[str, int], List[int]] = {}
            dst_row = edge_index[1].tolist()
            for local_i, dst_id in enumerate(dst_row):
                per_dst.setdefault((dst_type, dst_id), []).append(local_i)
            index[t] = per_dst
        return index

    def _choose_edges_for_node(
        self,
        per_relation_indices: Dict[EdgeTriple, List[int]],
        data,
        rng,
    ) -> Dict[EdgeTriple, List[int]]:
        counts = {t: len(idxs) for t, idxs in per_relation_indices.items()}
        degree = sum(counts.values())
        cfg = self.config

        if cfg.strategy == "full":
            budget = degree
        else:
            budget = decide_sample_size(degree, cfg.max_neighbors)

        if degree <= budget:
            return dict(per_relation_indices)

        if cfg.strategy == "uniform":
            quota = allocate_uniform_quota(counts, budget)
        else:  # "relation_aware" (and "full" never reaches here since degree<=budget above)
            quota = allocate_relation_quota(counts, budget, cfg.num_samples_per_relation)

        chosen: Dict[EdgeTriple, List[int]] = {}
        for t, k in quota.items():
            if k <= 0:
                continue
            available = per_relation_indices[t]
            if k >= len(available):
                chosen[t] = list(available)
                continue
            if cfg.within_relation_selection == "degree_weighted":
                chosen[t] = self._degree_weighted_sample(t, available, k, data, rng)
            else:
                chosen[t] = rng.sample(available, k)
        return chosen

    @staticmethod
    def _degree_weighted_sample(triple, available_local_idxs, k, data, rng) -> List[int]:
        """Non-label structural tie-break: prefer neighbors (source-side
        nodes) with higher OWN degree — computed here purely from
        edge_index counts on `triple`'s own source type within this
        triple, not from any node feature column, so it needs no
        knowledge of data_loader.py's NODE_FEATURE_SCHEMA column layout."""
        edge_index = data[triple].edge_index
        src_ids = [int(edge_index[0, li]) for li in available_local_idxs]
        from collections import Counter
        deg_within_triple = Counter(int(edge_index[0, i]) for i in range(edge_index.shape[1]))
        weights = [deg_within_triple[s] for s in src_ids]
        # Weighted sample without replacement via the standard "exponential
        # key" trick, seeded through `rng` for reproducibility — avoids
        # pulling in numpy's RNG (a second, differently-seeded source of
        # randomness) inside a torch-lazy-imported method.
        keyed = sorted(
            zip(available_local_idxs, weights),
            key=lambda pair: rng.random() ** (1.0 / max(pair[1], 1e-6)),
            reverse=True,
        )
        return [li for li, _ in keyed[:k]]

    @staticmethod
    def _build_induced_subgraph(data, edge_types, visited, sampled_edges, device):
        torch, HeteroData = _lazy_torch_imports()
        out = HeteroData()

        old_to_new: Dict[str, Dict[int, int]] = {}
        for ntype in data.node_types:
            ids = sorted(visited.get(ntype, set()))
            old_to_new[ntype] = {old: new for new, old in enumerate(ids)}
            if not ids:
                continue
            idx_tensor = torch.tensor(ids, dtype=torch.long)
            out[ntype].x = data[ntype].x[idx_tensor]
            if hasattr(data[ntype], "key"):
                full_key = data[ntype].key
                out[ntype].key = [full_key[i] for i in ids]

        for t in edge_types:
            if t not in data.edge_types:
                continue
            local_idxs = sorted(sampled_edges.get(t, set()))
            src_type, _, dst_type = t
            if not local_idxs:
                continue
            idx_tensor = torch.tensor(local_idxs, dtype=torch.long)
            old_edge_index = data[t].edge_index[:, idx_tensor]
            new_src = torch.tensor(
                [old_to_new[src_type][int(s)] for s in old_edge_index[0].tolist()], dtype=torch.long
            )
            new_dst = torch.tensor(
                [old_to_new[dst_type][int(d)] for d in old_edge_index[1].tolist()], dtype=torch.long
            )
            out[t].edge_index = torch.stack([new_src, new_dst], dim=0)
            if hasattr(data[t], "edge_attr"):
                out[t].edge_attr = data[t].edge_attr[idx_tensor]
            if hasattr(data[t], "y"):
                out[t].y = data[t].y[idx_tensor]
            if hasattr(data[t], "log_id"):
                full_log_id = data[t].log_id
                out[t].log_id = [full_log_id[i] for i in local_idxs]

        return out.to(device) if device is not None else out


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN-TIME CONVENIENCE — sample a training view seeded from train-mask
#    edges only, so val/test edges are never used as expansion seeds (this
#    keeps requirement 8's "only information available at detection time"
#    property AND avoids test/val leakage through subgraph structure).
# ══════════════════════════════════════════════════════════════════════════

def build_sampled_training_view(
    data,
    edge_types: List[EdgeTriple],
    train_mask_dict: Dict[EdgeTriple, "object"],
    config: SamplingConfig,
    device: Optional[str] = None,
):
    """
    Seeds expansion from the endpoints of every TRAIN-split edge only
    (never val/test edges — those stay evaluated on the full, unsampled
    graph, matching train_scalable.py's existing train_hgt() convention of
    evaluating on full val_data regardless of how training was sampled).
    Returns a HeteroData sized by `config`, plus the projected train mask
    for that returned graph's own (renumbered) edge indices (every edge in
    the returned graph that came from a train-split triple is True; the
    sampler cannot invent or drop WHICH edges are train vs val/test, since
    it only ever samples FROM already-train edges' neighborhoods on the
    context side, but validation/test edges can still appear in the
    returned subgraph as unlabeled context if they happen to lie on a
    sampled path — those are excluded from the returned mask explicitly).
    """
    torch, HeteroData = _lazy_torch_imports()

    seed_nodes: Dict[str, set] = {}
    for t, mask in train_mask_dict.items():
        if t not in data.edge_types:
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        edge_index = data[t].edge_index[:, idx]
        seed_nodes.setdefault(t[0], set()).update(edge_index[0].tolist())
        seed_nodes.setdefault(t[2], set()).update(edge_index[1].tolist())

    sampler = AdaptiveRelationAwareSampler(config)
    sampled = sampler.sample(data, edge_types, seed_nodes, device=device)

    # A sampled edge is "train" iff its (src,rel,dst) triple's ORIGINAL
    # local index was inside train_mask_dict — recovered via log_id, which
    # survives sampling unchanged (see _build_induced_subgraph), rather
    # than via position (sampling reorders/drops edges, so positional
    # alignment with the original mask cannot be assumed).
    projected_train_mask: Dict[EdgeTriple, "object"] = {}
    for t in sampled.edge_types:
        if t not in data.edge_types or not hasattr(data[t], "log_id"):
            projected_train_mask[t] = torch.zeros(sampled[t].edge_index.shape[1], dtype=torch.bool)
            continue
        train_log_ids = set()
        if t in train_mask_dict:
            mask = train_mask_dict[t]
            idx = mask.nonzero(as_tuple=True)[0].tolist()
            full_log_id = data[t].log_id
            train_log_ids = {full_log_id[i] for i in idx}
        sampled_log_ids = sampled[t].log_id
        projected_train_mask[t] = torch.tensor(
            [lid in train_log_ids for lid in sampled_log_ids], dtype=torch.bool
        )
    return sampled, projected_train_mask
