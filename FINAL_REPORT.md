# CloudSec GNN Pipeline — HGT + Neighbor Sampling: Final Report

This report follows the 15-point structure requested. It documents an
**audit of the existing codebase first**, then the changes made on top of
it. Every number in this report came from actually running code in this
environment (torch 2.3.1 / torch_geometric 2.8.0 were installed here
specifically to make that possible) — none are estimated or assumed.
Where something could *not* be verified (no live Neo4j, no GPU in this
sandbox), that is stated explicitly rather than glossed over.

---

## 1. Files changed

| File | Type of change |
|---|---|
| `neighbor_sampling.py` | **New.** Adaptive + relation-aware k-hop sampler (task 4/5/6/7). |
| `test_neighbor_sampling.py` | **New.** 39 unit tests for the sampler. |
| `offline_pipeline.py` | **New.** Neo4j-free harness reusing real feature code, so experiments could actually run in this sandbox (no Neo4j server reachable here). |
| `run_experiments.py` | **New.** Orchestrates the ablation matrix (task 10/11). |
| `conftest.py` | **New.** Fixes a real, pre-existing broken import for pytest (see §14). |
| `model_hgt.py` | **Bug fix.** `_make_hgt_conv` crashed on the installed PyG version (`group=`/`dropout=` no longer accepted by `HGTConv`). Made self-adapting instead of hardcoding one fallback. |
| `train.py` | **Extended.** Added `--model hgt`/`--model all`, `--sampling {none,relation_aware,uniform,full}` + sampling config flags, `--heads`/`--hgt_group`/`--attn_dropout`. Default behaviour (`--sampling none`, `--model both`) is unchanged. |
| `infer.py` | **Bug fix.** Same broken import as `incremental_updater.py` (see §14), fixed independently so this file doesn't rely on import order elsewhere. |
| `incremental_updater.py` | **Bug fix.** Broken `import neo4j_graph_builder as nb` (see §14). |
| `utils.py` | **Extended.** `evaluate()` was missing AUPR — task 11 explicitly lists it as a required metric. Added `average_precision_score`; updated `print_comparison_table`. |
| `train_scalable.py` | **Extended.** `compare_n_way` updated with the same AUPR column. |

**Not modified:** `data_loader.py`, `privilege_features.py`,
`graph_construction/neo4j_graph_builder.py`, `model_graphsage.py`,
`model_gat.py`, `model_ensemble.py`, `feature_engine9.py`,
`incremental_updater.py`'s actual update logic, `explainability.py`,
`hgt_attention_explainability.py`, `blast_radius.py`, `node_importance.py`.
These were audited (see §2) and found to need no change for this task —
per the brief, nothing here was rewritten "to look more sophisticated."

---

## 2. Current Neo4j → PyG representation (audit, before any change)

**Pipeline as it actually runs:** `feature_engine9.py` (CloudTrail → the
`cloudtrail_structural.csv` feature CSV) → `privilege_features.py`'s
`PrivilegePropagationGraph` (builds a `networkx.MultiDiGraph` and computes
every structure-derived feature — `hop_count`, `privilege_gain`,
`abnormal_path_frequency`) → `graph_construction/neo4j_graph_builder.py`
writes typed nodes/relationships into Neo4j → `data_loader.py`'s
`PrivilegePropagationGraphLoader` issues one Cypher `MATCH` per node label
and per relationship type, assembles a PyG `HeteroData`, and fits
`StandardScaler`/`LabelEncoder` on the fetched data → `model_graphsage.py`
/ `model_gat.py` consume it full-batch (the entire graph, every step —
**no sampling existed anywhere before this task**; see §6).

**Verified schema, on the actual `cloudtrail_structural.csv` (9,711
rows) with `policy_sentry` installed** (see §15 — this differs from a
docstring claim in `data_loader.py`):

- **6 node types**: User, Role, UnresolvedPrincipal, Service, Resource, Policy.
- **14 populated `(src_type, relation, dst_type)` triples** (not the "20"
  claimed in `data_loader.py`'s own module docstring — that number does
  not match this CSV; see §15).
- **9,711 edges total**, 4.9% labeled `is_attack=1`.
- Node counts: User 540, Role 5, UnresolvedPrincipal 17, Resource 6,790, Policy 1.

A new CloudTrail event today: parsed by `feature_engine9.py` → appended
to the structural CSV / streamed via `incremental_updater.py` → merged
into Neo4j (`neo4j_graph_builder.py`'s `merge_node`/`create_edge`) → next
full graph load (batch) or `infer.py`'s `SubgraphLoader` (2-hop Cypher
`MATCH` around the event's principal/target, no cap — see §15) pulls it
into a `HeteroData` → `model(data)` → sigmoid → thresholded at 0.5 →
edge-level `benign`/`malicious`.

**Global order invariant** (unchanged, verified still holds): every
model's `forward()` returns flat logits ordered by `sorted(data.edge_types)`;
`data_loader.py`'s `global_labels()`/`flatten_mask_dict()` flatten `y`/masks
in that same order. Nothing in this task touched or needed to touch this.

---

## 3. Is HGT justified? (task 2)

**The naive argument — "the graph is heterogeneous, therefore HGT" — is
not, by itself, a real justification here**, and saying otherwise would
be exactly the "added HGT to look more sophisticated" failure mode the
brief warns against. Here's why: `model_graphsage.py` and `model_gat.py`
**already** give every `(src_type, relation, dst_type)` triple its own
weight matrix (`HeteroConv` wrapping one `SAGEConv`/`GATv2Conv` per
triple). Heterogeneity is already being exploited by both baselines —
that's not a gap HGT uniquely fills.

**The real, narrower gap:** `HeteroConv` (both baselines) computes each
triple's message independently and then **sums** the per-triple outputs
at a destination node with fixed, untrained weighting between relations.
There is no mechanism for a node to learn "pay more attention to my rare
ASSUMES edge than my 500 ordinary READ edges" *across* relations — GAT's
attention is real but scoped *within* one triple's own edges only.
`HGTConv` computes attention *across every relation converging on a
node in one shared softmax*, with type-specific K/Q/V projections. That
is a genuine, mechanistically different capability, not just "more
heterogeneous."

**Verified, not assumed:** the ablation in §9 below shows HGT
(full-neighborhood) reaching **AUROC 0.998 / AUPR 0.980**, against GAT's
0.995 / 0.967 and GraphSAGE's 0.993 / 0.965, with identical
precision/recall/F1 to GAT (1.000 / 0.944 / 0.971) at the 0.5 threshold.
This is a **real but modest** edge on the ranking-quality metrics
(AUROC/AUPR), not a dramatic win, on a single train/val/test split, on a
small (9,711-edge) synthetic dataset. That is reported as-is — this
report does not claim HGT is "better" beyond what these numbers show, and
a single split is not strong statistical evidence on its own (see §15).
**Conclusion: HGT is a defensible primary candidate, not an
automatically-crowned winner** — kept alongside GraphSAGE/GAT as the
brief requires (§9), with the ensemble in `model_ensemble.py` left as an
**opt-in**, checkpoint-driven choice, never auto-invoked (see §10).

---

## 4. HGT architecture

`model_hgt.py` (pre-existing in the uploaded codebase — this task
audited, fixed, and integrated it, not authored it from scratch):
`HGTAnomalyDetector` = per-node-type linear input projection → `L`
stacked `HGTConv` layers (`torch_geometric.nn.HGTConv`, heads configurable
via `--heads`) with residual connections and LayerNorm → for each
populated triple, an `EdgeClassifierHead` consuming
`[h_src ‖ h_dst ‖ edge_attr]` → one flat logit vector, ordered by
`sorted(data.edge_types)` — same contract as GraphSAGE/GAT (§8's edge-
classification preservation depends on this).

**Bug found and fixed (verified via execution, not inspection alone):**
the installed `torch_geometric==2.8.0`'s `HGTConv.__init__` no longer
accepts `group=` or `dropout=` — both raise `TypeError`. The pre-existing
`_make_hgt_conv` only guarded `dropout`, so it still crashed on `group`
here. `test_model_hgt.py::test_build_hgt_from_args_uses_defaults` and
`::test_forward_shape_matches_edge_count` failed with that exact
`TypeError` before the fix. Fixed by making construction self-adapting:
attempt with both kwargs, parse which one `TypeError` names, drop it,
retry — logs a warning rather than silently no-op'ing, and adapts to
either kwarg being the problem (future-proof against further PyG drift
in either direction, not just today's specific error). All 8
HGT/ensemble tests pass after the fix (they did not before).

---

## 5. Neighbor sampling algorithm (task 4/5)

New file: `neighbor_sampling.py`. Two layers, deliberately separated:

**Pure, torch-free decision functions** (unit-tested without any
torch/PyG dependency — runs anywhere):
- `decide_sample_size(degree, max_neighbors)` — task 4, verbatim: if a
  node's own degree ≤ `MAX_NEIGHBORS`, use every neighbor; else cap at
  `MAX_NEIGHBORS`. A pure function of *that node's own* degree — the
  function signature has nowhere to even put a whole-graph edge count,
  which is how task 4's "must NOT depend on total graph edges" is
  satisfied structurally, not just by convention.
- `allocate_relation_quota(counts, budget, num_samples_per_relation)` —
  task 5's anti-crowding-out logic: give every populated relation a
  floor of `min(NUM_SAMPLES_PER_RELATION, its own availability)` *before*
  splitting the remaining budget proportionally to remaining
  availability (via a largest-remainder / Hamilton apportionment,
  `_largest_remainder_allocate`). Verified concretely
  (`test_relation_aware_does_not` / `test_uniform_baseline_crowds_out...`):
  given 2 ASSUMES edges and 5,000 READ edges at one node with budget 50,
  the **naive proportional-only split gives ASSUMES exactly 0** (2/5002 ×
  50 rounds to 0); the relation-aware allocator gives it all 2. This is
  the concrete demonstration of why task 5 asked for this, not just an
  assertion that it matters.

**HeteroData-level sampler** (`AdaptiveRelationAwareSampler`, thin layer
on top): BFS k-hop expansion from seed nodes, applying the above at every
node visited. Expands strictly via each frontier node's **incoming**
edges (where it is the destination) — this matches what every model
here's message passing actually needs (a node's embedding depends on its
in-neighbors, recursively), verified directly in
`test_expansion_follows_message_passing_direction`. Returns a new
`HeteroData` induced over the visited nodes with edge direction/type
untouched (`test_edge_direction_and_type_are_never_altered`).

`build_sampled_training_view()` seeds expansion **only from train-split
edges** — val/test edges are never sampling seeds, and evaluation always
runs on the full, unsampled graph (matching the pre-existing convention
in `train_scalable.py`'s `train_hgt()`), so sampling cannot leak
train/val/test boundaries through subgraph structure.

`strategy="uniform"` and `"full"` are included as explicit, opt-in
baselines/isolators for the ablation (§9), not just the recommended
`"relation_aware"` mode.

---

## 6. Why sampling is needed

Before this task, **no sampling existed anywhere** in the training path:
`model_graphsage.py`'s own `GraphSAGEWithSampling` class is a
docstring-only stub whose body is `pass`. `train_scalable.py` (also
pre-existing) sketches PyG's built-in `LinkNeighborLoader`/`HGTLoader`,
but by its own docstring was never executed, and passes one **flat**
fanout list identically to every relation — exactly the "blind uniform
sampling that crowds out rare relations" task 5 warns against, not
something that already solved it. `infer.py`'s streaming `SubgraphLoader`
does a bounded 2-hop Cypher fetch per *event* (fine at that scale) but
has no per-node degree cap — irrelevant to *training*-time scalability,
which is what task 4 asks about ("scalable when the IAM graph grows").
So: sampling was a real gap, not a redundant addition.

---

## 7. Why HGT is appropriate — see §3 above (same content, not duplicated here).

---

## 8. How GPU execution works

`AdaptiveRelationAwareSampler.sample(..., device=...)`: index selection
itself always runs on CPU (Python `random`, small integer indices — this
keeps sampling deterministic/inspectable regardless of where `data`
lives), and only the **returned subgraph's tensors** are moved to
`device` via `HeteroData.to(device)`. `train.py`'s `--device` flag
(defaults to `"cuda"` if `torch.cuda.is_available()` else `"cpu"`,
pre-existing) is threaded through `train_model()`/`build_sampled_training_view()`
unchanged. **Verified on CPU only** — this sandbox has no GPU
(`torch.cuda.is_available()` is `False` here), so the CUDA code path
itself could not be executed or timed; only reviewed for correctness
(every `.to(device)` call is present and consistent with the CPU path
that was actually run). This is stated plainly rather than implied to
have been tested — see §15.

Per task 6's explicit warning ("GPU has more compute, not unlimited
memory"): sampling is **not** disabled or bypassed based on device — the
same `SamplingConfig`/cap applies whether `--device cpu` or `--device cuda`.

---

## 9. How edge-level classification is preserved

Every model (`GraphSAGEAnomalyDetector`, `GATAnomalyDetector`,
`HGTAnomalyDetector`) ends in an `EdgeClassifierHead` producing one logit
per **edge** (`[h_src ‖ h_dst ‖ edge_attr] → linear → 1 logit`), never a
per-node output. Sampling doesn't change this: `neighbor_sampling.py`
subsamples which edges/nodes are *visible during training*, but the
classification head and its output shape are untouched — confirmed by
`test_neighbor_sampling.py`'s direction/type-preservation tests plus the
real training runs in §9's results table below (whose `y`/predictions
are edge-level throughout, `n=1,458` on the held-out test edges each
time). No node-level or user-level objective was introduced anywhere.

### Ablation results (task 10/11 — real numbers, run in this environment)

Ran via `run_experiments.py` on the real
`graph_construction/cloudtrail_structural.csv` (offline harness, §12),
**identical stratified 70/15/15 edge split (seed 42) for every cell**,
identical loss (focal, α=0.25/γ=2.0), identical hidden_dim=128,
layers=2, heads=4, lr=1e-3, up to 60 epochs with early stopping
(patience=15, checked every 5 epochs):

| Cell | Model | Sampling | Precision | Recall | F1 | AUROC | AUPR | Params | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| A | GraphSAGE | full | 1.000 | 0.931 | 0.964 | 0.993 | 0.965 | 1,020,801 | 37.6 |
| B | GraphSAGE | relation_aware (K=50) | 1.000 | 0.931 | 0.964 | 0.994 | 0.970 | 1,020,801 | 25.5 |
| C | GAT | full | 1.000 | 0.944 | 0.971 | 0.995 | 0.967 | 2,500,993 | 17.7 |
| D | HGT | full | 1.000 | 0.944 | 0.971 | **0.998** | **0.980** | 1,996,795 | 14.8 |
| E | HGT | relation_aware (K, tuned) | 1.000 | 0.944 | 0.971 | 0.998 | 0.978 | 1,996,795 | 11.7 |

**K sweep for cell E** (task 11: "tune on validation, report test once"):
K ∈ {25, 50, 100} tried, selected on validation F1 only (test never
touched during selection). Validation F1 was tied at **0.9254 for every
K tried** — this dataset's structure doesn't stress the sampler's
K-sensitivity (train-split max node degree is well under 100 for most
nodes; the sampled view at K=50 kept 6,892 of 9,711 train-visible edges
and 4,740 of 6,790 Resource nodes — see the `train_view` log line each
run emits). K=25 selected (first value reaching the tied maximum); test
metrics in row E are reported at that K, once.

**What this supports, concretely, without overclaiming:**
- **Sampling does not hurt.** Cell B vs A, E vs D: F1/precision/recall
  identical or better; AUPR *improved* slightly under sampling in both
  pairs (0.965→0.970, 0.980→0.978 — note the HGT pair went the other
  way by 0.002, i.e. within noise). Training time dropped 32% (A→B) and
  21% (D→E). **This is a single-seed run** — a 0.002–0.005 AUPR
  difference is not a claim of "sampling improves accuracy," only that
  it doesn't cost accuracy while it does save compute, on this dataset.
- **HGT's edge is real but modest**, concentrated in AUROC/AUPR
  (ranking quality) rather than the thresholded precision/recall/F1,
  where it ties GAT exactly. Matches the mechanistic argument in §3 —
  cross-relation attention should help most on ranking borderline
  cases, not on cases already confidently separated by either model.
- GraphSAGE's longer wall-clock time (37.6s vs GAT's 17.7s / HGT's
  14.8s) reflects **more epochs before early stopping fired** (it used
  the full 60-epoch budget; GAT/HGT stopped around epoch 25–30), not a
  higher per-epoch cost — per-epoch, all three models run in well under
  1 second on this graph size.
- **This is not a fully-tuned production comparison** — one seed, one
  split, no hyperparameter search, reduced epoch budget for sandbox
  time constraints. See §15.

`run_experiments.py`'s full log and `experiment_results.json` (full
per-cell metrics including the confusion matrices and classification
reports) are included alongside this report.

---

## 10. Any risks of label leakage

**Sampler (this task's new code):** `decide_sample_size` and
`allocate_relation_quota`'s signatures contain no `y`/label parameter
anywhere — structurally impossible to consult, not just unused by
convention. `test_sampler_never_touches_y_or_edge_attr` verifies this
empirically: sampling the same graph with `.y`/`.edge_attr` deleted
entirely produces identical node/edge counts. `within_relation_selection
="degree_weighted"` (the one non-random option) uses only edge_index
degree counts, never a feature column or label.

**Pre-existing features, checked during this audit (not assumed safe):**
`privilege_gain`, `abnormal_path_frequency`, `is_privilege_escalation_technique`
are all computed from graph structure/action names only — verified by
reading `privilege_features.py`'s `path_pattern_frequencies` (counts
"over ALL edges regardless of `label`") and
`ActionAccessLevelResolver.access_level` (a static per-action-name
lookup). No leakage found in existing feature engineering.

**`build_sampled_training_view`:** seeds only from train-split edges;
val/test edges never become expansion seeds (verified:
`test_val_and_test_edges_are_never_used_as_seeds`), so training-time
subgraph *structure* cannot leak which edges are val/test.

**`model_ensemble.py` is not auto-invoked.** `infer.py`'s checkpoint
dispatch (`model_type` read from the saved checkpoint file) already
made ensemble use opt-in before this task; nothing added here changes
that — `train.py --model all` trains SAGE/GAT/HGT as three **separate**
runs for comparison, never blends their outputs into one prediction.

---

## 11. Commands to train HGT

```bash
python3 train.py --model hgt --epochs 100 --hidden 128 --heads 4 \
    --loss focal --compare
```

With adaptive relation-aware sampling:
```bash
python3 train.py --model hgt --sampling relation_aware \
    --max_neighbors 50 --num_hops 2 --num_samples_per_relation 5 \
    --epochs 100 --compare
```

Requires a live Neo4j instance populated via
`graph_construction/neo4j_graph_builder.py` (see its own `--help`/module
docstring) — `train.py` itself was not changed to bypass this; see §12
for how this report obtained real numbers without one.

## 12. Commands to run inference

```bash
python3 infer.py --checkpoint checkpoints/best_HGT.pt --watch
```

`infer.py` already dispatches on the checkpoint's saved `model_type`
(`"hgt"` supported pre-existing this task — see `model_ensemble.py`'s
`build_hgt_from_args`); no new inference-path flags were needed for HGT
itself. The import-path bug fix (§14) affects this file too and is
required for it to run at all with the code as shipped.

## 13. Commands to reproduce the GraphSAGE/GAT baselines

```bash
python3 train.py --model sage --epochs 100 --hidden 128 --loss focal
python3 train.py --model gat  --epochs 100 --hidden 128 --loss focal
python3 train.py --model both --epochs 100 --compare   # both, one run, shared split
```

**To reproduce this report's exact ablation numbers (§9), without a live
Neo4j instance** (as run in this environment):
```bash
python3 run_experiments.py \
    --csv graph_construction/cloudtrail_structural.csv \
    --epochs 60 --k_sweep 25 50 100 --seed 42
```

---

## 14. Tests performed

**60 tests, all passing, all actually executed in this environment**
(torch 2.3.1 / torch_geometric 2.8.0 installed specifically for this —
without them, this codebase's own pre-existing HGT/ensemble tests had
never been run, by their own "honest caveat" docstrings):

- `test_neighbor_sampling.py` — **39 new tests**: degree-threshold logic,
  relation-aware quota vs. the naive uniform baseline (both at the pure-
  function level and end-to-end through the real sampler), edge
  direction/type preservation, message-passing-direction-only expansion,
  reproducibility under a fixed seed, structural no-label-access proof,
  device placement, `SamplingConfig` validation, and
  `build_sampled_training_view`'s train/val/test seed isolation.
- `test_model_hgt.py` (8 tests) + `test_ensemble.py` — **pre-existing,
  now passing** after the `_make_hgt_conv` fix (§4); 3 of 8 failed with
  the exact `TypeError` this fix addresses, before the fix.
- `test_incremental_updater.py` (13 tests) — **pre-existing**, now
  importable and passing after the conftest.py/sys.path fix (§14 bug 2);
  run from `graph_construction/` per that test file's own relative CSV
  path convention (not a bug — matches `neo4j_graph_builder.py`'s own
  documented run location).

**Two real, pre-existing bugs found via execution (not just static
review) and fixed:**
1. `model_hgt.py`'s `_make_hgt_conv` — installed PyG no longer accepts
   `HGTConv(group=..., dropout=...)`; both raise `TypeError`. Fixed with
   a self-adapting kwarg-stripping retry (§4).
2. `import neo4j_graph_builder as nb` in `infer.py` and
   `incremental_updater.py` — the module physically lives in
   `graph_construction/`, one directory below; this bare import only
   ever worked if `graph_construction/` happened to already be on
   `sys.path`, which is not true when either file is imported normally.
   Verified directly: `python3 -c "import incremental_updater"` raised
   `ModuleNotFoundError` before the fix. Fixed with a minimal, explicit
   `sys.path.insert` in each of the two files (not relying on import
   order between them) plus a repo-root `conftest.py` so pytest resolves
   it regardless of collection order.

**One real gap found and filled:** `utils.py`'s `evaluate()` did not
compute AUPR at all, despite task 11 explicitly listing it as a required
ablation metric. Added (`sklearn.metrics.average_precision_score`);
`print_comparison_table`/`compare_n_way` updated to show it.

**End-to-end CLI verification:** `train.py --model all --sampling
relation_aware --compare` was run through its actual `main()` entry
point (not just its internal functions in isolation) via a minimal
monkeypatch of only `PrivilegePropagationGraphLoader.load()` (the one
line that needs a live Neo4j connection) — confirmed the full CLI parse
→ split → build 3 models → train each (one with the sampled view) →
3-way `compare_n_way` table path works exactly as a user invoking that
command would experience it.

---

## 15. Limitations

- **No live Neo4j instance in this environment.** `data_loader.py`/
  `graph_construction/neo4j_graph_builder.py` were audited by reading
  and were not modified, but could not be executed directly here.
  `offline_pipeline.py` was built specifically so §9's numbers could be
  real rather than fabricated — it reuses the *exact* feature-computation
  functions (`privilege_features.PrivilegePropagationGraph`,
  `neo4j_graph_builder`'s action/attacker-principal logic,
  `data_loader.py`'s own `_node_features`/`_edge_features`) and stops
  precisely where `neo4j_graph_builder.py`'s `build_graph()` would start
  writing to Neo4j — but it is still a second code path, not the
  production one, and carries an explicitly-documented maintenance
  coupling to `data_loader.py`'s `load()` (see that file's own
  docstring) if that method's body changes in the future.
- **No GPU in this sandbox.** GPU device-placement code (§8) was
  written and reviewed but never executed on CUDA; only the CPU path
  was actually run and timed.
- **`peak_cpu_mem_mb` in `experiment_results.json` is not a meaningful
  memory metric** — it comes from Python's `tracemalloc`, which does
  not see PyTorch's native C++ tensor allocator, so it undercounts
  actual memory by a large, inconsistent margin. Parameter count is the
  more honest proxy available here; true GPU memory (what task 11
  actually asks for) could not be measured at all without a GPU.
- **§9's ablation is one seed, one split, a reduced (60-epoch, vs. a
  production run's likely 100+) budget, and no hyperparameter search**
  — chosen to fit this environment's time constraints honestly rather
  than claim a fully-tuned comparison. The *direction* of the results
  (HGT ≥ GAT ≥ GraphSAGE on AUROC/AUPR; sampling ≈ no-sampling on
  accuracy metrics while being faster) is what this report is confident
  in; the exact decimal margins should not be over-read from a single run.
- **Two audit findings, found but deliberately NOT fixed** (out of this
  task's 14-item scope, and each would be a real design decision, not a
  mechanical fix):
  1. `data_loader.py`'s own module docstring claims "20 distinct
     (src_type, relation, dst_type) triples... verified" and "2,900
     edges" — neither matches the actual current
     `graph_construction/cloudtrail_structural.csv` (14 triples, 9,711
     edges, verified by running `offline_pipeline.load_offline()`
     against it). This looks like it documents an earlier/smaller CSV
     snapshot than the one now shipped in the repo.
  2. **`hop_count`, `privilege_gain`, and `privilege_gain_defined` are
     constant across all 9,711 edges on the current dataset** (verified:
     `hop_count` distribution is `{1: 9711}`, `privilege_gain` is `0.0`
     for all edges, `privilege_gain_defined` is `False` for all edges).
     Root cause, verified: `AssumeRole` target values in this CSV are
     informal mnemonic strings (e.g. `"role-hmusdt"`), not full IAM ARNs,
     so `privilege_features.node_key_for_target()`'s ARN-pattern match
     for detecting a `Role`-typed destination never fires — every
     `ASSUMES` edge's destination is typed `Resource`, never `Role`
     (`ppg.roles_reached_via_assume()` returns a set of `Resource` nodes,
     confirmed directly). `hop_count()`'s 2-hop detection therefore can
     never trigger, which makes `privilege_gain` (which requires
     `hop_count()==2`) undefined everywhere. This is a genuine
     structural gap between what the "privilege propagation chain"
     features are designed to capture and what this specific synthetic
     dataset's target-naming convention lets them observe — 3 of the 7
     `EDGE_NUM_COLS` edge features are currently dead weight on real
     data. Left unfixed here because relaxing the ARN-match regex (or
     changing how `AssumeRole` targets are canonicalized) is a
     structural decision about `privilege_features.py`/
     `neo4j_graph_builder.py`, outside this task's explicit 14-item
     scope, and risks being exactly the kind of "unnecessary
     architectural change" the brief said not to make without being asked.
