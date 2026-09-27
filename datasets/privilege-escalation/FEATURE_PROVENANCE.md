# Feature provenance & label semantics

Answers, with code references, the feature-integrity questions raised in review.
Line numbers refer to `feature_engine9.py` unless noted.

## Which features feed which model (read this first)

There are **two** feature representations, and a review that inspects one does
not necessarily describe the other:

| Representation | Produced by | Consumed by | Example features |
|---|---|---|---|
| **Graph** (the headline GNN) | `graph_construction/privilege_features.py` via Neo4j | GraphSAGE/GAT | `hop_count`, `privilege_gain`, rank-normalized degree, `abnormal_path_frequency`, `resource_sensitivity` |
| **Streaming CSV** | `feature_engine9.py` | LSTM / temporal / ensemble track | `source_node_degree`, `edge_interaction_count`, `source_node_age_normalized`, `source_historical_risk` |

The GNN does **not** read `source_historical_risk`, `source_node_degree`, etc.
Those are online features for the sequence track. Keep this distinction when
attributing a concern to a result.

## label = 1 — definition

`label = 1` means **the API event is part of a known malicious attack chain**
(an event produced by an `ATTACK_CHAINS` step in the synthetic generator, or a
Stratus detonation step in the real capture). It is an **event/edge** label, not
a campaign label: benign setup, reconnaissance, and noise events *inside* an
attack session keep `label = 0`. So `AssumeRole` or `GetSecretValue` is labelled
`1` only when it is an actual chain step, `0` when it is benign activity — which
is exactly what makes the benign-vs-malicious distinction learnable rather than
a giveaway. The campaign/stage lineage (who/which-chain/which-stage) lives in
`synthetic_campaign_annotations.csv`, joined by `log_id`, and is **evaluation
ground truth, not a model feature**.

## source_historical_risk — NOT leakage (the important one)

`source_historical_risk` is an EWMA of a per-key **adaptive risk prior**
(`AdaptiveRiskPrior`), which is target encoding: its value is a function of
labels seen so far. Target encoding is legitimate only if it is strictly causal
and frozen at evaluation. Both hold here:

1. **Causal ordering — a row's own label never enters its own feature.** In the
   batch loop (lines 914–937): features are computed first —
   `get_structural_data` / `get_temporal_features`, which call `.score()`
   (lines 918–919) — and the label is folded in only afterwards, via
   `observe_label` -> `.update()` (line 937). So `score()` for row *i* sees only
   rows `< i`.
2. **Frozen on evaluation.** `freeze_priors = not is_training_input` (line 990):
   any input other than the training default is read-only. `update()` is a
   hard no-op when frozen (lines 268–269), and `save()` refuses to overwrite the
   training-fitted counts (lines 275–277). So dev/test labels never touch the
   prior.
3. **Fit-on-train, transform-only** — identical discipline to the
   `StandardScaler`s in `data_loader.py`, which are fit on the training graph
   and transform-only thereafter.

Net: `risk(t)` is a function of events strictly before `t`, fit on training data
only. There is no future/label leakage into evaluation features.

## source_node_degree / edge_interaction_count / source_node_age_normalized

These are **online, event-time** features, not static graph statistics. `fe9` is
a streaming engine (it mimics real-time ingestion), so:

- `source_node_degree` = the source's out-degree **observed so far at event
  time**, not the final whole-graph degree. On a single-source sequence it
  therefore climbs 1, 2, 3, … — that is correct for a causal feature, and is
  why it looked like a "running count" when the review ran it on a 10-row
  single-source fixture.
- `edge_interaction_count` = count of that `(source, target, edge_type)`
  interaction **seen so far**.
- `source_node_age_normalized` = normalized time since the source was first
  seen.

These are deliberately causal (a detector at inference time cannot know a node's
final degree). The **naming is the fair critique** — they read as static graph
metrics. They are documented here as event-time-historical; a future rename
(`source_out_degree_so_far`, etc.) would remove the ambiguity. They do **not**
feed the GNN.

## is_cross_account

Zero-variance on any single-account session (e.g. the fixture) by construction.
Meaningful only dataset-wide: the real capture spans multiple accounts, so the
combined dataset contains both values. Verify dataset-wide, never per-file.

## Node typing and target normalization

Node types are derived deterministically from principal/target ARNs in
`graph_construction/privilege_features.py`
(`node_key_for_principal` / `node_key_for_target`) and, for real logs, targets
are canonicalized out of `requestParameters` (`_canonicalize_target_from_params`
on the integration branch) so `AssumeRole`'s `roleArn`, `CreateRole`'s
`roleName`, etc. collapse onto one Role node. `role-*` identifiers resolve to
Role when the same name is seen acting as a principal in the session.

## Verifying all of the above

`verify_labels.py` checks, and fails non-zero otherwise:
- attack labels are preserved raw -> structural (no silent drop/flip),
- `structural.log_id` <-> `annotation.log_id` is a strict 1:1 join with agreeing
  labels,
- the corpus is the full diverse dataset, not the demo fixture.
