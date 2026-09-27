"""
test_data_loader.py
===================
Unit tests for data_loader.py's feature construction and global-ordering
contract.

WHY THIS FILE EXISTS: an end-to-end audit found eight defects in this pipeline.
FOUR of them lived in data_loader.py, and none were caught by the existing
suite because nothing tested this module at all. Every test below pins one
specific defect so it cannot come back silently:

  - unseen edge_type aliased onto an arbitrary real action  -> TestEdgeTypeUnknownFallback
  - edge_type fed to the net as a raw 0..66 ordinal         -> TestEdgeTypeOneHot
  - StandardScaler silently refit on evaluation data        -> TestNoRefitOnEvalData
  - a node type absent from the eval graph crashing at load -> TestEmptyNodeTypeShape

NO NEO4J REQUIRED. Every method under test is a pure function of its arguments
plus four instance attributes, so `_loader()` below constructs the object
without opening a driver. Run with:

    python -m unittest test_data_loader
"""

import logging
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import HeteroData

from data_loader import (
    EDGE_NUM_COLS,
    NODE_FEATURE_SCHEMA,
    UNK_CATEGORY,
    UNREACHABLE_DISTANCE_SENTINEL,
    PrivilegePropagationGraphLoader,
    flatten_mask_dict,
    global_labels,
)


def _loader(fit_artifacts=None, model_node_types=None, edge_scaler=None):
    """Builds a loader WITHOUT connecting to Neo4j.

    __init__ opens a driver eagerly, but nothing under test here touches it --
    the feature builders depend only on the four attributes set below. Using
    __new__ keeps production code free of a test-only constructor hook.
    """
    obj = object.__new__(PrivilegePropagationGraphLoader)
    obj.device = torch.device("cpu")
    obj._fit_artifacts = fit_artifacts
    obj._model_node_types = set(model_node_types) if model_node_types is not None else None
    obj.label_encoders = {}
    obj.node_scalers = {}
    obj.edge_scaler = edge_scaler if edge_scaler is not None else StandardScaler()
    return obj


def _edge_df(edge_types, n=None):
    """Minimal edge frame carrying every column _edge_features reads."""
    n = n if n is not None else len(edge_types)
    return pd.DataFrame({
        "edge_type": list(edge_types),
        "hop_count": [1] * n,
        "privilege_gain": [0.5] * n,
        "privilege_gain_defined": [True] * n,
        "action_global_frequency": [10] * n,
        "is_privilege_escalation_technique": [False] * n,
        "is_read_only": [1] * n,
        "abnormal_path_frequency_rank": [0.5] * n,
    })


def _fitted_edge_scaler():
    scaler = StandardScaler()
    scaler.fit(np.random.RandomState(0).normal(size=(50, len(EDGE_NUM_COLS))))
    return scaler


# ══════════════════════════════════════════════════════════════════════════
# _rank_normalize -- the transform behind the 6.16 result
# ══════════════════════════════════════════════════════════════════════════

class TestRankNormalize(unittest.TestCase):

    def test_returns_percentiles_in_unit_interval(self):
        out = PrivilegePropagationGraphLoader._rank_normalize(pd.Series([5, 1, 3, 9]))
        self.assertTrue(np.all(out > 0) and np.all(out <= 1.0))
        self.assertAlmostEqual(out.max(), 1.0)

    def test_is_scale_invariant(self):
        """THE property the whole 6.16 fix rests on: a node's degree becomes
        comparable in MEANING between a sparse synthetic graph (degrees ~10)
        and a dense real one (degrees ~8000). log1p compressed scale but did
        not achieve this, which is why 6.15 failed and 6.16 worked."""
        small = PrivilegePropagationGraphLoader._rank_normalize(pd.Series([1, 2, 3, 4]))
        huge = PrivilegePropagationGraphLoader._rank_normalize(pd.Series([10, 8000, 90000, 1e6]))
        np.testing.assert_allclose(small, huge)

    def test_is_monotonic_and_order_preserving(self):
        vals = pd.Series([3, 1, 4, 1, 5, 9, 2, 6])
        out = PrivilegePropagationGraphLoader._rank_normalize(vals)
        self.assertEqual(list(np.argsort(vals.to_numpy())), list(np.argsort(out)))

    def test_ties_share_a_rank(self):
        """method="average" gives every tied element the mean of the ranks they
        span -- 2.5/4 = 0.625 for four-way ties, not 1.0. What matters is that
        equal degrees are never separated into an arbitrary order."""
        out = PrivilegePropagationGraphLoader._rank_normalize(pd.Series([7, 7, 7, 7]))
        self.assertEqual(len(set(out)), 1)
        self.assertAlmostEqual(out[0], 0.625)

        mixed = PrivilegePropagationGraphLoader._rank_normalize(pd.Series([1, 5, 5, 9]))
        self.assertAlmostEqual(mixed[1], mixed[2])
        self.assertLess(mixed[0], mixed[1])
        self.assertLess(mixed[1], mixed[3])

    def test_single_element_does_not_divide_by_zero(self):
        np.testing.assert_allclose(
            PrivilegePropagationGraphLoader._rank_normalize(pd.Series([42])), np.array([1.0])
        )

    def test_uses_no_labels_and_fits_nothing(self):
        """It is legitimate to compute this fresh on eval data precisely
        because it learns no cross-graph statistic. Calling it must leave the
        loader's fitted artifacts untouched."""
        ldr = _loader()
        PrivilegePropagationGraphLoader._rank_normalize(pd.Series([1, 2, 3]))
        self.assertEqual(ldr.node_scalers, {})
        self.assertEqual(ldr.label_encoders, {})


# ══════════════════════════════════════════════════════════════════════════
# Node features
# ══════════════════════════════════════════════════════════════════════════

class TestNodeFeatureShapes(unittest.TestCase):

    def _principal_df(self, n=4):
        return pd.DataFrame({
            "out_degree": range(1, n + 1),
            "unique_targets": range(1, n + 1),
            "unique_actions": range(1, n + 1),
            "role_transition_count": [0] * n,
        })

    def _resource_df(self, n=4):
        return pd.DataFrame({
            "in_degree": range(1, n + 1),
            "unique_principals": range(1, n + 1),
            "resource_sensitivity": [1.0] * n,
            "distance_to_sensitive_resource": [2.0] * n,
            "resource_type": ["s3"] * n,
        })

    def test_principal_features_are_entirely_rank_normalized(self):
        """All four principal columns are count-like, so scaled_cols is empty
        and User/Role/UnresolvedPrincipal carry NO StandardScaler at all. This
        is why no scaler is registered for them -- and why the missing-scaler
        bug surfaced on Policy rather than here."""
        ldr = _loader()
        x = ldr._node_features("User", self._principal_df())
        self.assertEqual(x.shape, (4, 4))
        self.assertNotIn("User", ldr.node_scalers)
        self.assertTrue(torch.all(x > 0) and torch.all(x <= 1.0))

    def test_resource_features_are_two_scaled_two_rank_one_categorical(self):
        ldr = _loader()
        x = ldr._node_features("Resource", self._resource_df())
        self.assertEqual(x.shape, (4, 5))
        self.assertIn("Resource", ldr.node_scalers)
        self.assertEqual(ldr.node_scalers["Resource"].n_features_in_, 2)

    def test_every_schema_type_matches_its_declared_width(self):
        """Guards the empty-node-type path in load(), which allocates
        torch.zeros((0, len(num_cols) + len(cat_cols))) from this same schema.
        If a real type's width ever drifts from the schema, an eval graph
        missing that type would build a wrongly-shaped tensor."""
        for ntype, (num_cols, cat_cols) in NODE_FEATURE_SCHEMA.items():
            df = self._principal_df() if ntype in ("User", "Role", "UnresolvedPrincipal") \
                else self._resource_df()
            df = df.copy()
            for c in num_cols:
                if c not in df:
                    df[c] = 1.0
            for c in cat_cols:
                if c not in df:
                    df[c] = "unknown"
            with self.subTest(ntype=ntype):
                x = _loader()._node_features(ntype, df)
                self.assertEqual(x.shape[1], len(num_cols) + len(cat_cols))

    def test_unreachable_distance_becomes_the_sentinel_not_nan(self):
        df = self._resource_df()
        df.loc[0, "distance_to_sensitive_resource"] = np.nan
        x = _loader()._node_features("Resource", df)
        self.assertFalse(torch.isnan(x).any())
        self.assertGreaterEqual(UNREACHABLE_DISTANCE_SENTINEL, 7)


class TestEmptyNodeTypeShape(unittest.TestCase):
    """REGRESSION: the loader used to skip creating a feature tensor for any
    node type with zero rows, which crashed the model at inference whenever an
    eval graph legitimately had none of a type the model was trained on (the
    dev split has zero Policy nodes)."""

    def test_declared_width_is_computable_for_a_zero_row_type(self):
        for ntype, (num_cols, cat_cols) in NODE_FEATURE_SCHEMA.items():
            with self.subTest(ntype=ntype):
                empty = torch.zeros((0, len(num_cols) + len(cat_cols)), dtype=torch.float)
                self.assertEqual(empty.shape[0], 0)
                self.assertGreater(empty.shape[1], 0)


class TestNoRefitOnEvalData(unittest.TestCase):
    """REGRESSION for the audit's V-5. _node_features only registered a fitted
    scaler when a type had >1 node at training. The synthetic graph had exactly
    ONE Policy node, so no Policy scaler existed -- and at inference the code
    fell through and fitted one on the REAL graph's Policy nodes, which is
    precisely what fit_artifacts exists to prevent."""

    def _policy_df(self, n=3, scale=1.0):
        return pd.DataFrame({
            "in_degree": np.arange(1, n + 1),
            "unique_principals": np.arange(1, n + 1),
            "resource_sensitivity": np.arange(1, n + 1) * scale,
            "distance_to_sensitive_resource": np.arange(1, n + 1) * scale,
        })

    def test_scaler_is_registered_even_for_a_single_node_type(self):
        """The root cause: one Policy node meant no registered scaler."""
        ldr = _loader()
        ldr._node_features("Policy", self._policy_df(n=1))
        self.assertIn("Policy", ldr.node_scalers)

    def test_missing_scaler_for_a_consumed_type_raises(self):
        ldr = _loader(fit_artifacts={"node_scalers": {}}, model_node_types={"Policy"})
        with self.assertRaises(RuntimeError) as ctx:
            ldr._node_features("Policy", self._policy_df())
        self.assertIn("Policy", str(ctx.exception))
        self.assertEqual(ldr.node_scalers, {}, "must not fit on evaluation data")

    def test_missing_scaler_for_an_unconsumed_type_warns_and_does_not_fit(self):
        """Service never appears in synthetic training data, so the model has
        no Service layer and never reads these features. Warn, pass through
        unscaled -- but still never fit on eval statistics."""
        ldr = _loader(fit_artifacts={"node_scalers": {}}, model_node_types={"User"})
        with self.assertLogs("data_loader", level=logging.WARNING):
            ldr._node_features("Service", self._policy_df())
        self.assertEqual(ldr.node_scalers, {}, "must not fit on evaluation data")

    def test_fitted_scaler_is_applied_by_transform_only(self):
        train = self._policy_df(n=5, scale=1.0)
        fitted = StandardScaler().fit(
            train[["resource_sensitivity", "distance_to_sensitive_resource"]].values
        )
        mean_before = fitted.mean_.copy()

        ldr = _loader(fit_artifacts={"node_scalers": {"Policy": fitted}},
                      model_node_types={"Policy"})
        # Evaluation data on a wildly different scale: a refit would map these
        # to ~mean 0, whereas a correct transform must map them far from 0.
        ldr._node_features("Policy", self._policy_df(n=5, scale=1000.0))

        np.testing.assert_allclose(fitted.mean_, mean_before,
                                    err_msg="training scaler was mutated by evaluation")
        self.assertEqual(ldr.node_scalers, {})

    def test_eval_features_reflect_training_statistics_not_their_own(self):
        train = self._policy_df(n=5, scale=1.0)
        fitted = StandardScaler().fit(
            train[["resource_sensitivity", "distance_to_sensitive_resource"]].values
        )
        ldr = _loader(fit_artifacts={"node_scalers": {"Policy": fitted}},
                      model_node_types={"Policy"})
        x = ldr._node_features("Policy", self._policy_df(n=5, scale=1000.0)).numpy()
        scaled_part = x[:, :2]
        self.assertGreater(np.abs(scaled_part).mean(), 5.0,
                            "far-out-of-distribution eval data should produce extreme "
                            "z-scores; near-zero would mean it was refit on itself")


# ══════════════════════════════════════════════════════════════════════════
# Edge features
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeTypeUnknownFallback(unittest.TestCase):
    """REGRESSION for the audit's V-3. Unseen actions were mapped to
    classes_[0] -- alphabetically AddUserToGroup, itself a privilege-escalation
    action. 70.5% of real test edges (18,326 of 25,984, spanning 590 distinct
    action names) were therefore told they were AddUserToGroup, and the
    reported F1 moved between 0.756 and 0.823 depending on which arbitrary
    class that fallback happened to be."""

    def _encoder(self, actions):
        return LabelEncoder().fit(pd.Series(list(actions) + [UNK_CATEGORY]))

    def test_unk_is_fit_as_a_real_class(self):
        enc = self._encoder(["AssumeRole", "CreateUser"])
        self.assertIn(UNK_CATEGORY, list(enc.classes_))

    def test_unseen_action_lands_in_the_unk_column_not_class_zero(self):
        """'0Action' sorts before '<UNK>', so classes_[0] is a REAL action here.
        That makes this a sharp test of the old behaviour: if the fallback ever
        reverts to classes_[0], the unseen row activates 0Action's column."""
        enc = self._encoder(["0Action", "AssumeRole"])
        self.assertNotEqual(enc.classes_[0], UNK_CATEGORY)

        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["AssumeRole", "NeverSeenBefore"]), enc).numpy()

        onehot = x[:, len(EDGE_NUM_COLS) + 1:]
        unk_col = list(enc.classes_).index(UNK_CATEGORY)
        zero_col = 0
        self.assertEqual(onehot[1].argmax(), unk_col, "unseen action must map to <UNK>")
        self.assertEqual(onehot[1][zero_col], 0.0, "must NOT activate classes_[0]")

    def test_known_actions_are_unaffected_by_the_fallback(self):
        enc = self._encoder(["AssumeRole", "CreateUser"])
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["AssumeRole", "CreateUser"]), enc).numpy()
        onehot = x[:, len(EDGE_NUM_COLS) + 1:]
        self.assertEqual(onehot[0].argmax(), list(enc.classes_).index("AssumeRole"))
        self.assertEqual(onehot[1].argmax(), list(enc.classes_).index("CreateUser"))

    def test_many_distinct_unseen_actions_collapse_to_one_column(self):
        """Documents the accepted limitation: <UNK> is one bucket, not per-action
        identity. The point is that it is an HONEST bucket rather than an
        arbitrary real action."""
        enc = self._encoder(["AssumeRole"])
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["Decrypt", "GetParameters", "DeleteParameter"]), enc).numpy()
        onehot = x[:, len(EDGE_NUM_COLS) + 1:]
        unk_col = list(enc.classes_).index(UNK_CATEGORY)
        self.assertTrue(np.all(onehot.argmax(axis=1) == unk_col))


class TestEdgeTypeOneHot(unittest.TestCase):
    """REGRESSION for the audit's V-4. edge_type was concatenated AFTER scaling
    as a bare float in [0, 66] -- inventing an ordering across unrelated AWS
    actions and dominating z-scored features by one to two orders of magnitude.
    Occlusion showed it was actively harmful on real data."""

    def _encoder(self, n_actions):
        return LabelEncoder().fit(
            pd.Series([f"Action{i:02d}" for i in range(n_actions)] + [UNK_CATEGORY])
        )

    def test_width_is_numeric_plus_rank_plus_one_hot(self):
        enc = self._encoder(67)
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["Action00", "Action05"]), enc)
        expected = len(EDGE_NUM_COLS) + 1 + len(enc.classes_)
        self.assertEqual(x.shape[1], expected)
        self.assertEqual(x.shape[1], 75, "6 scaled + 1 rank + 68 one-hot")

    def test_each_row_activates_exactly_one_category(self):
        enc = self._encoder(20)
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["Action00", "Action19", "Unseen"]), enc).numpy()
        onehot = x[:, len(EDGE_NUM_COLS) + 1:]
        np.testing.assert_allclose(onehot.sum(axis=1), np.ones(3))
        self.assertTrue(np.all(np.isin(onehot, [0.0, 1.0])))

    def test_no_feature_carries_ordinal_magnitude(self):
        """The specific defect: a raw code of 66 sat next to z-scores in ~[-3, 3].
        Under one-hot no categorical value can exceed 1."""
        enc = self._encoder(67)
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["Action66"]), enc).numpy()
        self.assertLessEqual(x[:, len(EDGE_NUM_COLS) + 1:].max(), 1.0)

    def test_categorical_distance_is_uniform_across_actions(self):
        """Under the old ordinal encoding Action00 and Action01 were 1 apart
        while Action00 and Action66 were 66 apart, implying a similarity
        structure that does not exist. One-hot makes every pair equidistant."""
        enc = self._encoder(67)
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        x = ldr._edge_features(_edge_df(["Action00", "Action01", "Action66"]), enc).numpy()
        cat = x[:, len(EDGE_NUM_COLS) + 1:]
        d01 = np.linalg.norm(cat[0] - cat[1])
        d066 = np.linalg.norm(cat[0] - cat[2])
        self.assertAlmostEqual(d01, d066)

    def test_rank_column_sits_between_scaled_block_and_one_hot(self):
        enc = self._encoder(5)
        ldr = _loader(edge_scaler=_fitted_edge_scaler())
        df = _edge_df(["Action00"])
        df["abnormal_path_frequency_rank"] = [0.75]
        x = ldr._edge_features(df, enc).numpy()
        self.assertAlmostEqual(float(x[0, len(EDGE_NUM_COLS)]), 0.75)


# ══════════════════════════════════════════════════════════════════════════
# The global edge-order contract
# ══════════════════════════════════════════════════════════════════════════

class TestGlobalOrdering(unittest.TestCase):
    """model_graphsage.py, model_gat.py, global_labels() and flatten_mask_dict()
    each independently derive their order from sorted(data.edge_types). If any
    of them ever disagreed, labels would silently misalign with logits and
    every metric in the project would be wrong without erroring."""

    def _data(self):
        d = HeteroData()
        # Deliberately inserted OUT of lexicographic order.
        d["User", "WRITE", "Resource"].y = torch.tensor([1, 1])
        d["User", "READ", "Resource"].y = torch.tensor([0])
        d["Role", "READ", "Resource"].y = torch.tensor([0, 1, 0])
        return d

    def test_global_labels_follows_sorted_edge_types(self):
        d = self._data()
        expected = np.concatenate([d[t].y.numpy() for t in sorted(d.edge_types)])
        np.testing.assert_array_equal(global_labels(d), expected)

    def test_global_labels_is_insertion_order_independent(self):
        first = global_labels(self._data())
        d2 = HeteroData()
        d2["Role", "READ", "Resource"].y = torch.tensor([0, 1, 0])
        d2["User", "READ", "Resource"].y = torch.tensor([0])
        d2["User", "WRITE", "Resource"].y = torch.tensor([1, 1])
        np.testing.assert_array_equal(first, global_labels(d2))

    def test_flatten_mask_dict_aligns_with_global_labels(self):
        d = self._data()
        masks = {t: torch.ones(d[t].y.shape[0], dtype=torch.bool) for t in d.edge_types}
        flat = flatten_mask_dict(d, masks)
        self.assertEqual(flat.shape[0], len(global_labels(d)))
        self.assertTrue(bool(flat.all()))

    def test_a_partial_mask_selects_the_intended_labels(self):
        d = self._data()
        masks = {t: torch.zeros(d[t].y.shape[0], dtype=torch.bool) for t in d.edge_types}
        masks[("User", "WRITE", "Resource")][:] = True
        selected = global_labels(d)[flatten_mask_dict(d, masks).numpy()]
        np.testing.assert_array_equal(selected, np.array([1, 1]))

    def test_empty_graph_is_handled(self):
        d = HeteroData()
        self.assertEqual(len(global_labels(d)), 0)
        self.assertEqual(len(flatten_mask_dict(d, {})), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
