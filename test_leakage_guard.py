"""
test_leakage_guard.py
=====================
Regression tests for the two cross-branch leakage defects found when the
temporal and ensemble tracks were reviewed against the graph track.

C1 -- the sequence model was training on the graph model's held-out test set.
     train_temporal.csv = synthetic + the full invictus capture, and invictus
     had already been folded into real_dataset_combined.csv before that file
     was split into dev/test. 2,798 test rows and 60 dev rows ended up in the
     training set. Nothing errored, because the two tracks reach the same
     events through different files.

C2 -- AdaptiveRiskPrior is a LABEL-DERIVED feature (target encoding) with no
     eval-time freeze and a shared persisted file, so running feature
     engineering over a held-out split folded that split's labels into the
     features AND overwrote the training-fitted prior on disk.

The tests below also pin the near-miss that made the first version of the guard
useless: downstream builds namespace usernames by source ("inv:unknown_user"),
which silently broke key matching and made the guard report CLEAN on a
contaminated file. A guard that returns a false OK is worse than no guard, so
there is an explicit test for it.

    python -m unittest test_leakage_guard
"""

import json
import os
import tempfile
import unittest

import pandas as pd

import leakage_guard as lg
from feature_engine9 import AdaptiveRiskPrior


def _rows(pairs, prefix=""):
    """pairs: list of (timestamp, username)."""
    return pd.DataFrame([{"timestamp": t, "username": f"{prefix}{u}"} for t, u in pairs])


class TestHeldOutDetection(unittest.TestCase):
    """C1 regression, exercised against the real committed splits."""

    def setUp(self):
        self.test_df = pd.read_csv(lg.SPLIT_FILES["test"])
        self.dev_df = pd.read_csv(lg.SPLIT_FILES["dev"])

    def test_real_test_split_is_detected_as_held_out(self):
        hit = lg.find_heldout(self.test_df.head(200))
        self.assertEqual(hit["total"], 200)

    def test_real_dev_split_is_detected_as_held_out(self):
        hit = lg.find_heldout(self.dev_df.head(200))
        self.assertEqual(hit["total"], 200)

    def test_synthetic_training_data_is_not_flagged(self):
        """The false-positive check. Synthetic shares no events with the real
        splits; if the guard flagged it, the guard would be unusable."""
        syn = pd.read_csv(os.path.join(lg.DATA_DIR, "synthetic_cloudtrail.csv"))
        self.assertEqual(lg.find_heldout(syn)["total"], 0)

    def test_assert_raises_on_contaminated_frame(self):
        with self.assertRaises(SystemExit) as ctx:
            lg.assert_no_heldout(self.test_df.head(10), "fixture")
        self.assertIn("LEAKAGE", str(ctx.exception))

    def test_assert_passes_on_clean_frame(self):
        clean = _rows([("1999-01-01 00:00:00+00:00", "nobody")])
        lg.assert_no_heldout(clean, "fixture")

    def test_filter_removes_exactly_the_held_out_rows(self):
        clean_rows = _rows([(f"1999-01-0{i} 00:00:00+00:00", "nobody") for i in range(1, 6)])
        mixed = pd.concat(
            [clean_rows, self.test_df.head(7)[["timestamp", "username"]]], ignore_index=True
        )
        out, dropped = lg.filter_heldout(mixed, "fixture", verbose=False)
        self.assertEqual(dropped, 7)
        self.assertEqual(len(out), 5)
        lg.assert_no_heldout(out, "fixture")


class TestKeyNormalisation(unittest.TestCase):
    """The near-miss. An un-normalised key stopped matching once the LSTM build
    namespaced usernames, and the guard reported a contaminated file as clean."""

    def setUp(self):
        self.test_df = pd.read_csv(lg.SPLIT_FILES["test"])

    def test_source_namespaced_usernames_still_match(self):
        held = self.test_df.head(25)[["timestamp", "username"]].copy()
        held["username"] = "inv:" + held["username"].astype(str)
        self.assertEqual(lg.find_heldout(held)["total"], 25)

    def test_a_different_namespace_also_matches(self):
        held = self.test_df.head(25)[["timestamp", "username"]].copy()
        held["username"] = "fe:" + held["username"].astype(str)
        self.assertEqual(lg.find_heldout(held)["total"], 25)

    def test_guard_refuses_to_pass_when_only_timestamps_match(self):
        """If key normalisation ever drifts again, the timestamp cross-check
        must turn a false OK into a hard failure."""
        held = self.test_df.head(25)[["timestamp", "username"]].copy()
        held["username"] = "totally-unmatchable-" + held["username"].astype(str)
        with self.assertRaises(SystemExit) as ctx:
            lg.find_heldout(held)
        self.assertIn("GUARD INCONSISTENCY", str(ctx.exception))

    def test_reduced_key_is_used_when_event_name_absent(self):
        held = self.test_df.head(5)[["timestamp", "username"]]
        self.assertEqual(lg.find_heldout(held)["key"], lg.KEY_COLS_REDUCED)

    def test_full_key_is_preferred_when_available(self):
        held = self.test_df.head(5)[["timestamp", "event_name", "username"]]
        self.assertEqual(lg.find_heldout(held)["key"], lg.KEY_COLS)


class TestAdaptiveRiskPriorFreeze(unittest.TestCase):
    """C2 regression. The prior is target encoding: score() is a function of
    ground-truth labels and is consumed as a model feature."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "prior.json")

    def _fitted(self, n=20):
        p = AdaptiveRiskPrior({"CreateUser": 0.7}, path=self.path, frozen=False)
        for _ in range(n):
            p.update("CreateUser", "1")
        p.save()
        return p

    def test_unfrozen_prior_learns_from_labels(self):
        p = AdaptiveRiskPrior({"CreateUser": 0.7}, path=self.path, frozen=False)
        before = p.score("CreateUser")
        for _ in range(20):
            p.update("CreateUser", "1")
        self.assertGreater(p.score("CreateUser"), before)

    def test_frozen_prior_ignores_evaluation_labels(self):
        self._fitted()
        q = AdaptiveRiskPrior({"CreateUser": 0.7}, path=self.path, frozen=True)
        loaded = q.score("CreateUser")
        for _ in range(999):
            q.update("CreateUser", "0")
        self.assertAlmostEqual(q.score("CreateUser"), loaded, places=12)

    def test_frozen_prior_does_not_overwrite_the_fitted_file(self):
        self._fitted(n=20)
        q = AdaptiveRiskPrior({"CreateUser": 0.7}, path=self.path, frozen=True)
        for _ in range(50):
            q.update("CreateUser", "0")
        q.save()
        self.assertEqual(json.load(open(self.path))["CreateUser"], [20, 20])

    def test_frozen_prior_reuses_training_statistics(self):
        self._fitted()
        fitted_score = AdaptiveRiskPrior(
            {"CreateUser": 0.7}, path=self.path, frozen=False
        ).score("CreateUser")
        frozen_score = AdaptiveRiskPrior(
            {"CreateUser": 0.7}, path=self.path, frozen=True
        ).score("CreateUser")
        self.assertAlmostEqual(fitted_score, frozen_score, places=12)

    def test_frozen_without_a_fitted_file_fails_loudly(self):
        """Silently falling back to the hand-tuned prior would give the model a
        different feature distribution than it trained on, with no error."""
        with self.assertRaises(FileNotFoundError):
            AdaptiveRiskPrior({"X": 0.5}, path=os.path.join(self.tmp, "absent.json"), frozen=True)

    def test_unfrozen_without_a_file_is_fine(self):
        AdaptiveRiskPrior({"X": 0.5}, path=os.path.join(self.tmp, "absent.json"), frozen=False)


class TestResidualLeakCaughtByCrossCheck(unittest.TestCase):
    """Key matching alone left 42 invictus rows in the training set -- their
    username is the placeholder "unknown_user", which does not match the
    corresponding row in the split files. Only the timestamp cross-check found
    them. This pins that behaviour, since it is the reason the LSTM build
    excludes by SOURCE rather than relying on key matching."""

    def test_committed_train_temporal_is_still_contaminated(self):
        """Standing reminder: the build script is fixed, but the committed data
        file has not been regenerated yet. Delete this test once it has."""
        path = os.path.join("temporal-analysis", "data", "lstm", "train_temporal.csv")
        if not os.path.exists(path):
            self.skipTest("train_temporal.csv not present")
        hit = lg.find_heldout(pd.read_csv(path))
        self.assertGreater(hit["total"], 0,
                            "train_temporal.csv now looks clean -- regenerate confirmed, "
                            "delete this test")

    def test_dropping_real_capture_rows_yields_a_provably_clean_set(self):
        path = os.path.join("temporal-analysis", "data", "lstm", "train_temporal.csv")
        if not os.path.exists(path):
            self.skipTest("train_temporal.csv not present")
        df = pd.read_csv(path)
        synthetic_only = df.loc[
            ~df["username"].astype(str).str.startswith("inv:")
        ].reset_index(drop=True)
        self.assertEqual(lg.find_heldout(synthetic_only)["total"], 0)
        lg.assert_no_heldout(synthetic_only, "synthetic_only")


class TestCommittedArtefacts(unittest.TestCase):
    """Standing check on what is actually in the repo, so a stale contaminated
    file cannot sit there unnoticed."""

    def test_synthetic_training_artefacts_are_clean(self):
        for name in ("synthetic_cloudtrail.csv", "cloudtrail_temporal.csv"):
            path = os.path.join(lg.DATA_DIR, name)
            if os.path.exists(path):
                with self.subTest(file=name):
                    self.assertEqual(lg.find_heldout(pd.read_csv(path))["total"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
