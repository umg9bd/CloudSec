"""
test_campaign_split.py
======================
Guards campaign_split.py -- the ONE campaign-family assignment shared by the
graph track, the LSTM trainer, and the feature engine's prior fitting -- and
feature_engine9's --split-file / --tag mode.

Before: the graph split lived only in data_loader, the LSTM had no family
split at all, and the risk priors in the temporal features were fitted on every
synthetic row, so a family "held out" from the LSTM had already contributed its
labels to the LSTM's own input features.
"""

import csv
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import campaign_split as cs
import feature_engine9 as fe

SRC = "train.csv"
# (family, label, user): families have attack rows and label-0 context rows.
ROWS = ([("A", 1, "a1"), ("A", 0, "a1"), ("A", 1, "a2")] + [("B", 1, "b1"), ("B", 0, "b1")]
        + [("C", 1, "c1"), ("C", 0, "c1")] + [("D", 1, "d1")]
        + [("", 0, f"bg{i // 3}") for i in range(30)])


def annotations(rows=ROWS):
    return pd.DataFrame({"log_id": [f"{SRC}:{i}" for i in range(len(rows))],
                         "chain_name": [f for f, _, _ in rows],
                         "label": [str(y) for _, y, _ in rows]})


def users(rows=ROWS):
    return [u for _, _, u in rows]


class TestAssignment(unittest.TestCase):
    def test_every_row_is_assigned(self):
        a = cs.family_assignment(annotations(), groups=users())
        self.assertEqual(len(a), len(ROWS))
        self.assertTrue(set(a.values()) <= set(cs.SPLIT_NAMES))

    def test_family_rows_including_label_0_share_a_split(self):
        for seed in range(10):
            a = cs.family_assignment(annotations(), seed=seed, groups=users())
            for fam in "ABCD":
                got = {a[f"{SRC}:{i}"] for i, (f, _, _) in enumerate(ROWS) if f == fam}
                self.assertEqual(len(got), 1, (seed, fam))

    def test_background_users_never_straddle_splits(self):
        for seed in range(10):
            a = cs.family_assignment(annotations(), seed=seed, groups=users())
            by_user = {}
            for i, (_, _, u) in enumerate(ROWS):
                by_user.setdefault(u, set()).add(a[f"{SRC}:{i}"])
            self.assertTrue(all(len(s) == 1 for s in by_user.values()), seed)

    def test_background_reaches_every_split(self):
        a = cs.family_assignment(annotations(), seed=0, groups=users())
        got = {a[f"{SRC}:{i}"] for i, (f, _, _) in enumerate(ROWS) if not f}
        self.assertEqual(got, set(cs.SPLIT_NAMES))

    def test_seeded_family_assignment_matches_the_original_algorithm(self):
        for seed in range(5):
            families = ["A", "B", "C", "D"]
            rng = np.random.default_rng(seed)
            rng.shuffle(families)
            n_train = max(1, int(round(4 * 0.70)))
            n_val = max(1, int(round(4 * 0.15))) if 4 - n_train > 1 else 0
            expected = {f: "train" if i < n_train else ("val" if i < n_train + n_val else "test")
                        for i, f in enumerate(families)}
            a = cs.family_assignment(annotations(), seed=seed, groups=users())
            self.assertEqual(cs.family_of_split(annotations(), a), expected, seed)

    def test_explicit_families(self):
        a = cs.family_assignment(annotations(), val_families=["B"], test_families=["C", "D"])
        self.assertEqual(cs.family_of_split(annotations(), a),
                         {"A": "train", "B": "val", "C": "test", "D": "test"})

    def test_rejections(self):
        for kw in ({"val_families": ["B"], "test_families": ["B"]},
                   {"val_families": ["B"], "test_families": ["Z"]},
                   {"val_families": ["A", "B"], "test_families": ["C", "D"]}):
            with self.assertRaises(ValueError, msg=kw):
                cs.family_assignment(annotations(), **kw)

    def test_groups_must_align(self):
        with self.assertRaises(ValueError):
            cs.family_assignment(annotations(), groups=users()[:-1])


class TestSplitFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self):
        a = cs.family_assignment(annotations(), groups=users())
        path = os.path.join(self.tmp, "sub", "split.csv")
        cs.write_split_file(a, path)
        self.assertEqual(cs.read_split_file(path), a)

    def test_unknown_split_name_is_rejected(self):
        path = os.path.join(self.tmp, "split.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("log_id,split\nx:0,holdout\n")
        with self.assertRaises(ValueError):
            cs.read_split_file(path)

    def test_build_assignment_rejects_misaligned_files(self):
        ann = os.path.join(self.tmp, "ann.csv")
        ev = os.path.join(self.tmp, "ev.csv")
        annotations().to_csv(ann, index=False)
        pd.DataFrame({"username": users()[:-1]}).to_csv(ev, index=False)
        with self.assertRaises(ValueError):
            cs.build_assignment(ann, ev)


PATH_GLOBALS = ("DATA_DIR", "DEFAULT_INPUT", "STRUCT_OUT", "TEMPORAL_OUT", "STATE_FILE",
                "EVENT_NAME_VOCAB_FILE", "STATE_TRACKER_FILE", "GRAPH_NODE_STATE_FILE",
                "IDENTITY_STATE_FILE", "ACTION_PRIOR_FILE", "PRINCIPAL_PRIOR_FILE")


class TestFeatureEngineSplitFile(unittest.TestCase):
    """feature_engine9 --split-file --tag, in a sandbox (no repository file touched)."""

    def setUp(self):
        self.saved = {n: getattr(fe, n) for n in PATH_GLOBALS}
        self.tmp = d = tempfile.mkdtemp()
        fe.DATA_DIR = d
        fe.DEFAULT_INPUT = os.path.join(d, SRC)
        fe.STRUCT_OUT = os.path.join(d, "train_structural.csv")
        fe.TEMPORAL_OUT = os.path.join(d, "train_temporal.csv")
        fe.STATE_FILE = os.path.join(d, ".state.json")
        fe.EVENT_NAME_VOCAB_FILE = os.path.join(d, ".vocab.json")
        fe.STATE_TRACKER_FILE = os.path.join(d, ".tracker.json")
        fe.GRAPH_NODE_STATE_FILE = os.path.join(d, ".graph.json")
        fe.IDENTITY_STATE_FILE = os.path.join(d, ".identity.json")
        fe.ACTION_PRIOR_FILE = os.path.join(d, ".action_risk_prior.json")
        fe.PRINCIPAL_PRIOR_FILE = os.path.join(d, ".principal_risk_prior.json")
        with open(fe.DEFAULT_INPUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "event_name", "event_source", "principal_type",
                        "principal_arn", "username", "label"])
            for i, (_, y, u) in enumerate(ROWS):
                w.writerow([f"2024-01-01T10:{i:02d}:00Z", "ListRoles", "iam.amazonaws.com",
                            "IAMUser", f"arn:aws:iam::111122223333:user/{u}", u, y])
        self.assignment = cs.family_assignment(annotations(), groups=users())
        self.split_file = os.path.join(d, "split.csv")
        cs.write_split_file(self.assignment, self.split_file)

    def tearDown(self):
        for n, v in self.saved.items():
            setattr(fe, n, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def tagged(self, name):
        return os.path.join(self.tmp, name)

    def test_priors_fit_only_the_split_files_train_rows(self):
        fe.run_batch(fe.DEFAULT_INPUT, split_file=self.split_file, tag="cf")
        with open(self.tagged(".action_risk_prior_cf.json"), encoding="utf-8") as f:
            total = sum(t for _, t in json.load(f).values())
        self.assertEqual(total, sum(1 for s in self.assignment.values() if s == "train"))

    def test_outputs_are_tagged_and_defaults_untouched(self):
        fe.run_batch(fe.DEFAULT_INPUT, split_file=self.split_file, tag="cf")
        for name in ("train_cf_structural.csv", "train_cf_temporal.csv",
                     ".action_risk_prior_cf.json", ".principal_risk_prior_cf.json"):
            self.assertTrue(os.path.exists(self.tagged(name)), name)
        for name in ("train_structural.csv", ".action_risk_prior.json", ".principal_risk_prior.json"):
            self.assertFalse(os.path.exists(self.tagged(name)), name)

    def test_split_file_without_tag_is_refused(self):
        with self.assertRaises(SystemExit):
            fe.run_batch(fe.DEFAULT_INPUT, split_file=self.split_file)

    def test_row_missing_from_the_split_file_is_an_error(self):
        partial = dict(self.assignment)
        partial.pop(f"{SRC}:0")
        cs.write_split_file(partial, self.split_file)
        with self.assertRaises(ValueError):
            fe.run_batch(fe.DEFAULT_INPUT, split_file=self.split_file, tag="cf")

    def test_changed_split_file_marks_outputs_stale(self):
        fe.run_batch(fe.DEFAULT_INPUT, split_file=self.split_file, tag="cf")
        other = cs.family_assignment(annotations(), seed=7, groups=users())
        cs.write_split_file(other, self.split_file)
        with self.assertRaises(SystemExit):
            fe.run_batch(fe.DEFAULT_INPUT, split_file=self.split_file, tag="cf")


if __name__ == "__main__":
    unittest.main()
