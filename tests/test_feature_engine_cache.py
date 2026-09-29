"""
test_feature_engine_cache.py
============================
Guards feature_engine9.run_batch's cached state -- the "cached state files"
leakage path.

Before: a regenerated input with the same file name was skipped as "already
processed", so its old features and priors fitted on the OLD labels stayed in
place; fitting always started from whatever prior counts were on disk; and
every row of the training file (including its own val/test rows) was fitted.

Everything runs in a temporary directory: all of feature_engine9's path
constants are redirected, so no repository file is read or written.
"""

import csv
import json
import os
import shutil
import tempfile
import unittest

import feature_engine9 as fe

PATH_GLOBALS = ("DATA_DIR", "DEFAULT_INPUT", "STRUCT_OUT", "TEMPORAL_OUT", "STATE_FILE",
                "EVENT_NAME_VOCAB_FILE", "STATE_TRACKER_FILE", "GRAPH_NODE_STATE_FILE",
                "IDENTITY_STATE_FILE", "ACTION_PRIOR_FILE", "PRINCIPAL_PRIOR_FILE")

FIELDS = ["timestamp", "event_name", "event_source", "principal_type", "principal_arn",
          "username", "label", "split"]


def write_input(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def rows(n, splits=("train",), label_every=3):
    out = []
    for i in range(n):
        out.append({"timestamp": f"2024-01-01T10:{i:02d}:00Z", "event_name": "CreateUser" if i % 2 else "ListRoles",
                    "event_source": "iam.amazonaws.com", "principal_type": "IAMUser",
                    "principal_arn": f"arn:aws:iam::111122223333:user/u{i % 4}", "username": f"u{i % 4}",
                    "label": "1" if i % label_every == 0 else "0", "split": splits[i % len(splits)]})
    return out


def count_rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def prior_total(path):
    with open(path, encoding="utf-8") as f:
        return sum(t for _, t in json.load(f).values())


class FeatureEngineSandbox(unittest.TestCase):
    def setUp(self):
        self.saved = {name: getattr(fe, name) for name in PATH_GLOBALS}
        self.tmp = tempfile.mkdtemp()
        d = self.tmp
        fe.DATA_DIR = d
        fe.DEFAULT_INPUT = os.path.join(d, "train.csv")
        fe.STRUCT_OUT = os.path.join(d, "train_structural.csv")
        fe.TEMPORAL_OUT = os.path.join(d, "train_temporal.csv")
        fe.STATE_FILE = os.path.join(d, ".state.json")
        fe.EVENT_NAME_VOCAB_FILE = os.path.join(d, ".vocab.json")
        fe.STATE_TRACKER_FILE = os.path.join(d, ".tracker.json")
        fe.GRAPH_NODE_STATE_FILE = os.path.join(d, ".graph.json")
        fe.IDENTITY_STATE_FILE = os.path.join(d, ".identity.json")
        fe.ACTION_PRIOR_FILE = os.path.join(d, ".action_prior.json")
        fe.PRINCIPAL_PRIOR_FILE = os.path.join(d, ".principal_prior.json")
        self.train = fe.DEFAULT_INPUT
        self.struct = fe.STRUCT_OUT      # run_batch reassigns the globals; keep the originals

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(fe, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestTrainOnlyFitting(FeatureEngineSandbox):
    def test_priors_are_fitted_only_on_training_rows(self):
        write_input(self.train, rows(12, splits=("train", "train", "val", "test")))
        fe.run_batch(self.train)
        self.assertEqual(prior_total(fe.ACTION_PRIOR_FILE), 6)          # 6 of 12 rows are train
        meta = fe.read_prior_meta(fe.ACTION_PRIOR_FILE)
        self.assertEqual((meta["rows_fitted"], meta["rows_withheld"]), (6, 6))
        self.assertEqual(meta["sha256"], fe.file_fingerprint(self.train))

    def test_all_rows_are_still_featurized(self):
        write_input(self.train, rows(12, splits=("train", "test")))
        fe.run_batch(self.train)
        self.assertEqual(count_rows(self.struct), 12)

    def test_evaluation_input_never_fits(self):
        write_input(self.train, rows(8))
        fe.run_batch(self.train)
        before = prior_total(fe.ACTION_PRIOR_FILE)
        other = os.path.join(self.tmp, "heldout.csv")
        write_input(other, rows(8, splits=("",)))
        fe.run_batch(other)
        self.assertEqual(prior_total(fe.ACTION_PRIOR_FILE), before)


class TestContentFingerprints(FeatureEngineSandbox):
    def test_unchanged_input_is_skipped(self):
        write_input(self.train, rows(6))
        fe.run_batch(self.train)
        fe.run_batch(self.train)
        self.assertEqual(count_rows(self.struct), 6)                    # not appended twice

    def test_changed_input_is_refused(self):
        write_input(self.train, rows(6))
        fe.run_batch(self.train)
        write_input(self.train, rows(9))
        with self.assertRaises(SystemExit) as ctx:
            fe.run_batch(self.train)
        self.assertIn("--rebuild", str(ctx.exception))

    def test_rebuild_replaces_outputs_and_refits_priors(self):
        write_input(self.train, rows(6))
        fe.run_batch(self.train)
        write_input(self.train, rows(9))
        fe.run_batch(self.train, rebuild=True)
        self.assertEqual(count_rows(self.struct), 9)                    # replaced, not 6 + 9
        self.assertEqual(prior_total(fe.ACTION_PRIOR_FILE), 9)          # refitted, not 6 + 9

    def test_priors_from_another_input_block_fitting(self):
        with open(fe.ACTION_PRIOR_FILE, "w", encoding="utf-8") as f:
            json.dump({"CreateUser": [5, 5]}, f)                         # no provenance
        write_input(self.train, rows(6))
        with self.assertRaises(SystemExit):
            fe.run_batch(self.train)

    def test_legacy_state_without_fingerprints_is_skipped_with_notice(self):
        write_input(self.train, rows(6))
        fe.run_batch(self.train)
        with open(fe.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed_files": ["train.csv"]}, f)             # pre-fingerprint format
        fe.run_batch(self.train)                                         # must not raise or append
        self.assertEqual(count_rows(self.struct), 6)


class TestWatchModeIsolation(unittest.TestCase):
    def test_watch_outputs_never_alias_training_outputs(self):
        watch = fe._paths_for_stem("watch_incoming")
        training = fe._derive_paths(fe.DEFAULT_INPUT)
        for key in ("struct_out", "temporal_out", "state_file", "state_tracker_file",
                    "graph_node_state_file", "identity_state_file"):
            self.assertNotEqual(os.path.abspath(watch[key]), os.path.abspath(training[key]), key)


if __name__ == "__main__":
    unittest.main()
