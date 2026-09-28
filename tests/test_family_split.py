"""
test_family_split.py
====================
Guards the campaign-family holdout split (data_loader.campaign_family_split,
train.py --split campaign_family).

An "unseen campaign" claim is only meaningful if no edge of a held-out family
reaches training -- including the family's label-0 events (the attacker's own
recon/noise), which an earlier version split at random.
"""

import csv
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from data_loader import campaign_family_split, flatten_mask_dict, global_labels

SRC = "synthetic_cloudtrail.csv"
# (family, label) per row: every family has attack rows AND label-0 context rows.
ROWS = ([("A", 1), ("A", 0), ("A", 1), ("A", 0)] + [("B", 1), ("B", 0), ("B", 1)]
        + [("C", 1), ("C", 0), ("C", 1)] + [("D", 1), ("D", 1)] + [("", 0)] * 20)


def write_annotations(path, rows=ROWS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["log_id", "chain_name", "label"])
        for i, (fam, label) in enumerate(rows):
            w.writerow([f"{SRC}:{i}", fam, label])


def graph_for(rows=ROWS):
    """Two scored triples splitting the rows between them, log_ids as the feature engine writes them."""
    data = HeteroData()
    for ntype in ("User", "Role", "Resource"):
        data[ntype].x = torch.zeros((1, 1))
    halves = {("User", "READ", "Resource"): list(range(0, len(rows), 2)),
              ("Role", "WRITE", "Resource"): list(range(1, len(rows), 2))}
    labels = np.array([lab for _, lab in rows])
    for triple, idx in halves.items():
        data[triple].edge_index = torch.zeros((2, len(idx)), dtype=torch.long)
        data[triple].edge_attr = torch.zeros((len(idx), 1))
        data[triple].y = torch.tensor(labels[idx], dtype=torch.long)
        data[triple].log_id = [f"{SRC}:{i}" for i in idx]
    return data


def split_of_each_row(data, masks):
    out = {}
    for name, mask_dict in zip(("train", "val", "test"), masks):
        for triple, mask in mask_dict.items():
            for lid, m in zip(data[triple].log_id, mask.tolist()):
                if m:
                    out.setdefault(lid, []).append(name)
    return out


class SplitSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ann = os.path.join(self.tmp, "annotations.csv")
        write_annotations(self.ann)
        self.data = graph_for()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def split(self, **kw):
        return campaign_family_split(self.data, annotation_path=self.ann, **kw)


class TestFamilyIsolation(SplitSandbox):
    def test_masks_partition_every_scored_edge(self):
        rows = split_of_each_row(self.data, self.split(seed=0))
        self.assertEqual(len(rows), len(ROWS))
        self.assertTrue(all(len(v) == 1 for v in rows.values()))

    def test_every_row_of_a_family_including_label_0_lands_in_one_split(self):
        for seed in range(10):
            rows = split_of_each_row(self.data, self.split(seed=seed))
            for fam in "ABCD":
                got = {rows[f"{SRC}:{i}"][0] for i, (f, _) in enumerate(ROWS) if f == fam}
                self.assertEqual(len(got), 1, (seed, fam, got))

    def test_explicit_families(self):
        rows = split_of_each_row(self.data, self.split(val_families=["B"], test_families=["C", "D"]))
        expect = {"A": "train", "B": "val", "C": "test", "D": "test"}
        for i, (fam, _) in enumerate(ROWS):
            if fam:
                self.assertEqual(rows[f"{SRC}:{i}"][0], expect[fam], (i, fam))

    def test_held_out_rows_never_reach_training(self):
        train, _, _ = self.split(val_families=["B"], test_families=["C"])
        held = {f"{SRC}:{i}" for i, (f, _) in enumerate(ROWS) if f in ("B", "C")}
        for triple, mask in train.items():
            for lid, m in zip(self.data[triple].log_id, mask.tolist()):
                self.assertFalse(m and lid in held, lid)

    def test_background_rows_reach_every_split(self):
        rows = split_of_each_row(self.data, self.split(seed=0))
        got = {rows[f"{SRC}:{i}"][0] for i, (f, _) in enumerate(ROWS) if not f}
        self.assertEqual(got, {"train", "val", "test"})

    def test_masks_align_with_the_global_label_order(self):
        train, _, _ = self.split(seed=0)
        self.assertEqual(len(flatten_mask_dict(self.data, train)), len(global_labels(self.data)))


class TestSeededAssignment(SplitSandbox):
    def test_deterministic_for_a_seed(self):
        a = split_of_each_row(self.data, self.split(seed=3))
        b = split_of_each_row(self.data, self.split(seed=3))
        self.assertEqual(a, b)

    def test_family_assignment_matches_the_original_algorithm(self):
        """Seeded runs must assign FAMILIES exactly as the first version did, so
        the reported 11/2/3 family split stays reproducible."""
        for seed in range(5):
            families = ["A", "B", "C", "D"]
            rng = np.random.default_rng(seed)
            rng.shuffle(families)
            n_train = max(1, int(round(len(families) * 0.70)))
            n_val = max(1, int(round(len(families) * 0.15))) if len(families) - n_train > 1 else 0
            expected = {f: "train" if i < n_train else ("val" if i < n_train + n_val else "test")
                        for i, f in enumerate(families)}
            rows = split_of_each_row(self.data, self.split(seed=seed))
            for i, (fam, lab) in enumerate(ROWS):
                if fam and lab == 1:
                    self.assertEqual(rows[f"{SRC}:{i}"][0], expected[fam], (seed, fam))


class TestSplitFile(SplitSandbox):
    """split_file lets the graph track use the exact file the LSTM and the
    feature engine use (campaign_split.py)."""

    def test_masks_follow_the_split_file(self):
        import campaign_split as cs
        path = os.path.join(self.tmp, "split.csv")
        assignment = {f"{SRC}:{i}": ("test" if i % 3 == 0 else "train") for i in range(len(ROWS))}
        cs.write_split_file(assignment, path)
        rows = split_of_each_row(self.data, campaign_family_split(self.data, split_file=path))
        self.assertEqual({k: v[0] for k, v in rows.items()}, assignment)

    def test_split_file_missing_an_edge_is_an_error(self):
        import campaign_split as cs
        path = os.path.join(self.tmp, "split.csv")
        cs.write_split_file({f"{SRC}:{i}": "train" for i in range(len(ROWS) - 1)}, path)
        with self.assertRaises(ValueError):
            campaign_family_split(self.data, split_file=path)


class TestFailsLoudly(SplitSandbox):
    def test_family_in_both_val_and_test(self):
        with self.assertRaises(ValueError):
            self.split(val_families=["B"], test_families=["B"])

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            self.split(val_families=["B"], test_families=["Z"])

    def test_holding_out_every_family(self):
        with self.assertRaises(ValueError):
            self.split(val_families=["A", "B"], test_families=["C", "D"])

    def test_edge_missing_from_the_annotations(self):
        write_annotations(self.ann, ROWS[:-1])
        with self.assertRaises(ValueError):
            self.split(seed=0)


if __name__ == "__main__":
    unittest.main()
