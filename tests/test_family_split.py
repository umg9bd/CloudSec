"""
test_family_split.py
====================
Guards the campaign-family holdout split (data_loader.family_holdout_assignment
+ assignment_split, train.py --split family_holdout).

An "unseen campaign" claim is only meaningful if no row of a held-out family
reaches training and no session straddles two splits. Row-level random splits
(stratified_edge_split) give neither guarantee.
"""

import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from data_loader import (
    assignment_split, family_holdout_assignment, flatten_mask_dict, global_labels,
)

SRC = "synthetic.csv"


def raw_frame():
    fam = (["A"] * 4 + ["B"] * 3 + ["C"] * 3 + [None] * 10)
    sess = (["a1", "a1", "a2", "a2", "b1", "b1", "b2", "c1", "c1", "c2"]
            + [f"bg{i // 2}" for i in range(10)])
    return pd.DataFrame({"campaign_family": fam, "session_id": sess,
                         "label": [1] * 10 + [0] * 10})


def graph_for(df):
    """Two scored triples splitting the rows between them, log_ids as the feature engine writes them."""
    data = HeteroData()
    data["User"].x = torch.zeros((1, 1))
    data["Resource"].x = torch.zeros((1, 1))
    data["Role"].x = torch.zeros((1, 1))
    halves = {("User", "READ", "Resource"): range(0, len(df), 2),
              ("Role", "WRITE", "Resource"): range(1, len(df), 2)}
    for triple, idx in halves.items():
        idx = list(idx)
        data[triple].edge_index = torch.zeros((2, len(idx)), dtype=torch.long)
        data[triple].edge_attr = torch.zeros((len(idx), 1))
        data[triple].y = torch.tensor(df["label"].values[idx], dtype=torch.long)
        data[triple].log_id = [f"{SRC}:{i}" for i in idx]
    return data


class TestFamilyHoldoutAssignment(unittest.TestCase):
    def setUp(self):
        self.df = raw_frame()
        self.assign = family_holdout_assignment(self.df, SRC, ["B"], ["C"], seed=0)

    def test_every_row_is_assigned(self):
        self.assertEqual(len(self.assign), len(self.df))

    def test_each_family_lands_in_exactly_one_split(self):
        for fam, expected in {"A": {"train"}, "B": {"val"}, "C": {"test"}}.items():
            got = {self.assign[f"{SRC}:{i}"] for i in self.df.index[self.df.campaign_family == fam]}
            self.assertEqual(got, expected, fam)

    def test_no_session_straddles_splits(self):
        by_session = {}
        for i, s in enumerate(self.df.session_id):
            by_session.setdefault(s, set()).add(self.assign[f"{SRC}:{i}"])
        for s, splits in by_session.items():
            self.assertEqual(len(splits), 1, s)

    def test_deterministic_for_a_seed(self):
        self.assertEqual(self.assign, family_holdout_assignment(self.df, SRC, ["B"], ["C"], seed=0))

    def test_rejects_a_family_in_both_val_and_test(self):
        with self.assertRaises(ValueError):
            family_holdout_assignment(self.df, SRC, ["B"], ["B"])

    def test_rejects_unknown_family(self):
        with self.assertRaises(ValueError):
            family_holdout_assignment(self.df, SRC, ["B"], ["Z"])

    def test_rejects_holding_out_every_family(self):
        with self.assertRaises(ValueError):
            family_holdout_assignment(self.df, SRC, ["A", "B"], ["C"])

    def test_rejects_data_without_family_annotations(self):
        with self.assertRaises(ValueError):
            family_holdout_assignment(self.df.drop(columns=["campaign_family"]), SRC, ["B"], ["C"])


class TestAssignmentSplit(unittest.TestCase):
    def setUp(self):
        self.df = raw_frame()
        self.data = graph_for(self.df)
        self.assign = family_holdout_assignment(self.df, SRC, ["B"], ["C"], seed=0)

    def test_masks_partition_the_scored_edges(self):
        masks = assignment_split(self.data, self.assign)
        flat = np.stack([flatten_mask_dict(self.data, m).numpy() for m in masks])
        self.assertTrue((flat.sum(axis=0) == 1).all())

    def test_masks_follow_the_assignment(self):
        train, val, test = assignment_split(self.data, self.assign)
        for masks, name in [(train, "train"), (val, "val"), (test, "test")]:
            for triple, mask in masks.items():
                for lid, m in zip(self.data[triple].log_id, mask.tolist()):
                    self.assertEqual(m, self.assign[lid] == name, (triple, lid))

    def test_held_out_family_rows_never_reach_training(self):
        train, _, _ = assignment_split(self.data, self.assign)
        held_out = {f"{SRC}:{i}" for i in self.df.index[self.df.campaign_family.isin(["B", "C"])]}
        for triple, mask in train.items():
            for lid, m in zip(self.data[triple].log_id, mask.tolist()):
                self.assertFalse(m and lid in held_out, lid)

    def test_labels_line_up_with_the_global_order(self):
        train, _, _ = assignment_split(self.data, self.assign)
        y = global_labels(self.data)
        self.assertEqual(len(flatten_mask_dict(self.data, train)), len(y))

    def test_unassigned_edge_is_an_error(self):
        partial = dict(self.assign)
        partial.pop(f"{SRC}:0")
        with self.assertRaises(ValueError):
            assignment_split(self.data, partial)


if __name__ == "__main__":
    unittest.main()
