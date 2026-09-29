"""
campaign_split.py
=================
ONE campaign-family train/val/test assignment for the synthetic data, shared by
both tracks:

  * the graph track   -- data_loader.campaign_family_split (train.py --split campaign_family)
  * the sequence track -- train_lstm_transformer_v6.py --split campaign_family
  * the feature engine -- feature_engine9.py --split-file: label-fitted risk
    priors learn from TRAIN rows only

Every row that belongs to a campaign family follows its family, whatever its
label -- campaigns contain label-0 context events (the attacker's own recon and
noise), and splitting those at random put most of a held-out campaign's
context into training. Rows with no family (background activity) are split at
random BY USER, so benign behaviour appears in every split and no user's
history straddles two splits.

Families are assigned either explicitly (val_families / test_families; every
other family trains) or by a seeded shuffle in train/val/test ratios. The
seeded family assignment is the one data_loader.campaign_family_split has used
since it was introduced, so earlier family splits are reproduced.

The family is ground truth: it decides the split and is never a model feature.

CLI -- write the assignment once, then hand the same file to every consumer:

    python campaign_split.py --out splits/campaign_family_seed42.csv
    python campaign_split.py --out splits/cf.csv --val-families A B --test-families C
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANNOTATIONS = os.path.join(HERE, "datasets", "privilege-escalation",
                                   "synthetic_campaign_annotations.csv")
DEFAULT_EVENTS = os.path.join(HERE, "datasets", "privilege-escalation", "synthetic_cloudtrail.csv")
SPLIT_NAMES = ("train", "val", "test")


def load_annotations(path: str = DEFAULT_ANNOTATIONS) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def family_assignment(
    annotations: pd.DataFrame,
    seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15,
    val_families: Optional[Iterable[str]] = None,
    test_families: Optional[Iterable[str]] = None,
    family_col: str = "chain_name", label_col: str = "label",
    groups: Optional[Iterable[str]] = None,
) -> Dict[str, str]:
    """log_id -> "train" / "val" / "test" for every annotated row.

    groups (optional, one key per row, e.g. username): rows with no family are
    assigned per GROUP rather than per row, so one benign user's events never
    straddle splits -- the sequence model reads a user's history as input and
    selects evaluation windows by user."""
    for col in ("log_id", family_col, label_col):
        if col not in annotations.columns:
            raise ValueError(f"annotations need a {col!r} column; have {sorted(annotations.columns)}")
    log_ids = annotations["log_id"].astype(str).tolist()
    fams = annotations[family_col].fillna("").astype(str).tolist()
    labels = pd.to_numeric(annotations[label_col], errors="coerce").fillna(0).astype(int).tolist()

    families = sorted({f for f, y in zip(fams, labels) if y == 1 and f})
    rng = np.random.default_rng(seed)
    if val_families or test_families:
        val_families, test_families = set(val_families or ()), set(test_families or ())
        overlap = val_families & test_families
        if overlap:
            raise ValueError(f"families in both val and test: {sorted(overlap)}")
        unknown = (val_families | test_families) - set(families)
        if unknown:
            raise ValueError(f"unknown families {sorted(unknown)}; known: {families}")
        if not set(families) - val_families - test_families:
            raise ValueError("every campaign family is held out; nothing left to train on")
        fam_split = {f: ("val" if f in val_families else "test" if f in test_families else "train")
                     for f in families}
    else:
        shuffled = list(families)
        rng.shuffle(shuffled)
        n_train = max(1, int(round(len(shuffled) * train_ratio)))
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) - n_train > 1 else 0
        fam_split = {f: "train" if i < n_train else ("val" if i < n_train + n_val else "test")
                     for i, f in enumerate(shuffled)}

    free = [i for i, f in enumerate(fams) if f not in fam_split]
    if groups is None:
        units = [(i,) for i in free]
    else:
        groups = list(groups)
        if len(groups) != len(log_ids):
            raise ValueError(f"groups has {len(groups)} entries for {len(log_ids)} rows")
        by_group = {}
        for i in free:
            by_group.setdefault(str(groups[i]), []).append(i)
        units = [tuple(by_group[g]) for g in sorted(by_group)]
    order = np.arange(len(units))
    rng.shuffle(order)
    n_tr, n_va = int(len(units) * train_ratio), int(len(units) * val_ratio)
    free_split = {}
    for j, u in enumerate(order):
        split = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
        for i in units[u]:
            free_split[i] = split

    return {lid: (fam_split[f] if f in fam_split else free_split[i])
            for i, (lid, f) in enumerate(zip(log_ids, fams))}


def build_assignment(annotation_path: str = None, events_path: str = None, **kwargs) -> Dict[str, str]:
    """The project's default campaign-family assignment: annotations for the
    families, the synthetic events' usernames for grouping background rows.
    Used by the CLI and by data_loader.campaign_family_split when no split
    file is given, so both produce the same assignment."""
    annotations = load_annotations(annotation_path or DEFAULT_ANNOTATIONS)
    events = pd.read_csv(events_path or DEFAULT_EVENTS, usecols=["username"], dtype=str,
                         keep_default_na=False)
    if len(events) != len(annotations):
        raise ValueError(f"events ({len(events)}) and annotations ({len(annotations)}) do not align")
    return family_assignment(annotations, groups=events["username"].tolist(), **kwargs)


def family_of_split(annotations: pd.DataFrame, assignment: Dict[str, str],
                    family_col: str = "chain_name") -> Dict[str, str]:
    """family -> split, for reporting."""
    out = {}
    for lid, fam in zip(annotations["log_id"].astype(str), annotations[family_col].fillna("")):
        if fam:
            out[fam] = assignment[lid]
    return out


def write_split_file(assignment: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["log_id", "split"])
        for lid, split in assignment.items():
            w.writerow([lid, split])


def read_split_file(path: str) -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        out = {row["log_id"]: row["split"] for row in csv.DictReader(f)}
    bad = {s for s in out.values() if s not in SPLIT_NAMES}
    if bad:
        raise ValueError(f"{path}: unknown split names {sorted(bad)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--events", default=DEFAULT_EVENTS, help="synthetic events (usernames group background rows)")
    p.add_argument("--out", required=True, help="split file to write (log_id,split)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-families", nargs="+", default=None)
    p.add_argument("--test-families", nargs="+", default=None)
    args = p.parse_args()

    ann = load_annotations(args.annotations)
    assignment = build_assignment(args.annotations, args.events, seed=args.seed,
                                  val_families=args.val_families, test_families=args.test_families)
    write_split_file(assignment, args.out)
    by_family = family_of_split(ann, assignment)
    for name in SPLIT_NAMES:
        fams = sorted(f for f, s in by_family.items() if s == name)
        rows = sum(1 for s in assignment.values() if s == name)
        print(f"{name:5s}: {rows:6d} rows, {len(fams):2d} families {fams}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
