"""
Ground-truth integrity check for the synthetic dataset.

Answers three reviewer questions with evidence rather than assertion:

  * Are labels preserved end to end (raw CloudTrail -> structural graph)? A
    re-parse must never silently drop or flip an attack label (review point 1,
    and the "missing labels become benign" failure mode).
  * Is the structural.log_id <-> annotation.log_id join strict 1:1 (review
    points 5, 14)? Attack progression can only be evaluated if every structural
    edge maps to exactly one campaign/stage annotation.
  * Is the dataset actually diverse and labelled (not the tiny all-benign demo
    fixture the review was accidentally run against)?

Run after regenerating the dataset:
    python verify_labels.py

Exits non-zero if any invariant fails, so it can gate a pipeline.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "synthetic_cloudtrail.csv"
STRUCT = HERE / "cloudtrail_structural.csv"
ANNOT = HERE / "synthetic_campaign_annotations.csv"


def fail(msg):
    print(f"  FAIL: {msg}")
    return False


def main():
    ok = True
    raw = pd.read_csv(RAW, low_memory=False)
    struct = pd.read_csv(STRUCT, low_memory=False)
    annot = pd.read_csv(ANNOT, low_memory=False) if ANNOT.exists() else None

    print("1. Label preservation (raw -> structural)")
    raw_attacks = int((raw.label == 1).sum())
    struct_attacks = int((struct.label == 1).sum())
    # structural may have fewer rows than raw if fe9 skipped malformed rows, but
    # every attack row that survives must keep label==1. We check that the
    # structural attack count matches the raw attack count for the rows fe9 kept.
    kept = set(int(x.split(":")[-1]) for x in struct.log_id)
    raw_kept_attacks = int((raw.iloc[sorted(kept)].label == 1).sum())
    print(f"   raw attacks={raw_attacks}  structural attacks={struct_attacks}  "
          f"raw attacks among kept rows={raw_kept_attacks}")
    if struct_attacks != raw_kept_attacks:
        ok = fail(f"attack count drifted in re-parse: {raw_kept_attacks} -> {struct_attacks}")
    if struct_attacks == 0:
        ok = fail("structural has ZERO attack labels -- the 'all benign' failure mode")
    else:
        print("   OK: attack labels preserved through re-parse")

    print("\n2. structural.log_id <-> annotation.log_id join (strict 1:1)")
    if annot is None:
        ok = fail("annotation file missing")
    else:
        if annot.log_id.duplicated().any():
            ok = fail("annotation log_id not unique")
        struct_ids = set(struct.log_id)
        annot_ids = set(annot.log_id)
        missing = struct_ids - annot_ids
        if missing:
            ok = fail(f"{len(missing)} structural edges have NO annotation (e.g. {list(missing)[:3]})")
        else:
            print(f"   OK: all {len(struct_ids)} structural edges join to exactly one annotation")
        # labels must agree across the join
        merged = struct[["log_id", "label"]].merge(
            annot[["log_id", "label"]], on="log_id", suffixes=("_struct", "_annot"))
        disagree = int((merged.label_struct != merged.label_annot).sum())
        if disagree:
            ok = fail(f"{disagree} rows where structural label != annotation label")
        else:
            print("   OK: labels agree on every joined row")

    print("\n3. Dataset is diverse and labelled (not the demo fixture)")
    print(f"   rows={len(struct):,}  attacks={struct_attacks}  "
          f"distinct sources={struct.source_node.nunique()}  edge types={struct.edge_type.nunique()}")
    if len(struct) < 1000 or struct.source_node.nunique() < 50:
        ok = fail("dataset looks like a fixture, not a training corpus")
    else:
        print("   OK: full training corpus")

    if annot is not None:
        print("\n4. Campaign/stage annotation coverage")
        camp = annot[annot.campaign_id != ""]
        print(f"   campaigns={camp.campaign_id.nunique()}  "
              f"labelled chain-stage events={int((annot.stage_index >= 0).sum())}  "
              f"tactics={sorted(t for t in annot.attack_tactic.dropna().unique() if t)}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "INTEGRITY CHECK FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
