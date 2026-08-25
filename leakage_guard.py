"""
leakage_guard.py
================
One shared definition of "held out", and a hard check that no training artefact
contains it.

WHY THIS EXISTS
---------------
The project's real data is split once, by session, into
`real_dataset_dev.csv` (tuning) and `real_dataset_test.csv` (touched once).
That discipline is only worth anything if EVERY model in the ensemble respects
the same boundary -- and the boundary is easy to cross by accident, because the
same underlying events reach different tracks through different files.

That is exactly what happened. The temporal track's training set is built from
`cloudtrail_temporal.csv` (synthetic) + `invictus_temporal.csv`, and the
invictus capture was folded into `real_dataset_combined.csv` before the split.
So the sequence model was training on events that sit inside the graph model's
held-out test split:

    invictus events inside real_dataset_dev.csv  :    51
    invictus events inside real_dataset_test.csv : 1,502

Any ensemble combining the two, evaluated on the test split, would have a
component that had already seen the answer. Nothing errored; nothing warned.

WHAT COUNTS AS AN EVENT
-----------------------
Rows cannot be matched by index -- the combined dataset was deduplicated and
re-sessionised, so positions do not correspond. They are matched on the natural
key (timestamp, event_name, username), which was verified to align exactly:
1,553 unique invictus keys, 1,553 present in the combined dataset, zero
mismatches.

USAGE
-----
Audit any file(s) before training on them:

    python leakage_guard.py temporal-analysis/data/lstm/train_temporal.csv

In code:

    from leakage_guard import assert_no_heldout, filter_heldout
    assert_no_heldout(train_df, "train_temporal")        # raise if contaminated
    clean, dropped = filter_heldout(train_df, "train_temporal")   # or drop them
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

import pandas as pd

# The natural key for "the same CloudTrail event", verified to align exactly
# between invictus_enriched.csv and real_dataset_combined.csv (1,553/1,553).
#
# The temporal CSVs carry only log_id/username/timestamp + feature columns --
# event_name is attached later in the LSTM dataset build -- so a reduced key is
# needed there. It was validated not to over-match: on the synthetic training
# set, which shares no events with the real splits, the reduced key reports
# ZERO overlap (0/9,711), while correctly flagging 612/629 invictus keys. A
# reduced key can only ever over-report, never under-report, so a clean result
# under it is a strong result.
KEY_COLS = ("timestamp", "event_name", "username")
KEY_COLS_REDUCED = ("timestamp", "username")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "datasets", "privilege-escalation")
SPLIT_FILES = {
    "dev": os.path.join(DATA_DIR, "real_dataset_dev.csv"),
    "test": os.path.join(DATA_DIR, "real_dataset_test.csv"),
}


def choose_key(df: pd.DataFrame) -> tuple:
    """Richest key this frame supports. Prefers the full key; falls back to the
    reduced one for temporal feature files that carry no event_name."""
    if all(c in df.columns for c in KEY_COLS):
        return KEY_COLS
    if all(c in df.columns for c in KEY_COLS_REDUCED):
        return KEY_COLS_REDUCED
    missing = [c for c in KEY_COLS_REDUCED if c not in df.columns]
    raise KeyError(
        f"cannot check for held-out data: missing key column(s) {missing}. "
        f"Present: {sorted(df.columns)[:12]}..."
    )


# Downstream builds namespace usernames by source (invictus rows become
# "inv:unknown_user"). An un-normalised key therefore silently stops matching
# and the guard reports a clean bill of health on contaminated data -- which is
# strictly worse than no guard. Strip a leading "<source>:" namespace.
_NS_PREFIX = re.compile(r"^[a-z]{2,6}:")


def _norm(col: str, values: pd.Series) -> pd.Series:
    v = values.astype(str)
    if col == "username":
        v = v.str.replace(_NS_PREFIX, "", regex=True)
    return v


def _keys(df: pd.DataFrame, key_cols: tuple = KEY_COLS) -> set:
    cols = [_norm(c, df[c]) for c in key_cols]
    return set(zip(*cols))


def _timestamps(df: pd.DataFrame) -> set:
    return set(df["timestamp"].astype(str))


@lru_cache(maxsize=2)
def _keys_by_split(key_cols: tuple = KEY_COLS) -> dict:
    out = {}
    for name, path in SPLIT_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} split not found at {path}. leakage_guard cannot verify "
                f"anything without it -- do not train until this resolves."
            )
        out[name] = _keys(pd.read_csv(path), key_cols)
    return out


@lru_cache(maxsize=1)
def _timestamps_by_split() -> frozenset:
    out: set = set()
    for path in SPLIT_FILES.values():
        out |= _timestamps(pd.read_csv(path))
    return frozenset(out)


def heldout_keys(key_cols: tuple = KEY_COLS) -> frozenset:
    """Every event key in dev or test."""
    by_split = _keys_by_split(key_cols)
    return frozenset(by_split["dev"] | by_split["test"])


def find_heldout(df: pd.DataFrame) -> dict:
    """Returns {'dev': n, 'test': n, 'total': n, 'key': tuple, 'mask': BoolSeries}."""
    key_cols = choose_key(df)
    by_split = _keys_by_split(key_cols)
    cols = [_norm(c, df[c]) for c in key_cols]
    row_keys = pd.Series(list(zip(*cols)), index=df.index)
    dev_mask = row_keys.isin(by_split["dev"])
    test_mask = row_keys.isin(by_split["test"])
    both = dev_mask | test_mask

    # Independent, weaker cross-check. Real capture timestamps are second-
    # resolution wall-clock and share nothing with the synthetic generator's
    # (verified: 0 overlap on 9,711 synthetic rows), so a timestamp hit that
    # the keyed check missed means key normalisation has drifted again --
    # exactly the failure mode that made an earlier version of this file report
    # "clean" on a contaminated training set.
    ts_hits = int(df["timestamp"].astype(str).isin(_timestamps_by_split()).sum())
    if ts_hits and not int(both.sum()):
        raise SystemExit(
            f"\nGUARD INCONSISTENCY: {ts_hits} rows share a timestamp with the held-out "
            f"splits, but the {'+'.join(key_cols)} key matched none of them.\n"
            f"The key has almost certainly stopped matching (e.g. a new username "
            f"namespace). Do NOT treat this dataset as clean -- fix the key first."
        )
    return {
        "dev": int(dev_mask.sum()),
        "test": int(test_mask.sum()),
        "total": int(both.sum()),
        "key": key_cols,
        "mask": both,
    }


def assert_no_heldout(df: pd.DataFrame, name: str) -> None:
    """Raises if `df` contains any event from dev or test.

    Call this immediately before writing or training on any dataset that is
    meant to be training-only. Failing loudly here is the entire point: the
    alternative is a number that looks fine and is not."""
    hit = find_heldout(df)
    if hit["total"]:
        raise SystemExit(
            f"\nLEAKAGE: {name} contains {hit['total']} events from the held-out splits "
            f"({hit['dev']} from dev, {hit['test']} from test) out of {len(df)} rows.\n"
            f"Training on these invalidates any evaluation on real_dataset_test.csv,\n"
            f"including any ensemble this model feeds into.\n"
            f"Fix the dataset build (see filter_heldout) rather than suppressing this."
        )


def filter_heldout(df: pd.DataFrame, name: str, verbose: bool = True):
    """Drops every dev/test event from `df`. Returns (clean_df, n_dropped)."""
    hit = find_heldout(df)
    if not hit["total"]:
        if verbose:
            print(f"[leakage_guard] {name}: clean ({len(df)} rows, no held-out events)")
        return df, 0
    clean = df.loc[~hit["mask"]].reset_index(drop=True)
    if verbose:
        print(f"[leakage_guard] {name}: dropped {hit['total']} held-out events "
              f"({hit['dev']} dev, {hit['test']} test); {len(df)} -> {len(clean)} rows")
    return clean, hit["total"]


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for path in argv:
        df = pd.read_csv(path)
        try:
            hit = find_heldout(df)
        except KeyError as e:
            print(f"SKIP  {path}\n      {e}")
            continue
        label = os.path.basename(path)
        if hit["total"]:
            bad = 1
            print(f"LEAK  {label}: {hit['total']}/{len(df)} rows are held out "
                  f"({hit['dev']} dev, {hit['test']} test)  (key={'+'.join(hit['key'])})")
        else:
            print(f"OK    {label}: {len(df)} rows, none held out  (key={'+'.join(hit['key'])})")
    return bad


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
