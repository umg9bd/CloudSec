"""
One-time split of real_dataset_combined.csv into a dev slice (tuning:
ensemble fusion weights, thresholds, early stopping) and a test slice
(touched exactly once, at the end, for numbers that go in the paper).

Stratified by (data_source, session_label) so that:
  - every collector appears in both dev and test (a collector missing
    from one side would reopen the batch-effect risk already discussed)
  - the attack/benign ratio is preserved in both slices, not just overall

40% dev / 60% test: dev only needs to support tuning a handful of
parameters (the 3-weight ensemble fusion, a couple of thresholds); test
gets the larger share since it's what determines the final reported
confidence interval.

This is a ONE-TIME operation. Once real_dataset_dev.csv and
real_dataset_test.csv exist, don't regenerate them -- that would let
information leak from repeated re-splitting, defeating the point of a
held-out test set touched once.

Usage:
    python split_real_dev_test.py
"""

import random

import pandas as pd

INPUT_PATH = "real_dataset_combined.csv"
DEV_PATH = "real_dataset_dev.csv"
TEST_PATH = "real_dataset_test.csv"

DEV_FRACTION = 0.4
SEED = 42


def stratified_session_split(sessions_df, dev_fraction, seed):
    """sessions_df: one row per session_id, with data_source + session_label.
    Returns {session_id: 'dev'|'test'}, splitting within each
    (data_source, session_label) stratum independently."""
    rng = random.Random(seed)
    assignment = {}

    for (source, label), grp in sessions_df.groupby(["data_source", "session_label"]):
        ids = list(grp["session_id"])
        rng.shuffle(ids)
        n = len(ids)

        if n == 1:
            # can't split a single-member stratum both ways; test gets it,
            # since test is what the final reported numbers depend on
            n_dev = 0
        else:
            n_dev = round(n * dev_fraction)
            n_dev = max(1, min(n - 1, n_dev))  # guarantee at least 1 on each side

        for sid in ids[:n_dev]:
            assignment[sid] = "dev"
        for sid in ids[n_dev:]:
            assignment[sid] = "test"

    return assignment


def main():
    df = pd.read_csv(INPUT_PATH)

    sessions = (
        df.groupby("session_id")
        .agg(data_source=("data_source", "first"), session_label=("session_label", "max"))
        .reset_index()
    )

    assignment = stratified_session_split(sessions, DEV_FRACTION, SEED)
    df["split"] = df["session_id"].map(assignment)

    dev_df = df[df["split"] == "dev"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    dev_df.to_csv(DEV_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Wrote {DEV_PATH}: {dev_df['session_id'].nunique()} sessions, {len(dev_df)} events")
    print(f"Wrote {TEST_PATH}: {test_df['session_id'].nunique()} sessions, {len(test_df)} events")
    print()
    print("Per-collector breakdown (attack / benign sessions):")
    check = sessions.copy()
    check["split"] = check["session_id"].map(assignment)
    summary = check.groupby(["data_source", "split", "session_label"]).size().unstack(fill_value=0)
    print(summary)


if __name__ == "__main__":
    main()
