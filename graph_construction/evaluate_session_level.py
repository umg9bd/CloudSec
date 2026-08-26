"""
Evaluates a trained checkpoint against whatever graph currently exists in
Neo4j, aggregated to SESSION level -- the same unit evaluate_baselines.py
uses (a session is flagged attack if ANY of its events/edges is flagged),
so the resulting F1 is actually comparable to the rule-based baselines'
F1 numbers. evaluate_on_real.py's F1 is edge-level and is NOT directly
comparable to evaluate_baselines.py's session-level F1 -- this script
exists to close that gap honestly rather than compare across units.

log_id ("<raw_csv_filename>:<row_index>") is parsed to recover each edge's
originating row in the raw CSV, which carries session_id -- never assumed
positional alignment with the structural CSV, since log_id already encodes
the exact original row index.

Usage (from the repo root):
    python graph_construction/evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt \
        --model sage --raw-csv datasets/privilege-escalation/real_dataset_test.csv \
        --threshold 0.5
    # or sweep thresholds on the dev set:
    python graph_construction/evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt \
        --model sage --raw-csv datasets/privilege-escalation/real_dataset_dev.csv --sweep
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from data_loader import PrivilegePropagationGraphLoader
from evaluate_on_real import build_model_from_args
from utils import evaluate

LOG_ID_RE = re.compile(r"^(.*):(\d+)$")

# Rhino/GuardDuty-style 11-rule set, imported rather than restated so it can
# never drift from evaluate_baselines.py's definition. This file lives in
# graph_construction/, one level below the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "datasets", "privilege-escalation"))
from evaluate_baselines import RULES  # noqa: E402

GUARDDUTY = "GuardDuty-style (11 rules)"
N_BOOTSTRAP = 10000


def _prf(y_true: np.ndarray, y_pred: np.ndarray):
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


def check_graph_provenance(source_csv, raw_basename: str) -> None:
    """Neo4j holds exactly ONE graph at a time, and nothing used to stop this
    script from scoring (say) the test graph against the dev CSV: the row-index
    join below would still "work", because row indices exist in both files, and
    would produce a plausible and entirely meaningless F1. Raises SystemExit on
    a mismatch; warns (rather than failing) for graphs built before provenance
    stamping existed, so old graphs still load.

    Extracted from main() so it is unit-testable without a live Neo4j."""
    expected_struct = raw_basename.replace(".csv", "_structural.csv")
    if source_csv is None:
        print(f"WARNING: the graph in Neo4j predates provenance stamping; cannot verify it was "
              f"built from {expected_struct}. Rebuild with build_graph.py to enable this check.")
        return
    if source_csv != expected_struct:
        raise SystemExit(
            f"GRAPH MISMATCH: Neo4j holds a graph built from {source_csv!r}, but "
            f"--raw-csv is {raw_basename!r} (expected {expected_struct!r}).\n"
            f"Run:  python graph_construction/build_graph.py datasets/privilege-escalation/{expected_struct}"
        )


def parse_row_indices(log_ids, raw_basename: str, n_rows: int) -> np.ndarray:
    """log_id is "<raw_csv_filename>:<row_index>". Recovers the row index of
    each edge in the raw CSV, and enforces the two things the recovery silently
    assumed before: that the edge actually came from THIS csv, and that the
    index is in range. Indexing raw_df by a row number that originated in a
    different file is silent corruption, not an error.

    Extracted from main() so it is unit-testable without a live Neo4j."""
    row_idx = []
    for lid in log_ids:
        mobj = LOG_ID_RE.match(lid)
        if not mobj:
            raise ValueError(f"Unparseable log_id: {lid!r}")
        if mobj.group(1) != raw_basename:
            raise SystemExit(
                f"LOG_ID MISMATCH: edge log_id {lid!r} originates from {mobj.group(1)!r}, "
                f"but --raw-csv is {raw_basename!r}. These row indices are not comparable."
            )
        row_idx.append(int(mobj.group(2)))
    row_idx = np.array(row_idx, dtype=int)
    if len(row_idx) and row_idx.max() >= n_rows:
        raise SystemExit(
            f"LOG_ID OUT OF RANGE: max row index {row_idx.max()} exceeds {raw_basename} "
            f"({n_rows} rows). The structural CSV and the raw CSV are out of sync."
        )
    return row_idx


def session_max_scores(row_idx: np.ndarray, probs: np.ndarray, raw_df, session_index):
    """Session score = MAX edge probability within that session, mirroring the
    rule baseline's "flag the session if ANY event trips a rule" exactly, so the
    two are measured in the same unit. Sessions with zero in-schema edges score
    0.0 (predicted benign) rather than being dropped -- dropping them would
    quietly shrink the evaluation set.

    Extracted from main() so it is unit-testable without a live Neo4j."""
    edge_df = pd.DataFrame({
        "session_id": raw_df["session_id"].to_numpy()[row_idx],
        "prob": probs,
    })
    session_prob = edge_df.groupby("session_id")["prob"].max()
    return session_prob.reindex(session_index).fillna(0.0).to_numpy(), session_prob


def report_baseline_comparison(raw_df, sessions_true, y_true, y_model):
    """Computes the rule baseline on THE SAME sessions just scored, and reports
    a PAIRED bootstrap on the difference.

    This used to be a hardcoded `F1=0.732 [0.672, 0.790]` print statement. That
    number came from evaluate_baselines.py running on real_dataset_combined.csv
    -- all 397 dev+test sessions -- while the model above is scored on the 238
    test sessions only. Comparing them was apples-to-oranges: the same rule set
    scores 0.747 on test alone. Two independently-bootstrapped CIs are also not
    a significance test; the sessions are paired, so the difference must be
    resampled jointly. Computing both here makes the mismatch impossible."""
    rules = RULES[GUARDDUTY]
    events = raw_df.groupby("session_id")["event_name"].apply(set).reindex(sessions_true.index)
    y_rule = events.apply(lambda s: int(bool(s & rules))).to_numpy()

    mp, mr, mf = _prf(y_true, y_model)
    bp, br, bf = _prf(y_true, y_rule)

    # Fully vectorized: one (N_BOOTSTRAP x n) resample-index matrix, F1 computed
    # row-wise as 2TP / (2TP + FP + FN). Looping with sklearn's scorers here
    # would mean ~60k Python-level calls and take minutes for a number that
    # prints on every run.
    rng = np.random.default_rng(42)
    n = len(y_true)
    idx = rng.integers(0, n, (N_BOOTSTRAP, n))
    yt, ym, yr = y_true[idx], y_model[idx], y_rule[idx]

    def _f1_rows(t, p):
        tp = ((t == 1) & (p == 1)).sum(1)
        fp = ((t == 0) & (p == 1)).sum(1)
        fn = ((t == 1) & (p == 0)).sum(1)
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)

    deltas = _f1_rows(yt, ym) - _f1_rows(yt, yr)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_two_sided = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))

    print(f"\n{'':<28}{'P':>8}{'R':>8}{'F1':>8}")
    print(f"{'GNN (session-level)':<28}{mp:>8.3f}{mr:>8.3f}{mf:>8.3f}")
    print(f"{GUARDDUTY:<28}{bp:>8.3f}{br:>8.3f}{bf:>8.3f}   <- computed on THESE {n} sessions")
    print(f"\nPAIRED bootstrap on (GNN - rule) F1: {mf - bf:+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  two-sided p = {p_two_sided:.4f}")
    if lo > 0:
        print("The improvement is significant at the 5% level.")
    else:
        print("The improvement is NOT significant at the 5% level -- the CI includes zero. "
              "Report it as a point estimate with this interval, not as 'beats the baseline'.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["sage", "gat"], required=True)
    p.add_argument("--raw-csv", required=True, help="e.g. real_dataset_test.csv or real_dataset_dev.csv")
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="test1234")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sweep", action="store_true", help="sweep thresholds 0.05-0.95 and report best-F1")
    args = p.parse_args()

    raw_df = pd.read_csv(args.raw_csv)
    if "session_id" not in raw_df.columns or "session_label" not in raw_df.columns:
        raise ValueError(f"{args.raw_csv} is missing session_id/session_label columns")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    fit_artifacts = ckpt["fit_artifacts"]

    loader = PrivilegePropagationGraphLoader(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_pass,
        fit_artifacts=fit_artifacts,
        model_node_types=set(model_args["node_feat_dims"]),
    )
    data, meta = loader.load()

    raw_basename = os.path.basename(args.raw_csv)
    check_graph_provenance(meta.get("source_csv"), raw_basename)

    trained_triples = set(tuple(t) for t in model_args["edge_types"])
    real_triples = set(data.edge_types)
    untrained_triples = real_triples - trained_triples
    for t in untrained_triples:
        del data[t]

    model = build_model_from_args(args.model, model_args)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_true_masks = {t: torch.ones(data[t].y.shape[0], dtype=torch.bool) for t in data.edge_types}
    m = evaluate(model, data, all_true_masks, return_probs=True)
    probs = np.array(m["probs"])

    # Same flattening order utils.evaluate uses internally: sorted(data.edge_types).
    log_ids_flat = []
    for t in sorted(data.edge_types):
        log_ids_flat.extend(data[t].log_id)
    assert len(log_ids_flat) == len(probs), f"{len(log_ids_flat)} log_ids vs {len(probs)} probs"

    row_idx = parse_row_indices(log_ids_flat, raw_basename, len(raw_df))

    sessions_true = raw_df.drop_duplicates("session_id").set_index("session_id")["session_label"]
    y_prob_full, session_prob = session_max_scores(row_idx, probs, raw_df, sessions_true.index)
    scored_sessions = sessions_true.index.intersection(session_prob.index)
    unscored = sessions_true.index.difference(session_prob.index)

    print(f"Sessions: {len(sessions_true)} total | {len(scored_sessions)} have >=1 in-schema edge "
          f"| {len(unscored)} have ZERO in-schema edges (predicted benign by default)")

    y_true = sessions_true.to_numpy()

    def score_at(thr):
        y_pred = (y_prob_full >= thr).astype(int)
        return (
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
        )

    if args.sweep:
        print("\nthreshold  precision  recall  f1")
        best = (0.0, -1, None)
        for thr in np.arange(0.05, 0.96, 0.05):
            pr, rc, f1 = score_at(thr)
            print(f"{thr:.2f}       {pr:.3f}      {rc:.3f}   {f1:.3f}")
            if f1 > best[1]:
                best = (thr, f1, (pr, rc))
        print(f"\nBest threshold: {best[0]:.2f}  F1={best[1]:.3f}  (P={best[2][0]:.3f}, R={best[2][1]:.3f})")
    else:
        pr, rc, f1 = score_at(args.threshold)
        print(f"\nSESSION-LEVEL @ threshold={args.threshold}: P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")
        y_pred = (y_prob_full >= args.threshold).astype(int)
        report_baseline_comparison(raw_df, sessions_true, y_true, y_pred)


if __name__ == "__main__":
    main()
