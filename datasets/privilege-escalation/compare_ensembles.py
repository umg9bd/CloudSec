"""
Head-to-head comparison of the two ensemble candidates, for the final choice
between them:

  A. ensemble.py   -- fixed weighted sum, 0.5 * gnn_event_score + 0.5 * lstm_event_score
  B. ensemble1.py  -- stacked logistic-regression meta-learner (fit on synthetic only)

Both use the same per-event GNN/LSTM scores, so any difference is purely the
combiner. Same protocol as every other result in this project:

  - DEV (real_dataset_dev.csv): sweep each candidate's session-level threshold and report
    the dev-tuned value next to the SESSION_ALERT_THRESHOLD actually shipped in each file
    (the shipped constant is a rounded dev-tuned value -- this flags any drift between them).
  - TEST (real_dataset_test.csv, touched once): each candidate is scored EXACTLY as it runs in
    production -- 0-10 risk_score rounded to 2 decimals, compared against its file's shipped
    SESSION_ALERT_THRESHOLD, session flagged if any event reaches it -- so the reported
    numbers are the numbers of what actually ships.
  - PAIRED bootstrap on the 238 test sessions: A vs B (the comparison the choice hinges on),
    and each vs the curated rule baseline.

GNN/LSTM per-event scoring is the slow step (~3 min dev, ~10 min test on CPU). Pass
--cache-dir to reuse/write <stem>__gnn.parquet / <stem>__lstm.parquet there.

--lstm-ckpt evaluates both candidates with a different LSTM checkpoint (e.g. a retrain). The
shipped thresholds were tuned for the default checkpoint, so in that mode each candidate is
tested at its own dev-tuned threshold rounded to 2 decimals -- the value that would ship --
and candidate B's meta-learner is refit against that checkpoint (ensemble1.fit_meta_learner).

Usage (from the repo root, using the project venv):
    .venv/Scripts/python.exe datasets/privilege-escalation/compare_ensembles.py
    .venv/Scripts/python.exe datasets/privilege-escalation/compare_ensembles.py --cache-dir <dir>
    .venv/Scripts/python.exe datasets/privilege-escalation/compare_ensembles.py --lstm-ckpt <path.pt>
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "graph_construction"),
          os.path.join(REPO_ROOT, "temporal-analysis"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO_ROOT)  # feature_engine9's DATA_DIR is a relative path

import ensemble as ens
import ensemble1 as ens1
import feature_engine9 as fe9
import privilege_features as pf
from evaluate_baselines import RULES
from evaluate_ml_baselines import GUARDDUTY, best_f1_over_thresholds, session_probs

N_BOOTSTRAP = 10000
SEED = 42
A = "A: ensemble.py (fixed 0.5/0.5)"
B = "B: ensemble1.py (stacked)"
SHIPPED_THRESHOLD = {A: ens.SESSION_ALERT_THRESHOLD, B: ens1.SESSION_ALERT_THRESHOLD}


def load_scores(raw_csv_name: str, cache_dir: "str | None", lstm_ckpt: "str | None" = None) -> pd.DataFrame:
    """Per-event meta-features (incl. both branches' scores) for one real split, cached if possible."""
    stem = os.path.splitext(raw_csv_name)[0]
    # Cache LSTM scores under a hash of the checkpoint's bytes, so scores from one checkpoint can
    # never be silently reused for another (e.g. after the default checkpoint is replaced).
    with open(lstm_ckpt or ens.lstm_scorer.DEFAULT_CKPT, "rb") as f:
        lstm_tag = "__" + hashlib.sha1(f.read()).hexdigest()[:10]
    paths = fe9._derive_paths(os.path.join(fe9.DATA_DIR, raw_csv_name))
    if not os.path.exists(paths["struct_out"]):
        fe9.run_batch(os.path.join(fe9.DATA_DIR, raw_csv_name), freeze_vocab=True)
    structural_df = pd.read_csv(paths["struct_out"], dtype={"log_id": str})
    temporal_df = pd.read_csv(paths["temporal_out"], dtype={"log_id": str})

    gnn_cache = os.path.join(cache_dir, f"{stem}__gnn.parquet") if cache_dir else None
    lstm_cache = os.path.join(cache_dir, f"{stem}__lstm{lstm_tag}.parquet") if cache_dir else None

    if gnn_cache and os.path.exists(gnn_cache):
        gnn_df = pd.read_parquet(gnn_cache)
    else:
        print(f"[{raw_csv_name}] scoring GNN heuristic (slow)...", flush=True)
        gnn_df = ens.score_gnn_events(structural_df, source="csv", resolver=pf.ActionAccessLevelResolver())
        if gnn_cache:
            gnn_df.to_parquet(gnn_cache)

    if lstm_cache and os.path.exists(lstm_cache):
        lstm_df = pd.read_parquet(lstm_cache)
    else:
        print(f"[{raw_csv_name}] scoring LSTM...", flush=True)
        lstm_df = ens.score_lstm_events(temporal_df, fe9.EVENT_NAME_VOCAB_FILE, ckpt_path=lstm_ckpt)
        if lstm_cache:
            lstm_df.to_parquet(lstm_cache)

    merged = structural_df[["log_id", "label"]].copy()
    gnn_df = gnn_df.copy(); gnn_df["log_id"] = gnn_df["log_id"].astype(str)
    lstm_df = lstm_df.copy(); lstm_df["log_id"] = lstm_df["log_id"].astype(str)
    merged = merged.merge(gnn_df, on="log_id", how="left").merge(lstm_df, on="log_id", how="left")
    merged["gnn_event_score"] = merged["gnn_event_score"].fillna(0.0)
    merged["P_event"] = merged["P_event"].fillna(0.0)
    merged = merged.rename(columns={"P_event": "lstm_event_score"})
    return ens1._prepare_meta_features(merged)  # same feature prep ensemble1 uses at inference


def risk_scores(feats: pd.DataFrame, meta_model) -> dict:
    """Per-event 0-10 risk_score for each candidate, computed exactly as each file does it."""
    a = (0.5 * feats["gnn_event_score"] + 0.5 * feats["lstm_event_score"]).clip(0.0, 1.0) * 10
    b = meta_model.predict_proba(feats[ens1.META_FEATURE_COLS].to_numpy(dtype=float))[:, 1] * 10
    return {A: np.round(a.to_numpy(), 2), B: np.round(b, 2)}


def f1_rows(t, p):
    tp = ((t == 1) & (p == 1)).sum(1); fp = ((t == 0) & (p == 1)).sum(1); fn = ((t == 1) & (p == 0)).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def prf(t, p):
    tp = int(((t == 1) & (p == 1)).sum()); fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return pr, rc, f1


def paired_bootstrap(y_true, y_a, y_b):
    """F1(a) - F1(b), resampling sessions jointly (they're paired -- same sessions)."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y_true), (N_BOOTSTRAP, len(y_true)))
    deltas = f1_rows(y_true[idx], y_a[idx]) - f1_rows(y_true[idx], y_b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p2 = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return prf(y_true, y_a)[2] - prf(y_true, y_b)[2], lo, hi, p2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--lstm-ckpt", default=None,
                    help="Evaluate with this LSTM checkpoint instead of the default (see module docstring)")
    args = ap.parse_args()
    print(f"LSTM checkpoint: {args.lstm_ckpt or 'default (' + str(ens.lstm_scorer.DEFAULT_CKPT) + ')'}")

    meta_model = ens1.fit_meta_learner(lstm_ckpt=args.lstm_ckpt)
    print("Candidate B meta-learner coefficients (fit on synthetic):")
    for name, coef in zip(ens1.META_FEATURE_COLS, meta_model.coef_[0]):
        print(f"  {name:<26}{coef:+.3f}")

    dev_raw = pd.read_csv(os.path.join(HERE, "real_dataset_dev.csv"))
    test_raw = pd.read_csv(os.path.join(HERE, "real_dataset_test.csv"))
    dev_true = dev_raw.drop_duplicates("session_id").set_index("session_id")["session_label"]
    test_true = test_raw.drop_duplicates("session_id").set_index("session_id")["session_label"]

    dev_feats = load_scores("real_dataset_dev.csv", args.cache_dir, args.lstm_ckpt)
    test_feats = load_scores("real_dataset_test.csv", args.cache_dir, args.lstm_ckpt)
    dev_scores = risk_scores(dev_feats, meta_model)
    test_scores = risk_scores(test_feats, meta_model)

    y_test = test_true.to_numpy()
    preds = {}
    used_label = "proposed thr" if args.lstm_ckpt else "shipped thr"
    print(f"\n{'Candidate':<34}{'dev-tuned thr':>14}{used_label:>13}{'dev F1':>8}  |"
          f"{'test P':>8}{'test R':>8}{'test F1':>8}")
    for name in (A, B):
        dev_s = session_probs(dev_feats, dev_scores[name], "real_dataset_dev.csv", dev_raw, dev_true)
        f1, thr, _, _ = best_f1_over_thresholds(dev_s, dev_true.to_numpy())
        # The shipped constants belong to the default checkpoint; with another one, test at the
        # value that would ship for it: its own dev-tuned threshold, rounded like the constants.
        used = round(thr, 2) if args.lstm_ckpt else SHIPPED_THRESHOLD[name]
        test_s = session_probs(test_feats, test_scores[name], "real_dataset_test.csv", test_raw, test_true)
        preds[name] = (test_s >= used).astype(int)
        p, r, tf1 = prf(y_test, preds[name])
        print(f"{name:<34}{thr:>14.3f}{used:>13.2f}{f1:>8.3f}  |{p:>8.3f}{r:>8.3f}{tf1:>8.3f}")
        if not args.lstm_ckpt and abs(thr - used) > 0.1:
            print(f"  WARNING: shipped threshold {used} has drifted from the dev-tuned {thr:.3f} -- re-tune it.")

    rules = RULES[GUARDDUTY]
    events = test_raw.groupby("session_id")["event_name"].apply(set).reindex(test_true.index)
    preds[GUARDDUTY] = events.apply(lambda s: int(bool(s & rules))).to_numpy()
    p, r, f1 = prf(y_test, preds[GUARDDUTY])
    print(f"{GUARDDUTY:<34}{'--':>14}{'--':>13}{'--':>8}  |{p:>8.3f}{r:>8.3f}{f1:>8.3f}")

    print(f"\nPaired bootstrap on test F1 ({len(y_test)} sessions, {N_BOOTSTRAP} resamples):")
    for a, b in [(B, A), (A, GUARDDUTY), (B, GUARDDUTY)]:
        d, lo, hi, p2 = paired_bootstrap(y_test, preds[a], preds[b])
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant (CI includes 0)"
        print(f"  {a} - {b}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p2:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
