"""
Evaluates ensemble1.py's stacked meta-learner against the shipped ensemble.py
(fixed 0.5/0.5 weighted sum) and the curated rule baseline, on real data,
under the same protocol as every other result in this project:

  - The meta-learner is fit on synthetic_cloudtrail.csv only (ensemble1.fit_meta_learner).
  - Each method's session-level alert threshold is swept on real_dataset_dev.csv ONLY.
  - Each frozen threshold is applied ONCE to real_dataset_test.csv.
  - Reported with a PAIRED bootstrap against both the rule baseline and ensemble.py on the
    same 238 test sessions -- the ensemble1-vs-ensemble comparison is the one that answers
    "is the stacked method actually better", not just "does it also beat the rules".

GNN/LSTM per-event scoring is the slow step (~3 min dev, ~10 min test on CPU). Pass
--cache-dir to reuse/write <stem>__gnn.parquet / <stem>__lstm.parquet there; without it,
scores are computed fresh every run.

Usage (from the repo root, using the project venv):
    .venv/Scripts/python.exe datasets/privilege-escalation/evaluate_ensemble1.py
    .venv/Scripts/python.exe datasets/privilege-escalation/evaluate_ensemble1.py --cache-dir <dir>
"""
import argparse
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


def load_scores(raw_csv_name: str, cache_dir: "str | None"):
    """Per-event structural_df + gnn_df + lstm_df for one real split, from cache if present."""
    stem = os.path.splitext(raw_csv_name)[0]
    paths = fe9._derive_paths(os.path.join(fe9.DATA_DIR, raw_csv_name))
    if not os.path.exists(paths["struct_out"]):
        fe9.run_batch(os.path.join(fe9.DATA_DIR, raw_csv_name), freeze_vocab=True)
    structural_df = pd.read_csv(paths["struct_out"], dtype={"log_id": str})
    temporal_df = pd.read_csv(paths["temporal_out"], dtype={"log_id": str})

    gnn_cache = os.path.join(cache_dir, f"{stem}__gnn.parquet") if cache_dir else None
    lstm_cache = os.path.join(cache_dir, f"{stem}__lstm.parquet") if cache_dir else None

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
        lstm_df = ens.score_lstm_events(temporal_df, fe9.EVENT_NAME_VOCAB_FILE)
        if lstm_cache:
            lstm_df.to_parquet(lstm_cache)

    merged = structural_df[["log_id", "label"]].copy()
    gnn_df = gnn_df.copy(); gnn_df["log_id"] = gnn_df["log_id"].astype(str)
    lstm_df = lstm_df.copy(); lstm_df["log_id"] = lstm_df["log_id"].astype(str)
    merged = merged.merge(gnn_df, on="log_id", how="left").merge(lstm_df, on="log_id", how="left")
    merged["gnn_event_score"] = merged["gnn_event_score"].fillna(0.0)
    merged["P_event"] = merged["P_event"].fillna(0.0)
    merged = merged.rename(columns={"P_event": "lstm_event_score"})
    # Same feature preparation ensemble1 applies at inference time -- one shared code path.
    return ens1._prepare_meta_features(merged)


def method_scores(feats: pd.DataFrame, meta_model) -> dict:
    """Per-event 0-1 score under each method being compared."""
    X = feats[ens1.META_FEATURE_COLS].to_numpy(dtype=float)
    return {
        "ensemble.py (fixed 0.5/0.5)": 0.5 * feats["gnn_event_score"].to_numpy() + 0.5 * feats["lstm_event_score"].to_numpy(),
        "ensemble1.py (stacked)": meta_model.predict_proba(X)[:, 1],
    }


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
    args = ap.parse_args()

    meta_model = ens1.fit_meta_learner()
    print("Meta-learner coefficients (fit on synthetic):")
    for name, coef in zip(ens1.META_FEATURE_COLS, meta_model.coef_[0]):
        print(f"  {name:<26}{coef:+.3f}")

    dev_raw = pd.read_csv(os.path.join(HERE, "real_dataset_dev.csv"))
    test_raw = pd.read_csv(os.path.join(HERE, "real_dataset_test.csv"))
    dev_true = dev_raw.drop_duplicates("session_id").set_index("session_id")["session_label"]
    test_true = test_raw.drop_duplicates("session_id").set_index("session_id")["session_label"]

    dev_feats = load_scores("real_dataset_dev.csv", args.cache_dir)
    test_feats = load_scores("real_dataset_test.csv", args.cache_dir)
    dev_by_method = method_scores(dev_feats, meta_model)
    test_by_method = method_scores(test_feats, meta_model)

    y_test = test_true.to_numpy()
    preds = {}
    print(f"\n{'Method':<30}{'dev thr':>9}{'dev F1':>8}  |  {'test P':>7}{'test R':>8}{'test F1':>8}")
    for name in dev_by_method:
        dev_s = session_probs(dev_feats, dev_by_method[name], "real_dataset_dev.csv", dev_raw, dev_true)
        f1, thr, _, _ = best_f1_over_thresholds(dev_s, dev_true.to_numpy())
        test_s = session_probs(test_feats, test_by_method[name], "real_dataset_test.csv", test_raw, test_true)
        preds[name] = (test_s >= thr).astype(int)
        p, r, tf1 = prf(y_test, preds[name])
        print(f"{name:<30}{thr:>9.4f}{f1:>8.3f}  |  {p:>7.3f}{r:>8.3f}{tf1:>8.3f}")

    rules = RULES[GUARDDUTY]
    events = test_raw.groupby("session_id")["event_name"].apply(set).reindex(test_true.index)
    preds[GUARDDUTY] = events.apply(lambda s: int(bool(s & rules))).to_numpy()
    p, r, f1 = prf(y_test, preds[GUARDDUTY])
    print(f"{GUARDDUTY:<30}{'--':>9}{'--':>8}  |  {p:>7.3f}{r:>8.3f}{f1:>8.3f}")

    print(f"\nPaired bootstrap on test F1 ({len(y_test)} sessions, {N_BOOTSTRAP} resamples):")
    for a, b in [("ensemble1.py (stacked)", "ensemble.py (fixed 0.5/0.5)"),
                 ("ensemble1.py (stacked)", GUARDDUTY),
                 ("ensemble.py (fixed 0.5/0.5)", GUARDDUTY)]:
        d, lo, hi, p2 = paired_bootstrap(y_test, preds[a], preds[b])
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant (CI includes 0)"
        print(f"  {a} - {b}: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p2:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
