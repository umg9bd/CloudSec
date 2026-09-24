"""
Classical-ML baselines (Random Forest, XGBoost) trained on feature_engine9's
own temporal feature columns -- the standard comparison a reviewer expects
(the same style used by e.g. arxiv:2512.10280's RF/XGBoost/LSTM table), and a
more honest one than approximating a commercial product (see evaluate_baselines.py's
RULES -- "GuardDuty-style" was never validated against real GuardDuty output;
this project's data collection never enabled it, see stratus_collection/README.md).

Train-on-synthetic / evaluate-on-real, same paradigm as the GNN/LSTM models:
  - Fit on cloudtrail_temporal.csv (synthetic, event-level, label-balanced via
    class weighting rather than resampling).
  - Threshold swept on real_dataset_dev.csv ONLY (session-level max-pooled
    probability, same convention as evaluate_session_level.py/evaluate_baselines.py).
  - Applied ONCE, frozen, to real_dataset_test.csv. Reported with a paired
    bootstrap against the rule baseline on the same test sessions.

Usage:
    python evaluate_ml_baselines.py
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from evaluate_baselines import RULES

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ID_RE = re.compile(r"^(.*):(\d+)$")
GUARDDUTY = "Curated IAM rule baseline (11 rules)"

NON_FEATURE_COLS = {"log_id", "username", "timestamp", "label"}


def load_temporal(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(HERE, name))


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def session_probs(temporal_df: pd.DataFrame, probs: np.ndarray, raw_csv_name: str,
                   raw_df: pd.DataFrame, sessions_true: pd.Series) -> np.ndarray:
    row_idx = []
    for lid in temporal_df["log_id"].astype(str):
        m = LOG_ID_RE.match(lid)
        if not m or m.group(1) != raw_csv_name:
            raise SystemExit(f"log_id {lid!r} does not match expected source {raw_csv_name!r}")
        row_idx.append(int(m.group(2)))
    row_idx = np.array(row_idx, dtype=int)
    session_id = raw_df["session_id"].to_numpy()[row_idx]
    s = pd.Series(probs, index=session_id).groupby(level=0).max()
    return s.reindex(sessions_true.index).fillna(0.0).to_numpy()


def best_f1_over_thresholds(scores: np.ndarray, y_true: np.ndarray, max_thresholds: int = 300):
    cand = np.unique(scores)
    if len(cand) > max_thresholds:
        cand = np.unique(np.quantile(scores, np.linspace(0, 1, max_thresholds)))
    best = (-1.0, None, 0.0, 0.0)
    for thr in cand:
        pred = (scores >= thr).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best[0]:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            best = (f1, float(thr), p, r)
    return best


def paired_bootstrap_vs_rule(raw_df, sessions_true, y_true, y_model, n_bootstrap=10000, seed=42):
    rules = RULES[GUARDDUTY]
    events = raw_df.groupby("session_id")["event_name"].apply(set).reindex(sessions_true.index)
    y_rule = events.apply(lambda s: int(bool(s & rules))).to_numpy()

    def prf(t, p):
        tp = ((t == 1) & (p == 1)).sum(); fp = ((t == 0) & (p == 1)).sum(); fn = ((t == 1) & (p == 0)).sum()
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        return pr, rc, f1

    mp, mr, mf = prf(y_true, y_model)
    bp, br, bf = prf(y_true, y_rule)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, (n_bootstrap, n))
    yt, ym, yr = y_true[idx], y_model[idx], y_rule[idx]

    def f1_rows(t, p):
        tp = ((t == 1) & (p == 1)).sum(1); fp = ((t == 0) & (p == 1)).sum(1); fn = ((t == 1) & (p == 0)).sum(1)
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)

    deltas = f1_rows(yt, ym) - f1_rows(yt, yr)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p2 = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return (mp, mr, mf), (bp, br, bf), (mf - bf, lo, hi, p2)


def run_model(name, clf, train_df, cols, dev_temporal, dev_raw, dev_sessions_true,
              test_temporal, test_raw, test_sessions_true):
    X_train, y_train = train_df[cols].to_numpy(dtype=float), train_df["label"].to_numpy(int)
    clf.fit(X_train, y_train)

    dev_probs = clf.predict_proba(dev_temporal[cols].to_numpy(dtype=float))[:, 1]
    dev_sscore = session_probs(dev_temporal, dev_probs, "real_dataset_dev.csv", dev_raw, dev_sessions_true)
    dev_y_true = dev_sessions_true.to_numpy()
    f1, thr, p, r = best_f1_over_thresholds(dev_sscore, dev_y_true)
    print(f"\n[{name}] DEV sweep: best thr={thr:.4f}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    test_probs = clf.predict_proba(test_temporal[cols].to_numpy(dtype=float))[:, 1]
    test_sscore = session_probs(test_temporal, test_probs, "real_dataset_test.csv", test_raw, test_sessions_true)
    test_y_true = test_sessions_true.to_numpy()
    test_y_pred = (test_sscore >= thr).astype(int)

    (mp, mr, mf), (bp, br, bf), (delta, lo, hi, p2) = paired_bootstrap_vs_rule(
        test_raw, test_sessions_true, test_y_true, test_y_pred
    )
    print(f"[{name}] TEST @ frozen thr={thr:.4f}: P={mp:.3f} R={mr:.3f} F1={mf:.3f}")
    print(f"[{name}] paired bootstrap vs {GUARDDUTY} (F1={bf:.3f}): "
          f"delta={delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p2:.4f}")
    return {"model": name, "precision": mp, "recall": mr, "f1": mf,
            "delta_vs_rule": delta, "ci_lo": lo, "ci_hi": hi, "p_value": p2}


def main():
    train_df = load_temporal("cloudtrail_temporal.csv")
    dev_temporal = load_temporal("real_dataset_dev_temporal.csv")
    test_temporal = load_temporal("real_dataset_test_temporal.csv")
    cols = feature_cols(train_df)
    print(f"Feature columns ({len(cols)}): {cols}")

    dev_raw = pd.read_csv(os.path.join(HERE, "real_dataset_dev.csv"))
    test_raw = pd.read_csv(os.path.join(HERE, "real_dataset_test.csv"))
    dev_sessions_true = dev_raw.drop_duplicates("session_id").set_index("session_id")["session_label"]
    test_sessions_true = test_raw.drop_duplicates("session_id").set_index("session_id")["session_label"]
    print(f"DEV sessions: {len(dev_sessions_true)} ({int(dev_sessions_true.sum())} attack)")
    print(f"TEST sessions: {len(test_sessions_true)} ({int(test_sessions_true.sum())} attack)")

    neg, pos = (train_df["label"] == 0).sum(), (train_df["label"] == 1).sum()
    print(f"\nTrain label balance: {neg} benign / {pos} attack ({pos/(neg+pos):.1%} positive)")

    results = []
    results.append(run_model(
        "Random Forest",
        RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced",
                                random_state=42, n_jobs=-1),
        train_df, cols, dev_temporal, dev_raw, dev_sessions_true,
        test_temporal, test_raw, test_sessions_true,
    ))
    results.append(run_model(
        "XGBoost",
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=neg / pos, eval_metric="logloss",
                       random_state=42, n_jobs=-1),
        train_df, cols, dev_temporal, dev_raw, dev_sessions_true,
        test_temporal, test_raw, test_sessions_true,
    ))

    print(f"\n{'=' * 60}")
    print("SUMMARY -- classical ML baselines, real held-out test set")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for r in results:
        print(f"{r['model']:<20} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}")


if __name__ == "__main__":
    main()
