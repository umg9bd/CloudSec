"""
ensemble1.py -- alternative ensembling method: a STACKED meta-learner instead
of ensemble.py's fixed 0.5/0.5 weighted sum.

WHY A SEPARATE FILE
Nothing in ensemble.py's own weight/strategy sweep (linear blend 0.0-1.0, max,
geometric mean, impact-weighted) beat the shipped 0.5/0.5 default by more than
dev-set noise -- but every variant tried was still a fixed FORMULA applied to
the same two pre-combined scores (gnn_event_score, lstm_event_score). This
tries something structurally different: a small logistic-regression stacking
model that learns its own combination weights, using the GNN heuristic's
INDIVIDUAL raw sub-signals (not just its already-blended score) alongside the
LSTM's P_event, so it can pick up interactions a fixed linear blend of two
pre-summed scores cannot represent (e.g. "trust credential-access flags more
when the LSTM also agrees" is not expressible as w*gnn + (1-w)*lstm).

This file does not replace ensemble.py -- it is a second, independently
evaluated candidate. score_gnn_events/score_lstm_events/build_ppg are reused
directly from ensemble.py rather than reimplemented, so there is exactly one
place either scorer's logic lives.

PROTOCOL (same discipline as every other result in this project)
The meta-learner is fit on synthetic_cloudtrail.csv only (train-on-synthetic,
same as the GNN heuristic's hand-tuned weights and the LSTM checkpoint were
never fit on real data). Any reported real-data number must still be
threshold-tuned on real_dataset_dev.csv only and evaluated once on
real_dataset_test.csv -- this file does not do that sweep itself; see
datasets/privilege-escalation/evaluate_ensemble1.py (or run the equivalent
manually) before citing a real-data F1 for this method.

Usage:
    python ensemble1.py --input datasets/privilege-escalation/synthetic_cloudtrail.csv
    python ensemble1.py --input datasets/privilege-escalation/real_dataset_test.csv --out risk_scores1.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT,
          os.path.join(REPO_ROOT, "graph_construction"),
          os.path.join(REPO_ROOT, "temporal-analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import feature_engine9 as fe9
import privilege_features as pf
import ensemble as ens  # reuse score_gnn_events / score_lstm_events -- no duplicated scoring logic

DATA_DIR = os.path.join(REPO_ROOT, "datasets", "privilege-escalation")
META_MODEL_FILE = os.path.join(DATA_DIR, ".ensemble1_meta_model.pkl")

# gnn_event_score/lstm_event_score are the two branches' already-blended scores (kept as features
# so the meta-learner can still fall back to "trust the existing formula" if that's genuinely
# optimal); the rest are the GNN heuristic's INDIVIDUAL raw ingredients, exposed as separate
# columns by score_gnn_events -- see that function in ensemble.py for what each one means.
META_FEATURE_COLS = [
    "gnn_event_score", "lstm_event_score", "is_priv_esc_technique", "is_credential_access",
    "access_level_rank", "target_sensitivity_tier", "hop_count",
]


def _prepare_meta_features(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["is_priv_esc_technique"] = out["is_priv_esc_technique"].astype(int)
    out["is_credential_access"] = out["is_credential_access"].astype(int)
    # ACCESS_LEVEL_RANK has no entry for "no resolvable access level" (None) -- give it a value
    # (-1) below the real 0-4 rank range rather than dropping the row/imputing a real rank's value.
    out["access_level_rank"] = out["access_level"].map(pf.ACCESS_LEVEL_RANK).fillna(-1)
    out["target_sensitivity_tier"] = out["target_sensitivity_tier"].fillna(0)
    out["hop_count"] = out["hop_count"].fillna(0)
    return out


# Runs the GNN heuristic + LSTM scorers (reused from ensemble.py) and returns one merged
# per-event dataframe with label + every meta-feature, UNBLENDED -- the meta-learner's inputs.
def score_events_raw(structural_df: pd.DataFrame, temporal_df: pd.DataFrame,
                      gnn_source: str = "csv", resolver=None) -> pd.DataFrame:
    resolver = resolver or pf.ActionAccessLevelResolver()
    gnn_df = ens.score_gnn_events(structural_df, source=gnn_source, resolver=resolver)
    lstm_df = ens.score_lstm_events(temporal_df, fe9.EVENT_NAME_VOCAB_FILE)

    merged = structural_df[["log_id", "label"]].copy()
    merged["log_id"] = merged["log_id"].astype(str)
    gnn_df = gnn_df.copy(); gnn_df["log_id"] = gnn_df["log_id"].astype(str)
    lstm_df = lstm_df.copy(); lstm_df["log_id"] = lstm_df["log_id"].astype(str)
    merged = merged.merge(gnn_df, on="log_id", how="left").merge(lstm_df, on="log_id", how="left")
    merged["gnn_event_score"] = merged["gnn_event_score"].fillna(0.0)
    merged["P_event"] = merged["P_event"].fillna(0.0)
    merged = merged.rename(columns={"P_event": "lstm_event_score"})
    return _prepare_meta_features(merged)


# Fits the stacking meta-learner on synthetic training data (train-on-synthetic, same paradigm
# the GNN heuristic's hand-tuned weights and the LSTM checkpoint were built under). Cached to
# disk so repeated CLI invocations (e.g. --watch mode) don't refit on every run.
def fit_meta_learner(force_refit: bool = False) -> LogisticRegression:
    if not force_refit and os.path.exists(META_MODEL_FILE):
        with open(META_MODEL_FILE, "rb") as f:
            return pickle.load(f)

    fe9.run_batch(fe9.DEFAULT_INPUT)  # no-op if synthetic_cloudtrail.csv is already processed
    structural_df = pd.read_csv(fe9.STRUCT_OUT, dtype={"log_id": str})
    temporal_df = pd.read_csv(fe9.TEMPORAL_OUT, dtype={"log_id": str})

    print(f"[ensemble1] fitting meta-learner on {len(structural_df)} synthetic events...", flush=True)
    feats = score_events_raw(structural_df, temporal_df)
    X = feats[META_FEATURE_COLS].to_numpy(dtype=float)
    y = feats["label"].astype(int).to_numpy()

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X, y)
    print(f"[ensemble1] meta-learner coefficients: "
          f"{dict(zip(META_FEATURE_COLS, model.coef_[0].round(3)))}", flush=True)

    with open(META_MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


# Merges GNN and LSTM per-event scores via the fitted meta-learner into a single 0-10 risk_score.
def combine_events_stacked(structural_df: pd.DataFrame, temporal_df: pd.DataFrame,
                            model: LogisticRegression, gnn_source: str = "csv") -> pd.DataFrame:
    feats = score_events_raw(structural_df, temporal_df, gnn_source=gnn_source)
    X = feats[META_FEATURE_COLS].to_numpy(dtype=float)
    feats["risk_score"] = (model.predict_proba(X)[:, 1] * 10).round(2)

    merged = structural_df[["log_id", "source_node", "target_node", "edge_type"]].merge(
        temporal_df[["log_id", "username", "timestamp"]], on="log_id", how="left"
    )
    merged["log_id"] = merged["log_id"].astype(str)
    merged = merged.merge(feats, on="log_id", how="left")
    merged = merged.rename(columns={"edge_type": "event_name", "label": "ground_truth_label"})

    cols = ["log_id", "timestamp", "username", "source_node", "event_name", "target_node",
            "risk_score", "gnn_event_score", "lstm_event_score",
            "is_priv_esc_technique", "is_credential_access", "access_level_rank",
            "target_sensitivity_tier", "hop_count", "ground_truth_label"]
    return merged[cols].reset_index(drop=True)


# One-shot pipeline: feature engineering, both scorers, meta-learner combination.
def run(input_path: str, out_path: "str | None" = None, freeze_vocab: bool = False,
        gnn_source: str = "csv", force_refit: bool = False) -> pd.DataFrame:
    model = fit_meta_learner(force_refit=force_refit)

    fe9.run_batch(input_path, freeze_vocab=freeze_vocab)
    paths = fe9._derive_paths(input_path)
    structural_df = pd.read_csv(paths["struct_out"], dtype={"log_id": str})
    temporal_df = pd.read_csv(paths["temporal_out"], dtype={"log_id": str})

    print(f"[ensemble1] scoring {len(structural_df)} events (stacked meta-learner)...", flush=True)
    result = combine_events_stacked(structural_df, temporal_df, model, gnn_source=gnn_source)
    if out_path:
        result.to_csv(out_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Alternative ensembling method: stacked logistic-regression meta-learner "
                     "over the GNN heuristic's raw sub-signals + LSTM P_event, instead of "
                     "ensemble.py's fixed 0.5/0.5 weighted sum. See module docstring.")
    parser.add_argument("--input", default=fe9.DEFAULT_INPUT, help="Raw CloudTrail input")
    parser.add_argument("--out", default="risk_scores1.csv", help="Output CSV of per-event risk scores")
    parser.add_argument("--freeze-vocab", action="store_true")
    parser.add_argument("--source", choices=["csv", "neo4j"], default="csv")
    parser.add_argument("--refit-meta", action="store_true",
                         help="Refit the meta-learner on synthetic data even if a cached one exists")
    parser.add_argument("--show-table", action="store_true")
    args = parser.parse_args()

    result = run(args.input, args.out, args.freeze_vocab, args.source, force_refit=args.refit_meta)
    if args.show_table:
        with pd.option_context("display.max_rows", 50, "display.width", 200):
            print(result.to_string(index=False))
    print(f"\n{len(result)} events scored -> {args.out}")


if __name__ == "__main__":
    main()
