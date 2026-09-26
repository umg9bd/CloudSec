"""
ensemble1.py -- ENSEMBLE CANDIDATE B: stacked meta-learner.

This project has two candidate ensembles, kept as peers until a final choice
is made (see datasets/privilege-escalation/compare_ensembles.py, which
compares both on the same sessions):

  A. ensemble.py  -- fixed weighted sum: 0.5 * gnn_event_score + 0.5 * lstm_event_score.
  B. ensemble1.py -- this file: a logistic-regression stacking model that LEARNS the
     combination, using the GNN heuristic's INDIVIDUAL sub-signals (priv-esc and
     credential-access flags, access-level rank, target sensitivity, hop count) alongside
     both branches' blended scores. It can represent interactions a fixed blend of two
     pre-summed scores cannot (e.g. weighting credential-access flags differently
     depending on what the LSTM says).

Both candidates share the same per-event scorers -- score_gnn_events and
score_lstm_events are imported from ensemble.py, not reimplemented -- and the
same CLI, output columns, and watch loop, so they are drop-in interchangeable
and differ ONLY in how the two branches are combined.

PROTOCOL: the meta-learner is fit on synthetic_cloudtrail.csv only (train-on-
synthetic, like both branches themselves); SESSION_ALERT_THRESHOLD below was
tuned on real_dataset_dev.csv only and validated once on real_dataset_test.csv.

Usage (same as ensemble.py):
    python ensemble1.py --input datasets/privilege-escalation/real_dataset_test.csv --out risk_scores1.csv
    python ensemble1.py --watch incoming/ --simulate
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT,
          os.path.join(REPO_ROOT, "graph_construction"),
          os.path.join(REPO_ROOT, "temporal-analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import feature_engine9 as fe9
import privilege_features as pf
import ensemble as ens  # shared scorers + watch loop -- the only difference is the combiner

DATA_DIR = os.path.join(REPO_ROOT, "datasets", "privilege-escalation")
META_MODEL_FILE = os.path.join(DATA_DIR, ".ensemble1_meta_model.pkl")

# gnn_event_score/lstm_event_score are the two branches' already-blended scores (kept so the
# meta-learner can fall back to "trust the existing formula" if that's genuinely optimal); the
# rest are the GNN heuristic's individual raw ingredients, exposed as separate columns by
# score_gnn_events -- see that function in ensemble.py for what each one means.
META_FEATURE_COLS = [
    "gnn_event_score", "lstm_event_score", "is_priv_esc_technique", "is_credential_access",
    "access_level_rank", "target_sensitivity_tier", "hop_count",
]

# Session-level alerting threshold on the 0-10 risk_score (flag a session if ANY event reaches
# it). The meta-learner outputs a probability, so this sits on a different scale from
# ensemble.py's 5.5: it is the dev-swept session threshold 0.9278 x 10. Validated once on the
# 238 real test sessions: P=0.838 R=0.980 F1=0.903 (ensemble.py: 0.845/0.980/0.907; paired
# difference not significant, 95% CI [-0.019, +0.009]). Tied to the cached meta-model --
# re-tune it on dev if the meta-learner is ever refit (--refit-meta).
SESSION_ALERT_THRESHOLD = 9.28


def _prepare_meta_features(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["is_priv_esc_technique"] = out["is_priv_esc_technique"].astype(int)
    out["is_credential_access"] = out["is_credential_access"].astype(int)
    # ACCESS_LEVEL_RANK has no entry for "no resolvable access level" (None) -- give it a value
    # (-1) below the real 0-4 rank range rather than imputing a real rank's value.
    out["access_level_rank"] = out["access_level"].map(pf.ACCESS_LEVEL_RANK).fillna(-1)
    out["target_sensitivity_tier"] = out["target_sensitivity_tier"].fillna(0)
    out["hop_count"] = out["hop_count"].fillna(0)
    return out


# Runs the shared GNN heuristic + LSTM scorers and returns one merged per-event dataframe with
# label + every meta-feature, UNBLENDED -- the meta-learner's inputs.
def score_events_raw(structural_df: pd.DataFrame, temporal_df: pd.DataFrame,
                      gnn_source: str = "csv", resolver=None, lstm_ckpt=None) -> pd.DataFrame:
    resolver = resolver or pf.ActionAccessLevelResolver()
    gnn_df = ens.score_gnn_events(structural_df, source=gnn_source, resolver=resolver)
    lstm_df = ens.score_lstm_events(temporal_df, fe9.EVENT_NAME_VOCAB_FILE, ckpt_path=lstm_ckpt)

    merged = structural_df[["log_id", "label"]].copy()
    merged["log_id"] = merged["log_id"].astype(str)
    gnn_df = gnn_df.copy(); gnn_df["log_id"] = gnn_df["log_id"].astype(str)
    lstm_df = lstm_df.copy(); lstm_df["log_id"] = lstm_df["log_id"].astype(str)
    merged = merged.merge(gnn_df, on="log_id", how="left").merge(lstm_df, on="log_id", how="left")
    merged["gnn_event_score"] = merged["gnn_event_score"].fillna(0.0)
    merged["P_event"] = merged["P_event"].fillna(0.0)
    merged = merged.rename(columns={"P_event": "lstm_event_score"})
    return _prepare_meta_features(merged)


# Cached meta-model for a given LSTM checkpoint (None = the scorer's default checkpoint).
def _meta_model_file(lstm_ckpt) -> str:
    if lstm_ckpt is None:
        return META_MODEL_FILE
    return os.path.join(DATA_DIR, f".ensemble1_meta_model__{Path(lstm_ckpt).parent.name}.pkl")


# Fits the stacking meta-learner on synthetic training data, cached to disk so repeated runs
# (and --watch mode) don't refit every time.
def fit_meta_learner(force_refit: bool = False, lstm_ckpt=None) -> LogisticRegression:
    model_file = _meta_model_file(lstm_ckpt)
    if not force_refit and os.path.exists(model_file):
        with open(model_file, "rb") as f:
            return pickle.load(f)

    fe9.run_batch(fe9.DEFAULT_INPUT)  # no-op if synthetic_cloudtrail.csv is already processed
    structural_df = pd.read_csv(fe9.STRUCT_OUT, dtype={"log_id": str})
    temporal_df = pd.read_csv(fe9.TEMPORAL_OUT, dtype={"log_id": str})

    feats = score_events_raw(structural_df, temporal_df, lstm_ckpt=lstm_ckpt)
    feats = feats.merge(temporal_df[["log_id", "username"]], on="log_id", how="left")
    # Stacking needs base-model scores on data the base model never trained on. Checkpoints from
    # train_lstm_transformer.py record their user split; use only the LSTM's held-out test users
    # when that record exists (synthetic users appear there with an "fe:" prefix).
    ckpt = torch.load(lstm_ckpt or ens.lstm_scorer.DEFAULT_CKPT, map_location="cpu", weights_only=False)
    held_out = {u[3:] for u in (ckpt.get("split_users") or {}).get("test", []) if u.startswith("fe:")}
    if held_out:
        feats = feats[feats["username"].isin(held_out)]
        print(f"[ensemble1] fitting meta-learner on {len(feats)} synthetic events from the LSTM's "
              f"held-out users", flush=True)
    else:
        print(f"[ensemble1] fitting meta-learner on {len(feats)} synthetic events (checkpoint records no "
              f"split, so some LSTM scores are in-sample)", flush=True)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(feats[META_FEATURE_COLS].to_numpy(dtype=float), feats["label"].astype(int).to_numpy())
    print(f"[ensemble1] meta-learner coefficients: "
          f"{dict(zip(META_FEATURE_COLS, model.coef_[0].round(3).tolist()))}", flush=True)
    if force_refit:
        print(f"[ensemble1] WARNING: SESSION_ALERT_THRESHOLD={SESSION_ALERT_THRESHOLD} was tuned for "
              f"the previous meta-model -- re-tune it on dev before relying on it.", flush=True)

    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    return model


# Combines per-event GNN and LSTM scores via the fitted meta-learner into a 0-10 risk_score
# table with exactly the same columns as ensemble.combine_events.
def combine_events_stacked(structural_df: pd.DataFrame, temporal_df: pd.DataFrame,
                            model: LogisticRegression, gnn_source: str = "csv",
                            lstm_ckpt=None) -> pd.DataFrame:
    feats = score_events_raw(structural_df, temporal_df, gnn_source=gnn_source, lstm_ckpt=lstm_ckpt)
    feats["risk_score"] = (model.predict_proba(feats[META_FEATURE_COLS].to_numpy(dtype=float))[:, 1] * 10).round(2)

    merged = structural_df[["log_id", "source_node", "target_node", "edge_type"]].merge(
        temporal_df[["log_id", "username", "timestamp"]], on="log_id", how="left"
    )
    merged["log_id"] = merged["log_id"].astype(str)
    merged = merged.merge(feats, on="log_id", how="left")
    merged = merged.rename(columns={"edge_type": "event_name", "label": "ground_truth_label"})

    cols = ["log_id", "timestamp", "username", "source_node", "event_name", "target_node",
            "risk_score", "gnn_event_score", "lstm_event_score",
            "is_priv_esc_technique", "is_credential_access", "access_level",
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


# Watch mode: ensemble.py's watch loop, with the stacked combiner plugged in as its scorer.
def watch(directory: str, out_path: str = "risk_scores1.csv", freeze_vocab: bool = False,
          gnn_source: str = "csv", force_refit: bool = False) -> None:
    model = fit_meta_learner(force_refit=force_refit)
    ens.watch(directory, out_path=out_path, freeze_vocab=freeze_vocab, gnn_source=gnn_source,
              score_fn=lambda s, t: combine_events_stacked(s, t, model, gnn_source=gnn_source))


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble candidate B: stacked logistic-regression meta-learner over the GNN "
                     "heuristic's sub-signals + LSTM P_event (candidate A is ensemble.py's fixed "
                     "0.5/0.5 sum). One 0-10 risk score per EVENT, same output as ensemble.py.")
    parser.add_argument("--input", default=fe9.DEFAULT_INPUT,
                         help="Raw CloudTrail input (any file feature_engine9.py accepts). With "
                              "--watch --simulate: chunked into the watched directory.")
    parser.add_argument("--watch", metavar="DIR",
                         help="Watch DIR and re-run the full pipeline each time a new log file lands")
    parser.add_argument("--simulate", action="store_true",
                         help="With --watch, also chunk --input into small CSVs dropped into DIR")
    parser.add_argument("--out", default="risk_scores1.csv", help="Output CSV of per-event risk scores")
    parser.add_argument("--freeze-vocab", action="store_true",
                         help="Do not grow the shared event_name vocab on this run")
    parser.add_argument("--source", choices=["csv", "neo4j"], default="csv",
                         help="Where the GNN side builds its graph from (see ensemble.py --source)")
    parser.add_argument("--refit-meta", action="store_true",
                         help="Refit the meta-learner on synthetic data even if a cached one exists "
                              "(SESSION_ALERT_THRESHOLD then needs re-tuning on dev)")
    parser.add_argument("--show-table", action="store_true",
                         help="Also print the full per-event risk table. Ignored with --watch.")
    args = parser.parse_args()

    if args.watch:
        if args.simulate:
            fe9.simulate_incoming_files(args.watch, args.input)
        watch(args.watch, args.out, args.freeze_vocab, args.source, force_refit=args.refit_meta)
        return

    result = run(args.input, args.out, args.freeze_vocab, args.source, force_refit=args.refit_meta)
    if args.show_table:
        with pd.option_context("display.max_rows", 50, "display.width", 200):
            print(result.to_string(index=False))
    n_flagged = int((result["risk_score"] >= SESSION_ALERT_THRESHOLD).sum())
    print(f"\n{len(result)} events scored -> {args.out}")
    print(f"{n_flagged} events >= SESSION_ALERT_THRESHOLD ({SESSION_ALERT_THRESHOLD}) -- "
          f"flag the session containing any of these (see SESSION_ALERT_THRESHOLD comment)")


if __name__ == "__main__":
    main()
