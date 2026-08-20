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

Usage:
    python evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt \
        --model sage --raw-csv datasets/privilege-escalation/real_dataset_test.csv \
        --threshold 0.5
    # or sweep thresholds on the dev set:
    python evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt \
        --model sage --raw-csv datasets/privilege-escalation/real_dataset_dev.csv --sweep
"""

import argparse
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from data_loader import PrivilegePropagationGraphLoader
from evaluate_on_real import build_model_from_args
from utils import evaluate

LOG_ID_RE = re.compile(r"^(.*):(\d+)$")


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
    )
    data, meta = loader.load()

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

    row_idx = []
    for lid in log_ids_flat:
        mobj = LOG_ID_RE.match(lid)
        if not mobj:
            raise ValueError(f"Unparseable log_id: {lid!r}")
        row_idx.append(int(mobj.group(2)))
    row_idx = np.array(row_idx)

    session_id_per_edge = raw_df["session_id"].to_numpy()[row_idx]

    edge_df = pd.DataFrame({"session_id": session_id_per_edge, "prob": probs})
    session_prob = edge_df.groupby("session_id")["prob"].max()

    sessions_true = raw_df.drop_duplicates("session_id").set_index("session_id")["session_label"]
    scored_sessions = sessions_true.index.intersection(session_prob.index)
    unscored = sessions_true.index.difference(session_prob.index)

    print(f"Sessions: {len(sessions_true)} total | {len(scored_sessions)} have >=1 in-schema edge "
          f"| {len(unscored)} have ZERO in-schema edges (predicted benign by default)")

    y_true = sessions_true.reindex(sessions_true.index).to_numpy()
    y_prob_full = session_prob.reindex(sessions_true.index).fillna(0.0).to_numpy()

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
        print("Compare directly against GuardDuty-style rule baseline: F1=0.732 [95% CI: 0.672, 0.790]")


if __name__ == "__main__":
    main()
