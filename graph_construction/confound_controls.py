"""
Confound controls for the session-level result: is the win real, or an artifact
of session length (max-pooling over more edges trivially yields a higher score)?

Two controls, both on whatever graph is currently in Neo4j (the real test graph),
using the wrapped checkpoint's frozen scalers (never re-fit on eval data):

  1. Size-preserving permutation test. Session score = max edge probability in
     the session. Shuffle the per-edge probabilities across the WHOLE graph while
     keeping every session's edge COUNT identical, then recompute session-max AUC.
     This destroys the real edge->session assignment but leaves the "max over
     more edges" length effect intact. If the observed AUC beats the permuted
     distribution, the result is not a length artifact.

  2. Within-length-strata AUC. Bin sessions into quartiles by edge count and
     compute session-max AUC inside each bin, where length is ~constant. A model
     that only re-derives length collapses to chance within a stratum.

Usage (same args as evaluate_session_level.py):
    python graph_construction/confound_controls.py \
        --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage \
        --raw-csv datasets/privilege-escalation/real_dataset_test.csv --n-perm 200
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from data_loader import PrivilegePropagationGraphLoader, scored_edge_types
from evaluate_on_real import build_model_from_args
from evaluate_session_level import parse_row_indices, session_max_scores
from utils import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["sage", "gat"], default="sage")
    p.add_argument("--raw-csv", required=True)
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="test1234")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args, fit = ckpt["model_args"], ckpt["fit_artifacts"]
    loader = PrivilegePropagationGraphLoader(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_pass,
        fit_artifacts=fit, model_node_types=set(model_args["node_feat_dims"]),
        add_reverse_edges=any(str(t[1]).startswith("REV_") for t in model_args["edge_types"]),
    )
    data, meta = loader.load()
    trained = set(tuple(t) for t in model_args["edge_types"])
    for t in set(scored_edge_types(data)) - trained:
        del data[t]

    model = build_model_from_args(args.model, model_args)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    masks = {t: torch.ones(data[t].y.shape[0], dtype=torch.bool) for t in scored_edge_types(data)}
    probs = np.array(evaluate(model, data, masks, return_probs=True)["probs"])

    log_ids = []
    for t in scored_edge_types(data):
        log_ids.extend(data[t].log_id)
    raw_df = pd.read_csv(args.raw_csv, low_memory=False)
    raw_basename = os.path.basename(args.raw_csv)
    row_idx = parse_row_indices(log_ids, raw_basename, len(raw_df))

    sessions_true = raw_df.drop_duplicates("session_id").set_index("session_id")["session_label"]
    _, session_prob = session_max_scores(row_idx, probs, raw_df, sessions_true.index)
    scored = sessions_true.index.intersection(session_prob.index)
    y = sessions_true.loc[scored].to_numpy()
    obs = session_prob.loc[scored].to_numpy()
    observed_auc = roc_auc_score(y, obs)

    # session_id per edge, and edge counts per session (the sizes we preserve)
    edge_sessions = raw_df["session_id"].to_numpy()[row_idx]
    print(f"Sessions scored: {len(scored)}  attack: {int(y.sum())}  "
          f"edges: {len(probs)}")
    print(f"\nObserved session-max AUC: {observed_auc:.4f}")

    # ---- 1. size-preserving permutation ----
    rng = np.random.default_rng(args.seed)
    perm_aucs = []
    order = pd.Series(edge_sessions)
    for _ in range(args.n_perm):
        shuffled = rng.permutation(probs)  # global shuffle; session sizes unchanged
        sp = pd.DataFrame({"session_id": edge_sessions, "prob": shuffled}) \
            .groupby("session_id")["prob"].max().reindex(scored).fillna(0.0).to_numpy()
        # a permuted draw can be degenerate; guard the AUC
        if len(np.unique(y)) == 2:
            perm_aucs.append(roc_auc_score(y, sp))
    perm_aucs = np.array(perm_aucs)
    n_ge = int((perm_aucs >= observed_auc).sum())
    print(f"\n1. Size-preserving permutation ({len(perm_aucs)} draws)")
    print(f"   permuted AUC: mean {perm_aucs.mean():.4f}  max {perm_aucs.max():.4f}")
    print(f"   observed {observed_auc:.4f} beats {len(perm_aucs)-n_ge}/{len(perm_aucs)} "
          f"(empirical p = {(n_ge+1)/(len(perm_aucs)+1):.4f})")
    print("   -> not a length artifact" if n_ge == 0 else "   -> length may explain some of it")

    # ---- 2. within-length-strata AUC ----
    counts = pd.Series(edge_sessions).value_counts().reindex(scored).fillna(0).to_numpy()
    # quartile bins by edge count
    qs = np.quantile(counts, [0.25, 0.5, 0.75])
    bins = np.digitize(counts, qs)
    print("\n2. Within-length-strata session-max AUC (quartiles by edge count)")
    for b in range(4):
        m = bins == b
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            print(f"   stratum {b}: n={int(m.sum())}  (too few / single-class -- skipped)")
            continue
        print(f"   stratum {b}: n={int(m.sum())}  edges~[{int(counts[m].min())},"
              f"{int(counts[m].max())}]  AUC={roc_auc_score(y[m], obs[m]):.4f}")
    print("\n(If AUC stays high WITHIN strata, the win is not session length.)")


if __name__ == "__main__":
    main()
