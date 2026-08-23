"""
ensemble.py — single CloudTrail input in, single 0-10 risk score per
principal out.

Pipeline:

    raw CloudTrail input
            |
            v
    feature_engine9.FeatureEngineer   (existing, unmodified)
            |
            +----------------------+
            v                      v
    cloudtrail_structural.csv   cloudtrail_temporal.csv
            |                      |
            v                      v
    GNN side: blast radius     LSTM side: P_seq
    (blast_radius.py, via a    (temporal-analysis/prod/scorer.py,
     Neo4j-free in-memory       LSTMTransformerV5, max-pooled over
     PrivilegePropagationGraph  each principal's 10-min/stride-2
     -- see build_ppg() below)  windows)
            |                      |
            +----------+-----------+
                       v
        risk = clip(w_gnn*blast_score + w_lstm*p_seq, 0, 1) * 10

GNN side note: blast_radius.BlastRadiusEngine only needs an in-memory
privilege_features.PrivilegePropagationGraph -- it was already written
to be Neo4j-independent (see that class's docstring). Two ways to build
one are provided here:
  - build_ppg() -- parses the structural CSV directly (parse_principal/
    parse_target/node_key_for_*), same logic neo4j_graph_builder.py
    uses before it ever touches Neo4j. Zero infra required.
  - build_ppg_from_neo4j() -- reads the SAME graph back out of a live
    Neo4j instance that neo4j_graph_builder.build_graph() has already
    loaded (MATCH nodes/relationships via Cypher), then feeds the
    resulting rows through the identical PrivilegePropagationGraph.
    build_from_rows(). This is the "real" round-tripped-through-the-
    database path; --source neo4j selects it once Neo4j is up and
    build_graph.py has loaded the current structural CSV into it.
Both produce the same PrivilegePropagationGraph type, so
BlastRadiusEngine and everything downstream of it is identical either
way -- only how the graph's edges are sourced differs.

Note this is deliberately NOT data_loader.PrivilegePropagationGraphLoader
-- that class builds a PyTorch Geometric HeteroData for the trained
GraphSAGE/GAT classifier, a different object for a different job
(classification probability, not reachability). blast_radius.py has
never needed that class.

LSTM side notes (two fixes applied only in this file, not in the
vendored temporal-analysis/ code):
  1. cloudtrail_temporal.csv carries event_name_idx from THIS repo's
     own growable vocab (datasets/privilege-escalation/.event_name_vocab.json),
     which is a different index space than the LSTM checkpoint's own
     embedded vocab. Feeding event_name_idx straight through would
     silently score the wrong action for most events. Fixed by mapping
     event_name_idx -> event_name string via our vocab and letting the
     scorer re-map string -> its own vocab (prod/scorer.py already
     prefers "event_name" over "event_name_idx" when both are absent).
  2. pandas 3.x parses timestamp strings at microsecond resolution by
     default, but train_lstm_transformer.py's windowing code compares
     that against Timestamp.value (always nanoseconds) -- a 1000x unit
     mismatch that silently produces zero-length windows on this
     pandas version. Fixed by forcing datetime64[ns, UTC] before
     handing the frame to the scorer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT,
          os.path.join(REPO_ROOT, "graph_construction"),
          os.path.join(REPO_ROOT, "temporal-analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import feature_engine9 as fe9          # noqa: E402
import neo4j_graph_builder as nb       # noqa: E402
import privilege_features as pf        # noqa: E402
import blast_radius as br              # noqa: E402
import prod.scorer as lstm_scorer      # noqa: E402


# ── GNN side: Neo4j-free graph build + blast radius ─────────────────────────

def build_ppg(structural_df: pd.DataFrame, resolver: "pf.ActionAccessLevelResolver | None" = None):
    """Builds the same PrivilegePropagationGraph neo4j_graph_builder.build_graph()
    builds, minus the Neo4j write/read round-trip. Returns (ppg, principals),
    where principals is every (label, key) node that appears as a source_node
    in this batch -- i.e. every identity that was observed doing something.
    """
    resolver = resolver or pf.ActionAccessLevelResolver()

    principal_infos = structural_df["source_node"].apply(nb.parse_principal)
    target_infos = structural_df["target_node"].apply(nb.parse_target)

    src_keys = [
        pf.node_key_for_principal(arn, info.principal_type, info.name)
        for arn, info in zip(structural_df["source_node"], principal_infos)
    ]
    known_role_names = {
        info.name for info in principal_infos
        if info.principal_type in ("AssumedRole", "AWSServiceLinkedRole")
    }
    known_user_names = {
        info.name for info in principal_infos if info.principal_type == "IAMUser"
    }
    dst_keys = [
        pf.node_key_for_target(t.value, t.resource_type, t.service, known_role_names, known_user_names)
        for t in target_infos
    ]

    rows_for_graph = [
        {"log_id": lid, "source_key": sk, "target_key": dk, "edge_type": et, "label": int(lbl)}
        for lid, sk, dk, et, lbl in zip(
            structural_df["log_id"], src_keys, dst_keys,
            structural_df["edge_type"], structural_df["label"],
        )
    ]
    ppg = pf.PrivilegePropagationGraph(resolver).build_from_rows(rows_for_graph)
    principals = sorted({(sk.label, sk.key) for sk in src_keys})
    return ppg, principals


_NEO4J_SPECIFIC_LABELS = {"User", "Role", "UnresolvedPrincipal", "Service", "Resource", "Policy"}
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "test1234"


def build_ppg_from_neo4j(uri: str = DEFAULT_NEO4J_URI, user: str = DEFAULT_NEO4J_USER,
                          password: str = DEFAULT_NEO4J_PASSWORD,
                          resolver: "pf.ActionAccessLevelResolver | None" = None):
    """Same PrivilegePropagationGraph as build_ppg(), sourced from a live
    Neo4j instance that neo4j_graph_builder.build_graph() has already
    loaded (see build_graph.py). Requires the `neo4j` driver and a
    reachable Neo4j instance -- raises on connection failure rather than
    silently falling back, since a caller that asked for --source neo4j
    wants the real graph, not a quiet downgrade."""
    from neo4j import GraphDatabase

    resolver = resolver or pf.ActionAccessLevelResolver()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            node_label_by_key = {}
            for record in session.run("MATCH (n) WHERE n.key IS NOT NULL RETURN labels(n) AS labels, n.key AS key"):
                specific = [l for l in record["labels"] if l in _NEO4J_SPECIFIC_LABELS]
                if specific:
                    node_label_by_key[record["key"]] = specific[0]

            rows_for_graph = []
            edge_query = (
                "MATCH (src)-[r]->(dst) "
                "RETURN src.key AS src_key, dst.key AS dst_key, "
                "       r.log_id AS log_id, r.edge_type AS edge_type, r.is_attack AS is_attack"
            )
            for record in session.run(edge_query):
                src_key, dst_key = record["src_key"], record["dst_key"]
                src_label = node_label_by_key.get(src_key)
                dst_label = node_label_by_key.get(dst_key)
                if src_label is None or dst_label is None:
                    continue
                rows_for_graph.append({
                    "log_id": record["log_id"],
                    "source_key": pf.GraphNodeKey(src_label, src_key),
                    "target_key": pf.GraphNodeKey(dst_label, dst_key),
                    "edge_type": record["edge_type"],
                    "label": int(record["is_attack"]),
                })
    finally:
        driver.close()

    ppg = pf.PrivilegePropagationGraph(resolver).build_from_rows(rows_for_graph)
    principals = sorted({
        (row["source_key"].label, row["source_key"].key) for row in rows_for_graph
    })
    return ppg, principals


def score_gnn_blast_radius(structural_df: pd.DataFrame, source: str = "csv") -> dict:
    """Returns {(label, key): BlastRadiusReport} for every principal observed
    acting in this batch. source="csv" builds the graph directly from
    structural_df (no infra); source="neo4j" reads the same graph back
    out of a live Neo4j instance already loaded via build_graph.py."""
    # Built once and threaded through every compute() call below --
    # BlastRadiusEngine.compute() defaults to constructing a fresh
    # ActionAccessLevelResolver() per call when none is passed, and each
    # one re-parses policy_sentry's offline IAM action database (~0.4s).
    # Across hundreds of principals that's minutes of pure redundant
    # JSON parsing before any real reachability work happens.
    resolver = pf.ActionAccessLevelResolver()
    if source == "neo4j":
        ppg, principals = build_ppg_from_neo4j(resolver=resolver)
    else:
        ppg, principals = build_ppg(structural_df, resolver=resolver)
    engine = br.BlastRadiusEngine(ppg)
    return {node: engine.compute(node, resolver=resolver) for node in principals}


# ── LSTM side: P_seq via the existing prod scorer ────────────────────────────

def score_lstm_temporal(temporal_df: pd.DataFrame, event_vocab_path: str,
                         ckpt_path=None) -> pd.DataFrame:
    """Returns prod/scorer.py's per-(username, window) P_seq dataframe."""
    scorer = lstm_scorer.load_scorer(ckpt_path) if ckpt_path else lstm_scorer.load_scorer()

    vocab = json.loads(open(event_vocab_path, encoding="utf-8").read())
    inv = {v: k for k, v in vocab.items()}

    df = temporal_df.copy()
    df["event_name"] = df["event_name_idx"].map(inv).fillna("<UNK>")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")

    return lstm_scorer.score_dataframe(df, scorer)


# ── Ensemble ─────────────────────────────────────────────────────────────────

def combine(blast_reports: dict, lstm_windows: pd.DataFrame,
            weight_gnn: float = 0.5, weight_lstm: float = 0.5,
            agg: str = "max") -> pd.DataFrame:
    if not lstm_windows.empty:
        lstm_agg = lstm_windows.groupby("username")["P_seq"].agg(agg)
    else:
        lstm_agg = pd.Series(dtype=float)

    rows = []
    for (label, key), report in blast_reports.items():
        blast_score = float(report.score)
        lstm_score = float(lstm_agg.get(key, 0.0))
        combined = weight_gnn * blast_score + weight_lstm * lstm_score
        risk = round(min(10.0, max(0.0, combined * 10)), 2)
        rows.append({
            "principal_type": label,
            "principal": key,
            "gnn_blast_score": round(blast_score, 4),
            "lstm_p_seq_score": round(lstm_score, 4),
            "risk_score": risk,
            "reachable_assets": report.reachable_assets.total,
            "critical_assets_reachable": report.critical_assets.critical_asset_count,
            "administrator_reachable": report.privilege_reachability.administrator_reachable,
            "cross_service_reach": report.cross_service_count,
        })
    return (pd.DataFrame(rows)
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True))


def run(input_path: str, weight_gnn: float = 0.5, weight_lstm: float = 0.5,
        agg: str = "max", out_path: "str | None" = None,
        freeze_vocab: bool = False, gnn_source: str = "csv") -> pd.DataFrame:
    if not (0.0 <= weight_gnn <= 1.0 and 0.0 <= weight_lstm <= 1.0):
        raise ValueError("weights must be in [0, 1]")

    fe9.run_batch(input_path, freeze_vocab=freeze_vocab)
    structural_df = pd.read_csv(fe9.STRUCT_OUT)
    temporal_df = pd.read_csv(fe9.TEMPORAL_OUT)

    blast_reports = score_gnn_blast_radius(structural_df, source=gnn_source)
    lstm_windows = score_lstm_temporal(temporal_df, fe9.EVENT_NAME_VOCAB_FILE)

    result = combine(blast_reports, lstm_windows, weight_gnn, weight_lstm, agg)
    if out_path:
        result.to_csv(out_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Combined GNN blast-radius + LSTM temporal risk ensemble -- "
                     "one 0-10 risk score per principal.")
    parser.add_argument("--input", required=True,
                         help="Raw CloudTrail input (any file feature_engine9.py accepts)")
    parser.add_argument("--out", default="risk_scores.csv",
                         help="Output CSV of per-principal risk scores")
    parser.add_argument("--weight-gnn", type=float, default=0.5,
                         help="Weight on the blast-radius (GNN) score, default 0.5")
    parser.add_argument("--weight-lstm", type=float, default=0.5,
                         help="Weight on the P_seq (LSTM) score, default 0.5")
    parser.add_argument("--agg", choices=["max", "mean"], default="max",
                         help="How to aggregate a principal's LSTM window scores into one number")
    parser.add_argument("--freeze-vocab", action="store_true",
                         help="Do not grow the shared event_name vocab on this run")
    parser.add_argument("--source", choices=["csv", "neo4j"], default="csv",
                         help="Where the GNN side builds its graph from: 'csv' parses the "
                              "structural CSV directly (no infra); 'neo4j' reads the same "
                              "graph back out of a live Neo4j instance already loaded via "
                              "build_graph.py (bolt://localhost:7687, see build_ppg_from_neo4j)")
    args = parser.parse_args()

    result = run(args.input, args.weight_gnn, args.weight_lstm, args.agg,
                 args.out, args.freeze_vocab, gnn_source=args.source)
    with pd.option_context("display.max_rows", 50, "display.width", 160):
        print(result.to_string(index=False))
    print(f"\n{len(result)} principals scored -> {args.out}")


if __name__ == "__main__":
    main()
