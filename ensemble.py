from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT,
          os.path.join(REPO_ROOT, "graph_construction"),
          os.path.join(REPO_ROOT, "temporal-analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import feature_engine9 as fe9               
import neo4j_graph_builder as nb       
import privilege_features as pf          
import prod.scorer as lstm_scorer          
import train_lstm_transformer as tlt       


# Builds a PrivilegePropagationGraph directly from a structural DataFrame.
def build_ppg(structural_df: pd.DataFrame, resolver: "pf.ActionAccessLevelResolver | None" = None):
    """Builds the same PrivilegePropagationGraph neo4j_graph_builder.build_graph()
    builds"""
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
    return ppg


_NEO4J_SPECIFIC_LABELS = {"User", "Role", "UnresolvedPrincipal", "Service", "Resource", "Policy"}
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "test1234"


# Builds the same PrivilegePropagationGraph by reading nodes/edges back out of a live Neo4j instance.
def build_ppg_from_neo4j(uri: str = DEFAULT_NEO4J_URI, user: str = DEFAULT_NEO4J_USER,
                          password: str = DEFAULT_NEO4J_PASSWORD,
                          resolver: "pf.ActionAccessLevelResolver | None" = None):
    """Same PrivilegePropagationGraph as build_ppg(), sourced from a live
    Neo4j instance that neo4j_graph_builder.build_graph() has already
    loaded (see build_graph.py)."""
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

    return pf.PrivilegePropagationGraph(resolver).build_from_rows(rows_for_graph)


# ── GNN side: per-event structural risk (no trained model) ──────────────────
CREDENTIAL_ACCESS_ACTIONS = {
    "GetSecretValue", "GetPasswordData", "GetParameters", "GetParameter",
    "DescribeParameters", "ListSecrets", "BatchGetSecretValue",
}
CREDENTIAL_ACCESS_SERVICES = {"secretsmanager", "kms"}


# Computes a 0-1 structural risk score per event from graph features (hops, privilege gain, sensitivity).
def score_gnn_events(structural_df: pd.DataFrame, source: str = "csv",
                      resolver: "pf.ActionAccessLevelResolver | None" = None) -> pd.DataFrame:
    resolver = resolver or pf.ActionAccessLevelResolver()
    if source == "neo4j":
        ppg = build_ppg_from_neo4j(resolver=resolver)
    else:
        ppg = build_ppg(structural_df, resolver=resolver)

    edge_feats = ppg.compute_all_edge_features().set_index("log_id")
    target_infos = structural_df["target_node"].apply(nb.parse_target)
    sensitivity = target_infos.apply(lambda t: pf.resource_sensitivity_score(t.service, t.resource_type))

    rows = []
    for log_id, edge_type, sens, tinfo in zip(structural_df["log_id"], structural_df["edge_type"],
                                                sensitivity, target_infos):
        access_level = resolver.access_level(str(edge_type))
        is_priv_esc = str(edge_type) in nb.PRIVILEGE_ESCALATION_TECHNIQUES
        is_cred_access = (str(edge_type) in CREDENTIAL_ACCESS_ACTIONS
                           or tinfo.service in CREDENTIAL_ACCESS_SERVICES)

        if log_id in edge_feats.index:
            feats = edge_feats.loc[log_id]
            hop_count = int(feats["hop_count"])
            s_hop = min(1.0, max(0.0, hop_count - 1))
            s_gain = (min(1.0, max(0.0, float(feats["privilege_gain"]) / 4.0))
                      if bool(feats["privilege_gain_defined"]) else 0.0)
            s_abnormal = min(1.0, float(feats["abnormal_path_frequency"]) / 10.0)
        else:
            hop_count = None
            s_hop = s_gain = s_abnormal = 0.0

        s_priv_esc = 1.0 if is_priv_esc else 0.0
        s_cred_access = 1.0 if is_cred_access else 0.0
        s_access = (pf.ACCESS_LEVEL_RANK.get(access_level, 0) / 4.0) if access_level else 0.0
        s_target = float(sens) / 3.0

        score = (0.25 * s_priv_esc + 0.20 * s_cred_access + 0.10 * s_access + 0.15 * s_target
                 + 0.10 * s_hop + 0.10 * s_gain + 0.10 * s_abnormal)
        #privilege esc technique, access credentials, req access lvl of action, sensitivity of target resource
        #event is n privilege hops away, event produces privilege gain, path is abnormal
    

        rows.append({
            "log_id": log_id,
            "gnn_event_score": round(score, 4),
            "is_priv_esc_technique": is_priv_esc,
            "is_credential_access": is_cred_access,
            "access_level": access_level,
            "target_sensitivity_tier": int(sens),
            "hop_count": hop_count,
        })
    return pd.DataFrame(rows)


# ── LSTM side: per-event P_event (before window max-pooling) ────────────────

# Runs the trained LSTMTransformerV5 model to get a per-event probability (P_event) for each log row.
def score_lstm_events(temporal_df: pd.DataFrame, event_vocab_path: str,
                       ckpt_path=None) -> pd.DataFrame:
    """One row per log_id: P_event, this single event's own probability from
    LSTMTransformerV5 -- the per-event score the scorer computes internally
    before max-pooling into the per-window P_seq (see prod/scorer.py's
    score_dataframe -> train_lstm_transformer.score_events_to_windows)."""
    scorer = lstm_scorer.load_scorer(ckpt_path) if ckpt_path else lstm_scorer.load_scorer()

    our_vocab = json.loads(open(event_vocab_path, encoding="utf-8").read())
    inv = {v: k for k, v in our_vocab.items()}

    df = temporal_df.copy()
    df["event_name"] = df["event_name_idx"].map(inv).fillna("<UNK>")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").astype("datetime64[ns, UTC]")

    feature_cols = list(scorer.ckpt["feature_cols"])
    prepared = tlt.prepare_score_frame(df, scorer.vocab, feature_cols)
    seqs = tlt.build_event_sequences(prepared, feature_cols)
    if not seqs:
        raise ValueError("No event sequences built from input (need >=1 event per user)")
    event_df = tlt.score_seqs(scorer.model, seqs, scorer.device)
    return event_df[["log_id", "P_event"]]


# ── Ensemble ─────────────────────────────────────────────────────────────────

# Merges GNN and LSTM per-event scores into a single weighted 0-10 risk_score table.
def combine_events(structural_df: pd.DataFrame, temporal_df: pd.DataFrame,
                    gnn_df: pd.DataFrame, lstm_df: pd.DataFrame,
                    weight_gnn: float = 0.5, weight_lstm: float = 0.5) -> pd.DataFrame:
    merged = structural_df[["log_id", "source_node", "target_node", "edge_type", "label"]].merge(
        temporal_df[["log_id", "username", "timestamp"]], on="log_id", how="left"
    )
    merged["log_id"] = merged["log_id"].astype(str)
    gnn_df = gnn_df.copy()
    lstm_df = lstm_df.copy()
    gnn_df["log_id"] = gnn_df["log_id"].astype(str)
    lstm_df["log_id"] = lstm_df["log_id"].astype(str)

    merged = merged.merge(gnn_df, on="log_id", how="left")
    merged = merged.merge(lstm_df, on="log_id", how="left")
    merged["gnn_event_score"] = merged["gnn_event_score"].fillna(0.0)
    merged["P_event"] = merged["P_event"].fillna(0.0)

    combined = weight_gnn * merged["gnn_event_score"] + weight_lstm * merged["P_event"]
    merged["risk_score"] = (combined.clip(0.0, 1.0) * 10).round(2)
    merged = merged.rename(columns={"P_event": "lstm_event_score", "edge_type": "event_name",
                                     "label": "ground_truth_label"})

    cols = ["log_id", "timestamp", "username", "source_node", "event_name", "target_node",
            "risk_score", "gnn_event_score", "lstm_event_score",
            "is_priv_esc_technique", "is_credential_access", "access_level",
            "target_sensitivity_tier", "hop_count", "ground_truth_label"]
    # No sort: merge(how="left") preserves structural_df's row order, which
    # is arrival order (log_id = "<source_file>:<row_index>"). A real
    # deployment scores events interleaved across principals as they land
    # (--watch mode) -- sorting by (username, timestamp) would be a batch-mode
    # assumption that doesn't hold there, so this doesn't impose one either.
    return merged[cols].reset_index(drop=True)


# One-shot pipeline: runs feature engineering, scores with both models, and writes the combined risk table.
def run(input_path: str, weight_gnn: float = 0.5, weight_lstm: float = 0.5,
        out_path: "str | None" = None, freeze_vocab: bool = False,
        gnn_source: str = "csv") -> pd.DataFrame:
    if not (0.0 <= weight_gnn <= 1.0 and 0.0 <= weight_lstm <= 1.0):
        raise ValueError("weights must be in [0, 1]")
    if not math.isclose(weight_gnn + weight_lstm, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"weight_gnn + weight_lstm must sum to 1.0 (got {weight_gnn} + {weight_lstm} = "
            f"{weight_gnn + weight_lstm}) -- otherwise the 0-10 scale stops meaning what it "
            f"says (e.g. two independent 1.0/1.0 weights would let both signals count in "
            f"full, pushing everything toward the ceiling regardless of how strongly they "
            f"actually agree)."
        )

    fe9.run_batch(input_path, freeze_vocab=freeze_vocab)
    structural_df = pd.read_csv(fe9.STRUCT_OUT)
    temporal_df = pd.read_csv(fe9.TEMPORAL_OUT)
    n = len(structural_df)

    print(f"[GNN]  scoring {n} events from graph structure (source={gnn_source})...", flush=True)
    resolver = pf.ActionAccessLevelResolver()
    gnn_df = score_gnn_events(structural_df, source=gnn_source, resolver=resolver)

    print(f"[LSTM] scoring {n} events (LSTMTransformerV5, CPU -- this is the slow step, "
          f"can take a couple minutes)...", flush=True)
    lstm_df = score_lstm_events(temporal_df, fe9.EVENT_NAME_VOCAB_FILE)

    print("[ENSEMBLE] combining scores...", flush=True)
    result = combine_events(structural_df, temporal_df, gnn_df, lstm_df, weight_gnn, weight_lstm)
    if out_path:
        result.to_csv(out_path, index=False)
    return result

# ── Watch mode: re-score the full pipeline as new log files land ────────────

# Watches a directory and re-runs the full scoring pipeline each time a new log file arrives.
def watch(directory: str, weight_gnn: float = 0.5, weight_lstm: float = 0.5,
          out_path: str = "risk_scores.csv", freeze_vocab: bool = False,
          gnn_source: str = "csv") -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    os.makedirs(directory, exist_ok=True)
    engine = fe9.FeatureEngineer(
        event_name_vocab_path=fe9.EVENT_NAME_VOCAB_FILE,
        state_tracker_path=fe9.STATE_TRACKER_FILE,
        graph_state_path=fe9.GRAPH_NODE_STATE_FILE,
        action_prior_path=fe9.ACTION_PRIOR_FILE,
        principal_prior_path=fe9.PRINCIPAL_PRIOR_FILE,
        freeze_vocab=freeze_vocab,
        freeze_priors=True,
    )
    state = fe9.load_state()
    processed = set(state["processed_files"])
    resolver = pf.ActionAccessLevelResolver()

    # Re-scores all accumulated events so far and overwrites the output CSV.
    def rescore():
        structural_df = pd.read_csv(fe9.STRUCT_OUT)
        temporal_df = pd.read_csv(fe9.TEMPORAL_OUT)
        n = len(structural_df)
        print(f"[ENSEMBLE] rescoring {n} accumulated events...", flush=True)
        gnn_df = score_gnn_events(structural_df, source=gnn_source, resolver=resolver)
        lstm_df = score_lstm_events(temporal_df, fe9.EVENT_NAME_VOCAB_FILE)
        result = combine_events(structural_df, temporal_df, gnn_df, lstm_df, weight_gnn, weight_lstm)
        result.to_csv(out_path, index=False)
        print(f"[ENSEMBLE] {len(result)} events scored -> {out_path}", flush=True)

    # Filesystem event handler that reacts to new CloudTrail log files landing in the watched directory.
    class ArrivalHandler(FileSystemEventHandler):
        # Processes a newly-arrived, stable log file and triggers a rescore.
        def on_created(self, event):
            if event.is_directory:
                return
            path = event.src_path
            name = os.path.basename(path)
            if name in processed or not name.endswith(
                ('.csv', '.csv.gz', '.json', '.jsonl', '.ndjson', '.json.gz', '.jsonl.gz', '.ndjson.gz')
            ):
                return
            if not fe9._file_is_stable(path):
                return
            fe9.process_batch_file(engine, path)  # prints [FAST-LANE ALERT] as it goes
            processed.add(name)
            state["processed_files"] = sorted(processed)
            fe9.save_state(state)
            rescore()

    observer = Observer()
    observer.schedule(ArrivalHandler(), directory, recursive=False)
    observer.start()
    print(f"Watching {directory} for new CloudTrail log files (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        observer.stop()
        observer.join()


# CLI entry point: parses args and dispatches to either one-shot run() or watch().
def main():
    parser = argparse.ArgumentParser(
        description="Combined GNN structural + LSTM temporal risk ensemble -- "
                     "one 0-10 risk score per EVENT (not per principal), so an "
                     "attack chain shows up as a run of high scores in order.")
    parser.add_argument("--input", default=fe9.DEFAULT_INPUT,
                         help="Raw CloudTrail input (any file feature_engine9.py accepts). "
                              "One-shot mode: scored directly. With --watch --simulate: "
                              "chunked into the watched directory as the simulated source.")
    parser.add_argument("--watch", metavar="DIR",
                         help="Watch DIR and re-run the full pipeline (feature engineering "
                              "+ GNN + LSTM + ensemble) each time a new log file lands, "
                              "instead of scoring --input once. Ctrl+C to stop.")
    parser.add_argument("--simulate", action="store_true",
                         help="With --watch, also chunk --input into small CSVs dropped "
                              "into DIR periodically (see feature_engine9.py --simulate)")
    parser.add_argument("--out", default="risk_scores.csv",
                         help="Output CSV of per-event risk scores")
    parser.add_argument("--weight-gnn", type=float, default=0.5,
                         help="Weight on the structural (GNN) score, default 0.5")
    parser.add_argument("--weight-lstm", type=float, default=0.5,
                         help="Weight on the P_event (LSTM) score, default 0.5")
    parser.add_argument("--freeze-vocab", action="store_true",
                         help="Do not grow the shared event_name vocab on this run")
    parser.add_argument("--source", choices=["csv", "neo4j"], default="csv",
                         help="Where the GNN side builds its graph from: 'csv' parses the "
                              "structural CSV directly (no infra); 'neo4j' reads the same "
                              "graph back out of a live Neo4j instance already loaded via "
                              "build_graph.py (bolt://localhost:7687, see build_ppg_from_neo4j)")
    parser.add_argument("--show-table", action="store_true",
                         help="Also print the full per-event risk table to the terminal. "
                              "Off by default -- the run prints only feature_engine9's "
                              "[FAST-LANE ALERT] lines and a one-line summary; the full table "
                              "always goes to --out regardless of this flag. Ignored with --watch.")
    args = parser.parse_args()

    if args.watch:
        if args.simulate:
            fe9.simulate_incoming_files(args.watch, args.input)
        watch(args.watch, args.weight_gnn, args.weight_lstm, args.out,
              args.freeze_vocab, gnn_source=args.source)
        return

    result = run(args.input, args.weight_gnn, args.weight_lstm,
                 args.out, args.freeze_vocab, gnn_source=args.source)
    if args.show_table:
        with pd.option_context("display.max_rows", 50, "display.width", 200):
            print(result.to_string(index=False))
    print(f"\n{len(result)} events scored -> {args.out}")


if __name__ == "__main__":
    main()
