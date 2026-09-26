"""
pipeline.py -- the real-time detector. CloudTrail log files dropped into a
folder are scored event by event by two parallel branches, fused by the
ensemble, and turned into alerts.

    incoming/<file>   CloudTrail JSON ({"Records": [...]}), JSONL/NDJSON (optionally .gz),
        |             or a feature_engine9-format CSV
        v
    feature_engine9.FeatureEngineer     stateful per principal; vocabulary and risk priors
        |                               frozen from training (never updated by live data)
        +-- structural row --> graph over a rolling time window --> HGT  --> p_graph
        +-- temporal row   --> the principal's last hour of events --> LSTM --> p_sequence
                                                   |
        ensemble:  risk = w * p_graph + (1 - w) * p_sequence     (p_graph missing -> p_sequence)
        alert:     risk >= threshold  -->  alerts/alert_<id>.json, one per principal per file
        every event:                   -->  output/risk_scores.csv

Streaming gives the same scores as batch scoring of the same events:
  - LSTM: an event's input is its principal's last 10 minutes, the gap to the event before each of
    those (capped at 1 h), and the time since the principal's last privilege-escalation write
    (capped at 1 h). So each principal's last LSTM_HISTORY (1 h) of events, plus the one event
    before that, reproduces batch scores exactly; ties in timestamp keep arrival order, as in a
    batch pass over the files in order.
  - Graph: rebuilt with the batch code (graph_construction/offline_graph.py, verified
    tensor-identical to the Neo4j loader) over the events inside `graph_window_hours`, with the
    training-fitted scalers. It equals batch scoring of those same events; a window covering a
    whole dataset reproduces batch evaluation exactly.

Settings (checkpoints, ensemble weight, alert threshold, graph window) live in
pipeline_config.json; the weight and threshold there are chosen on real_dataset_dev.csv by
datasets/privilege-escalation/evaluate_pipeline.py.

Usage (inside the Docker image -- see Dockerfile; torch is blocked natively on this machine):
    python pipeline.py --watch incoming                # run forever on a folder
    python pipeline.py --files a.json b.json           # score files once
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "graph_construction"), os.path.join(ROOT, "temporal-analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import feature_engine9 as fe9                 # noqa: E402
import prod.scorer as lstm_scorer             # noqa: E402
import train_lstm_transformer as tlt          # noqa: E402
from gnn_scorer import GNNScorer              # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "pipeline_config.json")
INPUT_SUFFIXES = (".csv", ".csv.gz", ".json", ".jsonl", ".ndjson", ".json.gz", ".jsonl.gz", ".ndjson.gz")
LSTM_HISTORY = pd.Timedelta(hours=1)
STRUCT_COLS = ["source_node", "target_node", "edge_type"]


@dataclass
class PipelineConfig:
    hgt_checkpoint: str = "checkpoints/best_HGT_wrapped.pt"
    lstm_checkpoint: str = "temporal-analysis/artifacts/lstm_transformer_clean/temporal_lstm_transformer.pt"
    weight_graph: float = 0.5          # w; the sequence branch gets 1 - w
    alert_threshold: float = 0.5       # on the 0-1 ensemble risk
    graph_window_hours: float = 24.0
    state_dir: str = "runtime"
    alert_dir: str = "alerts"
    output_csv: str = "output/risk_scores.csv"
    tuned_on: str = "untuned defaults"
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "PipelineConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def ensemble_risk(p_graph: np.ndarray, p_sequence: np.ndarray, weight_graph: float) -> np.ndarray:
    """w * p_graph + (1 - w) * p_sequence; the sequence score alone where the graph branch has no
    score (the event's node/relation types never occurred in the GNN's training graph)."""
    p_graph = np.asarray(p_graph, dtype=float)
    p_sequence = np.asarray(p_sequence, dtype=float)
    return np.where(np.isnan(p_graph), p_sequence, weight_graph * p_graph + (1 - weight_graph) * p_sequence)


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None, write_outputs: bool = True):
        self.cfg = cfg or PipelineConfig.load()
        self.write_outputs = write_outputs
        path = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
        self.gnn = GNNScorer(path(self.cfg.hgt_checkpoint))
        self.lstm = lstm_scorer.load_scorer(path(self.cfg.lstm_checkpoint))
        self.lstm_features = list(self.lstm.ckpt["feature_cols"])
        self.state_dir = path(self.cfg.state_dir)
        self.alert_dir = path(self.cfg.alert_dir)
        self.output_csv = path(self.cfg.output_csv)
        os.makedirs(self.state_dir, exist_ok=True)
        # Vocabulary and risk priors are the training-time files, frozen: live (unlabeled) traffic
        # must never change what an index or a prior means. Per-principal tracking state is this
        # deployment's own.
        self.engine = fe9.FeatureEngineer(
            event_name_vocab_path=path(fe9.EVENT_NAME_VOCAB_FILE),
            state_tracker_path=os.path.join(self.state_dir, "state_tracker.json"),
            graph_state_path=os.path.join(self.state_dir, "graph_node_state.json"),
            action_prior_path=path(fe9.ACTION_PRIOR_FILE),
            principal_prior_path=path(fe9.PRINCIPAL_PRIOR_FILE),
            freeze_vocab=True, freeze_priors=True,
        )
        self.buffer_path = os.path.join(self.state_dir, "event_buffer.pkl")
        self.buffer = pd.read_pickle(self.buffer_path) if os.path.exists(self.buffer_path) else pd.DataFrame()
        self._arrival = int(self.buffer["arrival"].max()) + 1 if len(self.buffer) else 0

    # ── feature engineering ──────────────────────────────────────────────
    def featurize(self, rows, source_name: str) -> pd.DataFrame:
        """rows: iterable of (row_index, raw CloudTrail row dict). One record per event with its
        structural and temporal features plus the metadata the alerts report."""
        records = []
        for idx, row in rows:
            try:
                struct = self.engine.get_structural_data(row)
                temporal = self.engine.get_temporal_features(row)
            except ValueError as e:
                print(f"[SKIP] {source_name} row {idx}: {e}")
                continue
            rec = {
                "log_id": f"{source_name}:{idx}",
                "timestamp": row.get("timestamp"),
                "username": row.get("username") or "unknown_user",
                "event_name": row.get("event_name"),
                "principal_arn": row.get("principal_arn"),
                "source_ip": row.get("source_ip"),
                "label": row.get("label", 0),
                "fast_lane": row.get("event_name") in fe9.CRITICAL_ACTIONS,
            }
            rec.update({c: struct[c] for c in STRUCT_COLS})
            rec.update(dict(zip(fe9.TEMPORAL_COLS, temporal)))
            records.append(rec)
        df = pd.DataFrame(records)
        if len(df):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").astype("datetime64[ns, UTC]")
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
            df["arrival"] = np.arange(self._arrival, self._arrival + len(df))
            self._arrival += len(df)
        return df

    # ── the two branches + ensemble ──────────────────────────────────────
    def score(self, new: pd.DataFrame) -> pd.DataFrame:
        """Scores the new events against the buffered history, then adds them to the buffer."""
        buf = pd.concat([self.buffer, new], ignore_index=True) if len(self.buffer) else new.copy()
        latest = buf["timestamp"].max()

        graph_rows = buf[buf["timestamp"] >= latest - pd.Timedelta(hours=self.cfg.graph_window_hours)]
        graph_rows = pd.concat([graph_rows, new]).drop_duplicates("log_id")  # late-arriving events too
        p_graph = self.gnn.score(graph_rows[["log_id"] + STRUCT_COLS + ["label"]])

        # Each new event's principal: its last hour before the earliest new event, and the single
        # event before that (the true predecessor for the inter-event gap of the hour's first event).
        mine = buf[buf["username"].isin(set(new["username"]))]
        cutoff = new["timestamp"].min() - LSTM_HISTORY
        before = mine[mine["timestamp"] < cutoff].sort_values(["timestamp", "arrival"]).groupby("username").tail(1)
        hist = pd.concat([before, mine[mine["timestamp"] >= cutoff]]).sort_values("arrival")
        frame = tlt.prepare_score_frame(hist[["log_id", "username", "timestamp", "event_name", "label"]
                                             + fe9.TEMPORAL_COLS], self.lstm.vocab, self.lstm_features)
        seqs = tlt.build_event_sequences(frame, self.lstm_features)
        p_seq = tlt.score_seqs(self.lstm.model, seqs, self.lstm.device)[["log_id", "P_event"]]

        out = new.merge(p_graph, on="log_id", how="left").merge(p_seq, on="log_id", how="left")
        out = out.rename(columns={"gnn_prob": "p_graph", "P_event": "p_sequence"})
        out["p_sequence"] = out["p_sequence"].fillna(0.0)
        out["risk"] = ensemble_risk(out["p_graph"], out["p_sequence"], self.cfg.weight_graph)
        out["risk_score"] = (out["risk"] * 10).round(2)
        out["alert"] = out["risk"] >= self.cfg.alert_threshold

        # Keep the window, plus each principal's last event before it: any later history cutoff
        # falls inside the window, so that event is the only older one a future predecessor can be.
        keep_from = latest - max(pd.Timedelta(hours=self.cfg.graph_window_hours), LSTM_HISTORY)
        older = buf[buf["timestamp"] < keep_from]
        last_older = older.sort_values(["timestamp", "arrival"]).groupby("username").tail(1)
        self.buffer = (pd.concat([buf[buf["timestamp"] >= keep_from], last_older])
                       .sort_values("arrival").reset_index(drop=True))
        return out

    # ── alerts and outputs ───────────────────────────────────────────────
    def emit(self, scored: pd.DataFrame, source_name: str) -> list:
        alerts = []
        for _, ev in scored[scored["fast_lane"]].iterrows():
            print(f"[FAST-LANE ALERT] {ev['timestamp']} {ev['username']} {ev['event_name']}: "
                  f"{fe9.CRITICAL_ACTIONS[ev['event_name']]}", flush=True)
        flagged = scored[scored["alert"] | scored["fast_lane"]]
        for principal, g in flagged.groupby("username", sort=False):
            g = g.sort_values("risk", ascending=False)
            alert = {
                "alert_id": str(uuid.uuid4()),
                "source_file": source_name,
                "principal": principal,
                "principal_arn": next((a for a in g["principal_arn"] if isinstance(a, str)), None),
                "first_seen": str(g["timestamp"].min()), "last_seen": str(g["timestamp"].max()),
                "max_risk_score": float(g["risk_score"].max()),
                "n_flagged_events": int(len(g)),
                "fast_lane_reasons": sorted({fe9.CRITICAL_ACTIONS[e] for e in g.loc[g["fast_lane"], "event_name"]}),
                "threshold": self.cfg.alert_threshold,
                "events": [
                    {"log_id": r.log_id, "timestamp": str(r.timestamp), "event_name": r.event_name,
                     "target": r.target_node, "p_graph": None if pd.isna(r.p_graph) else round(float(r.p_graph), 4),
                     "p_sequence": round(float(r.p_sequence), 4), "risk_score": float(r.risk_score)}
                    for r in g.head(25).itertuples()
                ],
            }
            alerts.append(alert)
            print(f"[ALERT] {principal}: {len(g)} event(s), max risk {alert['max_risk_score']:.2f}/10 "
                  f"(top: {g.iloc[0]['event_name']})", flush=True)
            if self.write_outputs:
                os.makedirs(self.alert_dir, exist_ok=True)
                with open(os.path.join(self.alert_dir, f"alert_{alert['alert_id']}.json"), "w", encoding="utf-8") as f:
                    json.dump(alert, f, indent=2)
        if self.write_outputs:
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            cols = ["log_id", "timestamp", "username", "event_name", "source_node", "target_node", "edge_type",
                    "p_graph", "p_sequence", "risk_score", "alert"]
            scored[cols].to_csv(self.output_csv, mode="a", index=False, header=not os.path.exists(self.output_csv))
        return alerts

    def save_state(self) -> None:
        self.engine.tracker.save()
        self.engine.graph_tracker.save()
        self.buffer.to_pickle(self.buffer_path)

    def process_rows(self, rows, source_name: str) -> pd.DataFrame:
        new = self.featurize(rows, source_name)
        if new.empty:
            return new
        scored = self.score(new)
        self.emit(scored, source_name)
        return scored

    def process_file(self, path: str) -> pd.DataFrame:
        name = os.path.basename(path)
        t0 = time.time()
        scored = self.process_rows(enumerate(fe9.iter_input_rows(path)), name)
        if self.write_outputs:
            self.save_state()
        n_alert = int(scored["alert"].sum()) if len(scored) else 0
        print(f"[PIPELINE] {name}: {len(scored)} events scored, {n_alert} above threshold "
              f"({time.time() - t0:.1f}s)", flush=True)
        return scored


def watch(directory: str, pipeline: Pipeline, poll_seconds: float = 2.0) -> None:
    """Scores every file that lands in `directory`, oldest first, then moves it to
    <directory>/processed/ (or <directory>/failed/ if it could not be read)."""
    done_dir, failed_dir = os.path.join(directory, "processed"), os.path.join(directory, "failed")
    for d in (directory, done_dir, failed_dir):
        os.makedirs(d, exist_ok=True)
    print(f"Watching {directory}/ for CloudTrail files (Ctrl+C to stop) -- "
          f"alerts -> {pipeline.alert_dir}/, scores -> {pipeline.output_csv}", flush=True)
    try:
        while True:
            files = sorted((os.path.join(directory, n) for n in os.listdir(directory)
                            if n.endswith(INPUT_SUFFIXES) and os.path.isfile(os.path.join(directory, n))),
                           key=os.path.getmtime)
            for path in files:
                if not fe9._file_is_stable(path):
                    continue  # still being written; next poll
                try:
                    pipeline.process_file(path)
                    shutil.move(path, os.path.join(done_dir, os.path.basename(path)))
                except Exception as e:  # one bad file must not stop the detector
                    print(f"[ERROR] {path}: {type(e).__name__}: {e}", flush=True)
                    shutil.move(path, os.path.join(failed_dir, os.path.basename(path)))
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--watch", metavar="DIR", help="Watch DIR (e.g. incoming) and score each new file")
    src.add_argument("--files", nargs="+", metavar="FILE", help="Score these files once, in order")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--reset-state", action="store_true",
                    help="Forget per-principal history and the event buffer before starting")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    if args.reset_state:
        shutil.rmtree(os.path.join(ROOT, cfg.state_dir), ignore_errors=True)
    print(f"Ensemble: {cfg.weight_graph:g} x HGT + {1 - cfg.weight_graph:g} x LSTM, alert at "
          f"{cfg.alert_threshold * 10:.2f}/10 ({cfg.tuned_on})", flush=True)
    pipeline = Pipeline(cfg)
    if args.watch:
        watch(args.watch, pipeline)
    else:
        for f in args.files:
            pipeline.process_file(f)


if __name__ == "__main__":
    main()
