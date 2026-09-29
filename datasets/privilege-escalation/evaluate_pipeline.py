"""
Evaluates the real-time pipeline (pipeline.py) exactly as it runs: each real split is replayed
through Pipeline in file-sized chunks, in order, with fresh state -- featurization, the rolling
graph window, the LSTM history buffer, the HGT and LSTM branches and the ensemble all run as they
do on files landing in incoming/.

  1. DEV (real_dataset_dev.csv) replay. Checks that streaming reproduces batch where it should:
     the temporal features against real_dataset_dev_temporal.csv, and the LSTM scores against one
     batch scoring pass over all of dev.
  2. Ensemble weight w (HGT share) and alert threshold chosen on dev by session-level F1 (a session
     is flagged if any event alerts -- the convention every result in this project uses), and
     written to pipeline_config.json.
  3. --test: TEST (real_dataset_test.csv) replayed ONCE with the frozen config; P/R/F1 with a
     bootstrap CI and paired bootstraps against the rule baseline and the classical-ML baselines
     (evaluate_ml_baselines.py) on the same sessions.

Usage (inside the Docker image, from the repo root):
    python datasets/privilege-escalation/evaluate_pipeline.py            # dev: check + tune
    python datasets/privilege-escalation/evaluate_pipeline.py --test     # + the one test run
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "graph_construction"), os.path.join(REPO_ROOT, "temporal-analysis"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO_ROOT)  # feature_engine9's paths are repo-relative

import feature_engine9 as fe9                         # noqa: E402
import train_lstm_transformer as tlt                  # noqa: E402
from evaluate_ml_baselines import (GUARDDUTY, best_f1_over_thresholds, f1_ci, paired_bootstrap,  # noqa: E402
                                   prf, rule_predictions, load_real_split)
from pipeline import Pipeline, PipelineConfig, ensemble_risk  # noqa: E402

CHUNK_ROWS = 250                       # events per simulated incoming file
WEIGHTS = np.round(np.linspace(0, 1, 11), 2)


def replay(split: str, cfg: PipelineConfig) -> pd.DataFrame:
    """Every event of real_dataset_<split>.csv through a fresh Pipeline, CHUNK_ROWS at a time."""
    name = f"real_dataset_{split}.csv"
    with tempfile.TemporaryDirectory() as tmp:
        run_cfg = PipelineConfig(**{**cfg.__dict__, "state_dir": os.path.join(tmp, "state")})
        pipe = Pipeline(run_cfg, write_outputs=False)
        rows = list(enumerate(fe9.iter_input_rows(os.path.join(HERE, name))))
        t0, parts = time.time(), []
        for i in range(0, len(rows), CHUNK_ROWS):
            parts.append(pipe.process_rows(rows[i:i + CHUNK_ROWS], name))
        print(f"[{split}] replayed {len(rows)} events in {-(-len(rows) // CHUNK_ROWS)} chunks "
              f"({time.time() - t0:.0f}s, graph window {cfg.graph_window_hours:g}h)", flush=True)
        return pd.concat(parts, ignore_index=True), pipe


def check_against_batch(split: str, events: pd.DataFrame, pipe: Pipeline) -> None:
    batch = pd.read_csv(os.path.join(HERE, f"real_dataset_{split}_temporal.csv"), dtype={"log_id": str})
    m = events.merge(batch, on="log_id", suffixes=("", "_batch"))
    diffs = {c: float(np.abs(m[c].astype(float) - m[f"{c}_batch"].astype(float)).max()) for c in fe9.TEMPORAL_COLS}
    differing = {c: d for c, d in diffs.items() if d > 1e-9}
    print(f"  temporal features vs the committed batch {split} file: {len(m)}/{len(events)} events matched; "
          f"{len(fe9.TEMPORAL_COLS) - len(differing)}/{len(fe9.TEMPORAL_COLS)} columns identical"
          + (f", differing: {differing} -- these read the frozen vocabulary/risk-prior files, so a "
             f"difference means the committed file was built from an older snapshot of them" if differing else ""))
    frame = tlt.prepare_score_frame(events[["log_id", "username", "timestamp", "event_name", "label"] + fe9.TEMPORAL_COLS],
                                    pipe.lstm.vocab, pipe.lstm_features)
    whole = tlt.score_seqs(pipe.lstm.model, tlt.build_event_sequences(frame, pipe.lstm_features), pipe.lstm.device)
    m = events[["log_id", "p_sequence"]].merge(whole[["log_id", "P_event"]], on="log_id")
    print(f"  LSTM streaming vs one batch pass: max |diff| = {np.abs(m.p_sequence - m.P_event).max():.2e}")


def session_scores(events: pd.DataFrame, values, split: dict) -> np.ndarray:
    """Max over each session's events. Sessions come from the raw split's session_id, via each
    event's row index in its log_id ("real_dataset_dev.csv:<row>")."""
    rows = events["log_id"].str.rsplit(":", n=1).str[1].astype(int).to_numpy()
    sid = split["raw"]["session_id"].to_numpy()[rows]
    s = pd.Series(np.asarray(values, dtype=float), index=sid).groupby(level=0).max()
    return s.reindex(split["sessions_true"].index).fillna(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="After tuning on dev, replay TEST once with the frozen config")
    args = ap.parse_args()
    cfg = PipelineConfig.load()

    dev = load_real_split("dev")
    events, pipe = replay("dev", cfg)
    check_against_batch("dev", events, pipe)
    y = dev["y"]
    for label, v in [("HGT alone", events["p_graph"].fillna(0.0)), ("LSTM alone", events["p_sequence"])]:
        s = session_scores(events, v, dev)
        print(f"  {label:<11} dev session AUC={roc_auc_score(y, s):.3f} AP={average_precision_score(y, s):.3f} "
              f"best F1={best_f1_over_thresholds(s, y)[0]:.3f}")

    rows = []
    for w in WEIGHTS:
        s = session_scores(events, ensemble_risk(events["p_graph"], events["p_sequence"], w), dev)
        f1, thr, p, r = best_f1_over_thresholds(s, y)
        rows.append((f1, average_precision_score(y, s), -abs(w - 0.5), w, thr, p, r))
    print("\n  DEV sweep (w = HGT share):  " + "  ".join(f"{w:.1f}:{f1:.3f}" for f1, _, _, w, *_ in rows))
    f1, ap_, _, w, thr, p, r = max(rows)  # best F1, then AP, then the w closest to an even split
    print(f"  selected on dev: w={w:.1f}, alert threshold={thr:.4f} (risk {thr * 10:.2f}/10) -> "
          f"dev P={p:.3f} R={r:.3f} F1={f1:.3f} AP={ap_:.3f}")
    cfg.weight_graph, cfg.alert_threshold = float(w), round(float(thr), 4)
    cfg.tuned_on = (f"real_dataset_dev.csv, {len(y)} sessions: session-level F1={f1:.3f} "
                    f"(weight and threshold chosen there; test never used for tuning)")
    cfg.save()
    print(f"  -> written to pipeline_config.json")

    if not args.test:
        return
    test = load_real_split("test")
    events, _ = replay("test", cfg)
    s = session_scores(events, events["risk"], test)
    pred = (s >= cfg.alert_threshold).astype(int)
    p, r, f1 = prf(test["y"], pred)
    lo, hi = f1_ci(test["y"], pred)
    print(f"\nTEST ({len(test['y'])} sessions, frozen config): P={p:.3f} R={r:.3f} F1={f1:.3f} [95% CI {lo:.3f}, {hi:.3f}]  "
          f"session AUC={roc_auc_score(test['y'], s):.3f}")
    import evaluate_ml_baselines as mlb
    baselines = mlb.run()
    others = {GUARDDUTY: rule_predictions(test), **{k: v for k, v in baselines.items() if k not in ("sessions", "y")}}
    assert (baselines["sessions"] == test["sessions_true"].index).all()
    print("\nPaired bootstrap, pipeline - other (same test sessions):")
    for name, other in others.items():
        d, lo, hi, p2 = paired_bootstrap(test["y"], pred, other)
        print(f"  vs {name:<38} {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p2:.4f}")


if __name__ == "__main__":
    main()
