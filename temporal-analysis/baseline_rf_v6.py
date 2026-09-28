"""
Random Forest baseline for LSTM v6.1 — same features, users and real dev/test split.

Reads feature_cols, vocab and split_users from the saved v6.1 checkpoint, trains on the
synthetic train users, max-pools P_event into the same 10-min / stride-2 windows and tunes
thresholds on real dev. Gate: the LSTM must beat this on real dev before real test is used.

  python baseline_rf_v6.py              # dev (+ synthetic test) only
  python baseline_rf_v6.py --eval-test  # also score real test, once, after the gate
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

import train_lstm_transformer_v6 as t6
from train_lstm_transformer import (
    attach_pe_context,
    build_fusion_windows,
    metrics_dict,
    window_scores,
)
from train_lstm_transformer_v6 import tune_threshold

OUT_PATH = t6.OUT_DIR / "rf_baseline.json"
SEED = 42


@dataclass
class Setup:
    """Everything shared with the saved LSTM checkpoint: features, vocab, splits, data."""

    ckpt: dict
    cols: list[str]
    split: dict
    proto: t6.RealProtocol
    df: pd.DataFrame
    real: pd.DataFrame

    def part(self, name: str) -> tuple[pd.DataFrame, list[dict]]:
        return self.real[self.proto.event_mask(self.real, name)], t6.real_windows(self.real, self.proto, name)


def load_setup() -> Setup:
    ckpt = torch.load(t6.CKPT_PATH, map_location="cpu", weights_only=False)
    vocab, feature_cols = dict(ckpt["event_name_vocab"]), list(ckpt["feature_cols"])
    split = ckpt["config"]["split_users"]
    pe_ids, _ = t6.maybe_pe_ids(vocab)
    df, _, _ = t6.load_and_validate(t6.CSV_PATH)
    return Setup(
        ckpt=ckpt,
        cols=feature_cols + ["event_name_idx"],
        split=split,
        proto=t6.RealProtocol.from_config(split),
        df=attach_pe_context(df, pe_ids),
        real=attach_pe_context(t6.load_real(vocab, feature_cols), pe_ids),
    )


def train_rf(s: Setup) -> RandomForestClassifier:
    train = s.df[s.df["username"].isin(s.split["syn_train"])]
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
    ).fit(train[s.cols].to_numpy(dtype=np.float32), train["label"].to_numpy())
    print(f"RF trained on {len(train)} synthetic events, {len(s.cols)} features", flush=True)
    return rf


def rf_scores(rf, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return rf.predict_proba(df[cols].to_numpy(dtype=np.float32))[:, 1]


def rf_eval(rf, df: pd.DataFrame, cols: list[str], windows: list[dict], evt_thr=None, win_thr=None):
    p = rf_scores(rf, df, cols)
    y = df["label"].to_numpy()
    et = tune_threshold(y, p) if evt_thr is None else evt_thr
    ev = pd.DataFrame({"log_id": df["log_id"].to_numpy(), "P_event": p})
    pseq = window_scores(windows, ev, et)
    wy, wp = pseq["window_label"].to_numpy(), pseq["P_seq"].to_numpy()
    wt = tune_threshold(wy, wp) if win_thr is None else win_thr
    return metrics_dict(y, p, et), metrics_dict(wy, wp, wt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-test", action="store_true", help="also score real test (after the gate)")
    args = ap.parse_args()

    s = load_setup()
    rf = train_rf(s)
    cols, split = s.cols, s.split

    dev_df, dev_windows = s.part("dev")
    dev_evt, dev_win = rf_eval(rf, dev_df, cols, dev_windows)
    syn = s.df[s.df["username"].isin(split["syn_test"])]
    syn_evt, syn_win = rf_eval(rf, syn, cols, build_fusion_windows(syn), dev_evt["threshold"], dev_win["threshold"])
    out = {
        "model": "RandomForest(500, balanced_subsample)",
        "checkpoint_schema": s.ckpt.get("schema_version"),
        "features": cols,
        "event_threshold": dev_evt["threshold"],
        "window_threshold": dev_win["threshold"],
        "dev_event": dev_evt,
        "dev_window": dev_win,
        "syn_test_event": syn_evt,
        "syn_test_window": syn_win,
        "top_importances": dict(
            sorted(zip(cols, map(float, rf.feature_importances_)), key=lambda kv: -kv[1])[:15]
        ),
    }
    if args.eval_test:
        test_df, test_windows = s.part("test")
        out["test_event"], out["test_window"] = rf_eval(
            rf, test_df, cols, test_windows, dev_evt["threshold"], dev_win["threshold"]
        )
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lstm = json.loads(t6.METRICS_PATH.read_text(encoding="utf-8"))
    print("=== RF real DEV EVENT ===", dev_evt)
    print("=== RF real DEV WINDOW ===", dev_win)
    if args.eval_test:
        print("=== RF real TEST EVENT ===", out["test_event"])
        print("=== RF real TEST WINDOW ===", out["test_window"])
    fmt = lambda m: f"win_f1={m['dev_window']['f1']:.4f} win_ap={m['dev_window']['auc_pr']:.4f} evt_ap={m['dev_event']['auc_pr']:.4f} evt_f1={m['dev_event']['f1']:.4f}"
    print(f"GATE dev  LSTM {fmt(lstm)}")
    print(f"GATE dev  RF   {fmt(out)}")
    key = lambda m: (round(m["dev_window"]["f1"], 4), round(m["dev_event"]["auc_pr"], 4))
    verdict = "LSTM wins" if key(lstm) > key(out) else "RF wins" if key(lstm) < key(out) else "TIE (dev saturated)"
    print(f"GATE verdict: {verdict}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
