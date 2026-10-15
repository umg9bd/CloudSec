"""
LSTM v6.1 + Random Forest score fusion, tuned on real dev.

P_fused = w * P_lstm + (1 - w) * P_rf per event, then P_seq = max over the same 10-min / stride-2
windows. w (21-point grid) and both thresholds are tuned on real dev. Gate: the fusion must beat
the RF alone on dev (window F1, then event AUC-PR) before real test is scored. The dev comparison
is optimistic by construction (w is tuned on dev); real test is the unbiased check.

  python fusion_v6.py              # tune on dev + gate
  python fusion_v6.py --eval-test  # if the gate passed: score real test once (fusion, LSTM, RF)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import baseline_rf_v6 as rfb
import train_lstm_transformer as v5
import train_lstm_transformer_v6 as t6
from train_lstm_transformer import metrics_dict, score_seqs, window_scores
from train_lstm_transformer_v6 import tune_threshold

OUT_PATH = t6.OUT_DIR / "fusion.json"
W_GRID = [round(float(w), 2) for w in np.linspace(0.0, 1.0, 21)]


def part_scores(model, device, s: rfb.Setup, rf, part: str) -> tuple[pd.DataFrame, list[dict]]:
    """Per-event LSTM and RF scores on the same real events, plus that part's windows."""
    split = t6.RealSplit.build(s.real, s.proto, part, list(s.ckpt["feature_cols"]))
    ev = score_seqs(model, split.seqs, device).rename(columns={"P_event": "P_lstm"})
    rows = s.real.set_index("log_id").loc[ev["log_id"]].reset_index()
    ev["P_rf"] = rfb.rf_scores(rf, rows, s.cols)
    return ev, split.windows


def fused_eval(ev: pd.DataFrame, windows: list[dict], w: float, evt_thr=None, win_thr=None):
    e = ev.assign(P_event=w * ev["P_lstm"] + (1.0 - w) * ev["P_rf"])
    y, p = e["label"].to_numpy(), e["P_event"].to_numpy()
    et = tune_threshold(y, p) if evt_thr is None else float(evt_thr)
    pseq = window_scores(windows, e, et)
    wy, wp = pseq["window_label"].to_numpy(), pseq["P_seq"].to_numpy()
    wt = tune_threshold(wy, wp) if win_thr is None else float(win_thr)
    pseq["pred"] = (pseq["P_seq"] >= wt).astype(int)
    return metrics_dict(y, p, et), metrics_dict(wy, wp, wt), pseq


def gate_key(evt: dict, win: dict) -> tuple[float, float]:
    return round(win["f1"], 4), round(evt["auc_pr"], 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-test", action="store_true", help="score real test once (only if the dev gate passed)")
    args = ap.parse_args()

    s = rfb.load_setup()
    rf = rfb.train_rf(s)
    model, _, device = v5.load_checkpoint(t6.CKPT_PATH)
    dev_ev, dev_windows = part_scores(model, device, s, rf, "dev")

    grid = []
    for w in W_GRID:
        evt, win, _ = fused_eval(dev_ev, dev_windows, w)
        grid.append({"w_lstm": w, "dev_event": evt, "dev_window": win, "key": gate_key(evt, win)})
        print(
            f"w_lstm={w:.2f} dev evt_ap={evt['auc_pr']:.4f} evt_f1={evt['f1']:.4f} "
            f"win_ap={win['auc_pr']:.4f} win_f1={win['f1']:.4f}",
            flush=True,
        )
    # ties -> weight closest to an even mix
    best = max(grid, key=lambda g: (g["key"], -abs(g["w_lstm"] - 0.5)))
    rf_only, lstm_only = grid[0], grid[-1]
    passed = best["key"] > rf_only["key"]
    print(f"chosen w_lstm={best['w_lstm']:.2f} key={best['key']} | RF key={rf_only['key']} | LSTM key={lstm_only['key']}")
    print(f"GATE (fusion vs RF on dev): {'PASS' if passed else 'FAIL'}")

    out = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    if args.eval_test and "test_event" in out:
        raise SystemExit(f"real test already scored for this fusion ({OUT_PATH}); not scoring it again")
    out.update(
        {
            "formula": "P_fused = w_lstm * P_lstm + (1 - w_lstm) * P_rf; P_seq = max over window",
            "lstm_schema": s.ckpt.get("schema_version"),
            "w_lstm": best["w_lstm"],
            "event_threshold": best["dev_event"]["threshold"],
            "window_threshold": best["dev_window"]["threshold"],
            "dev_event": best["dev_event"],
            "dev_window": best["dev_window"],
            "dev_rf_only": {"event": rf_only["dev_event"], "window": rf_only["dev_window"]},
            "dev_lstm_only": {"event": lstm_only["dev_event"], "window": lstm_only["dev_window"]},
            "gate_vs_rf_on_dev": "PASS" if passed else "FAIL",
            "grid": [{k: g[k] for k in ("w_lstm", "key")} for g in grid],
            "note": "dev comparison is optimistic (w tuned on dev); real test is the unbiased check",
        }
    )

    if args.eval_test:
        if not passed:
            raise SystemExit("gate failed: fusion does not beat the RF on dev; real test not scored")
        test_ev, test_windows = part_scores(model, device, s, rf, "test")
        te_evt, te_win, te_pseq = fused_eval(
            test_ev, test_windows, best["w_lstm"], best["dev_event"]["threshold"], best["dev_window"]["threshold"]
        )
        out.update(test_event=te_evt, test_window=te_win)
        te_pseq.assign(split="real_test").to_csv(t6.OUT_DIR / "fusion_P_seq_test.csv", index=False)

        # single-model test scores, once, with their own dev-tuned thresholds
        t6.eval_test()
        rf_out = json.loads(rfb.OUT_PATH.read_text(encoding="utf-8"))
        rf_df, rf_windows = s.part("test")
        rf_out["test_event"], rf_out["test_window"] = rfb.rf_eval(
            rf, rf_df, s.cols, rf_windows, rf_out["event_threshold"], rf_out["window_threshold"]
        )
        rfb.OUT_PATH.write_text(json.dumps(rf_out, indent=2), encoding="utf-8")
        lstm = json.loads(t6.METRICS_PATH.read_text(encoding="utf-8"))
        fmt = lambda e, w_: f"evt_ap={e['auc_pr']:.4f} evt_f1={e['f1']:.4f} win_ap={w_['auc_pr']:.4f} win_f1={w_['f1']:.4f}"
        print("=== Real TEST (thresholds from dev) ===")
        print(f"  fusion w={best['w_lstm']:.2f}  {fmt(te_evt, te_win)}")
        print(f"  LSTM v6.1     {fmt(lstm['test_event'], lstm['test_window'])}")
        print(f"  RF            {fmt(rf_out['test_event'], rf_out['test_window'])}")

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    fused_dev = fused_eval(dev_ev, dev_windows, best["w_lstm"])[2]
    fused_dev.assign(split="real_dev").to_csv(t6.OUT_DIR / "fusion_P_seq_dev.csv", index=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
