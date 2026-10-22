"""
ablation_study.py
=================
Controlled ablation of the real-time pipeline (experiments A-G). Every experiment is scored on the
SAME real held-out sessions with the SAME protocol evaluate_pipeline.py uses:

  * each real split is replayed through pipeline.Pipeline in file-sized chunks (evaluate_pipeline.replay),
  * a session is flagged if any of its events alerts (max over events),
  * every free choice (ensemble weight, alert threshold) is made on DEV by session-level F1,
  * TEST is replayed once per model variant with that frozen choice (replays are cached),
  * test F1 gets a 95% bootstrap CI, and every experiment is compared with A by a paired bootstrap
    over the same test sessions (evaluate_ml_baselines.f1_ci / paired_bootstrap).

  A  full HGT + LSTM ensemble           live checkpoints; w and threshold tuned on dev
  B  HGT only                            3-seed HGT p_graph; threshold tuned on dev
  C  LSTM only                           LSTM v6.3-ft p_sequence; threshold tuned on dev
  D  HGT + LSTM without ensemble fusion  D-avg: unweighted mean (w=0.5, no weight tuning)
                                         D-or : alert if HGT-only OR LSTM-only alerts (B/C thresholds)
  E  without selected graph features     HGT retrained (3 seeds) with a group removed
     or relations                        (ablation_groups.GRAPH_GROUPS); E-control = same retrain
                                         with nothing removed, to separate retraining noise
  F  without selected temporal features  LSTM v6.3 retrained + head fine-tune with a group dropped
                                         (ablation_groups.TEMPORAL_GROUPS)
  G  conventional ML baselines           Random Forest and XGBoost (evaluate_ml_baselines.run)

Nothing here writes pipeline_config.json. Outputs: results/ablation/ablation.json and ablation.md.

    python datasets/privilege-escalation/ablation/ablation_study.py --dev-only   # dev selection only
    python datasets/privilege-escalation/ablation/ablation_study.py              # + the test replays
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ablation_groups import GRAPH_GROUPS, REPO_ROOT, TEMPORAL_GROUPS, graph_ablation

PE_DIR = os.path.join(REPO_ROOT, "datasets", "privilege-escalation")
if PE_DIR not in sys.path:
    sys.path.insert(0, PE_DIR)

import evaluate_pipeline as ep  # noqa: E402  (chdirs to the repo root, like the pipeline expects)
from evaluate_ml_baselines import best_f1_over_thresholds, f1_ci, load_real_split, paired_bootstrap, prf  # noqa: E402
from pipeline import PipelineConfig, ensemble_risk  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "results", "ablation")
CACHE_DIR = os.path.join(OUT_DIR, "replays")


# ── model variants: which checkpoints the pipeline serves ───────────────────────────────────────

def variants(cfg: PipelineConfig) -> dict:
    v = {"live": {"hgt": list(cfg.hgt_checkpoints), "lstm": cfg.lstm_checkpoint, "graph_group": None}}
    for g in ["none"] + list(GRAPH_GROUPS):
        ckpts = sorted(glob.glob(os.path.join(OUT_DIR, f"hgt_{g}", "best_HGT_seed*_wrapped.pt")))
        if ckpts:
            v[f"hgt_{g}"] = {"hgt": ckpts, "lstm": cfg.lstm_checkpoint, "graph_group": None if g == "none" else g}
    for g in ["control"] + list(TEMPORAL_GROUPS):
        ck = os.path.join(OUT_DIR, f"lstm_{g}_ft", "temporal_lstm_transformer.pt")
        if os.path.exists(ck):
            v[f"lstm_{g}"] = {"hgt": list(cfg.hgt_checkpoints), "lstm": ck, "graph_group": None}
    return v


def replay_cached(name: str, spec: dict, split: str, cfg: PipelineConfig) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{name}_{split}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"log_id": str})
    run_cfg = PipelineConfig(**{**cfg.__dict__, "hgt_checkpoints": spec["hgt"], "lstm_checkpoint": spec["lstm"]})
    t0 = time.time()
    with graph_ablation(spec["graph_group"]):
        events, _ = ep.replay(split, run_cfg)
    out = events[["log_id", "p_graph", "p_sequence"]].copy()
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"  [{name}/{split}] replay cached ({time.time() - t0:.0f}s)", flush=True)
    return out


# ── scoring rules ───────────────────────────────────────────────────────────────────────────────

def sess(events, values, split):
    return ep.session_scores(events, values, split)


def tune_ensemble(ev_dev, dev, fixed_w=None):
    """evaluate_pipeline's rule: best dev F1, then AP, then the w closest to 0.5."""
    ws = [fixed_w] if fixed_w is not None else ep.WEIGHTS
    rows = []
    for w in ws:
        s = sess(ev_dev, ensemble_risk(ev_dev["p_graph"], ev_dev["p_sequence"], w), dev)
        f1, thr, p, r = best_f1_over_thresholds(s, dev["y"])
        rows.append((f1, average_precision_score(dev["y"], s), -abs(w - 0.5), float(w), thr))
    f1, ap, _, w, thr = max(rows)
    return {"w": w, "thr": thr, "dev_f1": f1, "dev_ap": ap}


def single_branch(ev, col):
    return ev[col].fillna(0.0) if col == "p_graph" else ev[col]


def tune_single(ev_dev, dev, col):
    s = sess(ev_dev, single_branch(ev_dev, col), dev)
    f1, thr, _, _ = best_f1_over_thresholds(s, dev["y"])
    return {"thr": thr, "dev_f1": f1, "dev_ap": average_precision_score(dev["y"], s)}


def metrics(y, scores, pred):
    p, r, f1 = prf(y, pred)
    lo, hi = f1_ci(y, pred)
    out = {"P": p, "R": r, "F1": f1, "F1_CI": [float(lo), float(hi)]}
    if scores is not None:
        out.update({"AUC": float(roc_auc_score(y, scores)), "AP": float(average_precision_score(y, scores))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-only", action="store_true", help="dev replays + selection only; never load test")
    ap.add_argument("--skip-ml", action="store_true", help="skip G (Random Forest / XGBoost)")
    ap.add_argument("--replay", default=None, metavar="VARIANTS",
                    help="comma list: only cache these variants' replays of --split, then exit (parallel workers)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev", help="split for --replay")
    args = ap.parse_args()

    cfg = PipelineConfig.load()
    vs = variants(cfg)
    if args.replay:
        if args.split == "test":
            missing = [n for n in vs if not os.path.exists(os.path.join(CACHE_DIR, f"{n}_dev.csv"))]
            assert not missing, f"cache every dev replay before any test replay; missing: {missing}"
        for n in args.replay.split(","):
            replay_cached(n, vs[n], args.split, cfg)
        return
    print(f"model variants: {list(vs)}")
    dev = load_real_split("dev")
    ev_dev = {n: replay_cached(n, s, "dev", cfg) for n, s in vs.items()}

    # dev selection for every experiment (choice frozen before test is read)
    sel = {"A": ("live", "ens", tune_ensemble(ev_dev["live"], dev)),
           "B": ("live", "hgt", tune_single(ev_dev["live"], dev, "p_graph")),
           "C": ("live", "lstm", tune_single(ev_dev["live"], dev, "p_sequence")),
           "D-avg": ("live", "ens", tune_ensemble(ev_dev["live"], dev, fixed_w=0.5)),
           "D-or": ("live", "or", None)}
    for n in vs:
        if n.startswith("hgt_"):
            sel[f"E-{n[4:]}" if n != "hgt_none" else "E-control"] = (n, "ens", tune_ensemble(ev_dev[n], dev))
        elif n.startswith("lstm_"):
            sel[f"F-{n[5:]}"] = (n, "ens", tune_ensemble(ev_dev[n], dev))
    # The same E/F variants with A's fusion weight held fixed (only the threshold re-tuned), so the
    # comparison changes the features alone. Needed because the fine-tuned LSTM heads were fit on
    # dev labels: their dev scores are in-sample, which biases a re-tuned weight toward the LSTM.
    w_a = sel["A"][2]["w"]
    for k in [k for k in sel if k[:2] in ("E-", "F-")]:
        sel[f"{k}@wA"] = (sel[k][0], "ens", tune_ensemble(ev_dev[sel[k][0]], dev, fixed_w=w_a))

    live_cfg_check = sel["A"][2]
    print(f"\nA reproduces the live dev selection: w={live_cfg_check['w']:.1f} thr={live_cfg_check['thr']:.4f} "
          f"dev F1={live_cfg_check['dev_f1']:.3f} (pipeline_config.json: w={cfg.weight_graph}, thr={cfg.alert_threshold})")
    print("\nDEV selection:")
    for k, (n, kind, t) in sel.items():
        if t:
            print(f"  {k:<20} variant={n:<22} " + "  ".join(f"{a}={b:.4f}" for a, b in t.items()))

    result = {"protocol": __doc__.strip().splitlines()[2:9], "dev_sessions": int(len(dev["y"])),
              "dev_selection": {k: {"variant": n, "kind": kind, **(t or {})} for k, (n, kind, t) in sel.items()}}
    if args.dev_only:
        _write(result)
        return

    test = load_real_split("test")
    y = test["y"]
    ev_test = {n: replay_cached(n, s, "test", cfg) for n, s in vs.items()}
    preds, scores = {}, {}
    for k, (n, kind, t) in sel.items():
        ev = ev_test[n]
        if kind == "ens":
            s = sess(ev, ensemble_risk(ev["p_graph"], ev["p_sequence"], t["w"]), test)
        elif kind in ("hgt", "lstm"):
            s = sess(ev, single_branch(ev, "p_graph" if kind == "hgt" else "p_sequence"), test)
        else:  # D-or: either branch at its own dev threshold
            pb = scores["B"] >= sel["B"][2]["thr"]
            pc = scores["C"] >= sel["C"][2]["thr"]
            preds[k], scores[k] = (pb | pc).astype(int), None
            continue
        scores[k], preds[k] = s, (s >= t["thr"]).astype(int)

    if not args.skip_ml:
        import evaluate_ml_baselines as mlb
        base = mlb.run()
        assert (base["sessions"] == test["sessions_true"].index).all()
        for name in ("Random Forest", "XGBoost"):
            preds[f"G-{name}"], scores[f"G-{name}"] = base[name], None

    table = {}
    for k in preds:
        m = metrics(y, scores.get(k), preds[k])
        if k != "A":
            d, lo, hi, p2 = paired_bootstrap(y, preds[k], preds["A"])
            m["vs_A"] = {"dF1": float(d), "CI": [float(lo), float(hi)], "p": float(p2)}
        fam, suffix = k[:2], ("@wA" if k.endswith("@wA") else "")
        ctrl = f"{fam}control{suffix}"
        if fam in ("E-", "F-") and k != ctrl and ctrl in preds:
            d, lo, hi, p2 = paired_bootstrap(y, preds[k], preds[ctrl])
            m["vs_control"] = {"control": ctrl, "dF1": float(d), "CI": [float(lo), float(hi)], "p": float(p2)}
        table[k] = m
    result.update({"test_sessions": int(len(y)), "test_attack_sessions": int(y.sum()), "test": table})
    _write(result)


def _write(result):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ablation.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    lines = ["# Ablation A-G (real-time pipeline)", "",
             f"Dev sessions: {result['dev_sessions']}. Selection on dev only; test replayed once per variant.", ""]
    if "test" in result:
        lines += [f"Test: {result['test_sessions']} sessions ({result['test_attack_sessions']} attack).", "",
                  "| Exp | P | R | F1 [95% CI] | AUC | AP | dF1 vs A [95% CI], p |", "|---|---|---|---|---|---|---|"]
        for k, m in result["test"].items():
            va = m.get("vs_A")
            vs_txt = f"{va['dF1']:+.3f} [{va['CI'][0]:+.3f}, {va['CI'][1]:+.3f}], p={va['p']:.3f}" if va else "reference"
            lines.append(f"| {k} | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} [{m['F1_CI'][0]:.3f}, {m['F1_CI'][1]:.3f}] | "
                         f"{m.get('AUC', float('nan')):.3f} | {m.get('AP', float('nan')):.3f} | {vs_txt} |")
        ctrl = [(k, m["vs_control"]) for k, m in result["test"].items() if "vs_control" in m]
        if ctrl:
            lines += ["", "E/F groups vs their retrained control (same recipe and weight rule, nothing removed):", ""]
            lines += [f"- {k} vs {c['control']}: {c['dF1']:+.3f} [{c['CI'][0]:+.3f}, {c['CI'][1]:+.3f}], p={c['p']:.3f}"
                      for k, c in ctrl]
    lines += ["", "## Dev selection", ""]
    for k, d in result["dev_selection"].items():
        lines.append(f"- {k}: " + ", ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in d.items()))
    with open(os.path.join(OUT_DIR, "ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {os.path.join(OUT_DIR, 'ablation.md')}")


if __name__ == "__main__":
    main()
