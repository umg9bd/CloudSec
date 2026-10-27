"""
evaluate_checkpoint.py
======================
Score one or more trained checkpoints on the CURRENT dataset's held-out sets, each
with its own frozen gate_model.json thresholds (nothing is re-tuned here).

Used to compare checkpoints trained on different dataset versions on the same rows:
cloud_test is reported whole and split by dataset_version, because rows added in a
later version were written alongside that version's training rows -- the v0.2.0
rows are the like-for-like comparison with older checkpoints.

Run (repo root, module venv):
    prompt_injection_gate/.venv/Scripts/python.exe prompt_injection_gate/training/evaluate_checkpoint.py \
        prompt_injection_gate/artifacts/distilbert-mbert-pi-v1 prompt_injection_gate/artifacts/distilbert-mbert-pi-v2
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eval_transformer import encode, import_harness, predict  # noqa: E402


def evaluate(ckpt, sets, H):
    meta = json.load(open(os.path.join(ckpt, "gate_model.json"), encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    mal, safe = meta["malicious_threshold"], meta["safe_threshold"]
    out = {"model_version": meta["model_version"], "malicious_threshold": mal, "safe_threshold": safe, "sets": {}}
    for name, rs in sets.items():
        p = predict(model, encode(tok, [r["text"] for r in rs], meta["max_len"]))[:, 0]
        c = np.array([H.cls(r) for r in rs])
        atk, ben = c == "attack", c == "benign"
        verdict = np.where(p >= mal, "MALICIOUS", np.where(p < safe, "SAFE", "UNCERTAIN"))
        y = atk[atk | ben].astype(int)
        out["sets"][name] = {
            "n_attack": int(atk.sum()), "n_benign": int(ben.sum()),
            "roc_auc": float(roc_auc_score(y, p[atk | ben])) if 0 < y.sum() < len(y) else None,
            "attacks_blocked": float((verdict[atk] == "MALICIOUS").mean()) if atk.any() else None,
            "attacks_withheld": float((verdict[atk] != "SAFE").mean()) if atk.any() else None,
            "benign_safe": float((verdict[ben] == "SAFE").mean()) if ben.any() else None,
            "benign_review": float((verdict[ben] == "UNCERTAIN").mean()) if ben.any() else None,
            "benign_blocked": float((verdict[ben] == "MALICIOUS").mean()) if ben.any() else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--out", help="write the results as JSON here")
    args = ap.parse_args()
    H = import_harness()
    rows, _ = H.load()
    test = [r for r in rows if r["split"] == "test"]
    sets = {"cloud_test": test}
    for v in sorted({r["dataset_version"] for r in test}):
        sets[f"cloud_test:v{v}"] = [r for r in test if r["dataset_version"] == v]
    for r in rows:
        if r["split"] == "public_test":
            sets.setdefault("public_test:" + r["source"], []).append(r)

    results = {}
    for ckpt in args.checkpoints:
        res = evaluate(ckpt, sets, H)
        results[res["model_version"]] = res
        print(f"\n{res['model_version']}  (MALICIOUS >= {res['malicious_threshold']:.3f}, "
              f"SAFE < {res['safe_threshold']:.3f})")
        print(f"  {'set':<38}{'AUC':>6}{'atk blocked':>12}{'atk withheld':>13}"
              f"{'ben SAFE':>10}{'ben review':>11}{'ben blocked':>12}{'n atk/ben':>11}")
        f = lambda v: f"{v:.2f}" if v is not None else "  - "
        for name, s in res["sets"].items():
            print(f"  {name:<38}{f(s['roc_auc']):>6}{f(s['attacks_blocked']):>12}{f(s['attacks_withheld']):>13}"
                  f"{f(s['benign_safe']):>10}{f(s['benign_review']):>11}{f(s['benign_blocked']):>12}"
                  f"{s['n_attack']:>6}/{s['n_benign']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
