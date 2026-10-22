"""
train_eval_transformer.py
=========================
Fine-tunes the Stage-1 prompt-gate classifier and evaluates it with the SAME
protocol as datasets/prompt-injection/train_eval_baseline.py.

The harness file is imported, not edited: load(), cls(), dev_split(), metrics()
and TARGET_FPR come from it, so the split, the threshold rule and the reported
attack numbers are directly comparable with the TF-IDF baseline.

Model: one transformer encoder with 1 + K sigmoid outputs
  * output 0      = attack score (attack vs benign) -- the gate's main signal;
  * outputs 1..K  = one score per threat category (multi-label, from `labels`).
Ambiguous rows are the REVIEW_REQUIRED outcome: never trained on, reported apart.

Thresholds (all chosen on the grouped DEV slice, never on a *_test split):
  * malicious_threshold: the harness rule -- smallest dev score whose benign
    false-positive rate is <= TARGET_FPR;
  * safe_threshold: largest dev score at which at most SAFE_MISS of dev attacks
    would still be called SAFE; scores between the two are UNCERTAIN;
  * per-category thresholds: best dev F1 for that category (categories with no
    dev positives keep 0.5 and are marked uncalibrated).
Scores are not calibrated probabilities; only these dev-derived cut-offs are used.

Long inputs keep their head and tail (some public jailbreak prompts are ~2,000
words). On Windows the run asks the OS not to sleep (no admin rights needed).

Run (repo root, module venv):
    prompt_injection_gate/.venv/Scripts/python.exe -u prompt_injection_gate/training/train_eval_transformer.py

Writes:
    prompt_injection_gate/artifacts/<run_name>/     checkpoint, tokenizer, gate_model.json (gitignored)
    prompt_injection_gate/results/<run_name>.json   metrics + run manifest (committed)
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.dirname(HERE)
ROOT = os.path.dirname(GATE)
HARNESS = os.path.join(ROOT, "datasets", "prompt-injection", "train_eval_baseline.py")

# Dataset category labels the model scores (order = output columns 1..K).
CATEGORIES = [
    "prompt_injection_direct", "prompt_injection_indirect", "jailbreak", "system_prompt_extraction",
    "credential_harvesting", "exfiltration", "privilege_escalation", "defense_evasion_logging",
    "data_destruction", "obfuscation",
]
SAFE_MISS = 0.05          # at most 5% of dev attacks may score below safe_threshold


def import_harness():
    spec = importlib.util.spec_from_file_location("pi_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def keep_awake(on):
    """Windows only: stop the machine sleeping mid-run (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tok, texts, max_len):
    """Token ids per text; inputs longer than max_len keep head + tail."""
    out = []
    budget = max_len - 2
    head = budget // 2
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            ids = ids[:head] + ids[-(budget - head):]
        out.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    return out


def batches(ids, targets, bs, shuffle, rng):
    """Length-bucketed batches: shuffle, then sort inside chunks to cut padding."""
    order = list(range(len(ids)))
    if shuffle:
        rng.shuffle(order)
        chunk = bs * 50
        order = [i for c in range(0, len(order), chunk)
                 for i in sorted(order[c:c + chunk], key=lambda j: len(ids[j]))]
        groups = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(groups)
    else:
        order.sort(key=lambda j: len(ids[j]))
        groups = [order[i:i + bs] for i in range(0, len(order), bs)]
    for g in groups:
        width = max(len(ids[j]) for j in g)
        x = torch.zeros((len(g), width), dtype=torch.long)
        m = torch.zeros((len(g), width), dtype=torch.long)
        for k, j in enumerate(g):
            x[k, :len(ids[j])] = torch.tensor(ids[j])
            m[k, :len(ids[j])] = 1
        y = None if targets is None else torch.tensor(np.array([targets[j] for j in g]), dtype=torch.float)
        yield g, x, m, y


@torch.no_grad()
def predict(model, ids, bs=32):
    """Sigmoid scores, shape (n, 1 + K): column 0 = attack, then categories."""
    model.eval()
    out = None
    for g, x, m, _ in batches(ids, None, bs, False, None):
        p = torch.sigmoid(model(input_ids=x, attention_mask=m).logits).numpy()
        if out is None:
            out = np.zeros((len(ids), p.shape[1]))
        out[g] = p
    return out


def targets_for(rows, H):
    t = np.zeros((len(rows), 1 + len(CATEGORIES)), dtype=np.float32)
    for i, r in enumerate(rows):
        t[i, 0] = 1.0 if H.cls(r) == "attack" else 0.0
        for lab in r.get("labels", []):
            if lab in CATEGORIES:
                t[i, 1 + CATEGORIES.index(lab)] = 1.0
    return t


def best_f1_threshold(y, s):
    best = (0.5, -1.0)
    for t in sorted(set(np.round(s, 3))):
        yh = s >= t
        tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum())
        f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=os.path.join(GATE, "artifacts", "base_model"))
    ap.add_argument("--base-model-id",
                    default="distilbert/distilbert-base-multilingual-cased@45c032ab32cc946ad88a166f7cb282f58c753c2e")
    ap.add_argument("--run-name", default="distilbert-mbert-pi-v1")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--category-loss-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    keep_awake(True)
    try:
        run(args)
    finally:
        keep_awake(False)


def run(args):
    H = import_harness()
    rows, data_path = H.load()

    train_pool = [r for r in rows if r["split"] in ("train", "public_train")]
    test_sets = {"cloud_test": [r for r in rows if r["split"] == "test"]}
    for r in rows:
        if r["split"] == "public_test":
            test_sets.setdefault("public_test:" + r["source"], []).append(r)
    tr_all = [r for r in train_pool if H.cls(r) in ("attack", "benign")]
    tr, dev = H.dev_split(tr_all)
    print(f"train {len(tr)} | dev {len(dev)} | test sets: {', '.join(test_sets)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    id2label = {0: "attack", **{i + 1: c for i, c in enumerate(CATEGORIES)}}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1 + len(CATEGORIES), problem_type="multi_label_classification",
        id2label=id2label, label2id={v: k for k, v in id2label.items()})

    tr_ids = encode(tok, [r["text"] for r in tr], args.max_len)
    tr_t = targets_for(tr, H)
    dev_ids = encode(tok, [r["text"] for r in dev], args.max_len)
    dev_cls = [H.cls(r) for r in dev]
    dev_t = targets_for(dev, H)

    n_pos = int(tr_t[:, 0].sum()); n_neg = len(tr_t) - n_pos
    attack_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(1, n_pos)))
    cat_loss = torch.nn.BCEWithLogitsLoss()
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * math.ceil(len(tr_ids) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    rng = random.Random(args.seed)

    out_dir = os.path.join(GATE, "artifacts", args.run_name)
    best = {"auc": -1.0, "epoch": None}
    history = []
    t0 = time.time()
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for g, x, m, y in batches(tr_ids, tr_t, args.batch_size, True, rng):
            logits = model(input_ids=x, attention_mask=m).logits
            loss = attack_loss(logits[:, 0], y[:, 0]) + args.category_loss_weight * cat_loss(logits[:, 1:], y[:, 1:])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; stopping rather than training on it")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run_loss += loss.item(); n += 1; step += 1
            if step % 20 == 0:
                print(f"  epoch {ep} step {step}/{steps} loss {run_loss / n:.4f} elapsed {time.time() - t0:.0f}s",
                      flush=True)
                run_loss, n = 0.0, 0
        dp = predict(model, dev_ids)
        auc = roc_auc_score(dev_t[:, 0], dp[:, 0])
        history.append({"epoch": ep, "dev_attack_auc": round(float(auc), 4), "elapsed_s": round(time.time() - t0)})
        print(f"epoch {ep}: dev attack ROC-AUC {auc:.4f}", flush=True)
        if auc > best["auc"]:                                  # epoch selection on dev only
            best = {"auc": float(auc), "epoch": ep}
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)

    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    train_seconds = time.time() - t0
    dp = predict(model, dev_ids)
    da = dp[:, 0]

    # malicious threshold: exactly the harness rule
    cand = sorted(set(np.round(da, 3)))
    thr = next((t for t in cand if H.metrics(dev_cls, da, t)[1] <= H.TARGET_FPR), None)
    rule = f"smallest dev attack score with benign FPR <= {H.TARGET_FPR}"
    if thr is None:
        thr = max(cand, key=lambda t: H.metrics(dev_cls, da, t)[2]); rule = "best dev F1 (FPR target unreachable)"
    dr, df, dfi = H.metrics(dev_cls, da, thr)
    # safe threshold: at most SAFE_MISS of dev attacks below it
    atk_scores = np.sort(da[np.array([c == "attack" for c in dev_cls])])
    safe_thr = float(min(thr, atk_scores[int(math.floor(SAFE_MISS * len(atk_scores)))])) if len(atk_scores) else float(thr)
    ben = np.array([c == "benign" for c in dev_cls])
    dev_benign_uncertain = float(((da >= safe_thr) & (da < thr))[ben].mean())
    print(f"\nmalicious threshold (dev, benign FPR <= {H.TARGET_FPR:.0%}): {thr:.3f} -> dev attack-recall {dr:.2f} "
          f"benign-FPR {df:.2f} F1 {dfi:.2f}")
    print(f"safe threshold (dev, <= {SAFE_MISS:.0%} of attacks called SAFE): {safe_thr:.3f} "
          f"-> {dev_benign_uncertain:.0%} of dev benign land in UNCERTAIN")

    cat_thr = {}
    for i, c in enumerate(CATEGORIES):
        y = dev_t[:, 1 + i].astype(int)
        if y.sum() == 0:
            cat_thr[c] = {"threshold": 0.5, "calibrated": False, "dev_positives": 0}
        else:
            t, f1 = best_f1_threshold(y, dp[:, 1 + i])
            cat_thr[c] = {"threshold": t, "calibrated": True, "dev_positives": int(y.sum()), "dev_f1": round(f1, 3)}

    print("\n== held-out results (frozen thresholds) ==")
    print(f"{'set':<40}{'attack-recall':>14}{'benign-FPR':>12}{'F1':>7}{'AUC':>7}{'SAFE-rate':>10}{'n':>6}")
    results = {}
    scores_by_set = {}
    for name, rs in test_sets.items():
        if not rs:
            continue
        p = predict(model, encode(tok, [r["text"] for r in rs], args.max_len))
        scores_by_set[name] = p
        c = [H.cls(r) for r in rs]
        rec, fpr, f1 = H.metrics(c, p[:, 0], thr)
        mask = np.array([x in ("attack", "benign") for x in c])
        yb = np.array([1 if x == "attack" else 0 for x in c])[mask]
        auc = roc_auc_score(yb, p[mask, 0]) if 0 < yb.sum() < len(yb) else float("nan")
        verdict = np.where(p[:, 0] >= thr, "MALICIOUS", np.where(p[:, 0] < safe_thr, "SAFE", "UNCERTAIN"))
        atk = np.array([x == "attack" for x in c]); bn = np.array([x == "benign" for x in c])
        res = {"n": len(rs), "n_attack": int(atk.sum()), "n_benign": int(bn.sum()),
               "attack_recall_malicious": rec, "benign_fpr_malicious": fpr, "f1": f1, "roc_auc": auc,
               "attacks_called_safe": float((verdict[atk] == "SAFE").mean()) if atk.any() else None,
               "benign_called_safe": float((verdict[bn] == "SAFE").mean()) if bn.any() else None,
               "benign_called_uncertain": float((verdict[bn] == "UNCERTAIN").mean()) if bn.any() else None}
        results[name] = res
        safe_rate = res["benign_called_safe"] if res["benign_called_safe"] is not None else float("nan")
        print(f"{name:<40}{rec:>14.2f}{fpr:>12.2f}{f1:>7.2f}{auc:>7.3f}{safe_rate:>10.2f}{len(rs):>6}")

    # per-category recall (attack score >= malicious threshold) on cloud_test
    per_cat = {}
    ct = test_sets["cloud_test"]
    if ct:
        p = scores_by_set["cloud_test"]
        for r, s in zip(ct, p):
            for lab in r.get("labels", []):
                if lab not in CATEGORIES:
                    continue
                d = per_cat.setdefault(lab, {"n": 0, "blocked": 0, "not_safe": 0, "cat_hit": 0, "templates": set()})
                d["n"] += 1; d["blocked"] += int(s[0] >= thr); d["not_safe"] += int(s[0] >= safe_thr)
                d["cat_hit"] += int(s[1 + CATEGORIES.index(lab)] >= cat_thr[lab]["threshold"])
                d["templates"].add(r["template_id"])
        print("\ncloud_test by category (few templates each -- indicative only):")
        print(f"  {'category':28s}{'blocked':>9}{'not SAFE':>10}{'category head':>15}{'templates':>11}")
        for lab, d in sorted(per_cat.items()):
            print(f"  {lab:28s}{d['blocked']:>4}/{d['n']:<4}{d['not_safe']:>5}/{d['n']:<4}{d['cat_hit']:>9}/{d['n']:<5}"
                  f"{len(d['templates']):>8}")
        per_cat = {k: {"n": v["n"], "blocked_recall": v["blocked"] / v["n"], "not_safe_recall": v["not_safe"] / v["n"],
                       "category_head_recall": v["cat_hit"] / v["n"], "test_templates": len(v["templates"])}
                   for k, v in per_cat.items()}

    amb = [r for r in rows if H.cls(r) == "ambiguous" and r["split"] in ("train", "test")]
    amb_stats = None
    if amb:
        pa = predict(model, encode(tok, [r["text"] for r in amb], args.max_len))[:, 0]
        amb_stats = {"n": len(amb), "malicious": float((pa >= thr).mean()),
                     "uncertain": float(((pa >= safe_thr) & (pa < thr)).mean()), "safe": float((pa < safe_thr).mean())}
        print(f"\nambiguous rows (want REVIEW, not SAFE): SAFE {amb_stats['safe']:.0%}, "
              f"UNCERTAIN {amb_stats['uncertain']:.0%}, MALICIOUS {amb_stats['malicious']:.0%} (n={len(amb)})")

    sample = [r["text"] for r in random.Random(args.seed).sample(rows, min(200, len(rows)))]
    lat = []
    model.eval()
    with torch.no_grad():
        for s in encode(tok, sample, args.max_len):
            x = torch.tensor([s]); t1 = time.perf_counter()
            model(input_ids=x, attention_mask=torch.ones_like(x)); lat.append(1000 * (time.perf_counter() - t1))
    lat.sort()
    latency = {"p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(0.95 * len(lat)) - 1], "n": len(lat),
               "threads": torch.get_num_threads()}
    print(f"\nmodel latency, one prompt at a time on CPU: p50 {latency['p50_ms']:.0f} ms, p95 {latency['p95_ms']:.0f} ms")

    gate_model = {
        "model_version": f"{args.run_name}@{sha256(os.path.join(out_dir, 'model.safetensors'))[:12]}",
        "base_model": args.base_model_id,
        "categories": CATEGORIES,
        "max_len": args.max_len,
        "malicious_threshold": float(thr),
        "safe_threshold": safe_thr,
        "category_thresholds": cat_thr,
        "threshold_rule": {"malicious": rule, "safe": f"<= {SAFE_MISS:.0%} of dev attacks below it",
                           "categories": "best dev F1 per category; 0.5 where no dev positives (uncalibrated)"},
        "scores_are_calibrated_probabilities": False,
    }
    with open(os.path.join(out_dir, "gate_model.json"), "w", encoding="utf-8") as f:
        json.dump(gate_model, f, indent=2)

    manifest = {
        "run_name": args.run_name,
        "status": "fine-tuned checkpoint trained and evaluated in this run",
        "gate_model": gate_model,
        "task": "attack vs benign + multi-label threat categories; ambiguous not trained, reported separately",
        "data_file": os.path.relpath(data_path, ROOT).replace("\\", "/"),
        "data_sha256": sha256(data_path),
        "dataset_versions": sorted({str(r.get("dataset_version")) for r in rows}),
        "harness": os.path.relpath(HARNESS, ROOT).replace("\\", "/"),
        "harness_sha256": sha256(HARNESS),
        "n_train": len(tr), "n_dev": len(dev), "n_train_attack": n_pos, "n_train_benign": n_neg,
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "base_model"},
        "epoch_history": history, "selected_epoch": best["epoch"], "selected_dev_attack_auc": best["auc"],
        "train_seconds": round(train_seconds),
        "dev_at_malicious_threshold": {"attack_recall": dr, "benign_fpr": df, "f1": dfi},
        "dev_benign_uncertain_rate": dev_benign_uncertain,
        "held_out": results, "cloud_test_by_category": per_cat, "ambiguous": amb_stats,
        "latency_model_only_cpu": latency,
        "checkpoint_dir": os.path.relpath(out_dir, ROOT).replace("\\", "/"),
        "checkpoint_sha256": sha256(os.path.join(out_dir, "model.safetensors")),
        "torch": torch.__version__,
    }
    os.makedirs(os.path.join(GATE, "results"), exist_ok=True)
    out = os.path.join(GATE, "results", args.run_name + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=True)
    print(f"\nwrote {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
