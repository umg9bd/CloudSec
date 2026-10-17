"""
train_eval_baseline.py
======================
Reference harness for Stage-1 steps 1/2/5, done the honest way:
  * Train on `train` + `public_train`.
  * Pick the decision threshold on a DEV slice held out of train (grouped by
    template so it doesn't leak) -- never on any *_test split.
  * Report SEPARATELY on each held-out test set: cloud `test` and each
    `public_test` (out-of-distribution). This is the "does it generalize to
    human-written prompts" check the review measured at 45-85% FP.

This uses a TF-IDF + logistic-regression BASELINE so it runs on CPU in seconds.
It is a reference, not the production model: your teammate swaps the vectorizer
+ classifier for the DeBERTa fine-tune; the splitting / threshold / reporting
logic stays identical.

Binary gate: ATTACK (block) vs BENIGN (allow). `ambiguous` rows are the
REVIEW_REQUIRED outcome -- reported separately (not forced into either class).

Run (after merge_public.py, else cloud-only):
    python datasets/prompt-injection/train_eval_baseline.py
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED = os.path.join(HERE, "prompt_injection_merged.jsonl")
CLOUD = os.path.join(HERE, "prompt_injection_cloud.jsonl")
TARGET_FPR = 0.05          # pick threshold for <=5% false positives on dev benign


def load():
    path = MERGED if os.path.exists(MERGED) else CLOUD
    print(f"data: {os.path.basename(path)}"
          + ("" if path == MERGED else "   (run merge_public.py to add the public OOD sets)"))
    return [json.loads(l) for l in open(path, encoding="utf-8")], path


def cls(r):
    p = r["primary_label"]
    return "benign" if p == "benign" else ("ambiguous" if p == "ambiguous" else "attack")


def dev_split(train_rows, frac=0.2, seed=7):
    """Hold out `frac` of TRAIN templates (grouped) as dev -- not random rows."""
    import random
    tmpls = sorted({r["template_id"] for r in train_rows})
    random.Random(seed).shuffle(tmpls)
    dev_t = set(tmpls[:max(1, int(len(tmpls) * frac))])
    tr = [r for r in train_rows if r["template_id"] not in dev_t]
    dev = [r for r in train_rows if r["template_id"] in dev_t]
    return tr, dev


def metrics(y_true, p, thr):
    yhat = (p >= thr).astype(int)                       # 1 = attack
    atk = np.array([t == "attack" for t in y_true])
    ben = np.array([t == "benign" for t in y_true])
    rec = yhat[atk].mean() if atk.any() else float("nan")       # attack recall
    fpr = yhat[ben].mean() if ben.any() else float("nan")       # benign false-positive rate
    tp = (yhat[atk] == 1).sum(); fp = (yhat[ben] == 1).sum(); fn = (yhat[atk] == 0).sum()
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else float("nan")
    return rec, fpr, f1


def main():
    rows, path = load()

    train_pool = [r for r in rows if r["split"] in ("train", "public_train")]
    test_sets = {"cloud_test": [r for r in rows if r["split"] == "test"]}
    for r in rows:                                      # one bucket per public source
        if r["split"] == "public_test":
            test_sets.setdefault("public_test:" + r["source"], []).append(r)

    # train on attack+benign only; ambiguous is the review outcome (scored, not trained)
    tr_all = [r for r in train_pool if cls(r) in ("attack", "benign")]
    tr, dev = dev_split(tr_all)
    print(f"train {len(tr)} | dev {len(dev)} | test sets: {', '.join(test_sets)}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([r["text"] for r in tr])
    X = vec.transform([r["text"] for r in tr])
    y = np.array([cls(r) == "attack" for r in tr], dtype=int)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)

    def prob(rs):
        return clf.predict_proba(vec.transform([r["text"] for r in rs]))[:, 1]

    # threshold on DEV: smallest t with benign FPR <= target (fallback: best F1)
    dp = prob(dev); dy = [cls(r) for r in dev]
    cand = sorted(set(np.round(dp, 3)))
    thr = next((t for t in cand if metrics(dy, dp, t)[1] <= TARGET_FPR), None)
    if thr is None:
        thr = max(cand, key=lambda t: metrics(dy, dp, t)[2])
    dr, df, dfi = metrics(dy, dp, thr)
    print(f"\nthreshold (dev, FPR<= {TARGET_FPR:.0%}): {thr:.3f}  "
          f"-> dev attack-recall {dr:.2f}  benign-FPR {df:.2f}  F1 {dfi:.2f}")

    print("\n== held-out results (frozen threshold) ==")
    print(f"{'set':<28}{'attack-recall':>14}{'benign-FPR':>12}{'F1':>7}{'n':>7}")
    for name, rs in test_sets.items():
        if not rs:
            continue
        rec, fpr, f1 = metrics([cls(r) for r in rs], prob(rs), thr)
        print(f"{name:<28}{rec:>14.2f}{fpr:>12.2f}{f1:>7.2f}{len(rs):>7}")

    amb = [r for r in rows if cls(r) == "ambiguous" and r["split"] in ("train", "test")]
    if amb:
        flagged = (prob(amb) >= thr).mean()
        print(f"\nambiguous (REVIEW_REQUIRED): {flagged:.0%} would be flagged as attack "
              f"at this threshold (n={len(amb)}) -- these should route to human review, not auto-block.")
    print("\nSwap TfidfVectorizer+LogisticRegression for the DeBERTa fine-tune; keep the"
          " split/threshold/reporting logic. Report cloud_test AND each public_test.")


if __name__ == "__main__":
    main()
