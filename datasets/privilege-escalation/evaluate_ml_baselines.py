"""
Classical-ML baselines -- the standard comparison a reviewer expects (the same style used by
e.g. arxiv:2512.10280's RF/XGBoost/LSTM table), and a more honest one than approximating a
commercial product (see evaluate_baselines.py's RULES -- "GuardDuty-style" was never validated
against real GuardDuty output; this project's data collection never enabled it, see
stratus_collection/README.md).

  - Logistic regression over a bag of actions: session level, which API calls a session made.
  - Random Forest and XGBoost over feature_engine9's temporal columns: event level, a session's
    score is the max over its events (the convention every model in this project uses).

Train-on-synthetic / evaluate-on-real, under the same protocol as the proposed system:
  - Training data is the synthetic-only table the LSTM trains on (TRAIN_CSV: fe: rows are
    cloudtrail_temporal.csv, syn: rows are synthetic_pe_chains.csv), so no baseline sees less
    data than the model it is compared against.
  - Each baseline gets the tuning privilege the proposed system got: its configuration --
    hyperparameters and, for the event-level models, how the categorical IDs are encoded and
    whether the generator-artifact features are used (ID_COLS, ARTIFACT_FEATURES) -- is chosen
    on real_dataset_dev.csv ONLY, by session-level F1 at the dev-best threshold (ties: dev AP).
  - The dev-selected configuration and threshold are applied ONCE, frozen, to
    real_dataset_test.csv: bootstrap CI, paired bootstrap against the rule baseline on the same
    sessions, and the F1 spread over SEEDS for the stochastic models.

Usage (from the repo root, using the project venv):
    .venv/Scripts/python.exe datasets/privilege-escalation/evaluate_ml_baselines.py
    .venv/Scripts/python.exe datasets/privilege-escalation/evaluate_ml_baselines.py --dev-only
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from evaluate_baselines import RULES

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
LSTM_DATA = os.path.join(REPO_ROOT, "temporal-analysis", "data", "lstm")
TRAIN_CSV = os.path.join(LSTM_DATA, "train_temporal_aug.csv")
# The training table's event_name_idx uses the LSTM's vocabulary: IDs 0-67 are identical to
# feature_engine9's (FE9_VOCAB_JSON, which the real splits are encoded with); IDs above 67 are
# actions only the syn: rows contain.
TRAIN_VOCAB_JSON = os.path.join(LSTM_DATA, "event_name_vocab.json")
FE9_VOCAB_JSON = os.path.join(HERE, ".event_name_vocab.json")
LOG_ID_RE = re.compile(r"^(.*):(\d+)$")
GUARDDUTY = "Curated IAM rule baseline (11 rules)"

NON_FEATURE_COLS = {"log_id", "username", "timestamp", "label", "event_name"}
# Nominal IDs, not quantities (0 = <UNK>). Synthetic data covers 67 actions and real data
# hundreds, so ~54% of real dev events arrive as event_name_idx=0 -- which in training only the
# 173 syn: rows remapped by load_training_table ever show. How to encode them is part of each
# event-level model's dev selection.
ID_COLS = ("event_name_idx", "event_source_idx", "principal_type_idx")
ID_ENCODINGS = ("ordinal", "onehot", "drop")
# Hardcoded by generate_synthetic_data.py, so they separate the synthetic classes for a reason
# real CloudTrail doesn't have: every attack step is written with mfa_authenticated="False", and
# attack steps are the only rows given a request_params_raw. Univariate AUC, synthetic training
# table -> real dev: params_length_normalized 0.913 -> 0.415, no_mfa 0.829 -> 0.550,
# mfa_absent 0.358 -> 0.454.
ARTIFACT_FEATURES = ("no_mfa", "mfa_absent", "params_length_normalized")
EVENT_INPUT_KEYS = ("id_encoding", "drop_artifacts")

N_BOOTSTRAP = 10000
SEED = 42
SEEDS = (42, 43, 44, 45, 46)
SESSION_GAP = pd.Timedelta(minutes=30)
EVENT_INPUT_GRID = ParameterGrid({"id_encoding": list(ID_ENCODINGS), "drop_artifacts": [False, True]})
RF_GRID = ParameterGrid({"max_depth": [None, 8], "min_samples_leaf": [1, 10], "max_features": ["sqrt", 0.5]})
XGB_GRID = ParameterGrid({"max_depth": [3, 6], "n_estimators": [100, 300], "min_child_weight": [1, 10]})
LR_GRID = ParameterGrid({"C": [0.01, 0.1, 1.0, 10.0], "bag": ["binary", "log1p"]})


def load_training_table() -> pd.DataFrame:
    """TRAIN_CSV plus each row's action name, with the syn: rows' out-of-vocabulary action IDs set
    to 0 -- what feature_engine9's frozen vocabulary gives those same actions in the real splits."""
    df = pd.read_csv(TRAIN_CSV)
    with open(TRAIN_VOCAB_JSON) as f:
        id_to_name = {int(i): n for n, i in json.load(f).items()}
    with open(FE9_VOCAB_JSON) as f:
        fe9_max_id = max(json.load(f).values())
    df["event_name"] = df["event_name_idx"].astype(int).map(id_to_name)
    df.loc[df["event_name_idx"] > fe9_max_id, "event_name_idx"] = 0
    return df


def feature_cols(df: pd.DataFrame, drop_artifacts: bool = False) -> list:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and not (drop_artifacts and c in ARTIFACT_FEATURES)]


def event_matrix(df: pd.DataFrame, cols: list, id_encoding: str, id_categories: dict) -> np.ndarray:
    """Event-level design matrix. One-hot columns cover only the IDs seen in training, so an
    unseen ID is all zeros ("no evidence from identity") instead of an ordinal that sorts it next
    to whichever actions happen to have small IDs."""
    parts = [df[[c for c in cols if c not in ID_COLS]].to_numpy(dtype=float)]
    if id_encoding == "ordinal":
        parts.append(df[list(ID_COLS)].to_numpy(dtype=float))
    elif id_encoding == "onehot":
        parts += [(df[c].to_numpy()[:, None] == id_categories[c]).astype(float) for c in ID_COLS]
    return np.hstack(parts)


def session_ids(temporal_df: pd.DataFrame, raw_csv_name: str, raw_df: pd.DataFrame) -> np.ndarray:
    row_idx = []
    for lid in temporal_df["log_id"].astype(str):
        m = LOG_ID_RE.match(lid)
        if not m or m.group(1) != raw_csv_name:
            raise SystemExit(f"log_id {lid!r} does not match expected source {raw_csv_name!r}")
        row_idx.append(int(m.group(2)))
    return raw_df["session_id"].to_numpy()[np.array(row_idx, dtype=int)]


def session_max(probs: np.ndarray, sids: np.ndarray, sessions_true: pd.Series) -> np.ndarray:
    s = pd.Series(probs, index=sids).groupby(level=0).max()
    return s.reindex(sessions_true.index).fillna(0.0).to_numpy()


def session_probs(temporal_df: pd.DataFrame, probs: np.ndarray, raw_csv_name: str,
                   raw_df: pd.DataFrame, sessions_true: pd.Series) -> np.ndarray:
    return session_max(probs, session_ids(temporal_df, raw_csv_name, raw_df), sessions_true)


def best_f1_over_thresholds(scores: np.ndarray, y_true: np.ndarray, max_thresholds: int = 300):
    cand = np.unique(scores)
    if len(cand) > max_thresholds:
        cand = np.unique(np.quantile(scores, np.linspace(0, 1, max_thresholds)))
    best = (-1.0, None, 0.0, 0.0)
    for thr in cand:
        pred = (scores >= thr).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best[0]:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            best = (f1, float(thr), p, r)
    return best


def prf(t, p):
    tp = int(((t == 1) & (p == 1)).sum()); fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return pr, rc, f1


def f1_rows(t, p):
    tp = ((t == 1) & (p == 1)).sum(1); fp = ((t == 0) & (p == 1)).sum(1); fn = ((t == 1) & (p == 0)).sum(1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def bootstrap_idx(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, (N_BOOTSTRAP, n))


def f1_ci(y_true, y_pred):
    idx = bootstrap_idx(len(y_true))
    return tuple(np.percentile(f1_rows(y_true[idx], y_pred[idx]), [2.5, 97.5]))


def paired_bootstrap(y_true, y_a, y_b):
    """F1(a) - F1(b), resampling sessions jointly (same sessions, so the samples are paired)."""
    idx = bootstrap_idx(len(y_true))
    deltas = f1_rows(y_true[idx], y_a[idx]) - f1_rows(y_true[idx], y_b[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p2 = min(1.0, 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))))
    return prf(y_true, y_a)[2] - prf(y_true, y_b)[2], lo, hi, p2


def load_real_split(split: str) -> dict:
    name = f"real_dataset_{split}.csv"
    raw = pd.read_csv(os.path.join(HERE, name))
    temporal = pd.read_csv(os.path.join(HERE, f"real_dataset_{split}_temporal.csv"))
    sessions_true = raw.drop_duplicates("session_id").set_index("session_id")["session_label"]
    bags = raw.groupby("session_id")["event_name"].apply(list).reindex(sessions_true.index).tolist()
    return {"raw": raw, "temporal": temporal, "sessions_true": sessions_true, "y": sessions_true.to_numpy(),
            "sids": session_ids(temporal, name, raw), "bags": bags}


def rule_predictions(split: dict) -> np.ndarray:
    rules = RULES[GUARDDUTY]
    return np.array([int(bool(set(b) & rules)) for b in split["bags"]])


def synthetic_session_bags(train: pd.DataFrame):
    """The synthetic generators write no session id, so sessions are rebuilt the deployable way:
    one username's events, split wherever it goes quiet for SESSION_GAP. A session is an attack
    if any of its events is."""
    df = train[["username", "timestamp", "event_name", "label"]].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["username", "ts"], kind="stable")
    new = (df["username"] != df["username"].shift()) | (df["ts"].diff() > SESSION_GAP)
    g = df.groupby(new.cumsum())
    return g["event_name"].apply(list).tolist(), g["label"].max().to_numpy(int)


def bag_matrix(bags: list, vocab: dict, bag: str) -> np.ndarray:
    """Session x action counts over the training vocabulary. An action never seen in training
    has no learned weight either way, so it's dropped."""
    X = np.zeros((len(bags), len(vocab)))
    for i, names in enumerate(bags):
        for n in names:
            j = vocab.get(n)
            if j is not None:
                X[i, j] += 1
    return (X > 0).astype(float) if bag == "binary" else np.log1p(X)


def lr_session_scorer(cfg, train_bags, train_y, vocab):
    clf = LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=5000)
    clf.fit(bag_matrix(train_bags, vocab, cfg["bag"]), train_y)
    return lambda split: clf.predict_proba(bag_matrix(split["bags"], vocab, cfg["bag"]))[:, 1]


def make_rf(params, seed):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1, **params)


def make_xgb(pos_weight):
    def make(params, seed):
        return XGBClassifier(learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                             eval_metric="logloss", random_state=seed, n_jobs=-1, **params)
    return make


def event_session_scorer(make_clf, cfg, train, seed):
    params = {k: v for k, v in cfg.items() if k not in EVENT_INPUT_KEYS}
    cols = feature_cols(train, cfg["drop_artifacts"])
    cats = {c: np.unique(train[c].to_numpy()) for c in ID_COLS}
    clf = make_clf(params, seed)
    clf.fit(event_matrix(train, cols, cfg["id_encoding"], cats), train["label"].to_numpy(int))

    def score(split):
        probs = clf.predict_proba(event_matrix(split["temporal"], cols, cfg["id_encoding"], cats))[:, 1]
        return session_max(probs, split["sids"], split["sessions_true"])
    return score


def select_on_dev(candidates, fit, dev):
    """Fit every candidate configuration (seed SEED) and score it on dev, best first by (F1 at the
    dev-best threshold, average precision)."""
    rows = []
    for cfg in candidates:
        s = fit(cfg, SEED)(dev)
        f1, thr, p, r = best_f1_over_thresholds(s, dev["y"])
        rows.append({"cfg": cfg, "f1": f1, "thr": thr, "p": p, "r": r, "ap": average_precision_score(dev["y"], s)})
    return sorted(rows, key=lambda row: (row["f1"], row["ap"]), reverse=True)


def fmt_cfg(cfg: dict) -> str:
    names = {"id_encoding": lambda v: f"ids={v}",
             "drop_artifacts": lambda v: "artifacts=" + ("dropped" if v else "kept")}
    return ", ".join(names.get(k, lambda v, k=k: f"{k}={v}")(v) for k, v in cfg.items())


def print_dev_selection(name: str, rows: list):
    print(f"\n[{name}] DEV selection over {len(rows)} configurations (session F1 at the dev-best threshold):")
    for row in rows[:3]:
        print(f"    F1={row['f1']:.3f} AP={row['ap']:.3f} thr={row['thr']:.4f}  {fmt_cfg(row['cfg'])}")
    if "id_encoding" in rows[0]["cfg"]:
        print(f"    best dev F1 per input variant: {'artifacts kept':>16}{'artifacts dropped':>19}")
        for enc in ID_ENCODINGS:
            cells = [max(r["f1"] for r in rows
                         if r["cfg"]["id_encoding"] == enc and r["cfg"]["drop_artifacts"] == d) for d in (False, True)]
            print(f"    {'ids=' + enc:<31}{cells[0]:>16.3f}{cells[1]:>19.3f}")


def evaluate_on_test(fit, cfg, seeds, dev, test, rule_pred):
    """The dev-selected configuration, once per seed: threshold tuned on dev, applied frozen to
    test. The SEED run is the reported one; the others only measure seed sensitivity."""
    runs = {}
    for seed in seeds:
        score = fit(cfg, seed)
        _, thr, _, _ = best_f1_over_thresholds(score(dev), dev["y"])
        runs[seed] = (thr, (score(test) >= thr).astype(int))
    thr, pred = runs[SEED]
    p, r, f1 = prf(test["y"], pred)
    return {"thr": thr, "p": p, "r": r, "f1": f1, "ci": f1_ci(test["y"], pred),
            "vs_rules": paired_bootstrap(test["y"], pred, rule_pred),
            "seed_f1s": [prf(test["y"], pr)[2] for _, pr in runs.values()], "pred": pred}


def run(dev_only: bool = False) -> dict:
    train = load_training_table()
    neg, pos = int((train["label"] == 0).sum()), int((train["label"] == 1).sum())
    print(f"Training table: {len(train)} synthetic events ({pos} attack, {pos / len(train):.1%}) -- {TRAIN_CSV}")
    train_bags, train_bag_y = synthetic_session_bags(train)
    vocab = {n: j for j, n in enumerate(sorted({n for b in train_bags for n in b}))}
    print(f"Bag-of-actions training: {len(train_bags)} synthetic sessions ({int(train_bag_y.sum())} attack), "
          f"{len(vocab)} distinct actions")

    dev = load_real_split("dev")
    print(f"DEV sessions: {len(dev['y'])} ({int(dev['y'].sum())} attack)")

    models = {
        "LR (bag of actions)": (list(LR_GRID), (SEED,),
                                lambda cfg, seed: lr_session_scorer(cfg, train_bags, train_bag_y, vocab)),
        "Random Forest": ([{**i, **h} for i in EVENT_INPUT_GRID for h in RF_GRID], SEEDS,
                          lambda cfg, seed: event_session_scorer(make_rf, cfg, train, seed)),
        "XGBoost": ([{**i, **h} for i in EVENT_INPUT_GRID for h in XGB_GRID], SEEDS,
                    lambda cfg, seed: event_session_scorer(make_xgb(neg / pos), cfg, train, seed)),
    }
    selected = {}
    for name, (candidates, seeds, fit) in models.items():
        rows = select_on_dev(candidates, fit, dev)
        print_dev_selection(name, rows)
        selected[name] = (rows[0], seeds, fit)
    if dev_only:
        return {}

    test = load_real_split("test")
    rule_pred = rule_predictions(test)
    results = {name: evaluate_on_test(fit, best["cfg"], seeds, dev, test, rule_pred)
               for name, (best, seeds, fit) in selected.items()}

    n_att = int(test["y"].sum())
    print(f"\n{'=' * 104}")
    print(f"SUMMARY -- real held-out TEST set only, {len(test['y'])} sessions ({n_att} attack); "
          f"configuration and threshold frozen from dev")
    print(f"{'=' * 104}")
    print(f"{'Model':<38}{'P':>6}{'R':>7}{'F1':>7}  {'95% CI':<16}{'F1 - rules, paired [95% CI]':<38}{'F1 over seeds':<16}")
    for name, r in results.items():
        d, lo, hi, p2 = r["vs_rules"]
        seeds = (f"{np.mean(r['seed_f1s']):.3f} +/- {np.std(r['seed_f1s']):.3f}"
                 if len(r["seed_f1s"]) > 1 else "deterministic")
        p_txt = "p<0.0001" if p2 < 1e-4 else f"p={p2:.4f}"
        print(f"{name:<38}{r['p']:>6.3f}{r['r']:>7.3f}{r['f1']:>7.3f}  [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] {p_txt:<10}{seeds}")
    rp, rr, rf = prf(test["y"], rule_pred)
    lo, hi = f1_ci(test["y"], rule_pred)
    print(f"{GUARDDUTY:<38}{rp:>6.3f}{rr:>7.3f}{rf:>7.3f}  [{lo:.3f}, {hi:.3f}]  {'--':<38}{'--':<16}")
    print("\nSelected configurations (frozen from dev):")
    for name, (best, _, _) in selected.items():
        print(f"  {name:<22} thr={results[name]['thr']:.4f}  {fmt_cfg(best['cfg'])}")
    return {"sessions": test["sessions_true"].index, "y": test["y"],
            **{name: r["pred"] for name, r in results.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-only", action="store_true", help="Run the dev selection only; never load the test split")
    run(dev_only=ap.parse_args().dev_only)


if __name__ == "__main__":
    main()
