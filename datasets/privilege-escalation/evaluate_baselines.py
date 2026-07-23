"""
Rule-based baseline evaluation -- re-run against the corrected combined
real dataset (session_id-based grouping, 4 collectors + invictus) and the
synthetic training set.

Session grouping differs deliberately between the two datasets:
  - synthetic_cloudtrail.csv: one random username per session (by
    construction in the generator), so grouping by username is correct.
  - real_dataset_combined.csv: one IAM identity runs many sequential
    detonations, so username would collapse dozens of sessions into one.
    session_id (added when the real dataset was rebuilt) is the correct
    grouping key there.

Adds bootstrap 95% CIs on the real-data metrics, since a point estimate
on ~240 sessions isn't defensible on its own.

Usage:
    python evaluate_baselines.py
"""

import random

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RULES = {
    "Minimal SIEM (3 rules)": {
        "StopLogging", "DeleteTrail", "CreateLoginProfile",
    },
    "GuardDuty-style (11 rules)": {
        "CreateLoginProfile", "UpdateLoginProfile",
        "AttachUserPolicy", "AttachRolePolicy", "AttachGroupPolicy",
        "PutUserPolicy", "PutRolePolicy",
        "CreatePolicyVersion", "SetDefaultPolicyVersion",
        "StopLogging", "DeleteTrail",
    },
    "Post-incident rules (all 23)": {
        "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
        "AttachUserPolicy", "AttachRolePolicy", "AttachGroupPolicy",
        "PutUserPolicy", "PutRolePolicy", "PutGroupPolicy",
        "CreatePolicyVersion", "SetDefaultPolicyVersion", "AddUserToGroup",
        "CreateUser", "CreateRole", "UpdateAssumeRolePolicy",
        "StopLogging", "DeleteTrail", "UpdateTrail", "PutEventSelectors",
        "GetSecretValue", "GetPasswordData", "PutBucketPolicy", "DeleteBucketPolicy",
    },
}

N_BOOTSTRAP = 1000
SEED = 42


def build_sessions(df, group_col, label_col="session_label"):
    sessions = {}
    for key, grp in df.groupby(group_col):
        sessions[key] = {
            "events": list(grp["event_name"]),
            "true_label": int(grp[label_col].max()),
        }
    return sessions


def rule_predict(sessions, rule_set):
    return {k: int(any(e in rule_set for e in s["events"])) for k, s in sessions.items()}


def score(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f


def bootstrap_ci(sessions, preds, n=N_BOOTSTRAP, seed=SEED):
    """95% CI on F1 via resampling sessions with replacement."""
    rng = random.Random(seed)
    keys = list(sessions.keys())
    f1s = []
    for _ in range(n):
        sample = [rng.choice(keys) for _ in keys]
        y_true = [sessions[k]["true_label"] for k in sample]
        y_pred = [preds[k] for k in sample]
        _, _, f = score(y_true, y_pred)
        f1s.append(f)
    f1s.sort()
    lo = f1s[int(0.025 * n)]
    hi = f1s[int(0.975 * n)]
    return lo, hi


def evaluate(sessions, name_suffix, with_ci=False):
    y_true_all = [s["true_label"] for s in sessions.values()]
    n_attack = sum(y_true_all)
    n_benign = len(y_true_all) - n_attack
    print(f"\n{'=' * 60}")
    print(f"EVALUATION ON {name_suffix}")
    print(f"{'=' * 60}")
    print(f"Sessions: {len(sessions)}  |  Attack: {n_attack}  |  Benign: {n_benign}\n")

    results = []
    for rule_name, rule_set in RULES.items():
        preds = rule_predict(sessions, rule_set)
        y_true = [sessions[k]["true_label"] for k in sessions]
        y_pred = [preds[k] for k in sessions]
        p, r, f = score(y_true, y_pred)

        tp = sum(1 for k in sessions if preds[k] == 1 and sessions[k]["true_label"] == 1)
        fp = sum(1 for k in sessions if preds[k] == 1 and sessions[k]["true_label"] == 0)
        fn = sum(1 for k in sessions if preds[k] == 0 and sessions[k]["true_label"] == 1)

        line = f"{rule_name}\n  P={p:.3f}  R={r:.3f}  F1={f:.3f}  (TP={tp} FP={fp} FN={fn})"
        if with_ci and len(sessions) > 1:
            lo, hi = bootstrap_ci(sessions, preds)
            line += f"  |  95% CI on F1: [{lo:.3f}, {hi:.3f}]"
        print(line)
        results.append({"rule_set": rule_name, "precision": p, "recall": r, "f1": f})
    return results


def main():
    print("Loading synthetic_cloudtrail.csv ...")
    df_syn = pd.read_csv("synthetic_cloudtrail.csv")
    sessions_syn = build_sessions(df_syn, group_col="username")
    evaluate(sessions_syn, "SYNTHETIC DATA (train distribution)")

    print("\n\nLoading real_dataset_combined.csv ...")
    df_real = pd.read_csv("real_dataset_combined.csv")
    sessions_real = build_sessions(df_real, group_col="session_id")
    real_results = evaluate(sessions_real, "REAL DATA (held-out, session_id-corrected)", with_ci=True)

    print(f"\n{'=' * 60}")
    print("SUMMARY -- real held-out test set (the number that matters)")
    print(f"{'=' * 60}")
    print(f"{'Method':<32} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for r in real_results:
        print(f"{r['rule_set']:<32} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}")
    print(f"{'GNN + Sequence ensemble (ours)':<32} {'???':>10} {'???':>8} {'???':>8}")


if __name__ == "__main__":
    main()
