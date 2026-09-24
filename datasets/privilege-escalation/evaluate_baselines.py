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
    # Curated by reading AWS's public GuardDuty finding-type docs and picking related IAM
    # actions -- NOT validated against real GuardDuty output. This project's data collection
    # (see stratus_collection/README.md) never enabled GuardDuty during detonation, so there is
    # no real GuardDuty baseline to compare against; naming this "GuardDuty-style" would overclaim
    # what it actually is.
    "Curated IAM rule baseline (11 rules)": {
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
    evaluate(sessions_real, "REAL DATA, COMBINED dev+test (397 sessions) -- NOT comparable to the "
                             "test-only numbers below (see Fix A note)", with_ci=True)

    # The SUMMARY table below must be computed on real_dataset_test.csv alone (238 sessions), not
    # real_dataset_combined.csv (397 dev+test) -- this is exactly the population mismatch flagged as
    # "Fix A" elsewhere in this project's evaluation history (see docs/PROJECT_STATUS_REPORT.md
    # §6.17): comparing a rule baseline computed on 397 sessions against a model scored on 238
    # test sessions produces two numbers that LOOK comparable but are not measuring the same thing.
    # The classical-ML and ensemble rows below are all test-only (238 sessions), so the rule rows
    # must be too.
    print("\n\nLoading real_dataset_test.csv (238 sessions, for the apples-to-apples summary below) ...")
    df_test = pd.read_csv("real_dataset_test.csv")
    sessions_test = build_sessions(df_test, group_col="session_id")
    test_results = evaluate(sessions_test, "REAL DATA, TEST ONLY (238 sessions) -- matches the "
                                            "population every other row in the summary uses", with_ci=True)

    print(f"\n{'=' * 60}")
    print("SUMMARY -- real held-out TEST set only, 238 sessions (the number that matters)")
    print(f"{'=' * 60}")
    print(f"{'Method':<32} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for r in test_results:
        print(f"{r['rule_set']:<32} {r['precision']:>10.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}")
    # Classical-ML baselines (evaluate_ml_baselines.py) -- Random Forest / XGBoost trained on
    # feature_engine9's own temporal feature columns (train-on-synthetic, same paradigm as the
    # GNN/LSTM models), threshold swept on real_dataset_dev.csv only, applied once to test.
    # Both UNDERPERFORM the rule baseline above despite using real ML on the same features --
    # naive supervised learning on tabular features does not transfer from synthetic to real
    # attacks, which is the point: it motivates the ensemble's rule-injected + sequence approach
    # rather than a plain classifier on the same columns.
    print(f"{'Random Forest (temporal features)':<32} {0.508:>10.3f} {0.980:>8.3f} {0.669:>8.3f}")
    print(f"{'XGBoost (temporal features)':<32} {0.424:>10.3f} {1.000:>8.3f} {0.595:>8.3f}")
    # GNN + Sequence ensemble (ensemble.py, weight_gnn=weight_lstm=0.5, default
    # SESSION_ALERT_THRESHOLD=5.5), evaluated on real_dataset_test.csv's 238 sessions
    # (100 attack): P=0.845 R=0.980 F1=0.907. Paired bootstrap vs the curated-rule
    # row above, same 238 sessions: F1 gap +0.160, 95% CI [+0.091, +0.234], p<0.0001
    # -- significant. A weight/combination-strategy sweep (max, geometric mean,
    # impact-weighted) tuned only on real_dataset_dev.csv found nothing that beat the
    # 0.5/0.5 default by more than dev-set noise, so these are the unmodified defaults,
    # not a cherry-picked config (see ensemble.py's SESSION_ALERT_THRESHOLD comment).
    print(f"{'GNN + Sequence ensemble (ours)':<32} {0.845:>10.3f} {0.980:>8.3f} {0.907:>8.3f}")


if __name__ == "__main__":
    main()
