"""
Augment train_temporal.csv with diverse synthetic PE chains (CloudTrail-like rows).

These are labeled log sequences for detector training, not exploit code.
Users are prefixed syn: and stay out of the bert-jan test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leakage_guard import assert_no_heldout  # noqa: E402

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "lstm" / "train_temporal.csv"
VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab.json"
OUT = ROOT / "data" / "lstm" / "train_temporal_aug.csv"

META = {"log_id", "username", "timestamp", "label", "event_name_idx"}

# (event_name, label) steps. Recon stays 0; PE writes are 1 — same as Invictus.
CHAIN_TYPES: dict[str, list[tuple[str, int]]] = {
    "iam_user_persist": [
        ("ListUsers", 0),
        ("GetUser", 0),
        ("CreateUser", 1),
        ("CreateLoginProfile", 1),
        ("CreateAccessKey", 1),
        ("AttachUserPolicy", 1),
        ("ListAccessKeys", 0),
    ],
    "role_assume": [
        ("GetCallerIdentity", 0),
        ("ListRoles", 0),
        ("CreateRole", 1),
        ("AttachRolePolicy", 1),
        ("PutRolePolicy", 1),
        ("AssumeRole", 1),
        ("GetCallerIdentity", 0),
        ("GetRole", 0),
    ],
    "policy_version": [
        ("ListPolicies", 0),
        ("GetPolicy", 0),
        ("GetPolicyVersion", 0),
        ("CreatePolicyVersion", 1),
        ("SetDefaultPolicyVersion", 1),
        ("AttachUserPolicy", 1),
    ],
    "secrets_loot": [
        ("CreateSecret", 1),
        ("ListSecrets", 0),
        ("DescribeSecret", 0),
        ("GetSecretValue", 1),
        ("GetSecretValue", 1),
        ("Decrypt", 1),
        ("GetSecretValue", 1),
    ],
    "ec2_password": [
        ("DescribeInstances", 0),
        ("DescribeInstanceAttribute", 0),
        ("GetPasswordData", 1),
        ("GetCallerIdentity", 0),
    ],
    "trail_evasion": [
        ("DescribeTrails", 0),
        ("GetTrailStatus", 0),
        ("StopLogging", 1),
        ("PutEventSelectors", 1),
        ("DeleteTrail", 1),
        ("GetCallerIdentity", 0),
    ],
    "instance_profile": [
        ("ListRoles", 0),
        ("CreateRole", 1),
        ("CreateInstanceProfile", 0),
        ("AddRoleToInstanceProfile", 0),
        ("RunInstances", 0),
        ("AttachRolePolicy", 1),
    ],
    "access_key_existing": [
        ("ListUsers", 0),
        ("ListAccessKeys", 0),
        ("CreateAccessKey", 1),
        ("GetUser", 0),
    ],
    "bucket_policy": [
        ("ListBuckets", 0),
        ("GetBucketPolicy", 0),
        ("PutBucketPolicy", 1),
        ("GetBucketAcl", 0),
    ],
    "update_trust": [
        ("ListRoles", 0),
        ("GetRole", 0),
        ("UpdateAssumeRolePolicy", 1),
        ("AssumeRole", 1),
        ("GetCallerIdentity", 0),
    ],
    "recon_then_role": [
        ("DescribeVpcs", 0),
        ("DescribeSubnets", 0),
        ("DescribeSecurityGroups", 0),
        ("DescribeInstances", 0),
        ("GetCallerIdentity", 0),
        ("ListRoles", 0),
        ("GetRole", 0),
        ("CreateRole", 1),
        ("AttachRolePolicy", 1),
        ("PutRolePolicy", 1),
        ("AssumeRole", 1),
        ("GetSecretValue", 1),
        ("DescribeTrails", 0),
        ("StopLogging", 1),
    ],
    "recon_then_user": [
        ("ListUsers", 0),
        ("GetAccountSummary", 0),
        ("ListGroups", 0),
        ("GetCallerIdentity", 0),
        ("CreateUser", 1),
        ("AddUserToGroup", 1),
        ("AttachUserPolicy", 1),
        ("CreateAccessKey", 1),
        ("CreateLoginProfile", 1),
        ("GetSecretValue", 1),
    ],
}

IAM_WRITE = {
    "CreateUser",
    "CreateRole",
    "CreateAccessKey",
    "CreateLoginProfile",
    "CreatePolicyVersion",
    "AttachUserPolicy",
    "AttachRolePolicy",
    "PutRolePolicy",
    "PutBucketPolicy",
    "AddUserToGroup",
    "UpdateAssumeRolePolicy",
    "SetDefaultPolicyVersion",
    "StopLogging",
    "DeleteTrail",
    "PutEventSelectors",
}
PERM = {
    "AttachUserPolicy",
    "AttachRolePolicy",
    "PutRolePolicy",
    "PutBucketPolicy",
    "CreatePolicyVersion",
    "SetDefaultPolicyVersion",
    "UpdateAssumeRolePolicy",
    "AddUserToGroup",
}
HIGH_RISK = IAM_WRITE | {"GetSecretValue", "GetPasswordData", "Decrypt"}
USERS_PER_TYPE = 12
SEED = 7


def apply_api_flags(row: pd.Series, name: str, label: int, rng: np.random.RandomState) -> pd.Series:
    out = row.copy()
    out["is_iam_event"] = 1 if (
        name.startswith(("Create", "Attach", "PutRole", "ListUser", "GetUser", "GetRole", "Assume"))
        or name in IAM_WRITE
        or "Policy" in name
        or name in {"GetCallerIdentity", "AddUserToGroup"}
    ) else int(out.get("is_iam_event", 0) > 0.5)
    out["is_write_action"] = 1 if name in IAM_WRITE or name.startswith(("Create", "Put", "Delete", "Attach", "Update")) else int(out.get("is_write_action", 0) > 0.5)
    out["is_permission_modification"] = 1 if name in PERM else 0
    out["is_recon_action"] = 1 if name.startswith(("Describe", "List", "Get")) and name not in HIGH_RISK else 0
    out["is_get_caller_identity"] = 1 if name == "GetCallerIdentity" else 0
    out["is_create_key"] = 1 if name == "CreateAccessKey" else 0
    out["is_secrets_or_kms"] = 1 if name in {"GetSecretValue", "Decrypt", "ListSecrets", "DescribeSecret"} else 0
    out["is_defense_evasion"] = 1 if name in {"StopLogging", "DeleteTrail", "PutEventSelectors"} else 0
    if name in HIGH_RISK:
        out["action_risk_prior"] = float(np.clip(0.55 + 0.25 * rng.rand(), 0.4, 1.0))
    else:
        out["action_risk_prior"] = float(np.clip(0.08 + 0.2 * rng.rand(), 0.05, 0.45))
    out["no_mfa"] = 1 if rng.rand() < (0.85 if label else 0.35) else 0
    out["mfa_absent"] = int(out["no_mfa"])
    out["is_public_ip"] = 1 if rng.rand() < 0.55 else 0
    out["is_off_hours"] = 1 if rng.rand() < 0.4 else 0
    out["label"] = int(label)
    return out


def main() -> None:
    rng = np.random.RandomState(SEED)
    df = pd.read_csv(SRC)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    missing = sorted({n for steps in CHAIN_TYPES.values() for n, _ in steps if n not in vocab})
    if missing:
        raise SystemExit(f"vocab missing: {missing}")

    feat_cols = [c for c in df.columns if c not in META]
    by_idx: dict[int, pd.DataFrame] = {
        int(i): g[feat_cols] for i, g in df.groupby("event_name_idx")
    }
    pos_pool = df.loc[df["label"] == 1, feat_cols]
    neg_pool = df.loc[df["label"] == 0, feat_cols]

    rows = []
    log_i = 0
    t0 = pd.Timestamp("2024-09-01T00:00:00Z")
    n_users = 0
    for ctype, steps in CHAIN_TYPES.items():
        for u in range(USERS_PER_TYPE):
            n_users += 1
            username = f"syn:{ctype}-{u:03d}"
            t = t0 + pd.Timedelta(days=int(n_users), hours=int(rng.randint(0, 23)))
            extra_recon = rng.randint(0, 4)
            recon_names = ["DescribeInstances", "DescribeVpcs", "ListBuckets", "GetCallerIdentity", "DescribeRegions"]
            seq = list(steps)
            for _ in range(extra_recon):
                seq.insert(rng.randint(0, max(len(seq) - 1, 1)), (recon_names[rng.randint(len(recon_names))], 0))
            for name, lab in seq:
                idx = int(vocab[name])
                pool = by_idx.get(idx)
                if pool is not None and len(pool):
                    base = pool.iloc[rng.randint(len(pool))].copy()
                else:
                    base = (pos_pool if lab else neg_pool).iloc[rng.randint(len(pos_pool if lab else neg_pool))].copy()
                rec = apply_api_flags(base, name, lab, rng)
                rec["event_name_idx"] = idx
                rec["username"] = username
                rec["timestamp"] = t
                rec["log_id"] = f"synthetic_pe_chains.csv:{log_i}"
                rec["label"] = lab
                rows.append(rec)
                log_i += 1
                t = t + pd.Timedelta(seconds=int(rng.randint(4, 45)))

    syn = pd.DataFrame(rows)
    syn = syn[df.columns]
    out = pd.concat([df, syn], ignore_index=True)
    # The augmented set inherits whatever the base set contained, so it needs
    # the same guarantee -- an earlier version of this file carried 2,858
    # held-out rows purely because train_temporal.csv did.
    assert_no_heldout(out, "train_temporal_aug")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(f"base={len(df)} syn_events={len(syn)} syn_users={n_users} syn_pos={int(syn['label'].sum())} total={len(out)}")
    print("chain types", list(CHAIN_TYPES))


if __name__ == "__main__":
    main()
