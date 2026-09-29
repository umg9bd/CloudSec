"""
Build the v6.2 LSTM training CSV from cloudtrail_temporal_final.csv (synthetic only).

Fixes from the v6 audit (v6.1):
  - drop label-leaking columns (no_mfa, mfa_absent, params_length_normalized)
  - make per-name flags constant per event name (AssumeRole is_write_action was 1 only on attacks)
  - add read-only credential-theft chains (SSM / Secrets Manager / EC2 password) plus benign
    sessions calling the same APIs; every feature of a new row is drawn independently of its label
v6.2 (v6.1 rows are reproduced unchanged, new sessions appended):
  - long busy sessions with theft / escalation calls spread among routine reads, plus busy benign twins
  - benign inventory scans; routine reads drawn from COMMON_READS (generic AWS read-only calls)

Labels follow the Invictus rule: recon/discovery = 0, only the theft call = 1.
Chain templates follow public Stratus Red Team credential-access techniques. The Invictus
incident used for dev/test also contains Stratus Red Team activity - disclose this in the paper.
The v6.2 busy-session templates were added after reviewing v6.1 errors on Invictus dev and test.
Invictus rows are read only for fixed per-name encodings of names absent from the synthetic
data (never labels or per-event values).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "lstm" / "cloudtrail_temporal_final.csv"
REAL = ROOT / "data" / "lstm" / "train_temporal.csv"
V5_VOCAB = ROOT / "data" / "lstm" / "event_name_vocab.json"
OUT = ROOT / "data" / "lstm" / "cloudtrail_temporal_v6_2.csv"
MANIFEST = ROOT / "data" / "lstm" / "cloudtrail_temporal_v6_2.manifest.json"

SEED = 61
LEAKY_FEATURES = ["no_mfa", "mfa_absent", "params_length_normalized"]
# Fixed per event name (verified identical in synthetic and Invictus on the 47 shared names).
NAME_COLS = [
    "action_risk_prior",
    "event_source_idx",
    "is_write_action",
    "read_only_absent",
    "is_iam_event",
    "is_recon_action",
    "is_defense_evasion",
    "is_get_caller_identity",
    "is_create_key",
    "is_secrets_or_kms",
    "is_permission_modification",
]
# Constant per user -> copied from one donor user per new session.
PRINCIPAL_COLS = [
    "principal_type_prior_risk",
    "principal_type_idx",
    "has_access_key",
    "is_malicious_user_agent",
    "is_public_ip",
    "is_non_default_region",
]
TIME_COLS = ["time_sin", "time_cos", "is_weekend", "is_off_hours"]
# Per event -> sampled from same-name synthetic rows of either label.
EVENT_COLS = [
    "has_error",
    "is_access_denied",
    "targets_sensitive_resource",
    "policy_statement_count_normalized",
    "has_wildcard_action",
    "has_wildcard_resource",
    "privileged_action_reach",
    "action_velocity",
    "is_new_action",
    "session_duration_normalized",
    "events_per_minute_normalized",
]

# steps: (event_name, min_count, max_count, label)
TEMPLATES: dict[str, dict] = {
    "ssm_securestring_theft": {
        "technique": "aws.credential-access.ssm-retrieve-securestring-parameters",
        "users": 50,
        "steps": [("DescribeParameters", 1, 3, 0), ("GetParameters", 2, 5, 1)],
    },
    "secrets_bulk_theft": {
        "technique": "aws.credential-access.secretsmanager-retrieve-secrets",
        "users": 30,
        "steps": [("ListSecrets", 1, 1, 0), ("GetSecretValue", 3, 10, 1)],
    },
    "ec2_password_theft": {
        "technique": "aws.credential-access.ec2-get-password-data",
        "users": 30,
        "steps": [("DescribeInstances", 0, 1, 0), ("GetPasswordData", 3, 10, 1)],
    },
    "app_config_load": {"users": 40, "steps": [("GetParameters", 1, 4, 0)]},
    "admin_ssm_browse": {"users": 40, "steps": [("DescribeParameters", 1, 2, 0), ("GetParameter", 0, 2, 0)]},
    "app_secret_fetch": {"users": 40, "steps": [("GetSecretValue", 1, 4, 0)]},
    "admin_secret_browse": {
        "users": 40,
        "steps": [("ListSecrets", 1, 1, 0), ("DescribeSecret", 1, 2, 0), ("GetSecretValue", 0, 1, 0)],
    },
    "admin_password_recovery": {"users": 40, "steps": [("DescribeInstances", 1, 1, 0), ("GetPasswordData", 1, 1, 0)]},
}
STEP_GAP_MEDIAN = 2.0

# Generic read-only AWS calls (picked from AWS service docs, not from Invictus frequencies).
# Names without a known per-name encoding are skipped and listed in the manifest.
COMMON_READS = """
DescribeInstances DescribeVpcs DescribeSubnets DescribeSecurityGroups DescribeRouteTables DescribeNatGateways
DescribeInternetGateways DescribeNetworkAcls DescribeNetworkInterfaces DescribeAddresses DescribeAvailabilityZones
DescribeRegions DescribeImages DescribeSnapshots DescribeVolumes DescribeKeyPairs DescribeAccountAttributes
DescribeVpcAttribute DescribeInstanceStatus DescribeTags DescribeDBInstances DescribeDBClusters DescribeDBSnapshots
DescribeDBEngineVersions DescribeDBSubnetGroups ListBuckets GetBucketAcl GetBucketPolicy GetBucketLocation
GetBucketVersioning GetBucketLogging GetBucketEncryption ListObjects GetUser GetRole ListRoles ListUsers ListPolicies
ListAttachedRolePolicies ListRolePolicies GetPolicy GetPolicyVersion ListGroups GetAccountSummary ListKeys DescribeKey
ListAliases Decrypt GenerateDataKey Encrypt DescribeParameters GetParameter DescribeInstanceInformation ListCommands
ListSecrets DescribeSecret GetResourcePolicy DescribeTrails GetTrailStatus LookupEvents DescribeAlarms GetMetricData
DescribeLogGroups GetCallerIdentity ListFunctions GetFunction ListTagsForResource
""".split()

# v6.2 busy sessions: `background` routine reads (label 0, per-session Dirichlet mix over
# COMMON_READS) with `inserts` (event_name, min_count, max_count, label) at random positions.
BUSY_TEMPLATES: dict[str, dict] = {
    "busy_secret_theft": {
        "users": 12,
        "background": (60, 150),
        "inserts": [("GetSecretValue", 8, 25, 1), ("GetParameters", 0, 6, 1)],
    },
    "busy_escalation_theft": {
        "users": 12,
        "background": (60, 150),
        "inserts": [
            ("CreateRole", 1, 3, 1),
            ("AttachRolePolicy", 1, 3, 1),
            ("PutRolePolicy", 0, 2, 1),
            ("CreateUser", 0, 2, 1),
            ("CreateAccessKey", 0, 2, 1),
            ("GetSecretValue", 5, 20, 1),
        ],
    },
    "busy_benign_admin": {
        "users": 12,
        "background": (60, 150),
        "inserts": [("GetSecretValue", 0, 4, 0), ("GetParameters", 0, 3, 0), ("GetParameter", 0, 4, 0)],
    },
    "busy_benign_ops": {
        "users": 12,
        "background": (60, 150),
        "inserts": [("PutParameter", 0, 3, 0), ("StartInstances", 0, 2, 0), ("StopInstances", 0, 2, 0)],
    },
    "inventory_scan": {"users": 40, "background": (10, 40), "inserts": [], "gap_median": STEP_GAP_MEDIAN},
}
BUSY_GAP_MEDIAN = 6.0  # same for attack and benign busy sessions
DIRICHLET_ALPHA = 0.3


def load_synthetic(inv_vocab: dict[int, str]) -> pd.DataFrame:
    df = pd.read_csv(SRC).drop_duplicates(subset=["log_id"], keep="last").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["event_name"] = df["event_name_idx"].astype(int).map(inv_vocab)
    assert df["event_name"].notna().all(), "synthetic event_name_idx outside v5 vocab"
    return df.drop(columns=LEAKY_FEATURES)


def name_table(syn: pd.DataFrame, inv_vocab: dict[int, str]) -> pd.DataFrame:
    """Per-name encodings: synthetic mode first, Invictus mode only for names absent from synthetic."""
    mode = lambda g: g.mode().iloc[0]
    table = syn.groupby("event_name")[NAME_COLS].agg(mode)
    real = pd.read_csv(REAL)
    real = real[real["username"].str.startswith("inv:")].copy()
    real["event_name"] = real["event_name_idx"].astype(int).map(inv_vocab)
    extra = real[~real["event_name"].isin(table.index)].groupby("event_name")[NAME_COLS].agg(mode)
    return pd.concat([table, extra])


def normalize_name_cols(syn: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = syn.copy()
    fixed = table.loc[out["event_name"], NAME_COLS].to_numpy()
    changed = out[NAME_COLS].to_numpy() != fixed
    report = {}
    for j, c in enumerate(NAME_COLS):
        names = sorted(out.loc[changed[:, j], "event_name"].unique())
        if names:
            report[c] = {"names": names, "rows": int(changed[:, j].sum())}
    out[NAME_COLS] = fixed
    assert set(report) <= {"is_write_action"} and report.get("is_write_action", {}).get("names", ["AssumeRole"]) == ["AssumeRole"], report
    return out, report


def time_feats(ts: pd.Timestamp) -> dict[str, float]:
    h = ts.hour
    return {
        "time_sin": float(np.sin(2 * np.pi * h / 24)),
        "time_cos": float(np.cos(2 * np.pi * h / 24)),
        "is_weekend": int(ts.dayofweek >= 5),
        "is_off_hours": int(h < 9 or h >= 19),
    }


def generate_chains(syn: pd.DataFrame, table: pd.DataFrame, vocab: dict[str, int], reads: list[str]) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    users = syn["username"].unique()
    by_user = {u: g for u, g in syn.groupby("username")}
    by_name = {n: g for n, g in syn.groupby("event_name")}
    read_pool = syn[syn["is_write_action"] == 0]
    starts = syn["timestamp"].to_numpy()
    rows, log_i = [], 0

    def emit(user: str, log_file: str, make_seq, gap_median: float) -> None:
        nonlocal log_i
        donor = by_user[users[rng.randint(len(users))]]
        t = pd.Timestamp(starts[rng.randint(len(starts))]) + pd.Timedelta(seconds=round(float(rng.uniform(0, 1800)), 3))
        for name, lab in make_seq():
            pool = by_name.get(name, read_pool)
            ev = pool.iloc[rng.randint(len(pool))]
            pr = donor.iloc[rng.randint(len(donor))]
            rec = {c: table.at[name, c] for c in NAME_COLS}
            rec.update({c: pr[c] for c in PRINCIPAL_COLS})
            rec.update({c: ev[c] for c in EVENT_COLS})
            rec.update(time_feats(t))
            rec.update(
                log_id=f"{log_file}:{log_i}",
                username=user,
                timestamp=t,
                event_name=name,
                event_name_idx=int(vocab[name]),
                label=int(lab),
            )
            rows.append(rec)
            log_i += 1
            # gap distribution depends on the template family, never on the label
            t = t + pd.Timedelta(seconds=round(float(np.clip(np.exp(rng.normal(np.log(gap_median), 1.0)), 0.05, 120)), 3))

    def step_seq(spec: dict):
        seq = [(n, lab) for n, lo, hi, lab in spec["steps"] for _ in range(rng.randint(lo, hi + 1))]
        return seq or [(spec["steps"][-1][0], spec["steps"][-1][3])]

    def busy_seq(spec: dict):
        mix = rng.dirichlet(np.full(len(reads), DIRICHLET_ALPHA))
        n_bg = rng.randint(spec["background"][0], spec["background"][1] + 1)
        seq = [(reads[i], 0) for i in rng.choice(len(reads), size=n_bg, p=mix)]
        for name, lo, hi, lab in spec["inserts"]:
            for _ in range(rng.randint(lo, hi + 1)):
                seq.insert(rng.randint(0, len(seq) + 1), (name, lab))
        return seq

    for tname, spec in TEMPLATES.items():
        for u in range(spec["users"]):
            emit(f"syn61-{tname}-{u:03d}", "synthetic_v6_1_chains.csv", lambda: step_seq(spec), STEP_GAP_MEDIAN)
    for tname, spec in BUSY_TEMPLATES.items():
        for u in range(spec["users"]):
            emit(
                f"syn62-{tname}-{u:03d}",
                "synthetic_v6_2_chains.csv",
                lambda: busy_seq(spec),
                spec.get("gap_median", BUSY_GAP_MEDIAN),
            )
    return pd.DataFrame(rows).assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True).dt.round("ms"))


def feat_auc(df: pd.DataFrame, feats: list[str]) -> dict[str, float]:
    y = df["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return {}
    return {c: round(float(roc_auc_score(y, df[c])), 3) if df[c].nunique() > 1 else 0.5 for c in feats}


def within_name_check(new: pd.DataFrame, feats: list[str]) -> dict[str, float]:
    """Max |AUC - 0.5| of non-time features between labels, per name, on the new rows only."""
    out = {}
    for name, g in new.groupby("event_name"):
        if g["label"].nunique() < 2:
            continue
        aucs = feat_auc(g, [c for c in feats if c not in TIME_COLS])
        worst = max(aucs, key=lambda c: abs(aucs[c] - 0.5))
        out[name] = {"worst_feature": worst, "auc": aucs[worst], "n_pos": int(g["label"].sum()), "n_neg": int((g["label"] == 0).sum())}
    return out


def main() -> None:
    vocab = {str(k): int(v) for k, v in json.loads(V5_VOCAB.read_text(encoding="utf-8")).items()}
    inv_vocab = {v: k for k, v in vocab.items()}
    names = {n for s in TEMPLATES.values() for n, *_ in s["steps"]} | {
        n for s in BUSY_TEMPLATES.values() for n, *_ in s["inserts"]
    }
    missing = sorted(n for n in names if n not in vocab)
    assert not missing, f"template names missing from v5 vocab: {missing}"

    syn = load_synthetic(inv_vocab)
    table = name_table(syn, inv_vocab)
    syn, name_fix = normalize_name_cols(syn, table)
    reads = [n for n in COMMON_READS if n in vocab and n in table.index]
    skipped = [n for n in COMMON_READS if n not in reads]
    new = generate_chains(syn, table, vocab, reads)

    feats = [c for c in syn.columns if c not in {"log_id", "username", "timestamp", "label", "event_name_idx", "event_name"}]
    assert len(feats) == 32 and set(feats) == set(NAME_COLS + PRINCIPAL_COLS + TIME_COLS + EVENT_COLS), feats
    cols = ["log_id", "username", "timestamp", *feats, "event_name_idx", "event_name", "label"]
    out = pd.concat([syn[cols], new[cols]], ignore_index=True)
    assert out["log_id"].is_unique and str(out["timestamp"].dtype).startswith("datetime64"), out["timestamp"].dtype
    out.to_csv(OUT, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")

    check = within_name_check(new, feats)
    manifest = {
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "label_rule": "Invictus rule: recon/discovery = 0; PE writes and credential-theft calls = 1",
        "dropped_features": LEAKY_FEATURES,
        "name_col_fix": name_fix,
        "n_rows": int(len(out)),
        "n_rows_base": int(len(syn)),
        "n_rows_new": int(len(new)),
        "n_users": int(out["username"].nunique()),
        "n_pos": int(out["label"].sum()),
        "new_by_template": {
            t: {
                "technique": s.get("technique", "attack" if any(x[-1] for x in s.get("steps", s.get("inserts", []))) else "benign"),
                "users": s["users"],
                "events": int(new["username"].str.startswith(f"{p}-{t}-").sum()),
                "pos": int(new.loc[new["username"].str.startswith(f"{p}-{t}-"), "label"].sum()),
            }
            for p, group in (("syn61", TEMPLATES), ("syn62", BUSY_TEMPLATES))
            for t, s in group.items()
        },
        "common_reads_used": reads,
        "common_reads_skipped": skipped,
        "within_name_label_check_new_rows": check,
        "single_feature_auc_all_rows": feat_auc(out, feats),
        "disclosure": (
            "Chain templates follow public Stratus Red Team credential-access techniques; the "
            "Invictus dev/test incident also contains Stratus Red Team activity. v6.2 busy-session "
            "templates were added after reviewing v6.1 errors on Invictus dev and test."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {OUT} rows={len(out)} (base={len(syn)} new={len(new)}) pos={int(out['label'].sum())}")
    print(f"name-col fix: {name_fix}")
    print(f"common reads used={len(reads)} skipped={skipped}")
    print("new rows by template:", json.dumps(manifest["new_by_template"]))
    print("within-name check (new rows, non-time features):")
    for n, r in check.items():
        print(f"  {n}: worst={r['worst_feature']} auc={r['auc']} pos/neg={r['n_pos']}/{r['n_neg']}")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
