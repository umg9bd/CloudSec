"""
Synthetic CloudTrail session generator -- extracted from explore.ipynb
(Section 2) into a standalone, re-runnable script, with two distribution
patches applied against the real data:

  target_resource:    was ~94% null (only attack-chain steps got one).
                       Real data (recomputed against the full combined
                       real dataset, not the old invictus-only estimate)
                       is only ~28% null. Benign/recon events now get a
                       plausible resource name at a rate calibrated per
                       event_source to match.

  mfa_authenticated:   was 0% null (every row got "true"/"false").
                       Real data is ~69% null, and when populated is
                       overwhelmingly "False" (96.4%) -- MFA-authenticated
                       sessions are rare in this data. Patched to match.

Everything else (attack chains, recon events, benign event pool, error
code distributions) is unchanged from the original notebook logic, so
this remains comparable to previous runs except for these two fields.

Usage:
    python generate_synthetic_data.py
"""

import random
import string
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

random.seed(99)

# ── Attack chain library (unchanged from explore.ipynb) ───────────────────────
ATTACK_CHAINS = {
    "create_role_attach_managed_policy": [
        {"event_name": "CreateRole",        "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "AttachRolePolicy",  "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
    ],
    "create_role_inline_policy": [
        {"event_name": "CreateRole",    "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "PutRolePolicy", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
    ],
    "create_user_accesskey_policy": [
        {"event_name": "CreateUser",       "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "user"},
        {"event_name": "CreateAccessKey",  "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "user"},
        {"event_name": "AttachUserPolicy", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "user"},
    ],
    "create_user_console_access": [
        {"event_name": "CreateUser",         "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "user"},
        {"event_name": "CreateLoginProfile", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "user"},
        {"event_name": "AttachUserPolicy",   "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "user"},
    ],
    "add_user_to_admin_group": [
        {"event_name": "AddUserToGroup", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "group"},
    ],
    "update_role_inline_policy": [
        {"event_name": "PutRolePolicy", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
    ],
    "create_policy_version": [
        {"event_name": "CreatePolicyVersion",     "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "policy"},
        {"event_name": "SetDefaultPolicyVersion", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "policy"},
    ],
    "update_assume_role_policy": [
        {"event_name": "UpdateAssumeRolePolicy", "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "AssumeRole",             "event_source": "sts.amazonaws.com",  "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
    ],
    "full_kill_chain": [
        {"event_name": "CreateRole",       "event_source": "iam.amazonaws.com",            "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "AttachRolePolicy", "event_source": "iam.amazonaws.com",            "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "GetSecretValue",   "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access",    "read_only": True,  "target_key": "secret"},
        {"event_name": "PutBucketPolicy",  "event_source": "s3.amazonaws.com",             "attack_technique": "exfiltration",         "read_only": False, "target_key": "bucket"},
        {"event_name": "StopLogging",      "event_source": "cloudtrail.amazonaws.com",     "attack_technique": "defense-evasion",      "read_only": False, "target_key": "trail", "error_probability": 0.4},
    ],
    "ec2_password_data": [
        {"event_name": "CreateRole",      "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "PutRolePolicy",   "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "GetPasswordData", "event_source": "ec2.amazonaws.com", "attack_technique": "credential-access",    "read_only": True,  "target_key": "instance", "error_probability": 0.9},
    ],
}

RECON_EVENTS = [
    ("GetAccountSummary",             "iam.amazonaws.com",            True),
    ("ListUsers",                     "iam.amazonaws.com",            True),
    ("ListRoles",                     "iam.amazonaws.com",            True),
    ("ListGroups",                    "iam.amazonaws.com",            True),
    ("ListPolicies",                  "iam.amazonaws.com",            True),
    ("GetAccountAuthorizationDetails","iam.amazonaws.com",            True),
    ("ListAttachedUserPolicies",      "iam.amazonaws.com",            True),
    ("ListAttachedRolePolicies",      "iam.amazonaws.com",            True),
    ("ListBuckets",                   "s3.amazonaws.com",             True),
    ("DescribeInstances",             "ec2.amazonaws.com",            True),
    ("ListSecrets",                   "secretsmanager.amazonaws.com", True),
    ("DescribeTrails",                "cloudtrail.amazonaws.com",     True),
    ("GetCallerIdentity",             "sts.amazonaws.com",            True),
    ("ListAccessKeys",                "iam.amazonaws.com",            True),
]

BENIGN_EVENTS_WEIGHTED = [
    ("GetBucketLogging",              "s3.amazonaws.com",             True,  4),
    ("GetBucketPolicy",               "s3.amazonaws.com",             True,  4),
    ("GetBucketAcl",                  "s3.amazonaws.com",             True,  3),
    ("DescribeSecurityGroups",        "ec2.amazonaws.com",            True,  5),
    ("DescribeVpcs",                  "ec2.amazonaws.com",            True,  4),
    ("DescribeSubnets",               "ec2.amazonaws.com",            True,  4),
    ("DescribeInstances",             "ec2.amazonaws.com",            True,  5),
    ("GetRegionOptStatus",            "account.amazonaws.com",        True,  2),
    ("DescribeDBInstances",           "rds.amazonaws.com",            True,  3),
    ("ListKeys",                      "kms.amazonaws.com",            True,  4),
    ("DescribeKey",                   "kms.amazonaws.com",            True,  3),
    ("GetParameter",                  "ssm.amazonaws.com",            True,  5),
    ("DescribeInstanceInformation",   "ssm.amazonaws.com",            True,  4),
    ("GetSecretValue",                "secretsmanager.amazonaws.com", True,  4),
    ("ListFunctions",                 "lambda.amazonaws.com",         True,  2),
    ("DescribeLoadBalancers",         "elasticloadbalancing.amazonaws.com", True, 2),
    ("GetRole",           "iam.amazonaws.com", True,  3),
    ("GetUser",           "iam.amazonaws.com", True,  3),
    ("ListRolePolicies",  "iam.amazonaws.com", True,  2),
    ("GetRolePolicy",     "iam.amazonaws.com", True,  2),
    ("GetUserPolicy",     "iam.amazonaws.com", True,  1),
    ("ListGroupsForUser", "iam.amazonaws.com", True,  1),
    ("PutParameter",                  "ssm.amazonaws.com",            False, 3),
    ("SendCommand",                   "ssm.amazonaws.com",            False, 3),
    ("StartInstances",                "ec2.amazonaws.com",            False, 2),
    ("StopInstances",                 "ec2.amazonaws.com",            False, 2),
    ("RebootInstances",               "ec2.amazonaws.com",            False, 1),
    ("ModifyInstanceAttribute",       "ec2.amazonaws.com",            False, 2),
    ("RotateSecret",                  "secretsmanager.amazonaws.com", False, 2),
    ("PutSecretValue",                "secretsmanager.amazonaws.com", False, 2),
    ("CreateSnapshot",                "ec2.amazonaws.com",            False, 1),
    ("ModifyDBInstance",              "rds.amazonaws.com",            False, 1),
]

BENIGN_ADMIN_IAM_EVENTS_WEIGHTED = [
    ("CreateRole",             "iam.amazonaws.com", False, 6),
    ("AttachRolePolicy",       "iam.amazonaws.com", False, 6),
    ("PutRolePolicy",          "iam.amazonaws.com", False, 4),
    ("CreateUser",             "iam.amazonaws.com", False, 5),
    ("AttachUserPolicy",       "iam.amazonaws.com", False, 5),
    ("CreateAccessKey",        "iam.amazonaws.com", False, 3),
    ("CreateLoginProfile",     "iam.amazonaws.com", False, 2),
    ("AddUserToGroup",         "iam.amazonaws.com", False, 4),
    ("CreatePolicyVersion",    "iam.amazonaws.com", False, 2),
    ("SetDefaultPolicyVersion","iam.amazonaws.com", False, 2),
    ("UpdateAssumeRolePolicy", "iam.amazonaws.com", False, 2),
    ("PutBucketPolicy",        "s3.amazonaws.com",  False, 2),
]

ASSUMED_ROLE_BENIGN = [
    ("DescribeInstances",   "ec2.amazonaws.com",            True,  5),
    ("GetParameter",        "ssm.amazonaws.com",            True,  5),
    ("GetSecretValue",      "secretsmanager.amazonaws.com", True,  4),
    ("ListKeys",            "kms.amazonaws.com",            True,  3),
    ("SendCommand",         "ssm.amazonaws.com",            False, 3),
    ("PutParameter",        "ssm.amazonaws.com",            False, 2),
    ("DescribeDBInstances", "rds.amazonaws.com",            True,  2),
]

BENIGN_ERROR_CODES = [
    ("ThrottlingException",                  40),
    ("Client.UnauthorizedOperation",         17),
    ("AccessDenied",                          6),
    ("NoSuchBucketPolicy",                    5),
    ("Client.InvalidRouteTableID.NotFound",   5),
    ("NoSuchPublicAccessBlockConfiguration",  5),
    ("NoSuchWebsiteConfiguration",            4),
    ("NoSuchCORSConfiguration",               4),
    ("NoSuchLifecycleConfiguration",          4),
]
_benign_err_pool  = [code for code, w in BENIGN_ERROR_CODES for _ in range(w)]
ATTACK_ERROR_CODES = ["AccessDenied", "NoSuchEntity", "ThrottlingException", "InvalidParameterValue"]

USER_AGENTS = [
    "aws-cli/2.13.0 Python/3.11.4 Linux/5.15.0 botocore/2.0.0",
    "Boto3/1.28.0 Python/3.10.6 Linux/5.19.0 Botocore/1.31.0",
    "Boto3/1.26.165 Python/3.10.6 Linux/5.19.0-46-generic Botocore/1.29.165",
    "aws-cli/1.29.0 Python/3.9.0 Darwin/22.0.0 botocore/1.31.0",
    "Terraform/1.5.0 aws-sdk-go/1.44.300",
    "console.amazonaws.com",
    "AWS Internal",
]
ATTACKER_UAS = [
    "aws-cli/2.13.0 Python/3.11.4 Linux/5.15.0 botocore/2.0.0",
    "Boto3/1.28.0 Python/3.10.6 Linux/5.19.0 Botocore/1.31.0",
    "python-requests/2.28.0",
]

def rand_str(n=8):   return "".join(random.choices(string.ascii_lowercase, k=n))
def rand_account():  return "".join(random.choices(string.digits, k=12))
def rand_ip():       return f"{random.randint(10,203)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
def rand_key():      return "AKIA" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
def rand_role_key(): return "ASIA" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
def rand_resource(kind):
    s = rand_str(6)
    return {"role": f"role-{s}", "user": f"svc-{s}", "group": f"admins-{s}",
            "policy": "arn:aws:iam::aws:policy/AdministratorAccess",
            "secret": f"prod/db/{s}", "bucket": f"data-{s}-bucket",
            "trail": f"mgmt-trail-{s}", "instance": f"i-{rand_str(17)}"}.get(kind, s)
def jitter(lo=2, hi=45): return timedelta(seconds=random.randint(lo, hi))

def _weighted_sample(pool, n):
    items  = [x[:-1] if len(x)==4 else x for x in pool]
    weights= [x[-1] for x in pool]
    return random.choices(items, weights=weights, k=n)


# ── PATCH 1: target_resource for benign/recon events ──────────────────────────
# Calibrated per event_source against the real combined dataset's non-null
# contribution by source (ssm/kms/sts/s3/secretsmanager/iam dominate; others
# rarely carry an identifiable single resource). Probability tuned so the
# overall non-null rate lands near the real ~72% (28% null).
_TARGET_RESOURCE_PROB_BY_SOURCE = {
    "ssm.amazonaws.com": 0.9,
    "kms.amazonaws.com": 0.9,
    "sts.amazonaws.com": 0.9,
    "s3.amazonaws.com": 0.9,
    "secretsmanager.amazonaws.com": 0.9,
    "iam.amazonaws.com": 0.62,
    "ec2.amazonaws.com": 0.4,
    "lambda.amazonaws.com": 0.57,
    "cloudtrail.amazonaws.com": 0.57,
    "rds.amazonaws.com": 0.33,
    "elasticloadbalancing.amazonaws.com": 0.11,
    "account.amazonaws.com": 0.0,
}

def _benign_target_resource(event_source):
    prob = _TARGET_RESOURCE_PROB_BY_SOURCE.get(event_source, 0.2)
    if random.random() > prob:
        return None
    s = rand_str(6)
    return {
        "ssm.amazonaws.com": f"/app/{s}/config",
        "kms.amazonaws.com": f"alias/{s}",
        "sts.amazonaws.com": f"role-{s}",
        "s3.amazonaws.com": f"data-{s}-bucket",
        "secretsmanager.amazonaws.com": f"prod/db/{s}",
        "iam.amazonaws.com": f"role-{s}",
        "ec2.amazonaws.com": f"i-{rand_str(17)}",
        "lambda.amazonaws.com": f"fn-{s}",
        "cloudtrail.amazonaws.com": f"mgmt-trail-{s}",
        "rds.amazonaws.com": f"db-{s}",
    }.get(event_source, s)


# ── PATCH 2: mfa_authenticated -- mostly null, rarely True when present ───────
def _mfa_value():
    # Threshold raised above the raw real null rate (0.6913) to compensate:
    # attack-chain steps and assumed-role-benign rows always get a non-null
    # value regardless of this function, which dilutes the overall rate
    # below target if this used 0.6913 directly (empirically calibrated).
    if random.random() < 0.719:
        return None
    return "True" if random.random() < 0.036 else "False"  # 96.4% False when present


# ── Attack session generator (unchanged logic; recon/noise now use the patches) ─
def generate_attack_session(chain_name, recon_events=5, noise_events=3):
    chain      = ATTACK_CHAINS[chain_name]
    account_id = rand_account(); attacker = rand_str(random.randint(4, 12))
    source_ip  = rand_ip(); access_key = rand_key()
    ua         = random.choice(ATTACKER_UAS)
    t          = datetime(2024, random.randint(1,12), random.randint(1,28),
                          random.randint(7,19), 0, 0, tzinfo=timezone.utc)
    resources  = {s["target_key"]: rand_resource(s["target_key"]) for s in chain}
    rows = []

    for name, source, ro in random.sample(RECON_EVENTS, min(recon_events, len(RECON_EVENTS))):
        t += jitter(3, 30)
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": None,
            "label": 0, "attack_technique": None, "read_only": ro, "user_agent": ua,
            "access_key_id": access_key, "mfa_authenticated": _mfa_value(),
            "target_resource": _benign_target_resource(source), "request_params_raw": None,
            "principal_type": "IAMUser",
            "principal_arn": f"arn:aws:iam::{account_id}:user/{attacker}",
            "username": attacker, "session_label": 1, "synthetic": True})

    for step in chain:
        t += jitter(5, 60)
        ep = step.get("error_probability", 0)
        ec = random.choice(ATTACK_ERROR_CODES) if random.random() < ep else None
        rows.append({"timestamp": t.isoformat(), "event_name": step["event_name"],
            "event_source": step["event_source"], "aws_region": "us-east-1",
            "source_ip": source_ip, "error_code": ec, "label": 1,
            "attack_technique": step["attack_technique"], "read_only": step["read_only"],
            "user_agent": ua, "access_key_id": access_key, "mfa_authenticated": "False",
            "target_resource": resources[step["target_key"]],
            "request_params_raw": json.dumps({step["target_key"]+"Name": resources[step["target_key"]]}),
            "principal_type": "IAMUser",
            "principal_arn": f"arn:aws:iam::{account_id}:user/{attacker}",
            "username": attacker, "session_label": 1, "synthetic": True})

    for name, source, ro in _weighted_sample(BENIGN_EVENTS_WEIGHTED, noise_events):
        t += jitter(10, 90)
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": None,
            "label": 0, "attack_technique": None, "read_only": ro, "user_agent": ua,
            "access_key_id": access_key, "mfa_authenticated": _mfa_value(),
            "target_resource": _benign_target_resource(source), "request_params_raw": None,
            "principal_type": "IAMUser",
            "principal_arn": f"arn:aws:iam::{account_id}:user/{attacker}",
            "username": attacker, "session_label": 1, "synthetic": True})
    return rows


def generate_benign_iamuser(n_events=15):
    account_id = rand_account(); username = rand_str(6)
    source_ip  = rand_ip(); access_key = rand_key()
    ua         = random.choice(USER_AGENTS)
    t          = datetime(2024, random.randint(1,12), random.randint(1,28),
                          random.randint(7,19), 0, 0, tzinfo=timezone.utc)
    rows = []
    for name, source, ro in _weighted_sample(BENIGN_EVENTS_WEIGHTED, n_events):
        t += jitter(5, 120)
        ec = random.choice(_benign_err_pool) if random.random() < 0.12 else None
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": ec,
            "label": 0, "attack_technique": None, "read_only": ro, "user_agent": ua,
            "access_key_id": access_key, "mfa_authenticated": _mfa_value(),
            "target_resource": _benign_target_resource(source), "request_params_raw": None,
            "principal_type": "IAMUser",
            "principal_arn": f"arn:aws:iam::{account_id}:user/{username}",
            "username": username, "session_label": 0, "synthetic": True})
    return rows


def generate_benign_assumed_role(n_events=12):
    account_id = rand_account()
    role_name  = random.choice(["AWSServiceRoleForEC2", "LambdaExecutionRole",
                                 "ECSTaskRole", "CodeDeployRole", "AutoScalingRole"])
    session_id = rand_str(16)
    source_ip  = random.choice(["AWS Internal", rand_ip()])
    access_key = rand_role_key()
    ua         = random.choice(["AWS Internal", "aws-sdk-java/1.11.x", "aws-sdk-go/1.44.x"])
    t          = datetime(2024, random.randint(1,12), random.randint(1,28),
                          random.randint(7,19), 0, 0, tzinfo=timezone.utc)
    arn        = f"arn:aws:sts::{account_id}:assumed-role/{role_name}/{session_id}"
    rows = []
    for name, source, ro in _weighted_sample(ASSUMED_ROLE_BENIGN, n_events):
        t += jitter(1, 30)
        ec = random.choice(_benign_err_pool) if random.random() < 0.08 else None
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": ec,
            "label": 0, "attack_technique": None, "read_only": ro, "user_agent": ua,
            "access_key_id": access_key, "mfa_authenticated": "False",
            "target_resource": _benign_target_resource(source), "request_params_raw": None,
            "principal_type": "AssumedRole", "principal_arn": arn,
            "username": role_name, "session_label": 0, "synthetic": True})
    return rows


# ── PATCH 3: legitimate admin/IaC IAM mutations ────────────────────────────────
# CreateRole/AttachUserPolicy/CreateAccessKey/etc. are routine in real accounts
# (Terraform applies, onboarding a new engineer) -- but before this patch, every
# occurrence of these event names in this dataset came from ATTACK_CHAINS, so
# event_name alone was a perfect (100%-accurate, non-generalizing) predictor of
# label. This session type gives the same event names a legitimate context with
# a genuinely different behavioral signature: MFA present, slow/deliberate
# pacing, legit tooling UA -- instead of the attack chains' no-MFA/rapid/
# attacker-UA signature. Forces any model (or hand-tuned prior) to learn from
# behavior, not just which API was called.
#
# StopLogging and GetPasswordData are deliberately left out of this pool --
# disabling trail logging and retrieving a Windows instance password are rare
# enough even for legitimate admins that keeping them attack-exclusive is a
# reasonable modeling choice, not an oversight.
def generate_benign_admin_iam_session(n_events=4):
    account_id = rand_account(); admin = rand_str(random.randint(4, 10))
    source_ip  = rand_ip(); access_key = rand_key()
    ua         = random.choice(USER_AGENTS)
    t          = datetime(2024, random.randint(1,12), random.randint(1,28),
                          random.randint(7,19), 0, 0, tzinfo=timezone.utc)
    rows = []
    for name, source, ro in _weighted_sample(BENIGN_ADMIN_IAM_EVENTS_WEIGHTED, n_events):
        t += jitter(120, 900)  # deliberate/slow admin pacing, not a rapid attack chain
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": None,
            "label": 0, "attack_technique": None, "read_only": ro, "user_agent": ua,
            "access_key_id": access_key,
            "mfa_authenticated": "True" if random.random() < 0.85 else "False",
            "target_resource": _benign_target_resource(source), "request_params_raw": None,
            "principal_type": "IAMUser",
            "principal_arn": f"arn:aws:iam::{account_id}:user/{admin}",
            "username": admin, "session_label": 0, "synthetic": True})
    return rows


# ── AWS service/root background noise (closes the principal_type gap) ────────
# Patterns below are taken directly from what real_dataset_combined.csv
# actually contains for each principal_type, not invented:
#   AWSService: resource-explorer periodic AssumeRole (64%),
#               cloudtrail periodic GetBucketAcl on its log bucket (36%)
#   unknown:    Secrets Manager's own lifecycle events (StartSecretVersionDelete
#               / EndSecretVersionDelete pairs), always read_only=False
#   Root:       account-level billing/cost/notification background calls,
#               read_only mostly True, mfa_authenticated populated ~69% False
#               / ~31% True (unlike AWSService/unknown, which never have it)

SERVICE_NOISE_EVENTS_WEIGHTED = [
    ("AssumeRole",   "sts.amazonaws.com", "resource-explorer-2.amazonaws.com", 64),
    ("GetBucketAcl", "s3.amazonaws.com",  "cloudtrail.amazonaws.com",          36),
]

ROOT_BILLING_EVENTS_WEIGHTED = [
    ("ListManagedNotificationEvents", "notifications.amazonaws.com", 180),
    ("GetCostAndUsage",               "ce.amazonaws.com",             37),
    ("DescribeBudgets",               "budgets.amazonaws.com",        28),
    ("GetAccountPlanState",           "iam.amazonaws.com",            23),
    ("ListEnrollmentStatuses",        "freetier.amazonaws.com",       23),
]


def generate_service_noise_session(n_events=10):
    rows = []
    t = datetime(2024, random.randint(1,12), random.randint(1,28), random.randint(0,23), 0, 0, tzinfo=timezone.utc)
    for _ in range(n_events):
        name, source, invoked_by = random.choices(
            [(n, s, i) for n, s, i, _ in SERVICE_NOISE_EVENTS_WEIGHTED],
            weights=[w for *_, w in SERVICE_NOISE_EVENTS_WEIGHTED], k=1)[0]
        t += jitter(30, 300)
        # AWSService rows are ALWAYS non-null for target_resource in real data
        target = f"role-{rand_str(6)}" if name == "AssumeRole" else f"data-{rand_str(6)}-bucket"
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": invoked_by, "error_code": None,
            "label": 0, "attack_technique": None, "read_only": True, "user_agent": invoked_by,
            "access_key_id": None, "mfa_authenticated": None,
            "target_resource": target, "request_params_raw": None,
            "principal_type": "AWSService", "principal_arn": None,
            "username": invoked_by, "session_label": 0, "synthetic": True})
    return rows


def generate_secretsmanager_lifecycle_session(n_pairs=3):
    rows = []
    t = datetime(2024, random.randint(1,12), random.randint(1,28), random.randint(0,23), 0, 0, tzinfo=timezone.utc)
    for _ in range(n_pairs):
        for name in ("StartSecretVersionDelete", "EndSecretVersionDelete"):
            t += jitter(5, 60)
            rows.append({"timestamp": t.isoformat(), "event_name": name,
                "event_source": "secretsmanager.amazonaws.com",
                "aws_region": "us-east-1", "source_ip": "secretsmanager.amazonaws.com", "error_code": None,
                "label": 0, "attack_technique": None, "read_only": False,
                "user_agent": "secretsmanager.amazonaws.com",
                "access_key_id": None, "mfa_authenticated": None,
                "target_resource": None, "request_params_raw": None,
                "principal_type": "unknown", "principal_arn": None,
                "username": None, "session_label": 0, "synthetic": True})
    return rows


def generate_root_billing_session(n_events=8):
    account_id = rand_account()
    rows = []
    t = datetime(2024, random.randint(1,12), random.randint(1,28), random.randint(0,23), 0, 0, tzinfo=timezone.utc)
    for _ in range(n_events):
        name, source = random.choices(
            [(n, s) for n, s, _ in ROOT_BILLING_EVENTS_WEIGHTED],
            weights=[w for *_, w in ROOT_BILLING_EVENTS_WEIGHTED], k=1)[0]
        t += jitter(60, 600)
        mfa = "False" if random.random() < 0.6875 else "True"  # matches real Root split
        # Root rows are non-null for target_resource ~64.5% of the time in real data
        target = f"budget-{rand_str(6)}" if random.random() < 0.645 else None
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": rand_ip(), "error_code": None,
            "label": 0, "attack_technique": None, "read_only": True, "user_agent": "console.amazonaws.com",
            "access_key_id": rand_key(), "mfa_authenticated": mfa,
            "target_resource": target, "request_params_raw": None,
            "principal_type": "Root", "principal_arn": f"arn:aws:iam::{account_id}:root",
            "username": None, "session_label": 0, "synthetic": True})
    return rows


def main():
    N_PER_CHAIN       = 20
    N_BENIGN_IAMUSER  = 340
    N_BENIGN_ASSUMED  = 60
    # Sized so the IAM-mutation event names shared with ATTACK_CHAINS (CreateRole,
    # AttachUserPolicy, CreateAccessKey, etc.) land around a ~25-30% benign share
    # instead of the 100%-attack they'd otherwise have -- see PATCH 3.
    N_BENIGN_ADMIN_IAM = 45
    # Sized so principal_type ends up ~18.5% AWSService / ~1.7% unknown /
    # ~1.3% Root of the final dataset, matching real_dataset_combined.csv.
    N_SERVICE_NOISE       = 178
    N_SECRETSMANAGER_NOISE = 27
    N_ROOT_NOISE          = 16

    all_rows = []
    for chain_name in ATTACK_CHAINS:
        for _ in range(N_PER_CHAIN):
            all_rows.extend(generate_attack_session(
                chain_name, recon_events=random.randint(3, 8), noise_events=random.randint(2, 5),
            ))
    for _ in range(N_BENIGN_IAMUSER):
        all_rows.extend(generate_benign_iamuser(n_events=random.randint(8, 20)))
    for _ in range(N_BENIGN_ASSUMED):
        all_rows.extend(generate_benign_assumed_role(n_events=random.randint(6, 15)))
    for _ in range(N_BENIGN_ADMIN_IAM):
        all_rows.extend(generate_benign_admin_iam_session(n_events=random.randint(2, 6)))
    for _ in range(N_SERVICE_NOISE):
        all_rows.extend(generate_service_noise_session(n_events=random.randint(6, 14)))
    for _ in range(N_SECRETSMANAGER_NOISE):
        all_rows.extend(generate_secretsmanager_lifecycle_session(n_pairs=random.randint(2, 4)))
    for _ in range(N_ROOT_NOISE):
        all_rows.extend(generate_root_billing_session(n_events=random.randint(6, 10)))

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Shape: {df.shape}")
    print(f"Label split: benign={  (df['label']==0).sum() }  attack={ (df['label']==1).sum() }")
    print(f"target_resource null rate: {df['target_resource'].isnull().mean():.4f}  (target: 0.2824)")
    print(f"mfa_authenticated null rate: {df['mfa_authenticated'].isnull().mean():.4f}  (target: 0.6913)")
    print(f"principal_type distribution:\n{(df['principal_type'].value_counts(normalize=True)*100).round(2)}")
    print("(real targets: IAMUser 66.93 / AWSService 18.45 / AssumedRole 11.66 / unknown 1.67 / Root 1.30)")

    df.to_csv("synthetic_cloudtrail.csv", index=False)
    print("\nSaved synthetic_cloudtrail.csv")


if __name__ == "__main__":
    main()
