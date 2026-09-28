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
        {"event_name": "GetSecretValue",         "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "secret"},
    ],
    "full_kill_chain": [
        {"event_name": "CreateRole",       "event_source": "iam.amazonaws.com",            "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "AttachRolePolicy", "event_source": "iam.amazonaws.com",            "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "AssumeRole",       "event_source": "sts.amazonaws.com",            "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "GetSecretValue",   "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access",    "read_only": True,  "target_key": "secret"},
        {"event_name": "PutBucketPolicy",  "event_source": "s3.amazonaws.com",             "attack_technique": "exfiltration",         "read_only": False, "target_key": "bucket"},
        {"event_name": "StopLogging",      "event_source": "cloudtrail.amazonaws.com",     "attack_technique": "defense-evasion",      "read_only": False, "target_key": "trail", "error_probability": 0.4},
    ],
    # ── Credential-access chains ─────────────────────────────────────────────
    # ADDED after measuring a tactic-coverage gap between synthetic and real
    # attack data. The chains above are IAM-manipulation shaped: of 26 attack
    # steps, only 3 touched credentials, and each was the LAST step of an IAM
    # chain rather than the objective. Real Stratus data is the mirror image --
    # 92.5% of its attack events are credential-access, dominated by
    # DescribeParameters / GetParameters / GetSecretValue.
    #
    # The consequence was measurable: the model learned "READ = reconnaissance
    # = benign" and scored real credential-access attacks LOW. Within-group AUC
    # on (User, READ, Resource) -- which carries 87% of real attack edges -- was
    # 0.275, while every other edge type scored 0.76-0.98.
    #
    # These three chains make credential retrieval the OBJECTIVE, and every step
    # is read_only so they land in exactly that under-represented triple.
    # DescribeParameters and GetParameters appeared in NO chain before, despite
    # being 72% of real attack edges.
    # ── genuine privilege ESCALATION (privilege_gain > 0) ──────────────────
    # privilege_features.privilege_gain() = rank(current action) - rank(the
    # AssumeRole that granted the role). AssumeRole is rank 3 (Write); the
    # permissions-management actions below are rank 4, so these chains are the
    # only ones that yield a POSITIVE gain. Every pre-existing chain either put
    # AssumeRole last (nothing follows as the role) or followed it with
    # read-only theft (rank 1, gain -2), so "privilege escalation" was a label
    # the corpus asserted but its own structural feature never witnessed.
    "assume_then_attach_admin": [
        {"event_name": "AssumeRole",       "event_source": "sts.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "AttachRolePolicy", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "AttachUserPolicy", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "user"},
    ],
    "assume_then_backdoor_key": [
        {"event_name": "AssumeRole",      "event_source": "sts.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "CreateAccessKey", "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "user"},
        {"event_name": "PutUserPolicy",   "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "user"},
    ],
    # TRUE multi-principal propagation (review point 3): the actor changes TWICE.
    # User assumes AdminRole (role); AS AdminRole it creates and empowers a second
    # role (role2); then it ASSUMES role2 and acts as THAT. This yields two
    # ASSUMES edges -- User->role and role->role2 -- so the graph carries a real
    # User -> AdminRole -> BackdoorRole -> Resource principal chain, not a role
    # merely manipulating another role as an object. The AttachRolePolicy step
    # (rank 4) performed by the assumed role (assume = rank 3) is a +1 gain edge.
    "assume_admin_then_backdoor_role": [
        {"event_name": "AssumeRole",       "event_source": "sts.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "CreateRole",       "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "role2"},
        {"event_name": "AttachRolePolicy", "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role2"},
        {"event_name": "AssumeRole",       "event_source": "sts.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role2"},
        {"event_name": "GetSecretValue",   "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "secret"},
    ],
    "ssm_parameter_harvest": [
        {"event_name": "DescribeParameters", "event_source": "ssm.amazonaws.com",  "attack_technique": "credential-access", "read_only": True, "target_key": "parameter"},
        {"event_name": "GetParameters",      "event_source": "ssm.amazonaws.com",  "attack_technique": "credential-access", "read_only": True, "target_key": "parameter"},
        {"event_name": "Decrypt",            "event_source": "kms.amazonaws.com",  "attack_technique": "credential-access", "read_only": True, "target_key": "key"},
    ],
    "secrets_manager_sweep": [
        {"event_name": "ListSecrets",    "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "secret"},
        {"event_name": "DescribeSecret", "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "secret"},
        {"event_name": "GetSecretValue", "event_source": "secretsmanager.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "secret"},
    ],
    "ec2_credential_extraction": [
        {"event_name": "DescribeInstances", "event_source": "ec2.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "instance"},
        {"event_name": "GetPasswordData",   "event_source": "ec2.amazonaws.com", "attack_technique": "credential-access", "read_only": True, "target_key": "instance", "error_probability": 0.3},
    ],
    "ec2_password_data": [
        {"event_name": "CreateRole",      "event_source": "iam.amazonaws.com", "attack_technique": "persistence",          "read_only": False, "target_key": "role"},
        {"event_name": "PutRolePolicy",   "event_source": "iam.amazonaws.com", "attack_technique": "privilege-escalation", "read_only": False, "target_key": "role"},
        {"event_name": "GetPasswordData", "event_source": "ec2.amazonaws.com", "attack_technique": "credential-access",    "read_only": True,  "target_key": "instance", "error_probability": 0.9},
    ],
}

# Per-chain metadata (review point 8): credential-access sweeps fire in tight
# sub-second bursts in the real capture (DescribeParameters/GetParameters
# repeated at effectively the same second), so those chains are marked bursty;
# the rest keep realistic multi-second spacing. This is timing only, not labels.
ATTACK_CHAINS_META = {
    "ssm_parameter_harvest":     {"burst": True},
    "secrets_manager_sweep":     {"burst": True},
    "ec2_credential_extraction": {"burst": True},
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
def rand_key():      return "SYNAK" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))  # non-AWS-format so synthetic data never trips secret scanners
def rand_role_key(): return "SYNAS" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))  # non-AWS-format (was ASIA...)
def rand_resource(kind):
    s = rand_str(6)
    # role2 is a SECOND, distinct role in the same session -- used by chains that
    # assume one role and then create/assume another (true multi-principal
    # propagation). It must be role-formatted so the graph builder types it as a
    # Role, not a bare Resource.
    return {"role": f"role-{s}", "role2": f"role-{s}", "user": f"svc-{s}", "group": f"admins-{s}",
            "policy": "arn:aws:iam::aws:policy/AdministratorAccess",
            "secret": f"prod/db/{s}", "bucket": f"data-{s}-bucket",
            "trail": f"mgmt-trail-{s}", "instance": f"i-{rand_str(17)}",
            "parameter": f"/prod/app/{s}", "key": f"alias/{s}"}.get(kind, s)
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


# Mirrors privilege_features.ASSUME_ACTIONS -- kept as a local copy since this
# script is standalone and has no dependency on the graph-construction package.
_ASSUME_ACTIONS = {"AssumeRole", "AssumeRoleWithSAML", "AssumeRoleWithWebIdentity"}


# ── Attack session generator (unchanged logic; recon/noise now use the patches) ─
# ── Real AWS requestParameters + embedded policy state (review points 5, 4) ────
# Each attack event carries the parameter shape CloudTrail actually emits (so the
# graph builder's target extraction and any permission-aware feature has real
# structure to read), plus, for permission-mutating events, the after-state
# policy document. Before/after permission SETS are attached to the row
# separately (PERM_CHANGE) so privilege_delta is derivable as ground truth.
_ADMIN_ARN = "arn:aws:iam::aws:policy/AdministratorAccess"
_PRIV_DOC = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": ["iam:*", "s3:*", "sts:AssumeRole"], "Resource": "*"}]}

def _trust_doc(account_id):
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
         "Action": "sts:AssumeRole"}]}

# event_name -> (before_permissions, after_permissions). Ground truth only; the
# feature engine must infer privilege_delta from the observable event, not read
# these directly (they are annotation, like campaign_id).
PERM_CHANGE = {
    "AttachRolePolicy":       (["s3:GetObject", "s3:ListBucket"], ["*"]),
    "AttachUserPolicy":       (["s3:GetObject", "s3:ListBucket"], ["*"]),
    "PutRolePolicy":          (["s3:GetObject"], ["iam:*", "s3:*", "sts:AssumeRole"]),
    "PutUserPolicy":          (["s3:GetObject"], ["iam:*", "s3:*", "sts:AssumeRole"]),
    "CreatePolicyVersion":    (["s3:GetObject"], ["iam:*", "s3:*"]),
    "UpdateAssumeRolePolicy": ([], ["sts:AssumeRole/*"]),
}

def _attack_request_params(event_name, target_key, resources, account_id):
    r = resources.get(target_key)
    role_arn = f"arn:aws:iam::{account_id}:role/{r}"
    policy_arn = resources.get("policy", _ADMIN_ARN)
    m = {
        "AssumeRole":               {"roleArn": role_arn, "roleSessionName": rand_str(8)},
        "AssumeRoleWithSAML":       {"roleArn": role_arn, "principalArn": f"arn:aws:iam::{account_id}:saml-provider/idp"},
        "GetSecretValue":           {"secretId": r},
        "DescribeSecret":           {"secretId": r},
        "ListSecrets":              {},
        "GetPasswordData":          {"instanceId": r},
        "DescribeInstances":        {},
        "DescribeParameters":       {},
        "GetParameters":            {"names": [r]},
        "GetParameter":             {"name": r},
        "Decrypt":                  {"keyId": r},
        "StopLogging":              {"name": r},
        "DeleteTrail":              {"name": r},
        "PutEventSelectors":        {"trailName": r, "eventSelectors": [{"readWriteType": "None"}]},
        "CreateRole":               {"roleName": r, "assumeRolePolicyDocument": json.dumps(_trust_doc(account_id))},
        "CreateUser":               {"userName": r},
        "CreateAccessKey":          {"userName": r},
        "CreateLoginProfile":       {"userName": r},
        "UpdateLoginProfile":       {"userName": r},
        "AddUserToGroup":           {"userName": r, "groupName": resources.get("group", "admins")},
        "AttachRolePolicy":         {"roleName": r, "policyArn": policy_arn},
        "AttachUserPolicy":         {"userName": r, "policyArn": policy_arn},
        "PutRolePolicy":            {"roleName": r, "policyName": "inline-esc", "policyDocument": json.dumps(_PRIV_DOC)},
        "PutUserPolicy":            {"userName": r, "policyName": "inline-esc", "policyDocument": json.dumps(_PRIV_DOC)},
        "CreatePolicyVersion":      {"policyArn": policy_arn, "policyDocument": json.dumps(_PRIV_DOC), "setAsDefault": True},
        "SetDefaultPolicyVersion":  {"policyArn": policy_arn, "versionId": "v2"},
        "UpdateAssumeRolePolicy":   {"roleName": r, "policyDocument": json.dumps(_trust_doc(account_id))},
        "PutBucketPolicy":          {"bucketName": r, "policy": json.dumps(
            {"Statement": [{"Effect": "Allow", "Principal": "*", "Action": "s3:GetObject",
                            "Resource": f"arn:aws:s3:::{r}/*"}]})},
        "DeleteBucketPolicy":       {"bucketName": r},
        "AddPermission20150331v2":  {"functionName": r, "statementId": rand_str(6), "action": "lambda:InvokeFunction"},
    }
    return m.get(event_name, {(target_key or "resource") + "Name": r})


def generate_attack_session(chain_name, recon_events=5, noise_events=3):
    chain      = ATTACK_CHAINS[chain_name]
    account_id = rand_account(); attacker = rand_str(random.randint(4, 12))
    source_ip  = rand_ip(); access_key = rand_key()
    ua         = random.choice(ATTACKER_UAS)
    t          = datetime(2024, random.randint(1,12), random.randint(1,28),
                          random.randint(7,19), 0, 0, tzinfo=timezone.utc)
    resources  = {s["target_key"]: rand_resource(s["target_key"]) for s in chain}
    # Campaign/lineage ground truth (review points 2, 14, 15): a stable id per
    # attack session, the chain family it belongs to, and -- on the labelled
    # chain steps -- a 0-based stage index and separated tactic/technique. These
    # travel with each row and are re-emitted as a joinable annotation layer in
    # main(); they are ground truth for evaluating attack progression and are NOT
    # consumed as model features.
    campaign_id = "camp-" + rand_str(12)
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
            "username": attacker, "session_label": 1, "synthetic": True,
            "campaign_id": campaign_id, "chain_name": chain_name, "stage_index": -1,
            "attack_tactic": "reconnaissance", "attack_technique_id": chain_name})

    # Identity pivots to the assumed role once an AssumeRole-family step is
    # processed -- every action after that point in a real AssumeRole-based
    # escalation is performed AS the role (temp STS credentials), not as the
    # original user. Without this, no attack-labeled chain ever produced a
    # Role-sourced action edge, even though "assume then abuse" is the
    # canonical AWS privilege-escalation shape (this was the root cause of
    # the GNN never learning that pattern -- see privilege_features.py's
    # node_key_for_principal/node_key_for_target for the matching graph-side
    # canonicalization this now lines up with).
    principal_type = "IAMUser"
    principal_arn  = f"arn:aws:iam::{account_id}:user/{attacker}"
    principal_name = attacker

    # Full lineage (review point 2): a stable id per event, its parent (the prior
    # chain step), the campaign root, and a hop id that increments on every
    # principal handoff (AssumeRole). All ground truth, never model features.
    # burst: some campaigns fire in tight sub-second bursts (review point 8).
    burst = ATTACK_CHAINS_META.get(chain_name, {}).get("burst", False)
    event_uids = ["evt-" + rand_str(12) for _ in chain]
    root_uid = event_uids[0] if event_uids else None
    hop_id = 0
    for stage_index, step in enumerate(chain):
        t += (timedelta(seconds=random.choice([0, 0, 1])) if burst else jitter(5, 60))
        ep = step.get("error_probability", 0)
        ec = random.choice(ATTACK_ERROR_CODES) if random.random() < ep else None
        before_perms, after_perms = PERM_CHANGE.get(step["event_name"], ([], []))
        rows.append({"timestamp": t.isoformat(), "event_name": step["event_name"],
            "event_source": step["event_source"], "aws_region": "us-east-1",
            "source_ip": source_ip, "error_code": ec, "label": 1,
            "attack_technique": step["attack_technique"], "read_only": step["read_only"],
            "user_agent": ua, "access_key_id": access_key, "mfa_authenticated": "False",
            "target_resource": resources[step["target_key"]],
            "request_params_raw": json.dumps(
                _attack_request_params(step["event_name"], step["target_key"], resources, account_id)),
            "principal_type": principal_type,
            "principal_arn": principal_arn,
            "username": principal_name, "session_label": 1, "synthetic": True,
            # attack_tactic is the MITRE tactic (step["attack_technique"] holds a
            # tactic string); attack_technique_id is the specific chain family --
            # kept separate per review point 15 (don't encode a tactic as a
            # technique id).
            "campaign_id": campaign_id, "chain_name": chain_name, "stage_index": stage_index,
            "attack_tactic": step["attack_technique"], "attack_technique_id": chain_name,
            "event_uid": event_uids[stage_index], "hop_id": hop_id,
            "parent_event_id": event_uids[stage_index - 1] if stage_index > 0 else "",
            "root_event_id": root_uid,
            "before_permissions": ";".join(before_perms), "after_permissions": ";".join(after_perms)})

        if step["event_name"] in _ASSUME_ACTIONS:
            role_name = resources[step["target_key"]]
            principal_type = "AssumedRole"
            principal_arn  = f"arn:aws:sts::{account_id}:assumed-role/{role_name}/{rand_str(16)}"
            principal_name = role_name
            hop_id += 1  # a real principal handoff -- the next event acts as the new principal

    for name, source, ro in _weighted_sample(BENIGN_EVENTS_WEIGHTED, noise_events):
        t += jitter(10, 90)
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": None,
            "label": 0, "attack_technique": None, "read_only": ro, "user_agent": ua,
            "access_key_id": access_key, "mfa_authenticated": _mfa_value(),
            "target_resource": _benign_target_resource(source), "request_params_raw": None,
            "principal_type": principal_type,
            "principal_arn": principal_arn,
            "username": principal_name, "session_label": 1, "synthetic": True,
            "campaign_id": campaign_id, "chain_name": chain_name, "stage_index": -1,
            "attack_tactic": "", "attack_technique_id": chain_name})
    return rows


# CAMPAIGN_LIBRARY: declarative multi-principal campaign object model (review #17).
# Unlike ATTACK_CHAINS (a flat step list with an implicit linear actor pivot),
# each event declares its ACTOR and, where relevant, the principal it CREATES or
# ASSUMES. The generator resolves named principals to concrete identities,
# changes the acting identity when a role is assumed, and DERIVES the full
# lineage (event_uid / parent_event_id / root_event_id / hop_id / stage_index)
# from the declaration. Supports branching and genuine multi-principal handoffs
# (user -> role_A -> role_B, or a user creating and empowering a second USER)
# that a linear chain cannot express.
EVENT_META = {
    "CreateRole": ("iam.amazonaws.com", False), "AttachRolePolicy": ("iam.amazonaws.com", False),
    "PutRolePolicy": ("iam.amazonaws.com", False), "UpdateAssumeRolePolicy": ("iam.amazonaws.com", False),
    "AssumeRole": ("sts.amazonaws.com", False), "CreateUser": ("iam.amazonaws.com", False),
    "CreateAccessKey": ("iam.amazonaws.com", False), "AttachUserPolicy": ("iam.amazonaws.com", False),
    "PutUserPolicy": ("iam.amazonaws.com", False), "CreateLoginProfile": ("iam.amazonaws.com", False),
    "GetSecretValue": ("secretsmanager.amazonaws.com", True), "GetPasswordData": ("ec2.amazonaws.com", True),
    "GetParameters": ("ssm.amazonaws.com", True), "DescribeParameters": ("ssm.amazonaws.com", True),
    "PutBucketPolicy": ("s3.amazonaws.com", False), "StopLogging": ("cloudtrail.amazonaws.com", False),
}

CAMPAIGN_LIBRARY = {
    "role_escalation_secret_access": {
        "events": [
            {"event": "CreateRole",       "actor": "user_A", "target": "role_A",  "tactic": "persistence"},
            {"event": "AttachRolePolicy", "actor": "user_A", "target": "role_A",  "tactic": "privilege-escalation"},
            {"event": "AssumeRole",       "actor": "user_A", "assumes": "role_A", "target": "role_A", "tactic": "privilege-escalation"},
            {"event": "GetSecretValue",   "actor": "role_A", "target": "secret_A", "tactic": "credential-access"},
        ]},
    "double_role_pivot": {
        "events": [
            {"event": "AssumeRole",       "actor": "user_A", "assumes": "role_A", "target": "role_A", "tactic": "privilege-escalation"},
            {"event": "CreateRole",       "actor": "role_A", "target": "role_B",  "tactic": "persistence"},
            {"event": "AttachRolePolicy", "actor": "role_A", "target": "role_B",  "tactic": "privilege-escalation"},
            {"event": "AssumeRole",       "actor": "role_A", "assumes": "role_B", "target": "role_B", "tactic": "privilege-escalation"},
            {"event": "GetPasswordData",  "actor": "role_B", "target": "instance_A", "tactic": "credential-access"},
        ]},
    "user_persistence_handoff": {
        "events": [
            {"event": "CreateUser",       "actor": "user_A", "creates": "user_B", "target": "user_B", "tactic": "persistence"},
            {"event": "CreateAccessKey",  "actor": "user_A", "target": "user_B",  "tactic": "persistence"},
            {"event": "AttachUserPolicy", "actor": "user_A", "target": "user_B",  "tactic": "privilege-escalation"},
            {"event": "GetParameters",    "actor": "user_B", "target": "param_A", "tactic": "credential-access"},
        ]},
}


def _campaign_resource(named):
    if named.startswith("role"):     return "role", "role-" + rand_str(6)
    if named.startswith("user"):     return "user", "user-" + rand_str(6)
    if named.startswith("secret"):   return "secret", "prod/db/" + rand_str(6)
    if named.startswith("instance"): return "instance", "i-" + rand_str(17)
    if named.startswith("param"):    return "parameter", "/prod/app/" + rand_str(6)
    if named.startswith("bucket"):   return "bucket", "data-" + rand_str(6) + "-bucket"
    return "resource", rand_str(8)


def generate_campaign_session(campaign_name, noise_events=3):
    camp = CAMPAIGN_LIBRARY[campaign_name]
    account_id = rand_account()
    attacker = rand_str(random.randint(4, 12))
    source_ip = rand_ip(); access_key = rand_key(); ua = random.choice(ATTACKER_UAS)
    t = datetime(2024, random.randint(1, 12), random.randint(1, 28),
                 random.randint(7, 19), 0, 0, tzinfo=timezone.utc)
    campaign_id = "camp-" + rand_str(12)

    names = set()
    for e in camp["events"]:
        names.add(e["actor"]); names.add(e.get("target", ""))
    resolved, key_of = {}, {}
    for nm in names:
        if not nm:
            continue
        if nm == "user_A":
            resolved[nm], key_of[nm] = attacker, "user"
        else:
            key_of[nm], resolved[nm] = _campaign_resource(nm)

    identity = {"user_A": ("IAMUser", "arn:aws:iam::" + account_id + ":user/" + attacker, attacker)}
    event_uids = ["evt-" + rand_str(12) for _ in camp["events"]]
    root_uid = event_uids[0]
    rows, hop_id, last_actor = [], 0, None
    for stage_index, e in enumerate(camp["events"]):
        actor = e["actor"]
        if actor not in identity:
            rname = resolved.get(actor, actor)
            identity[actor] = ("AssumedRole",
                               "arn:aws:sts::" + account_id + ":assumed-role/" + rname + "/" + rand_str(16), rname)
        if last_actor is not None and actor != last_actor:
            hop_id += 1
        last_actor = actor
        p_type, p_arn, p_name = identity[actor]

        t += (timedelta(seconds=random.choice([0, 1])) if campaign_name.endswith("access") else jitter(5, 60))
        tgt = e.get("target", "")
        tgt_key = key_of.get(tgt, "resource")
        step_resources = {tgt_key: resolved.get(tgt, tgt), "policy": _ADMIN_ARN, "group": "admins-" + rand_str(4)}
        before_perms, after_perms = PERM_CHANGE.get(e["event"], ([], []))
        rows.append({"timestamp": t.isoformat(), "event_name": e["event"],
            "event_source": EVENT_META.get(e["event"], ("iam.amazonaws.com", False))[0],
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": None, "label": 1,
            "attack_technique": e["tactic"], "read_only": EVENT_META.get(e["event"], ("", False))[1],
            "user_agent": ua, "access_key_id": access_key, "mfa_authenticated": "False",
            "target_resource": resolved.get(tgt, tgt),
            "request_params_raw": json.dumps(_attack_request_params(e["event"], tgt_key, step_resources, account_id)),
            "principal_type": p_type, "principal_arn": p_arn, "username": p_name,
            "session_label": 1, "synthetic": True,
            "campaign_id": campaign_id, "chain_name": campaign_name, "stage_index": stage_index,
            "attack_tactic": e["tactic"], "attack_technique_id": campaign_name,
            "event_uid": event_uids[stage_index], "hop_id": hop_id,
            "parent_event_id": event_uids[stage_index - 1] if stage_index > 0 else "",
            "root_event_id": root_uid,
            "before_permissions": ";".join(before_perms), "after_permissions": ";".join(after_perms)})

        if e["event"] == "AssumeRole" and e.get("assumes"):
            rr = e["assumes"]; rname = resolved.get(rr, rr)
            identity[rr] = ("AssumedRole",
                            "arn:aws:sts::" + account_id + ":assumed-role/" + rname + "/" + rand_str(16), rname)
        if e.get("creates", "").startswith("user"):
            cu = e["creates"]; uname = resolved.get(cu, cu)
            identity[cu] = ("IAMUser", "arn:aws:iam::" + account_id + ":user/" + uname, uname)

    for name, src, ro in _weighted_sample(BENIGN_EVENTS_WEIGHTED, noise_events):
        t += jitter(10, 90)
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": src,
            "aws_region": "us-east-1", "source_ip": source_ip, "error_code": None, "label": 0,
            "attack_technique": None, "read_only": ro, "user_agent": ua, "access_key_id": access_key,
            "mfa_authenticated": _mfa_value(), "target_resource": _benign_target_resource(src),
            "request_params_raw": None, "principal_type": identity[last_actor][0],
            "principal_arn": identity[last_actor][1], "username": identity[last_actor][2],
            "session_label": 1, "synthetic": True, "campaign_id": campaign_id,
            "chain_name": campaign_name, "stage_index": -1, "attack_tactic": "",
            "attack_technique_id": campaign_name})
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


# Each service-linked role is assumed by the AWS service that owns it. Without
# the assumption event these roles ACT but are never ASSUMED, so
# privilege_features.hop_count() -- which asks "was this Role the target of an
# ASSUMES edge?" -- returns 1 for every edge they emit, and privilege_gain is
# undefined for all of them. Real CloudTrail always shows the pairing (e.g.
# resource-explorer-2 assumes AWSServiceRoleForResourceExplorer, then that role
# does the work), so the missing half was a generator artefact, not a property
# of AWS.
BENIGN_ROLE_OWNERS = {
    "AWSServiceRoleForEC2": "ec2.amazonaws.com",
    "LambdaExecutionRole":  "lambda.amazonaws.com",
    "ECSTaskRole":          "ecs-tasks.amazonaws.com",
    "CodeDeployRole":       "codedeploy.amazonaws.com",
    "AutoScalingRole":      "autoscaling.amazonaws.com",
}


def generate_benign_assumed_role(n_events=12):
    account_id = rand_account()
    role_name  = random.choice(list(BENIGN_ROLE_OWNERS))
    session_id = rand_str(16)
    source_ip  = random.choice(["AWS Internal", rand_ip()])
    access_key = rand_role_key()
    ua         = random.choice(["AWS Internal", "aws-sdk-java/1.11.x", "aws-sdk-go/1.44.x"])
    t          = datetime(2024, random.randint(1,12), random.randint(1,28),
                          random.randint(7,19), 0, 0, tzinfo=timezone.utc)
    arn        = f"arn:aws:sts::{account_id}:assumed-role/{role_name}/{session_id}"
    owner      = BENIGN_ROLE_OWNERS[role_name]
    # The assumption itself, emitted by the owning service. This is the edge
    # that makes role_name a 2-hop node in the propagation graph.
    rows = [{"timestamp": t.isoformat(), "event_name": "AssumeRole",
        "event_source": "sts.amazonaws.com", "aws_region": "us-east-1",
        "source_ip": owner, "error_code": None, "label": 0,
        "attack_technique": None, "read_only": True, "user_agent": owner,
        "access_key_id": None, "mfa_authenticated": None,
        "target_resource": role_name, "request_params_raw": None,
        "principal_type": "AWSService", "principal_arn": None,
        "username": owner, "session_label": 0, "synthetic": True}]
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


# Work the service-linked role performs once it has been assumed. Mirrors what
# resource-explorer's role actually does in the real capture (read-only
# inventory sweeps), so the second leg of the chain is realistic rather than
# invented.
SERVICE_ROLE_FOLLOWUP = [
    ("ListResources",  "resource-explorer-2.amazonaws.com", True, 5),
    ("Search",         "resource-explorer-2.amazonaws.com", True, 3),
    ("ListIndexes",    "resource-explorer-2.amazonaws.com", True, 2),
]

# The service-linked role each AWS service assumes. One STABLE name per service,
# not a fresh random one per event: previously every AssumeRole here minted a
# throwaway `role-xxxxxx` that appeared exactly once and never acted, so the
# graph filled with thousands of dead-end Role targets that could never be the
# first leg of a privilege chain.
SERVICE_LINKED_ROLES = {
    "resource-explorer-2.amazonaws.com": "AWSServiceRoleForResourceExplorer",
    "cloudtrail.amazonaws.com":          "AWSServiceRoleForCloudTrail",
}


def generate_service_noise_session(n_events=10):
    rows = []
    account_id = rand_account()
    t = datetime(2024, random.randint(1,12), random.randint(1,28), random.randint(0,23), 0, 0, tzinfo=timezone.utc)
    assumed = {}  # invoked_by -> (role_name, session_arn), set on first AssumeRole
    for _ in range(n_events):
        name, source, invoked_by = random.choices(
            [(n, s, i) for n, s, i, _ in SERVICE_NOISE_EVENTS_WEIGHTED],
            weights=[w for *_, w in SERVICE_NOISE_EVENTS_WEIGHTED], k=1)[0]
        t += jitter(30, 300)
        role_name = SERVICE_LINKED_ROLES.get(invoked_by, "AWSServiceRoleForResourceExplorer")
        # AWSService rows are ALWAYS non-null for target_resource in real data
        target = role_name if name == "AssumeRole" else f"data-{rand_str(6)}-bucket"
        rows.append({"timestamp": t.isoformat(), "event_name": name, "event_source": source,
            "aws_region": "us-east-1", "source_ip": invoked_by, "error_code": None,
            "label": 0, "attack_technique": None, "read_only": True, "user_agent": invoked_by,
            "access_key_id": None, "mfa_authenticated": None,
            "target_resource": target, "request_params_raw": None,
            "principal_type": "AWSService", "principal_arn": None,
            "username": invoked_by, "session_label": 0, "synthetic": True})

        # Second leg, emitted ONCE per service per session: the role it just
        # assumed does the actual work. Without this the assumption is a dead
        # end and hop_count never reaches 2. Emitting it on every AssumeRole
        # (AssumeRole is 64% of this stream) over-produced AssumedRole rows by
        # ~3x and wrecked the principal_type calibration, so it fires only on
        # the first assumption.
        if name == "AssumeRole" and invoked_by not in assumed:
            role_arn = f"arn:aws:sts::{account_id}:assumed-role/{role_name}/{rand_str(16)}"
            assumed[invoked_by] = (role_name, role_arn)
            for fname, fsource, fro in _weighted_sample(SERVICE_ROLE_FOLLOWUP, random.randint(1, 2)):
                t += jitter(5, 45)
                rows.append({"timestamp": t.isoformat(), "event_name": fname,
                    "event_source": fsource, "aws_region": "us-east-1",
                    "source_ip": invoked_by, "error_code": None, "label": 0,
                    "attack_technique": None, "read_only": fro, "user_agent": invoked_by,
                    "access_key_id": rand_role_key(), "mfa_authenticated": None,
                    # Explicit, never null: real AWSService/AssumedRole rows
                    # always carry a target_resource, and _benign_target_resource
                    # would default an unlisted source to 80% null.
                    "target_resource": f"index/{rand_str(8)}",
                    "request_params_raw": None,
                    "principal_type": "AssumedRole", "principal_arn": role_arn,
                    "username": role_name, "session_label": 0, "synthetic": True})
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
    # Deliberate, DOCUMENTED malicious mixture (review point 1). The real capture
    # is ~95% credential-access, but blindly copying that would starve the
    # multi-hop privilege-escalation research question. Instead we choose a
    # mixture that is credential-access-leaning (matching the real dominance
    # DIRECTION) while retaining enough privilege-escalation and multi-hop
    # campaigns to test the mechanism. Per-chain repetition counts below are set
    # to approximate, over malicious EVENTS:
    #   credential-access ~45% | privilege-escalation ~30% | persistence ~15%
    #   defense-evasion/exfiltration ~10%.  Tune here, not by multiplying one
    #   tactic blindly. Default (no entry) = N_PER_CHAIN_DEFAULT.
    N_PER_CHAIN_DEFAULT = 12
    N_PER_CHAIN = {
        # credential-access (dominant, like the real data) -- these are short
        # chains, so they need higher reps to dominate the event count
        "ssm_parameter_harvest":     40,
        "secrets_manager_sweep":     40,
        "ec2_credential_extraction": 40,
        # privilege-escalation / multi-hop (the research question) -- kept
        # substantial but no longer the majority
        "assume_admin_then_backdoor_role": 16,
        "assume_then_attach_admin":        14,
        "assume_then_backdoor_key":        14,
        "full_kill_chain":                 12,
        # persistence-leaning IAM chains
        "create_user_accesskey_policy": 10,
        "create_user_console_access":   8,
        "add_user_to_admin_group":      8,
    }
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
        reps = N_PER_CHAIN.get(chain_name, N_PER_CHAIN_DEFAULT)
        for _ in range(reps):
            all_rows.extend(generate_attack_session(
                chain_name, recon_events=random.randint(3, 8), noise_events=random.randint(2, 5),
            ))
    # Declarative CAMPAIGN_LIBRARY campaigns (review #17): explicit multi-principal
    # object model with auto-derived lineage, generated alongside ATTACK_CHAINS.
    N_PER_CAMPAIGN = 14
    for campaign_name in CAMPAIGN_LIBRARY:
        for _ in range(N_PER_CAMPAIGN):
            all_rows.extend(generate_campaign_session(
                campaign_name, noise_events=random.randint(2, 5)))
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

    # Benign sessions carry no campaign; give the lineage columns explicit empty
    # values (not NaN) so the annotation layer is clean.
    LINEAGE_COLS = ["campaign_id", "chain_name", "attack_tactic", "attack_technique_id",
                    "event_uid", "parent_event_id", "root_event_id",
                    "before_permissions", "after_permissions"]
    for c in LINEAGE_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    for c in ("stage_index", "hop_id"):
        if c not in df.columns:
            df[c] = -1
        df[c] = df[c].fillna(-1).astype(int)

    print(f"Shape: {df.shape}")
    print(f"Label split: benign={  (df['label']==0).sum() }  attack={ (df['label']==1).sum() }")
    print(f"target_resource null rate: {df['target_resource'].isnull().mean():.4f}  (target: 0.2824)")
    print(f"mfa_authenticated null rate: {df['mfa_authenticated'].isnull().mean():.4f}  (target: 0.6913)")
    print(f"principal_type distribution:\n{(df['principal_type'].value_counts(normalize=True)*100).round(2)}")
    print("(real targets: IAMUser 66.93 / AWSService 18.45 / AssumedRole 11.66 / unknown 1.67 / Root 1.30)")

    df.to_csv("synthetic_cloudtrail.csv", index=False)
    print("\nSaved synthetic_cloudtrail.csv")

    # Joinable campaign/lineage annotation layer (review points 2, 5, 14, 16).
    # feature_engine9 assigns log_id = "<input filename>:<row index>", reading
    # rows top-to-bottom in THIS written order, so row i here == log_id
    # "synthetic_cloudtrail.csv:i". Emitting the annotation with that exact key
    # gives a strict 1:1 join to the structural graph. These columns are ground
    # truth for evaluating attack progression and are NOT model features.
    ann = df[LINEAGE_COLS + ["stage_index", "hop_id", "label"]].copy()
    ann.insert(0, "log_id", [f"synthetic_cloudtrail.csv:{i}" for i in range(len(df))])
    ann.to_csv("synthetic_campaign_annotations.csv", index=False)
    n_camp = df.loc[df.campaign_id != "", "campaign_id"].nunique()
    print(f"Saved synthetic_campaign_annotations.csv ({n_camp} campaigns, "
          f"{int((df.stage_index >= 0).sum())} labelled chain-stage events)")


if __name__ == "__main__":
    main()
