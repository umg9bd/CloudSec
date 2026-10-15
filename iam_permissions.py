"""
iam_permissions.py
==================
IAM permission state that can be tracked from a CloudTrail event stream, shared
by the synthetic generator (to emit permission changes with real semantics) and
by the feature engine (to infer what those changes did, from the events alone).

WHY THIS EXISTS
---------------
The synthetic data used to contain AttachRolePolicy / PutRolePolicy /
CreatePolicyVersion rows whose request parameters were just {"roleName": ...}:
no policy ARN, no policy document, no before/after. A feature such as
"privilege delta" could therefore only be faked from the event NAME, never
derived from what the event actually granted. With this module:

  * a permission-changing event carries what real CloudTrail carries
    (policyArn, policyDocument, groupName, versionId ...), and
  * `PermissionState.apply()` turns that into a before/after permission set
    for the identity it changed, so `new_permission_count`,
    `permission_expansion_score` and `privilege_delta` are computed from
    observed policy content.

WHAT IS AND IS NOT MODELED (state it in a methods section as-is)
----------------------------------------------------------------
  * Allow statements, with Action or NotAction. Deny statements, conditions,
    permission boundaries and SCPs are NOT evaluated.
  * Resource scope does not enter coverage: coverage measures the BREADTH of
    actions granted. Resource wildcards are exposed separately
    (feature_engine9.parse_policy_features -> has_wildcard_resource).
  * Only permissions visible in the stream are known. A pre-existing role's
    policies, attached before the log starts, are unknown -- `effective()`
    reports that as `known=False` instead of assuming "no permissions".
  * AWS managed policies are known by ARN (their documents are public).
    MANAGED_POLICIES holds ABRIDGED action lists for the ones this project
    uses -- faithful to each policy's intent and breadth, not verbatim copies.

Coverage and access level are measured against AWS's own action catalogue
(policy_sentry's offline copy of the Service Authorization Reference, ~20k
actions), not against a hand-picked list. The access-level ORDER used for
`max_access_rank` is privilege_features.ACCESS_LEVEL_RANK, the project's one
documented convention for it.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_construction"))
from privilege_features import ACCESS_LEVEL_RANK  # noqa: E402

log = logging.getLogger(__name__)

ALLOW_ALL = ({"Effect": "Allow", "Action": ["*"]},)

# ── AWS managed policies (abridged, see module docstring) ────────────────────
_P = "arn:aws:iam::aws:policy/"
MANAGED_POLICIES = {
    _P + "AdministratorAccess": [{"Effect": "Allow", "Action": ["*"]}],
    _P + "PowerUserAccess": [
        {"Effect": "Allow", "NotAction": ["iam:*", "organizations:*", "account:*"]},
        {"Effect": "Allow", "Action": ["iam:CreateServiceLinkedRole", "iam:DeleteServiceLinkedRole",
                                        "iam:ListRoles", "organizations:DescribeOrganization",
                                        "account:ListRegions"]},
    ],
    _P + "IAMFullAccess": [{"Effect": "Allow", "Action": ["iam:*", "organizations:DescribeAccount",
                                                           "organizations:DescribeOrganization",
                                                           "organizations:ListRoots"]}],
    _P + "ReadOnlyAccess": [{"Effect": "Allow", "Action": ["*:Describe*", "*:Get*", "*:List*",
                                                            "*:BatchGet*", "*:Lookup*"]}],
    _P + "SecurityAudit": [{"Effect": "Allow", "Action": ["*:Describe*", "*:List*",
                                                           "*:GetBucketPolicy", "*:GetAccountSummary",
                                                           "iam:GetAccountAuthorizationDetails"]}],
    _P + "SecretsManagerReadWrite": [{"Effect": "Allow", "Action": [
        "secretsmanager:*", "kms:Decrypt", "kms:DescribeKey", "kms:ListAliases",
        "kms:ListKeys", "rds:DescribeDBInstances", "lambda:ListFunctions", "ec2:DescribeVpcs"]}],
    _P + "AmazonSSMReadOnlyAccess": [{"Effect": "Allow", "Action": ["ssm:Describe*", "ssm:Get*",
                                                                     "ssm:List*"]}],
    _P + "AmazonSSMFullAccess": [{"Effect": "Allow", "Action": ["ssm:*", "ec2:DescribeInstances",
                                                                 "ds:CreateComputer", "ds:DescribeDirectories"]}],
    _P + "AmazonSSMManagedInstanceCore": [{"Effect": "Allow", "Action": [
        "ssm:DescribeAssociation", "ssm:GetDeployablePatchSnapshotForInstance", "ssm:GetDocument",
        "ssm:DescribeDocument", "ssm:GetManifest", "ssm:GetParameter", "ssm:GetParameters",
        "ssm:ListAssociations", "ssm:ListInstanceAssociations", "ssm:PutInventory",
        "ssm:UpdateInstanceInformation", "ssmmessages:*", "ec2messages:*"]}],
    _P + "AmazonS3ReadOnlyAccess": [{"Effect": "Allow", "Action": ["s3:Get*", "s3:List*",
                                                                    "s3-object-lambda:Get*",
                                                                    "s3-object-lambda:List*"]}],
    _P + "AmazonS3FullAccess": [{"Effect": "Allow", "Action": ["s3:*", "s3-object-lambda:*"]}],
    _P + "AmazonEC2ReadOnlyAccess": [{"Effect": "Allow", "Action": [
        "ec2:Describe*", "elasticloadbalancing:Describe*", "cloudwatch:ListMetrics",
        "cloudwatch:GetMetricStatistics", "cloudwatch:Describe*", "autoscaling:Describe*"]}],
    _P + "AmazonEC2FullAccess": [{"Effect": "Allow", "Action": [
        "ec2:*", "elasticloadbalancing:*", "cloudwatch:*", "autoscaling:*",
        "iam:CreateServiceLinkedRole"]}],
    _P + "AWSLambda_ReadOnlyAccess": [{"Effect": "Allow", "Action": ["lambda:Get*", "lambda:List*",
                                                                      "iam:GetRole", "iam:ListRoles",
                                                                      "logs:DescribeLogGroups"]}],
    _P + "AWSCloudTrail_ReadOnlyAccess": [{"Effect": "Allow", "Action": ["cloudtrail:Get*",
                                                                          "cloudtrail:Describe*",
                                                                          "cloudtrail:List*",
                                                                          "cloudtrail:LookupEvents"]}],
    _P + "AmazonRDSReadOnlyAccess": [{"Effect": "Allow", "Action": ["rds:Describe*", "rds:List*",
                                                                     "cloudwatch:GetMetricStatistics"]}],
}

# Policies an attacker attaches to gain broad control, vs. the scoped ones an
# engineer attaches for a job. Used by the GENERATOR to pick realistic policies;
# the feature engine never sees this split, only the policy contents.
PRIVILEGED_MANAGED = [_P + "AdministratorAccess", _P + "PowerUserAccess", _P + "IAMFullAccess"]
SCOPED_MANAGED = [p for p in MANAGED_POLICIES if p not in PRIVILEGED_MANAGED]


# ── AWS action universe (policy_sentry, offline) ─────────────────────────────

@lru_cache(maxsize=1)
def _action_universe():
    """([ 'service:action' lower-case ], [access-level rank]) from policy_sentry."""
    try:
        from policy_sentry.shared.constants import DATASTORE_FILE_PATH
        with open(DATASTORE_FILE_PATH, encoding="utf-8") as f:
            db = json.load(f)
    except Exception as exc:  # ImportError, FileNotFoundError
        log.warning("policy_sentry unavailable (%s): permission coverage falls back to a "
                    "tiny universe and is not comparable across environments.", exc)
        names = ["iam:attachrolepolicy", "iam:createuser", "s3:getobject", "ec2:describeinstances",
                 "secretsmanager:getsecretvalue", "ssm:getparameters", "sts:assumerole"]
        return names, [4, 4, 1, 0, 1, 1, 3]
    names, ranks = [], []
    for svc in db.values():
        if not isinstance(svc, dict) or "privileges" not in svc:
            continue
        prefix = svc["prefix"].lower()
        for action, info in svc["privileges"].items():
            names.append(f"{prefix}:{action.lower()}")
            ranks.append(ACCESS_LEVEL_RANK.get(info.get("access_level"), 0))
    return names, ranks


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _pattern_regex(patterns):
    parts = [fnmatch.translate(str(p).lower()) for p in patterns]
    return re.compile("|".join(parts)) if parts else None


def canonical_statements(statements) -> tuple:
    """Allow-statements reduced to a hashable, order-independent form:
    (("Action", ("s3:get*", ...)), ("NotAction", (...)), ...)."""
    out = set()
    for st in statements or ():
        if not isinstance(st, dict) or str(st.get("Effect", "Allow")).lower() != "allow":
            continue
        if "NotAction" in st:
            out.add(("NotAction", tuple(sorted(str(a).lower() for a in _as_list(st["NotAction"])))))
        else:
            acts = tuple(sorted(str(a).lower() for a in _as_list(st.get("Action"))))
            if acts:
                out.add(("Action", acts))
    return tuple(sorted(out))


@lru_cache(maxsize=4096)
def _grant_mask(canon: tuple) -> frozenset:
    """Indices into the action universe that `canon` allows."""
    names, _ = _action_universe()
    allowed = set()
    for kind, patterns in canon:
        rx = _pattern_regex(patterns)
        hits = {i for i, n in enumerate(names) if rx.fullmatch(n)}
        allowed |= hits if kind == "Action" else set(range(len(names))) - hits
    return frozenset(allowed)


def coverage(canon: tuple) -> float:
    """Fraction of AWS's action catalogue that these statements allow (0..1)."""
    names, _ = _action_universe()
    return len(_grant_mask(canon)) / len(names) if canon else 0.0


def max_access_rank(canon: tuple) -> int:
    """Highest ACCESS_LEVEL_RANK among allowed actions, -1 when nothing is allowed."""
    _, ranks = _action_universe()
    mask = _grant_mask(canon)
    return max((ranks[i] for i in mask), default=-1)


def allowed_count(canon: tuple) -> int:
    return len(_grant_mask(canon))


# ── Identity keys ─────────────────────────────────────────────────────────────

_ASSUMED_ROLE_RE = re.compile(r":assumed-role/([^/]+)/(.+)$")


def actor_key(principal_type, principal_arn, username=None):
    """Permission-state key of the identity that PERFORMED an event, or None."""
    ptype = str(principal_type or "")
    arn = str(principal_arn or "")
    if ptype == "Root" or arn.endswith(":root"):
        return "root"
    m = _ASSUMED_ROLE_RE.search(arn)
    if m:
        return f"role/{m.group(1)}"
    if ptype == "IAMUser":
        name = arn.rsplit("/", 1)[-1] if "/" in arn else username
        return f"user/{name}" if name else None
    return None


def assumed_session(principal_arn):
    """(role_name, session_name) for an STS assumed-role ARN, else None."""
    m = _ASSUMED_ROLE_RE.search(str(principal_arn or ""))
    return (m.group(1), m.group(2)) if m else None


def role_name_from_arn(arn):
    return str(arn).rsplit("/", 1)[-1] if ":role/" in str(arn) else None


def _parse_doc(doc):
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except ValueError:
            return []
    if not isinstance(doc, dict):
        return []
    st = doc.get("Statement", [])
    return [st] if isinstance(st, dict) else [s for s in st if isinstance(s, dict)]


# ── Permission state ──────────────────────────────────────────────────────────

class PermissionState:
    """What the event stream has revealed about each identity's permissions."""

    def __init__(self):
        self.attached = defaultdict(set)        # identity key -> {policy ARN}
        self.inline = defaultdict(dict)         # identity key -> {policy name: statements}
        self.groups_of = defaultdict(set)       # user key -> {group key}
        self.policy_versions = {}               # customer policy ARN -> {"versions": {vid: stmts}, "default": vid}
        self.observed = set()                   # identity keys whose permissions were changed in-stream

    # -- policy content -------------------------------------------------------
    def policy_statements(self, policy_arn):
        """Statements of a policy, or None if its content is unknown."""
        if policy_arn in MANAGED_POLICIES:
            return MANAGED_POLICIES[policy_arn]
        pv = self.policy_versions.get(policy_arn)
        if pv and pv["default"] in pv["versions"]:
            return pv["versions"][pv["default"]]
        return None

    def effective(self, key):
        """(canonical statements, known) for an identity. `known` is False when
        the identity has an attached policy whose content was never observed,
        or nothing about it was observed at all."""
        if key == "root":
            return canonical_statements(ALLOW_ALL), True
        statements, known = [], key in self.observed
        sources = [key] + sorted(self.groups_of.get(key, ()))
        for src in sources:
            for arn in self.attached.get(src, ()):
                st = self.policy_statements(arn)
                if st is None:
                    known = False
                else:
                    statements.extend(st)
            for st in self.inline.get(src, {}).values():
                statements.extend(st)
        return canonical_statements(statements), known

    # -- applying one event ---------------------------------------------------
    def apply(self, event_name, params, account_id=None):
        """Updates state for one successful event. Returns [(identity_key, before, after)]
        -- one entry per identity whose effective permissions the event could change."""
        if not isinstance(params, dict):
            return []
        acct = account_id or "unknown"
        role = params.get("roleName")
        user = params.get("userName")
        group = params.get("groupName")
        targets = []

        def _target_key():
            if role:
                return f"role/{role}"
            if user:
                return f"user/{user}"
            if group:
                return f"group/{group}"
            return None

        key = _target_key()
        affected = [key] if key else []
        if key and key.startswith("group/"):
            affected = [u for u, gs in self.groups_of.items() if key in gs] + [key]
        before = {k: self.effective(k)[0] for k in affected}

        if event_name in ("AttachRolePolicy", "AttachUserPolicy", "AttachGroupPolicy") and key:
            self.attached[key].add(params.get("policyArn"))
        elif event_name in ("DetachRolePolicy", "DetachUserPolicy", "DetachGroupPolicy") and key:
            self.attached[key].discard(params.get("policyArn"))
        elif event_name in ("PutRolePolicy", "PutUserPolicy", "PutGroupPolicy") and key:
            self.inline[key][params.get("policyName", "inline")] = _parse_doc(params.get("policyDocument"))
        elif event_name in ("DeleteRolePolicy", "DeleteUserPolicy", "DeleteGroupPolicy") and key:
            self.inline[key].pop(params.get("policyName", "inline"), None)
        elif event_name == "AddUserToGroup" and user and group:
            self.groups_of[f"user/{user}"].add(f"group/{group}")
            affected, key = [f"user/{user}"], f"user/{user}"
            before = {key: before.get(key, self.effective(key)[0])}
        elif event_name == "RemoveUserFromGroup" and user and group:
            self.groups_of[f"user/{user}"].discard(f"group/{group}")
            affected, key = [f"user/{user}"], f"user/{user}"
        elif event_name == "CreatePolicy":
            arn = f"arn:aws:iam::{acct}:policy/{params.get('policyName')}"
            self.policy_versions[arn] = {"versions": {"v1": _parse_doc(params.get("policyDocument"))},
                                         "default": "v1"}
            return []
        elif event_name in ("CreatePolicyVersion", "SetDefaultPolicyVersion"):
            arn = params.get("policyArn")
            holders = [k for k, arns in self.attached.items() if arn in arns]
            affected = sorted(set(holders) | {u for u, gs in self.groups_of.items()
                                              if any(g in holders for g in gs)})
            before = {k: self.effective(k)[0] for k in affected}
            pv = self.policy_versions.setdefault(arn, {"versions": {}, "default": "v1"})
            if event_name == "CreatePolicyVersion":
                # versionId is only in the RESPONSE, which this schema does not keep; a
                # policy's versions are numbered v1, v2 ... by AWS, and the pre-log v1
                # is by definition unobserved, so observed versions are numbered after it.
                vid = f"v{len(pv['versions']) + 2}" if "v1" not in pv["versions"] else f"v{len(pv['versions']) + 1}"
                pv["versions"][vid] = _parse_doc(params.get("policyDocument"))
                if str(params.get("setAsDefault", "")).lower() == "true":
                    pv["default"] = vid
            else:
                vid = params.get("versionId")
                if vid not in pv["versions"] and pv["versions"]:
                    vid = sorted(pv["versions"], key=lambda v: int(v.lstrip("v") or 0))[-1]
                pv["default"] = vid
        else:
            return []

        for k in affected:
            if not k.startswith("group/"):
                self.observed.add(k)
                targets.append((k, before.get(k, ()), self.effective(k)[0]))
        return targets

    # -- persistence ----------------------------------------------------------
    def to_json(self):
        return {
            "attached": {k: sorted(v) for k, v in self.attached.items() if v},
            "inline": {k: v for k, v in self.inline.items() if v},
            "groups_of": {k: sorted(v) for k, v in self.groups_of.items() if v},
            "policy_versions": self.policy_versions,
            "observed": sorted(self.observed),
        }

    @classmethod
    def from_json(cls, data):
        st = cls()
        for k, v in (data or {}).get("attached", {}).items():
            st.attached[k] = set(v)
        for k, v in (data or {}).get("inline", {}).items():
            st.inline[k] = dict(v)
        for k, v in (data or {}).get("groups_of", {}).items():
            st.groups_of[k] = set(v)
        st.policy_versions = dict((data or {}).get("policy_versions", {}))
        st.observed = set((data or {}).get("observed", []))
        return st
