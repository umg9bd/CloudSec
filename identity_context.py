"""
identity_context.py
===================
Per-event identity and permission context, inferred ONLY from what the event
stream itself shows (the observable prefix). Feeds feature_engine9.

WHAT IT INFERS, AND FROM WHICH OBSERVABLE FIELDS
------------------------------------------------
  * Identity handoffs. CloudTrail links an assumed-role session to the call
    that created it: AssumeRole's requestParameters carry roleArn and
    roleSessionName, and every later call made with those credentials is
    recorded as arn:aws:sts::ACCT:assumed-role/ROLE/SESSION. Matching
    (ROLE, SESSION) is an exact, observable link -- not a guess from row order.
    Likewise a user whose credentials were issued by someone else
    (CreateUser / CreateAccessKey / CreateLoginProfile / UpdateLoginProfile
    naming that userName) is linked to the issuer when the user later acts.
  * Lineage depth: how many observed handoffs separate the acting identity
    from an identity with no observed origin.
  * Lineage enabling steps: how many identity-enabling steps (a permission
    change detected from policy CONTENT, a credential issuance, a role
    assumption) the acting identity and its ancestors performed in the last
    hour.
  * Permission changes: before/after permission sets from iam_permissions,
    giving new_permission_count, permission_expansion_score, privilege_delta.
  * Privilege level: highest AWS access level the actor is known to hold
    (from observed grants) or has demonstrated (successfully performed), on
    the project's one documented access-level scale
    (privilege_features.ACCESS_LEVEL_RANK, levels from AWS's Service
    Authorization Reference via policy_sentry).

NO GROUND TRUTH IS READ. Labels, splits and any campaign / lineage annotations
a dataset carries are never consulted here -- see
feature_engine9.GROUND_TRUTH_COLUMNS and tests/test_identity_features.py.

Failed calls (non-empty error_code) change nothing: a denied AttachRolePolicy
granted nothing, and a denied AssumeRole issued no session.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import iam_permissions as ip
from privilege_features import ASSUME_ACTIONS, ActionAccessLevelResolver

# API calls that hand an identity's credentials to whoever called them. This is
# what the calls DO (AWS API semantics), not a judgment about how risky they are.
CREDENTIAL_ISSUING = {"CreateUser", "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile"}

LINEAGE_WINDOW = timedelta(hours=1)
MAX_LINEAGE_DEPTH = 10          # cycle / runaway guard when walking ancestors
DEPTH_NORMALIZER = 3.0          # depth 3+ -> 1.0
STEPS_NORMALIZER = 5.0          # 5+ enabling steps in the window -> 1.0

_resolver = None


def _access_rank(event_name):
    global _resolver
    if _resolver is None:
        _resolver = ActionAccessLevelResolver()
    rank = _resolver.rank(event_name or "")
    return -1 if rank is None else rank


def _params(log):
    raw = log.get("request_params_raw")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _target_identity(params):
    """Permission-state key of the identity an event acts ON (roleName, roleArn or userName)."""
    if params.get("roleName"):
        return f"role/{params['roleName']}"
    role = ip.role_name_from_arn(params.get("roleArn") or "")
    if role:
        return f"role/{role}"
    if params.get("userName"):
        return f"user/{params['userName']}"
    return None


def actor_node(log):
    """Stable node id for the identity that performed an event."""
    arn = log.get("principal_arn")
    if arn and arn != "unknown_principal":
        return str(arn)
    return f"service:{log.get('username') or 'unknown'}"


def _account_id(log):
    acct = log.get("recipient_account_id")
    if acct:
        return str(acct)
    parts = str(log.get("principal_arn") or "").split(":")
    return parts[4] if len(parts) >= 5 and parts[4].isdigit() else None


class IdentityContext:
    """Stateful over the stream; persisted as JSON at `path` like the other trackers."""

    def __init__(self, path=None):
        self.path = path
        self.perms = ip.PermissionState()
        self.pending_sessions = {}   # "ROLE|SESSION" -> issuing node
        self.issued = {}             # "user/NAME" -> issuing node
        self.parent = {}             # node -> parent node
        self.depth = {}              # node -> lineage depth
        self.steps = {}              # node -> [iso timestamps of enabling steps]
        self.demonstrated = {}       # node -> highest access rank successfully performed
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._load(json.load(f))

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self, raw):
        self.perms = ip.PermissionState.from_json(raw.get("permissions"))
        self.pending_sessions = raw.get("pending_sessions", {})
        self.issued = raw.get("issued", {})
        self.parent = raw.get("parent", {})
        self.depth = raw.get("depth", {})
        self.steps = raw.get("steps", {})
        self.demonstrated = raw.get("demonstrated", {})

    def save(self):
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({
                "permissions": self.perms.to_json(),
                "pending_sessions": self.pending_sessions,
                "issued": self.issued,
                "parent": self.parent,
                "depth": self.depth,
                "steps": self.steps,
                "demonstrated": self.demonstrated,
            }, f, indent=2, sort_keys=True)

    # ── lineage ─────────────────────────────────────────────────────────────
    def _resolve_parent(self, node, log):
        if node in self.parent:
            return
        origin = None
        session = ip.assumed_session(log.get("principal_arn"))
        if session:
            origin = self.pending_sessions.get(f"{session[0]}|{session[1]}")
        else:
            key = ip.actor_key(log.get("principal_type"), log.get("principal_arn"), log.get("username"))
            if key and key.startswith("user/"):
                origin = self.issued.get(key)
        if origin and origin != node:
            self.parent[node] = origin
            self.depth[node] = min(self.depth.get(origin, 0) + 1, MAX_LINEAGE_DEPTH)

    def _lineage(self, node):
        chain, seen = [node], {node}
        while len(chain) <= MAX_LINEAGE_DEPTH:
            up = self.parent.get(chain[-1])
            if up is None or up in seen:
                break
            chain.append(up)
            seen.add(up)
        return chain

    def _recent_steps(self, node, ts):
        cutoff = ts - LINEAGE_WINDOW
        return sum(1 for n in self._lineage(node) for t in self.steps.get(n, ())
                   if datetime.fromisoformat(t) >= cutoff)

    def _record_step(self, node, ts):
        cutoff = ts - LINEAGE_WINDOW
        kept = [t for t in self.steps.get(node, ()) if datetime.fromisoformat(t) >= cutoff]
        kept.append(ts.isoformat())
        self.steps[node] = kept

    # ── one event ───────────────────────────────────────────────────────────
    def observe(self, log, ts):
        """Features for one event, computed from the state BEFORE it (actor
        context) and from the event's own observable effect (permission change).
        Then folds the event into the state."""
        node = actor_node(log)
        self._resolve_parent(node, log)
        key = ip.actor_key(log.get("principal_type"), log.get("principal_arn"), log.get("username"))
        event_name = log.get("event_name") or ""
        succeeded = not log.get("error_code")
        params = _params(log)

        known_canon, _ = self.perms.effective(key) if key else ((), False)
        known_rank = ip.max_access_rank(known_canon) if known_canon else -1

        features = {
            "principal_handoff": 1 if node in self.parent else 0,
            "causal_depth_normalized": min(self.depth.get(node, 0) / DEPTH_NORMALIZER, 1.0),
            "lineage_enabling_steps_normalized": min(self._recent_steps(node, ts) / STEPS_NORMALIZER, 1.0),
            "actor_permission_coverage": ip.coverage(known_canon) if known_canon else 0.0,
            "new_permission_count_log": 0.0,
            "permission_expansion_score": 0.0,
            "privilege_delta": 0.0,
            "target_permission_coverage": 0.0,
        }

        if succeeded:
            enabling = False
            changes = self.perms.apply(event_name, params, _account_id(log)) if params else []
            if changes:
                enabling = True
                target_key = _target_identity(params)
                target, before, after = next((c for c in changes if c[0] == target_key), changes[0])
                universe = len(ip._action_universe()[0])
                gained = max(0, ip.allowed_count(after) - ip.allowed_count(before))
                features["new_permission_count_log"] = math.log1p(gained) / math.log1p(universe)
                features["permission_expansion_score"] = max(0.0, ip.coverage(after) - ip.coverage(before))
                rank_before = ip.max_access_rank(before) if before else -1
                rank_after = ip.max_access_rank(after) if after else -1
                features["privilege_delta"] = max(-1.0, min(1.0, (rank_after - rank_before) / 5.0))
            if params and event_name in ASSUME_ACTIONS and params.get("roleArn") and params.get("roleSessionName"):
                role = ip.role_name_from_arn(params["roleArn"])
                if role:
                    self.pending_sessions[f"{role}|{params['roleSessionName']}"] = node
                    enabling = True
            if params and event_name in CREDENTIAL_ISSUING and params.get("userName"):
                self.issued[f"user/{params['userName']}"] = node
                enabling = True
            if enabling:
                self._record_step(node, ts)
            self.demonstrated[node] = max(self.demonstrated.get(node, -1), _access_rank(event_name))

        if params:
            target_key = _target_identity(params)
            if target_key:
                canon, _ = self.perms.effective(target_key)
                features["target_permission_coverage"] = ip.coverage(canon) if canon else 0.0

        # 0 = nothing known; 1..5 = ACCESS_LEVEL_RANK + 1 (List .. Permissions management).
        features["source_privilege_level"] = 1 + max(known_rank, self.demonstrated.get(node, -1))
        return features
