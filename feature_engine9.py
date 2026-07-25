"""
feature_engine9.py
===================
Adds per-node graph attributes to the structural (GNN triples) output,
computed incrementally as each event streams through -- no change to the
temporal branch, no new dataset needed.

New columns on cloudtrail_structural.csv (one row per event/edge, same as
fe8), all derived from the acting principal's (source_node's) event
history so far:

  - source_node_degree          running count of edges touching this
                                 principal (both as source and target)
  - edge_interaction_count      running count of this exact
                                 (source_node, target_node) pair
  - source_node_age_normalized  time since this principal's first-ever
                                 event, capped at 1 day -> [0, 1]
  - source_privilege_level      0-4 tier (ReadOnly..Root), a running MAX
                                 per principal -- inferred from principal_type
                                 plus the same policy signals fe7/fe8 already
                                 parse (wildcard actions, AdministratorAccess-
                                 style policyArn, AssumeRole/PassRole). Not a
                                 raw log field: CloudTrail (and AWS itself)
                                 has no "current privilege level" field, so
                                 this is a heuristic, same spirit as
                                 action_risk_prior/principal_type_prior_risk.
  - source_historical_risk      EWMA of action_risk_prior per principal
  - target_resource_criticality static hand-scored lookup on
                                 event_source/target_resource (same style as
                                 the existing action_map/principal_risk_prior
                                 hand-tuned priors)
  - is_cross_account            1 if the principal's AWS account (parsed out
                                 of principal_arn) differs from the trail's
                                 recipientAccountId (real CloudTrail JSON) --
                                 or, when that field isn't present (this
                                 project's synthetic CSV), from whichever
                                 account has been seen most often so far

See GraphNodeTracker below for the implementation and persistence.

Deliberately NOT added here: reachable-asset counts, privilege escalation
depth, cross-service reachability, lateral movement score, shortest attack
path, blast-radius/propagation score. Those all require traversing the
*whole* graph (BFS/shortest-path/reachability), not just the current event,
so they can't be computed correctly -- or cheaply -- inside a one-event-at-
a-time streaming engine. They belong in a separate downstream "Blast Radius
Engine" that reads the structural CSV (or graph_node_state.json) after
enough of the graph has accumulated, not here.

fe8's design (unchanged, still in effect): the JSON/CSV ingestion, the
event_name/event_source/principal_type vocab indices, StateTracker's
velocity/session tracking, the policy/permission parsing, the read_only/
mfa_authenticated absence-vs-false distinction, and the temporal branch's
feature set (TEMPORAL_COLS) are all carried over as-is. See feature_engine8.py
for that history.

Run:
    python feature_engine9.py
        One-shot batch over synthetic_cloudtrail.csv.

    python feature_engine9.py --watch incoming/ [--simulate]
        Same watch-folder mode as fe6-fe8.

NOTE: cloudtrail_structural.csv's column layout changed (7 new columns).
Delete any existing cloudtrail_structural.csv before the first fe9 run --
_check_output_schema will refuse to append mismatched rows otherwise.
cloudtrail_temporal.csv's schema is unchanged, so it does NOT need to be
deleted.
"""

import argparse
import csv
import gzip
import json
import os
import time
import uuid
from datetime import datetime, timezone

import ipaddress
import numpy as np


def _parse_json_text(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def _is_cloudtrail_json_path(input_path):
    lower_path = input_path.lower()
    return lower_path.endswith(('.json', '.jsonl', '.ndjson', '.json.gz', '.jsonl.gz', '.ndjson.gz'))


def _is_cloudtrail_csv_path(input_path):
    lower_path = input_path.lower()
    return lower_path.endswith(('.csv', '.csv.gz'))


def _open_input_text(input_path):
    if input_path.lower().endswith('.gz'):
        return gzip.open(input_path, mode='rt', encoding='utf-8')
    return open(input_path, mode='r', encoding='utf-8')


def _absent_or_present(value):
    """None/''/missing -> None (absent). Anything else -> the raw value."""
    if value is None or value == '':
        return None
    return value


def normalize_cloudtrail_row(row):
    """Map raw CloudTrail fields or enriched aliases onto the internal schema."""

    user_identity = _parse_json_text(row.get('userIdentity')) or {}
    session_context = user_identity.get('sessionContext') or {}
    session_attributes = session_context.get('attributes') or {}

    # For AssumedRole/FederatedUser events, userIdentity.arn is suffixed with
    # a unique session name (e.g. .../EC2_Service_Role/i-0abc123), so it's
    # different on every session. sessionIssuer.arn is the stable underlying
    # role ARN and is what should identify the principal across sessions --
    # falling back to userIdentity.arn only for principal types (IAMUser,
    # Root) that don't have a sessionIssuer at all.
    session_issuer = session_context.get('sessionIssuer') or {}
    principal_arn = row.get('principal_arn') or session_issuer.get('arn') or user_identity.get('arn')
    principal_type = row.get('principal_type') or user_identity.get('type') or 'Unknown'

    # Same stability concern as principal_arn: for AssumedRole/FederatedUser
    # events userIdentity.userName is usually absent, so fall back to the
    # role name off sessionIssuer.arn (the human/service identity that
    # actually owns the role), then to the tail of principal_arn itself.
    username = row.get('username') or user_identity.get('userName') or session_issuer.get('userName')
    if not username and principal_arn:
        username = principal_arn.rsplit('/', 1)[-1]

    request_params = row.get('request_params_raw')
    if request_params is None:
        request_params = row.get('requestParameters')
        if isinstance(request_params, (dict, list)):
            request_params = json.dumps(request_params)

    target_resource = row.get('target_resource')
    if not target_resource:
        resources = row.get('resources') or []
        if isinstance(resources, str):
            resources = _parse_json_text(resources) or []
        if isinstance(resources, list) and resources:
            first_resource = resources[0]
            if isinstance(first_resource, dict):
                target_resource = first_resource.get('ARN') or first_resource.get('arn') or first_resource.get('name')

    # read_only / mfa_authenticated: kept as None when genuinely absent from
    # the source log, instead of defaulting to a guessed value -- see fe8.
    read_only = _absent_or_present(row.get('read_only'))
    if read_only is None:
        read_only = _absent_or_present(row.get('readOnly'))

    mfa_authenticated = _absent_or_present(row.get('mfa_authenticated'))
    if mfa_authenticated is None:
        mfa_authenticated = _absent_or_present(session_attributes.get('mfaAuthenticated'))

    # recipientAccountId is the account a real CloudTrail trail belongs to --
    # present at the top level of real CloudTrail JSON, absent from this
    # project's synthetic CSV. Used as the authoritative "home account" for
    # is_cross_account when available (see GraphNodeTracker.cross_account_flag).
    recipient_account_id = row.get('recipient_account_id') or row.get('recipientAccountId')

    normalized = {
        'timestamp': row.get('timestamp') or row.get('eventTime'),
        'event_name': row.get('event_name') or row.get('eventName'),
        'event_source': row.get('event_source') or row.get('eventSource') or '',
        'principal_type': principal_type,
        'principal_arn': principal_arn or 'unknown_principal',
        'username': username or 'unknown_user',
        'source_ip': row.get('source_ip') or row.get('sourceIPAddress') or '0.0.0.0',
        'user_agent': row.get('user_agent') or row.get('userAgent') or '',
        'read_only': read_only,
        'aws_region': row.get('aws_region') or row.get('awsRegion') or 'us-east-1',
        'mfa_authenticated': mfa_authenticated,
        'error_code': row.get('error_code') or row.get('errorCode') or '',
        'target_resource': target_resource or 'aws_service',
        'label': row.get('label', '0'),
        'request_params_raw': request_params or '{}',
        'access_key_id': row.get('access_key_id') or row.get('accessKeyId') or user_identity.get('accessKeyId') or '',
        'recipient_account_id': str(recipient_account_id) if recipient_account_id else '',
    }

    return normalized


def iter_input_rows(input_path):
    with _open_input_text(input_path) as infile:
        if _is_cloudtrail_json_path(input_path):
            first_non_ws = ''
            pos = infile.tell()
            while True:
                char = infile.read(1)
                if not char:
                    break
                if not char.isspace():
                    first_non_ws = char
                    break
            infile.seek(pos)

            # A real CloudTrail file delivered to S3 is ONE JSON object shaped
            # like {"Records": [...]}, not a top-level array -- so both '{'
            # and '[' need to attempt the whole-document parse. If the file is
            # actually NDJSON (one object per line), json.load() will fail
            # with "Extra data" once it hits the second line's worth of
            # content, and we fall back to line-by-line parsing below.
            if first_non_ws in ('{', '['):
                try:
                    payload = json.load(infile)
                except json.JSONDecodeError:
                    infile.seek(pos)
                    payload = None

                if payload is not None:
                    if isinstance(payload, dict) and 'Records' in payload:
                        rows = payload.get('Records')
                    elif isinstance(payload, list):
                        rows = payload
                    else:
                        rows = [payload]  # single raw event object, no wrapper

                    if isinstance(rows, list):
                        for row in rows:
                            if isinstance(row, dict):
                                yield normalize_cloudtrail_row(row)
                    return

            for line in infile:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield normalize_cloudtrail_row(row)
            return

        if not _is_cloudtrail_csv_path(input_path):
            raise ValueError(f"Unsupported input format: {input_path}")

        reader = csv.DictReader(infile)
        for row in reader:
            yield normalize_cloudtrail_row(row)


# ── Vocabularies for categorical -> index encoding (for a future embedding
#    layer) ─────────────────────────────────────────────────────────────────

UNK_TOKEN = "<UNK>"

# AWS's actual documented userIdentity.type values. Fixed/closed: anything
# not on this list (there shouldn't be anything) maps to <UNK> rather than
# growing the vocab, since this is a small, AWS-defined enum.
FIXED_PRINCIPAL_TYPES = [
    UNK_TOKEN, "Root", "IAMUser", "AssumedRole", "FederatedUser",
    "AWSAccount", "AWSService", "SAMLUser", "WebIdentityUser", "Unknown",
]

# Scoped to the services this project's Stratus Red Team techniques and
# background traffic actually touch (see stratus_collection/stratus_techniques.py),
# not all ~300 AWS service endpoints. Fixed/closed with <UNK> fallback.
FIXED_EVENT_SOURCES = [
    UNK_TOKEN, "iam.amazonaws.com", "sts.amazonaws.com", "ec2.amazonaws.com",
    "s3.amazonaws.com", "secretsmanager.amazonaws.com", "ssm.amazonaws.com",
    "cloudtrail.amazonaws.com", "lambda.amazonaws.com", "kms.amazonaws.com",
]


class VocabIndex:
    """Maps categorical string values to stable integer indices.

    Fixed vocabs (growable=False) never add new entries -- unseen values
    map to <UNK>. Growable vocabs (event_name) start from a persisted JSON
    file and append newly-seen values, so indices stay stable across runs.
    Once a model is trained against a growable vocab file, stop growing it
    (or the embedding table size drifts out from under the trained model).
    """

    def __init__(self, fixed_tokens=None, path=None):
        self.path = path
        self.growable = fixed_tokens is None
        if fixed_tokens is not None:
            self.token_to_idx = {t: i for i, t in enumerate(fixed_tokens)}
        elif path and os.path.exists(path):
            self.token_to_idx = json.loads(open(path, encoding='utf-8').read())
        else:
            self.token_to_idx = {UNK_TOKEN: 0}

    def index(self, token):
        token = token or UNK_TOKEN
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        if self.growable:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            return idx
        return self.token_to_idx[UNK_TOKEN]

    def save(self):
        if self.path:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.token_to_idx, f, indent=2, sort_keys=True)


# ── Policy/permission feature parsing ─────────────────────────────────────

PRIVILEGED_POLICY_ARN_MARKERS = ("administratoraccess", "poweruseraccess", "iamfullaccess")


def _extract_statements(policy_doc):
    """policy_doc is a (possibly JSON-string-encoded) IAM policy document."""
    if not policy_doc:
        return []
    doc = policy_doc
    if isinstance(doc, str):
        doc = _parse_json_text(doc)
    if not isinstance(doc, dict):
        return []
    statements = doc.get('Statement')
    if statements is None:
        return []
    if isinstance(statements, dict):
        statements = [statements]
    return statements if isinstance(statements, list) else []


def parse_policy_features(request_params_raw):
    """Returns (statement_count, has_wildcard_action, has_wildcard_resource,
    privileged_action_reach) parsed from a request's policy-bearing params.

    Handles both real CloudTrail's nested policyDocument/
    assumeRolePolicyDocument JSON (with real Statement/Action/Resource) and
    this project's synthetic data, which instead just attaches a
    policyArn (often literally .../AdministratorAccess) with no nested doc.
    """
    params = _parse_json_text(request_params_raw)
    if not isinstance(params, dict):
        return 0, 0, 0, 0.0

    statements = []
    for key in ("policyDocument", "assumeRolePolicyDocument"):
        statements.extend(_extract_statements(params.get(key)))

    has_wildcard_action = 0
    has_wildcard_resource = 0
    reach = 0.0

    for stmt in statements:
        if not isinstance(stmt, dict) or str(stmt.get('Effect', '')).lower() != 'allow':
            continue

        actions = stmt.get('Action', [])
        if isinstance(actions, str):
            actions = [actions]
        for action in actions:
            action_lower = str(action).lower()
            if action_lower == '*' or action_lower.endswith(':*'):
                has_wildcard_action = 1
                reach = max(reach, 1.0)
            elif action_lower in ("sts:assumerole", "iam:passrole"):
                reach = max(reach, 0.6)

        resources = stmt.get('Resource', [])
        if isinstance(resources, str):
            resources = [resources]
        if any(str(r) == '*' for r in resources):
            has_wildcard_resource = 1

    policy_arn = str(params.get('policyArn', '')).lower()
    if any(marker in policy_arn for marker in PRIVILEGED_POLICY_ARN_MARKERS):
        reach = max(reach, 1.0)

    return len(statements), has_wildcard_action, has_wildcard_resource, reach


# ── fe9: resource criticality + privilege-level scoring ───────────────────

DEFAULT_RESOURCE_CRITICALITY = 2


def get_resource_criticality(event_source, target_resource):
    """Static, hand-scored criticality per resource -- same style as the
    existing action_map/principal_risk_prior hand-tuned priors. Computed
    once per event; a downstream Blast Radius Engine reads this instead of
    re-deriving criticality from event_source/target_resource itself."""
    source = (event_source or '').lower()
    target = str(target_resource or '').lower()

    if 'secretsmanager' in source or 'kms' in source:
        return 5
    if 'iam' in source:
        return 5 if any(x in target for x in ('admin', 'root')) else 4
    if 's3' in source:
        return 4 if 'prod' in target else 3
    if 'ec2' in source:
        return 3
    if 'cloudwatch' in source or 'logs' in source:
        return 1
    return DEFAULT_RESOURCE_CRITICALITY


PRIVILEGE_TIERS = {"ReadOnly": 0, "Developer": 1, "PowerUser": 2, "Admin": 3, "Root": 4}


def derive_privilege_signal(principal_type, is_write_action, wildcard_action, privileged_reach):
    """One event's evidence about its principal's privilege tier -- fed into
    GraphNodeTracker's running MAX per principal. CloudTrail only shows
    permission grants as they happen, so a principal's inferred tier can
    only climb as escalating events are observed, never silently regress."""
    if principal_type == 'Root':
        return PRIVILEGE_TIERS['Root']
    if wildcard_action or privileged_reach >= 1.0:
        return PRIVILEGE_TIERS['Admin']
    if privileged_reach >= 0.6:
        return PRIVILEGE_TIERS['PowerUser']
    if is_write_action:
        return PRIVILEGE_TIERS['Developer']
    return PRIVILEGE_TIERS['ReadOnly']


def _extract_account_id(arn):
    """arn:aws:{service}:{region}:{account_id}:{resource} -> account_id."""
    if not arn:
        return None
    parts = arn.split(':')
    if len(parts) >= 5 and parts[0] == 'arn' and parts[4].isdigit():
        return parts[4]
    return None


class GraphNodeTracker:
    """Tracks per-node graph attributes (degree, age, privilege level,
    historical risk) and per-edge interaction counts, incrementally -- one
    CloudTrail event at a time, the same streaming pattern StateTracker
    already uses for velocity/session state.

    These are exactly the attributes flagged as cheap/streamable (Node
    Degree, Edge Interaction Count, Node Age, Privilege Level, Historical
    Risk, Cross-Account Flag) -- as opposed to graph-traversal features
    (reachability, shortest attack path, blast radius), which need the
    whole graph and belong in a separate downstream Blast Radius Engine.

    Note: degree is tracked for every node this project's graph touches
    (principals AND resources), even though the structural CSV only
    surfaces source_node_degree per row -- a resource's degree is still
    available in the persisted state file for that future Blast Radius
    Engine to read directly, without re-deriving it.

    Persisted to `path` so a --watch restart doesn't reset every node back
    to "brand new" (same reasoning as StateTracker/VocabIndex).
    """

    EWMA_ALPHA = 0.3  # weight on the newest risk sample
    NODE_AGE_CAP_SECONDS = 86400.0  # normalize node age against 1 day

    def __init__(self, path=None):
        self.path = path
        self.nodes = {}
        self.edge_counts = {}
        self.account_id_counts = {}
        if path and os.path.exists(path):
            raw = json.loads(open(path, encoding='utf-8').read())
            for node_id, data in raw.get('nodes', {}).items():
                self.nodes[node_id] = {
                    'degree': data['degree'],
                    'first_seen': datetime.fromisoformat(data['first_seen']),
                    'last_seen': datetime.fromisoformat(data['last_seen']),
                    'privilege_tier': data['privilege_tier'],
                    'historical_risk': data['historical_risk'],
                }
            self.edge_counts = raw.get('edge_counts', {})
            self.account_id_counts = raw.get('account_id_counts', {})

    def save(self):
        if not self.path:
            return
        serializable = {
            'nodes': {
                node_id: {
                    'degree': data['degree'],
                    'first_seen': data['first_seen'].isoformat(),
                    'last_seen': data['last_seen'].isoformat(),
                    'privilege_tier': data['privilege_tier'],
                    'historical_risk': data['historical_risk'],
                }
                for node_id, data in self.nodes.items()
            },
            'edge_counts': self.edge_counts,
            'account_id_counts': self.account_id_counts,
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, sort_keys=True)

    def _touch_node(self, node_id, current_ts):
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                'degree': 0,
                'first_seen': current_ts,
                'last_seen': current_ts,
                'privilege_tier': 0,
                'historical_risk': 0.0,
            }
        node = self.nodes[node_id]
        node['degree'] += 1
        node['last_seen'] = current_ts
        return node

    def record_edge(self, source_node, target_node, current_ts, event_risk, privilege_signal):
        """Updates degree/age/interaction-count/risk/privilege for one
        CloudTrail event, and returns the snapshot values to attach to
        that event's structural row."""
        source = self._touch_node(source_node, current_ts)
        self._touch_node(target_node, current_ts)  # target's degree also grows

        edge_key = f"{source_node}||{target_node}"
        self.edge_counts[edge_key] = self.edge_counts.get(edge_key, 0) + 1

        source['historical_risk'] = (
            self.EWMA_ALPHA * event_risk + (1 - self.EWMA_ALPHA) * source['historical_risk']
        )
        source['privilege_tier'] = max(source['privilege_tier'], privilege_signal)

        age_seconds = (current_ts - source['first_seen']).total_seconds()

        return {
            'source_node_degree': source['degree'],
            'edge_interaction_count': self.edge_counts[edge_key],
            'source_node_age_normalized': min(age_seconds / self.NODE_AGE_CAP_SECONDS, 1.0),
            'source_privilege_level': source['privilege_tier'],
            'source_historical_risk': round(source['historical_risk'], 4),
        }

    def cross_account_flag(self, principal_arn, recipient_account_id):
        account_id = _extract_account_id(principal_arn)
        if not account_id:
            return 0
        self.account_id_counts[account_id] = self.account_id_counts.get(account_id, 0) + 1

        if recipient_account_id:
            return 1 if account_id != recipient_account_id else 0

        # Fallback for sources without a trail-level account id (e.g. this
        # project's synthetic CSV): treat whichever account has been seen
        # most often so far as "home", flag anything else as cross-account.
        home_account = max(self.account_id_counts, key=self.account_id_counts.get)
        return 1 if account_id != home_account else 0


# ── Shared feature engineering ─────────────────────────────────────────────

def _parse_timestamp(timestamp_str):
    if 'T' in timestamp_str:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S%z")  # 2026-03-31 10:00:00+0000


SESSION_GAP_SECONDS = 1800  # inactivity gap that starts a new session


class StateTracker:
    """Tracks historical + session behavior to detect Privilege Escalation
    and Velocity (both single-hop and session-level).

    Persisted to `path` (like VocabIndex) so a watch-mode restart doesn't
    reset every principal's velocity/session state back to "brand new" --
    without this, action_velocity/is_new_action/session_duration_normalized
    would all silently reset to their first-ever-seen values on restart,
    even mid-session for a principal that had been active for hours.
    """

    def __init__(self, path=None):
        self.path = path
        self.user_registry = {}
        if path and os.path.exists(path):
            raw = json.loads(open(path, encoding='utf-8').read())
            for p_arn, data in raw.items():
                self.user_registry[p_arn] = {
                    'last_ts': datetime.fromisoformat(data['last_ts']),
                    'actions': set(data['actions']),
                    'session_start_ts': datetime.fromisoformat(data['session_start_ts']),
                    'session_event_count': data['session_event_count'],
                }

    def save(self):
        if not self.path:
            return
        serializable = {
            p_arn: {
                'last_ts': data['last_ts'].isoformat(),
                'actions': sorted(data['actions']),
                'session_start_ts': data['session_start_ts'].isoformat(),
                'session_event_count': data['session_event_count'],
            }
            for p_arn, data in self.user_registry.items()
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, sort_keys=True)

    def get_metrics(self, p_arn, event_name, current_ts):
        if p_arn not in self.user_registry:
            self.user_registry[p_arn] = {
                'last_ts': current_ts,
                'actions': {event_name},
                'session_start_ts': current_ts,
                'session_event_count': 1,
            }
            return 0.0, 1, 0.0, 1.0  # velocity, is_new_action, session_duration_s, events_per_minute

        user_data = self.user_registry[p_arn]

        delta = (current_ts - user_data['last_ts']).total_seconds()
        velocity = max(0, 1 - (delta / 3600))
        is_new_action = 1 if event_name not in user_data['actions'] else 0

        if delta > SESSION_GAP_SECONDS:
            user_data['session_start_ts'] = current_ts
            user_data['session_event_count'] = 1
        else:
            user_data['session_event_count'] += 1

        session_duration = (current_ts - user_data['session_start_ts']).total_seconds()
        session_minutes = max(session_duration / 60.0, 1.0 / 60.0)  # avoid /0 on a session's first event
        events_per_minute = user_data['session_event_count'] / session_minutes

        user_data['last_ts'] = current_ts
        user_data['actions'].add(event_name)

        return velocity, is_new_action, session_duration, events_per_minute


class FeatureEngineer:
    def __init__(self, event_name_vocab_path=None, state_tracker_path=None, graph_state_path=None):
        self.tracker = StateTracker(path=state_tracker_path)
        self.graph_tracker = GraphNodeTracker(path=graph_state_path)
        self.principal_type_vocab = VocabIndex(fixed_tokens=FIXED_PRINCIPAL_TYPES)
        self.event_source_vocab = VocabIndex(fixed_tokens=FIXED_EVENT_SOURCES)
        self.event_name_vocab = VocabIndex(path=event_name_vocab_path)

        self.action_map = {
            # --- 1. RECONNAISSANCE ---
            "GetCallerIdentity": 2,
            "ListBuckets": 2,
            "DescribeInstances": 2,
            "ListUsers": 2,
            "GetAccountAuthorizationDetails": 4,

            # --- 2. PERSISTENCE (Creating Backdoors) ---
            "CreateUser": 7,
            "CreateRole": 7,
            "CreateAccessKey": 8,
            "CreateLoginProfile": 8,

            # --- 3. PRIVILEGE ESCALATION---
            "PutUserPolicy": 10,
            "AttachUserPolicy": 10,
            "UpdateAssumeRolePolicy": 10,
            "PassRole": 9,
            "CreatePolicyVersion": 9,
            "SetDefaultPolicyVersion": 9,

            # --- 4. CREDENTIAL ACCESS & EXFILTRATION ---
            "GetSecretValue": 8,
            "Decrypt": 7,
            "AssumeRole": 6,

            # --- 5. DEFENSE EVASION ---
            "DeleteTrail": 10,
            "StopLogging": 10,
            "UpdateDetector": 9,
            "DeleteFlowLogs": 8,

            # --- 6. COMMAND & CONTROL / EXECUTION ---
            "SendCommand": 8,
            "InvokeFunction": 7,
        }
        self.default_risk = 1

        self.principal_risk_prior = {
            "Root": 1.0, "IAMUser": 0.8, "FederatedUser": 0.6,
            "AssumedRole": 0.5, "AWSService": 0.1,
        }

    def save_state(self):
        self.event_name_vocab.save()
        self.tracker.save()
        self.graph_tracker.save()

    def get_structural_data(self, log):
        """Generates GNN triples, extended (fe9) with per-node graph
        attributes -- see GraphNodeTracker and the module docstring for
        which attributes live here vs. in a future Blast Radius Engine."""
        p_arn = log.get('principal_arn', 'unknown_principal')
        event = log.get('event_name', 'unknown_action')
        target = log.get('target_resource') or 'aws_service'

        timestamp_str = log.get('timestamp') or log.get('eventTime')
        if not timestamp_str:
            raise ValueError(f"row for principal {p_arn!r} has no timestamp/eventTime")
        dt = _parse_timestamp(timestamp_str)

        event_risk = self.action_map.get(event, self.default_risk) / 10.0

        read_only = log.get('read_only')
        is_write_action = 0 if read_only is None else (1 if str(read_only).lower() == 'false' else 0)

        request_params_raw = str(log.get('request_params_raw', '{}'))
        _, wildcard_action, _, privileged_reach = parse_policy_features(request_params_raw)

        privilege_signal = derive_privilege_signal(
            log.get('principal_type', ''), is_write_action, wildcard_action, privileged_reach
        )

        graph_attrs = self.graph_tracker.record_edge(p_arn, target, dt, event_risk, privilege_signal)
        graph_attrs['target_resource_criticality'] = get_resource_criticality(log.get('event_source'), target)
        graph_attrs['is_cross_account'] = self.graph_tracker.cross_account_flag(
            p_arn, log.get('recipient_account_id')
        )

        return {
            'source_node': p_arn,
            'target_node': target,
            'edge_type': event,
            **graph_attrs,
        }

    def get_temporal_features(self, log):
        """Generates the temporal-branch feature vector (see TEMPORAL_COLS).
        Unchanged from fe8."""
        f = []
        p_arn = log.get('principal_arn', 'unknown')

        timestamp_str = log.get('timestamp') or log.get('eventTime')
        if not timestamp_str:
            raise ValueError(f"row for principal {p_arn!r} has no timestamp/eventTime")

        dt = _parse_timestamp(timestamp_str)

        # PILLAR 1: IDENTITY & DYNAMICS
        mfa_raw = log.get('mfa_authenticated')
        if mfa_raw is None:
            f.append(0)  # no_mfa: unknown, not claimed true or false
            f.append(1)  # mfa_absent
        else:
            mfa = str(mfa_raw).lower()
            f.append(1 if mfa in ['false', 'no', 'none', ''] else 0)  # no_mfa
            f.append(0)  # mfa_absent

        p_type = log.get('principal_type', '')
        f.append(self.principal_risk_prior.get(p_type, 0.3))  # principal_type_prior_risk
        f.append(self.principal_type_vocab.index(p_type))  # principal_type_idx

        f.append(1 if log.get('access_key_id') else 0)  # has_access_key

        velocity, is_new_action, session_duration, events_per_minute = self.tracker.get_metrics(
            p_arn, log.get('event_name'), dt
        )
        f.append(velocity)  # action_velocity
        f.append(is_new_action)  # is_new_action
        f.append(min(session_duration / 3600.0, 1.0))  # session_duration_normalized
        f.append(min(events_per_minute / 10.0, 1.0))  # events_per_minute_normalized

        # PILLAR 2: TEMPORAL CONTEXT
        h_rad = 2 * np.pi * dt.hour / 24.0
        f.append(np.sin(h_rad))  # time_sin
        f.append(np.cos(h_rad))  # time_cos
        f.append(1 if dt.weekday() >= 5 else 0)  # is_weekend
        f.append(1 if dt.hour < 9 or dt.hour > 18 else 0)  # is_off_hours

        # PILLAR 3: ACTION INTENT
        event_name = log.get('event_name')
        f.append(self.action_map.get(event_name, self.default_risk) / 10.0)  # action_risk_prior
        f.append(self.event_name_vocab.index(event_name))  # event_name_idx
        f.append(self.event_source_vocab.index(log.get('event_source')))  # event_source_idx

        read_only = log.get('read_only')
        if read_only is None:
            f.append(0)  # is_write_action: unknown, default to the safer (read) assumption
            f.append(1)  # read_only_absent
        else:
            f.append(1 if str(read_only).lower() == 'false' else 0)  # is_write_action
            f.append(0)  # read_only_absent

        err = log.get('error_code', '')
        f.append(1 if err else 0)  # has_error
        f.append(1 if err == "AccessDenied" else 0)  # is_access_denied

        f.append(1 if "iam" in log.get('event_source', '') else 0)  # is_iam_event
        f.append(1 if any(x in (event_name or '') for x in ['Describe', 'List', 'Get']) else 0)  # is_recon_action
        f.append(1 if event_name in ["DeleteTrail", "StopLogging"] else 0)  # is_defense_evasion
        f.append(1 if event_name == "GetCallerIdentity" else 0)  # is_get_caller_identity

        # PILLAR 4: METADATA ANOMALIES
        ua = str(log.get('user_agent', '')).lower()
        f.append(1 if any(x in ua for x in ['kali', 'pacu', 'metasploit', 'requests']) else 0)  # is_malicious_user_agent

        try:
            ip = log.get('source_ip', '0.0.0.0')
            f.append(1 if not ipaddress.ip_address(ip).is_private else 0)  # is_public_ip
        except Exception:
            f.append(1)

        request_params_raw = str(log.get('request_params_raw', '{}'))
        f.append(min(len(request_params_raw) / 750.0, 1.0))  # params_length_normalized

        target_str = str(log.get('target_resource', '')).lower()
        f.append(1 if any(x in target_str for x in ['admin', 'vault', 'prod']) else 0)  # targets_sensitive_resource
        f.append(1 if log.get('aws_region') != "us-east-1" else 0)  # is_non_default_region
        f.append(1 if "Create" in (event_name or '') and "Key" in (event_name or '') else 0)  # is_create_key
        f.append(1 if any(x in log.get('event_source', '') for x in ['secrets', 'kms']) else 0)  # is_secrets_or_kms
        f.append(1 if "Attach" in (event_name or '') or "Put" in (event_name or '') else 0)  # is_permission_modification

        # PILLAR 5: POLICY STRUCTURE
        stmt_count, wildcard_action, wildcard_resource, privileged_reach = parse_policy_features(request_params_raw)
        f.append(min(stmt_count / 5.0, 1.0))  # policy_statement_count_normalized
        f.append(wildcard_action)  # has_wildcard_action
        f.append(wildcard_resource)  # has_wildcard_resource
        f.append(privileged_reach)  # privileged_action_reach

        return f


TEMPORAL_COLS = [
    "no_mfa", "mfa_absent", "principal_type_prior_risk", "principal_type_idx",
    "has_access_key", "action_velocity", "is_new_action",
    "session_duration_normalized", "events_per_minute_normalized",
    "time_sin", "time_cos", "is_weekend", "is_off_hours",
    "action_risk_prior", "event_name_idx", "event_source_idx",
    "is_write_action", "read_only_absent", "has_error", "is_access_denied",
    "is_iam_event", "is_recon_action", "is_defense_evasion", "is_get_caller_identity",
    "is_malicious_user_agent", "is_public_ip", "params_length_normalized",
    "targets_sensitive_resource", "is_non_default_region", "is_create_key",
    "is_secrets_or_kms", "is_permission_modification",
    "policy_statement_count_normalized", "has_wildcard_action",
    "has_wildcard_resource", "privileged_action_reach",
]

# fe9: 7 new per-node/per-edge graph attribute columns, appended after the
# fe8 struct fields -- see the module docstring for what each one means and
# why the graph-traversal ("blast radius") features are NOT here.
GRAPH_ATTR_FIELDS = [
    "source_node_degree", "edge_interaction_count", "source_node_age_normalized",
    "source_privilege_level", "source_historical_risk",
    "target_resource_criticality", "is_cross_account",
]
STRUCT_FIELDS = ["log_id", "source_node", "target_node", "edge_type", "label"] + GRAPH_ATTR_FIELDS

# username/timestamp are raw (not normalized-to-[0,1]) metadata, not model
# input features -- they're here so a downstream script can group by
# username and build sliding windows over timestamp without having to
# re-join against the structural CSV or the original input file.
TEMP_FIELDS = ["log_id", "username", "timestamp"] + TEMPORAL_COLS + ["label"]

DATA_DIR = "datasets/privilege-escalation"
DEFAULT_INPUT = os.path.join(DATA_DIR, "synthetic_cloudtrail.csv")
STRUCT_OUT = os.path.join(DATA_DIR, "cloudtrail_structural.csv")
TEMPORAL_OUT = os.path.join(DATA_DIR, "cloudtrail_temporal.csv")
STATE_FILE = os.path.join(DATA_DIR, ".feature_engine9_state.json")
EVENT_NAME_VOCAB_FILE = os.path.join(DATA_DIR, ".event_name_vocab.json")
STATE_TRACKER_FILE = os.path.join(DATA_DIR, ".state_tracker.json")
GRAPH_NODE_STATE_FILE = os.path.join(DATA_DIR, ".graph_node_state.json")

# ── Fast-lane: defense-evasion actions that get an immediate alert ───────────
# (subset of action_map's DEFENSE EVASION tier -- MITRE ATT&CK for Cloud)
CRITICAL_ACTIONS = {
    "StopLogging":     "CloudTrail logging disabled",
    "DeleteTrail":     "CloudTrail trail deleted",
    "UpdateDetector":  "GuardDuty detector reconfigured/disabled",
    "DeleteFlowLogs":  "VPC flow logs deleted",
}


def fast_lane_alert(row) -> None:
    """Fires the instant a defense-evasion action is seen, ahead of the
    rest of that row's feature computation. Models a low-latency
    EventBridge rule running alongside the batched feature pipeline."""
    event_name = row.get('event_name')
    reason = CRITICAL_ACTIONS.get(event_name)
    if reason:
        principal = row.get('principal_arn', 'unknown_principal')
        print(f"[FAST-LANE ALERT] {event_name} by {principal} - {reason}")


# ── State (processed-file tracking, so re-running / restarting the watcher
#    never reprocesses a CloudTrail log file twice) ────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        return json.loads(open(STATE_FILE, encoding='utf-8').read())
    return {"processed_files": []}


def save_state(state: dict) -> None:
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f)


# ── Micro-batch processing: one CloudTrail log file = one unit of work ───────

def _check_output_schema(path, expected_fields):
    """Refuses to append rows with a different column layout than what's
    already in the file -- the structural schema changed from fe8 (7 new
    graph-attribute columns)."""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        existing_header = next(csv.reader(f), None)
    if existing_header is not None and existing_header != expected_fields:
        raise SystemExit(
            f"{path} has a different column layout than feature_engine9 produces.\n"
            f"Delete it (and its structural/temporal sibling) and re-run to regenerate from scratch."
        )


def _rewrite_temporal_sorted(new_rows) -> None:
    """Keeps cloudtrail_temporal.csv globally sorted by (username,
    timestamp), not just in file-arrival order.

    CloudTrail doesn't guarantee event order within a single delivered log
    file, let alone across files -- and in --watch mode, files can land out
    of chronological order. A downstream sliding-window/LSTM script needs to
    group by username and walk events in order, so that invariant is
    maintained here rather than left for every consumer to re-sort.

    Re-sorts the whole file on every batch (existing rows + new rows). Fine
    at this project's data scale (thousands of rows); a high-volume
    production stream would want a merge step or an external sort instead.

    Written to a .tmp file in the same directory and then os.replace()'d
    into place, rather than truncating cloudtrail_temporal.csv directly --
    os.replace is atomic on both POSIX and Windows, so a crash mid-rewrite
    leaves either the old complete file or the new complete file, never a
    half-written one. (The append-only STRUCT_OUT doesn't need this: losing
    the tail of an in-progress append only risks the single row being
    written, not the whole file's history.)
    """
    existing_rows = []
    if os.path.exists(TEMPORAL_OUT):
        with open(TEMPORAL_OUT, encoding='utf-8') as f:
            existing_rows = list(csv.DictReader(f))

    combined = existing_rows + new_rows
    combined.sort(key=lambda r: (r['username'], _parse_timestamp(r['timestamp'])))

    tmp_path = TEMPORAL_OUT + ".tmp"
    with open(tmp_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TEMP_FIELDS)
        writer.writeheader()
        writer.writerows(combined)
    os.replace(tmp_path, TEMPORAL_OUT)


def process_batch_file(engine: FeatureEngineer, input_path: str) -> int:
    """Processes a single arrived log file, appending features to the
    structural/temporal output CSVs. Returns rows processed."""
    _check_output_schema(STRUCT_OUT, STRUCT_FIELDS)
    _check_output_schema(TEMPORAL_OUT, TEMP_FIELDS)

    struct_is_new = not os.path.exists(STRUCT_OUT)
    os.makedirs(DATA_DIR, exist_ok=True)

    count = 0
    new_temp_rows = []
    with open(STRUCT_OUT, mode='a', newline='', encoding='utf-8') as struct_out:
        struct_writer = csv.DictWriter(struct_out, fieldnames=STRUCT_FIELDS)
        if struct_is_new:
            struct_writer.writeheader()

        for row in iter_input_rows(input_path):
            fast_lane_alert(row)  # low-latency lane, runs before full feature computation

            try:
                struct = engine.get_structural_data(row)
                temporal = engine.get_temporal_features(row)
            except ValueError as e:
                print(f"[SKIP] {input_path}: {e}")
                continue

            label = row.get("label", "0")
            log_id = f"{os.path.basename(input_path)}:{count}"

            struct_row = {
                "log_id": log_id,
                "source_node": struct["source_node"],
                "target_node": struct["target_node"],
                "edge_type": struct["edge_type"],
                "label": label,
            }
            for field in GRAPH_ATTR_FIELDS:
                struct_row[field] = struct[field]
            struct_writer.writerow(struct_row)

            temp_row = {
                "log_id": log_id,
                "username": row.get("username", "unknown_user"),
                "timestamp": row.get("timestamp"),
                "label": label,
            }
            for i, val in enumerate(temporal):
                temp_row[TEMPORAL_COLS[i]] = val
            new_temp_rows.append(temp_row)

            count += 1

    if new_temp_rows:
        _rewrite_temporal_sorted(new_temp_rows)

    engine.save_state()
    print(f"[BATCH] {input_path} -> {count} rows (struct/temporal features written)")
    return count


def run_batch(input_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        raise SystemExit(1)
    engine = FeatureEngineer(
        event_name_vocab_path=EVENT_NAME_VOCAB_FILE,
        state_tracker_path=STATE_TRACKER_FILE,
        graph_state_path=GRAPH_NODE_STATE_FILE,
    )
    print(f"Reading logs in BATCH mode from {input_path}...")
    process_batch_file(engine, input_path)


# ── Watch mode: react to new log files landing (stand-in for S3 trigger) ─────

def _file_is_stable(path: str, checks: int = 2, interval: float = 0.3) -> bool:
    """Waits until a file's size stops changing, so we don't read a
    CloudTrail log file mid-write."""
    last_size = -1
    stable_count = 0
    while stable_count < checks:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = size
        time.sleep(interval)
    return True


def watch_folder(directory: str) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    os.makedirs(directory, exist_ok=True)
    engine = FeatureEngineer(
        event_name_vocab_path=EVENT_NAME_VOCAB_FILE,
        state_tracker_path=STATE_TRACKER_FILE,
        graph_state_path=GRAPH_NODE_STATE_FILE,
    )
    state = load_state()
    processed = set(state["processed_files"])

    class ArrivalHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = event.src_path
            name = os.path.basename(path)
            if name in processed or not name.endswith(('.csv', '.csv.gz', '.json', '.jsonl', '.ndjson', '.json.gz', '.jsonl.gz', '.ndjson.gz')):
                return
            if not _file_is_stable(path):
                return
            process_batch_file(engine, path)
            processed.add(name)
            state["processed_files"] = sorted(processed)
            save_state(state)

    observer = Observer()
    observer.schedule(ArrivalHandler(), directory, recursive=False)
    observer.start()
    print(f"Watching {directory} for new CloudTrail log files (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        observer.stop()
        observer.join()


# ── Optional: drop mock log files into the watched folder, for demoing
#    without a real S3 bucket / EventBridge rule ──────────────────────────────

def _generate_mock_log():
    import random

    actions = [
        ("GetCallerIdentity", "sts.amazonaws.com", "true"),
        ("ListBuckets", "s3.amazonaws.com", "true"),
        ("DescribeInstances", "ec2.amazonaws.com", "true"),
        ("CreateAccessKey", "iam.amazonaws.com", "false"),
        ("PutUserPolicy", "iam.amazonaws.com", "false"),
        ("AttachUserPolicy", "iam.amazonaws.com", "false"),
        ("GetSecretValue", "secretsmanager.amazonaws.com", "true"),
        ("AssumeRole", "sts.amazonaws.com", "false"),
        ("DeleteTrail", "cloudtrail.amazonaws.com", "false"),
        ("StopLogging", "cloudtrail.amazonaws.com", "false"),
    ]
    event_name, event_source, read_only = random.choice(actions)
    principals = [
        {"arn": "arn:aws:iam::123456789012:user/DevUser", "type": "IAMUser"},
        {"arn": "arn:aws:iam::123456789012:user/AdminUser", "type": "IAMUser"},
        {"arn": "arn:aws:iam::123456789012:role/EC2_Service_Role", "type": "AssumedRole"},
        {"arn": "arn:aws:iam::123456789012:user/Attacker", "type": "IAMUser"},
    ]
    principal = random.choice(principals)
    ips = ["10.0.0.15", "10.0.1.200", "192.168.1.100", "203.0.113.45", "198.51.100.22"]
    is_evil = principal["arn"].endswith("Attacker")

    return {
        "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "event_name": event_name,
        "event_source": event_source,
        "principal_type": principal["type"],
        "principal_arn": principal["arn"],
        "source_ip": random.choice(ips),
        "user_agent": "aws-cli/2.0 Python/3.8",
        "read_only": read_only,
        "aws_region": random.choice(["us-east-1", "us-east-1", "us-west-2"]),
        "mfa_authenticated": str(random.choice([True, True, False])).lower(),
        "error_code": "AccessDenied" if random.random() < 0.05 else "",
        "target_resource": f"resource-{random.randint(1, 50)}",
        "label": "1" if is_evil else "0",
        "request_params_raw": "{}",
    }


def simulate_incoming_files(directory: str, min_rows: int = 5, max_rows: int = 20,
                             min_interval: float = 3.0, max_interval: float = 8.0) -> None:
    """Periodically drops a new CSV into `directory`, each containing a
    handful of mock events -- a stand-in for a CloudTrail log file landing
    in S3 every ~5 minutes, compressed down to seconds for demo purposes."""
    import random
    import threading

    os.makedirs(directory, exist_ok=True)
    fieldnames = list(_generate_mock_log().keys())

    def loop():
        while True:
            time.sleep(random.uniform(min_interval, max_interval))
            n_rows = random.randint(min_rows, max_rows)
            fname = f"cloudtrail_{int(time.time())}_{uuid.uuid4().hex[:8]}.csv"
            path = os.path.join(directory, fname)
            with open(path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for _ in range(n_rows):
                    writer.writerow(_generate_mock_log())
            print(f"[SIMULATE] dropped {fname} ({n_rows} events)")

    t = threading.Thread(target=loop, daemon=True)
    t.start()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Micro-batch CloudTrail feature engine")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV to process in one-shot batch mode")
    parser.add_argument("--watch", metavar="DIR", help="Watch DIR and process each new log file as it lands")
    parser.add_argument("--simulate", action="store_true",
                         help="With --watch, also drop mock CloudTrail log files into DIR periodically")
    args = parser.parse_args()

    if args.watch:
        if args.simulate:
            simulate_incoming_files(args.watch)
        watch_folder(args.watch)
    else:
        run_batch(args.input)


if __name__ == "__main__":
    main()
