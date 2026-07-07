"""
feature_engine6.py
===================
Ingestion model matches how CloudTrail actually delivers logs: files land in
S3 roughly every 5 minutes, not one event at a time. So this engine works in
two lanes instead of fe5's per-event mock stream:

  1. MICRO-BATCH lane (the backbone)
     One CloudTrail log file = one unit of work. A directory watcher
     (watchdog) stands in for an S3 "ObjectCreated" / EventBridge trigger:
     whenever a new log file appears in --watch DIR, it gets processed as a
     batch, same as the original batch mode.

  2. FAST-LANE lane (defense-evasion alerting)
     A handful of high-severity actions (StopLogging, DeleteTrail, ...) are
     checked the instant a row is read, before the rest of that row's
     features are even computed, and print an immediate alert. This models
     a separate low-latency EventBridge rule that runs alongside the
     batched feature pipeline rather than waiting on the next micro-batch.

Feature logic (StateTracker / FeatureEngineer) is unchanged from
feature_engine4.py / feature_engine5.py.

Run:
    python feature_engine6.py
        One-shot batch over invictus_enriched.csv (same as fe4/fe5 batch mode).

    python feature_engine6.py --watch incoming/
        Watch incoming/ and process each new log file as it lands.

    python feature_engine6.py --watch incoming/ --simulate
        Same, but also drops a mock CloudTrail log file into incoming/ every
        few seconds so the watch loop has something to react to without a
        real S3 bucket.
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
    session_issuer_arn = (session_context.get('sessionIssuer') or {}).get('arn')
    principal_arn = row.get('principal_arn') or session_issuer_arn or user_identity.get('arn')
    principal_type = row.get('principal_type') or user_identity.get('type') or 'Unknown'

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

    normalized = {
        'timestamp': row.get('timestamp') or row.get('eventTime'),
        'event_name': row.get('event_name') or row.get('eventName'),
        'event_source': row.get('event_source') or row.get('eventSource') or '',
        'principal_type': principal_type,
        'principal_arn': principal_arn or 'unknown_principal',
        'source_ip': row.get('source_ip') or row.get('sourceIPAddress') or '0.0.0.0',
        'user_agent': row.get('user_agent') or row.get('userAgent') or '',
        'read_only': row.get('read_only') or row.get('readOnly') or 'true',
        'aws_region': row.get('aws_region') or row.get('awsRegion') or 'us-east-1',
        'mfa_authenticated': row.get('mfa_authenticated') or session_attributes.get('mfaAuthenticated') or 'false',
        'error_code': row.get('error_code') or row.get('errorCode') or '',
        'target_resource': target_resource or 'aws_service',
        'label': row.get('label', '0'),
        'request_params_raw': request_params or '{}',
        'access_key_id': row.get('access_key_id') or row.get('accessKeyId') or '',
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


# ── Shared feature engineering (unchanged from fe4/fe5) ───────────────────────

class StateTracker:
    """Tracks historical behavior to detect Privilege Escalation and Velocity."""
    def __init__(self):
        self.user_registry = {}

    def get_metrics(self, p_arn, event_name, current_ts):
        if p_arn not in self.user_registry:
            self.user_registry[p_arn] = {
                'last_ts': current_ts,
                'actions': {event_name}
            }
            return 0.0, 1  # No previous history = No velocity, but is NEW action

        user_data = self.user_registry[p_arn]

        # Calculate Velocity (Time delta normalized)
        delta = (current_ts - user_data['last_ts']).total_seconds()
        velocity = max(0, 1 - (delta / 3600))

        # Check for Privilege Creep
        is_new_action = 1 if event_name not in user_data['actions'] else 0

        # Update State for next log
        user_data['last_ts'] = current_ts
        user_data['actions'].add(event_name)

        return velocity, is_new_action


class FeatureEngineer:
    def __init__(self):
        self.tracker = StateTracker()
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

    def get_structural_data(self, log):
        """Generates GNN Triples"""
        p_arn = log.get('principal_arn', 'unknown_principal')
        event = log.get('event_name', 'unknown_action')
        target = log.get('target_resource') or 'aws_service'

        return {
            'source_node': p_arn,
            'target_node': target,
            'edge_type': event
        }

    def get_temporal_features(self, log):
        """Generates 25D Vector."""
        f = []
        p_arn = log.get('principal_arn', 'unknown')

        timestamp_str = log.get('timestamp') or log.get('eventTime')

        if 'T' in timestamp_str:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:  # Standard format (2026-03-31 10:00:00+0000)
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S%z")

        # PILLAR 1: IDENTITY & DYNAMICS (5 Features)
        mfa = str(log.get('mfa_authenticated', 'false')).lower()
        f.append(1 if mfa in ['false', 'no', 'none', ''] else 0)  # 1

        p_type = log.get('principal_type', '')
        p_risk = {"Root": 1.0, "IAMUser": 0.8, "AssumedRole": 0.5}.get(p_type, 0.3)
        f.append(p_risk)  # 2

        f.append(1 if log.get('access_key_id') else 0)  # 3

        velocity, is_new_action = self.tracker.get_metrics(p_arn, log.get('event_name'), dt)
        f.append(velocity)  # 4
        f.append(is_new_action)  # 5

        # PILLAR 2: TEMPORAL CONTEXT (4 Features)
        h_rad = 2 * np.pi * dt.hour / 24.0
        f.append(np.sin(h_rad))  # 6
        f.append(np.cos(h_rad))  # 7
        f.append(1 if dt.weekday() >= 5 else 0)  # 8
        f.append(1 if dt.hour < 9 or dt.hour > 18 else 0)  # 9

        # PILLAR 3: ACTION INTENT (8 Features)
        f.append(self.action_map.get(log.get('event_name'), self.default_risk) / 10.0)  # 10
        f.append(1 if str(log.get('read_only', 'true')).lower() == 'false' else 0)  # 11

        err = log.get('error_code', '')
        f.append(1 if err else 0)  # 12
        f.append(1 if err == "AccessDenied" else 0)  # 13

        f.append(1 if "iam" in log.get('event_source', '') else 0)  # 14
        f.append(1 if any(x in log.get('event_name', '') for x in ['Describe', 'List', 'Get']) else 0)  # 15
        f.append(1 if log.get('event_name') in ["DeleteTrail", "StopLogging"] else 0)  # 16
        f.append(1 if log.get('event_name') == "GetCallerIdentity" else 0)  # 17

        # PILLAR 4: METADATA ANOMALIES (8 Features)
        ua = str(log.get('user_agent', '')).lower()
        f.append(1 if any(x in ua for x in ['kali', 'pacu', 'metasploit', 'requests']) else 0)  # 18

        try:
            ip = log.get('source_ip', '0.0.0.0')
            f.append(1 if not ipaddress.ip_address(ip).is_private else 0)  # 19
        except Exception:
            f.append(1)

        params = str(log.get('request_params_raw', '{}'))
        f.append(min(len(params) / 750.0, 1.0))  # 20

        target_str = str(log.get('target_resource', '')).lower()
        f.append(1 if any(x in target_str for x in ['admin', 'vault', 'prod']) else 0)  # 21
        f.append(1 if log.get('aws_region') != "us-east-1" else 0)  # 22
        f.append(1 if "Create" in log.get('event_name', '') and "Key" in log.get('event_name', '') else 0)  # 23
        f.append(1 if any(x in log.get('event_source', '') for x in ['secrets', 'kms']) else 0)  # 24
        f.append(1 if "Attach" in log.get('event_name', '') or "Put" in log.get('event_name', '') else 0)  # 25

        return f


TEMPORAL_COLS = [
    "no_mfa", "principal_risk", "has_access_key", "action_velocity", "is_new_action",
    "time_sin", "time_cos", "is_weekend", "is_off_hours",
    "mitre_action_risk", "is_write_action", "has_error", "is_access_denied",
    "is_iam_event", "is_recon_action", "is_defense_evasion", "is_get_caller_identity",
    "is_malicious_user_agent", "is_public_ip", "params_length_normalized",
    "targets_sensitive_resource", "is_non_default_region", "is_create_key",
    "is_secrets_or_kms", "is_permission_modification"
]
STRUCT_FIELDS = ["log_id", "source_node", "target_node", "edge_type", "label"]
TEMP_FIELDS = ["log_id"] + TEMPORAL_COLS + ["label"]

DATA_DIR = "datasets/privilege-escalation"
DEFAULT_INPUT = os.path.join(DATA_DIR, "invictus_enriched.csv")
STRUCT_OUT = os.path.join(DATA_DIR, "invictus_structural.csv")
TEMPORAL_OUT = os.path.join(DATA_DIR, "invictus_temporal.csv")
STATE_FILE = os.path.join(DATA_DIR, ".feature_engine6_state.json")

# ── Fast-lane: defense-evasion actions that get an immediate alert ───────────
# (subset of action_map's DEFENSE EVASION tier — MITRE ATT&CK for Cloud)
CRITICAL_ACTIONS = {
    "StopLogging":     "CloudTrail logging disabled",
    "DeleteTrail":     "CloudTrail trail deleted",
    "UpdateDetector":  "GuardDuty detector reconfigured/disabled",
    "DeleteFlowLogs":  "VPC flow logs deleted",
}


def fast_lane_alert(row) -> None:
    """Fires the instant a defense-evasion action is seen, ahead of the
    rest of that row's feature computation. Models a low-latency
    EventBridge rule running alongside the batched pipeline."""
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

def process_batch_file(engine: FeatureEngineer, input_path: str) -> int:
    """Processes a single arrived log file, appending features to the
    structural/temporal output CSVs. Returns rows processed."""
    struct_is_new = not os.path.exists(STRUCT_OUT)
    temp_is_new = not os.path.exists(TEMPORAL_OUT)
    os.makedirs(DATA_DIR, exist_ok=True)

    count = 0
    with open(STRUCT_OUT, mode='a', newline='', encoding='utf-8') as struct_out, \
         open(TEMPORAL_OUT, mode='a', newline='', encoding='utf-8') as temp_out:

        struct_writer = csv.DictWriter(struct_out, fieldnames=STRUCT_FIELDS)
        temp_writer = csv.DictWriter(temp_out, fieldnames=TEMP_FIELDS)
        if struct_is_new:
            struct_writer.writeheader()
        if temp_is_new:
            temp_writer.writeheader()

        for row in iter_input_rows(input_path):
            fast_lane_alert(row)  # low-latency lane, runs before full feature computation

            struct = engine.get_structural_data(row)
            temporal = engine.get_temporal_features(row)
            label = row.get("label", "0")
            log_id = f"{os.path.basename(input_path)}:{count}"

            struct_writer.writerow({
                "log_id": log_id,
                "source_node": struct["source_node"],
                "target_node": struct["target_node"],
                "edge_type": struct["edge_type"],
                "label": label,
            })

            temp_row = {"log_id": log_id, "label": label}
            for i, val in enumerate(temporal):
                temp_row[TEMPORAL_COLS[i]] = val
            temp_writer.writerow(temp_row)

            count += 1

    print(f"[BATCH] {input_path} -> {count} rows (struct/temporal features written)")
    return count


def run_batch(input_path: str) -> None:
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        raise SystemExit(1)
    engine = FeatureEngineer()
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
    engine = FeatureEngineer()
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
    handful of mock events — a stand-in for a CloudTrail log file landing
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