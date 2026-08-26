"""
Builds ONE combined, labeled real dataset from:
  1. invictus_enriched.csv (deduped) -- the original 2023 real capture
  2. every collector's Stratus-collected sessions (manifest_*.csv +
     stratus_own_runs/CloudTrail/*.json)

Why this isn't a simple concatenation:
  invictus_enriched.csv labels events via a fixed dictionary of "attack
  event names" (StopLogging, CreateLoginProfile, ...). Several of the new
  techniques' actual detonation events aren't in that dictionary
  (DescribeParameters, ListSecrets), and two of them (AssumeRole,
  AddPermission20150331v2) are common in ordinary AWS activity too --
  blanket-labeling every occurrence would mislabel real benign events.

  Instead, for the newly-collected data, an event is labeled an attack
  only if it falls inside a specific manifest-recorded [start_ts, end_ts]
  window AND matches that technique's expected_events -- the same check
  collect_real_logs.py uses to confirm delivery, now driving labels.

Output: ../real_dataset_combined.csv
"""

import csv
import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from stratus_techniques import TECHNIQUES

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent
CLOUDTRAIL_DIR = DATASET_DIR / "stratus_own_runs" / "CloudTrail"
INVICTUS_PATH = DATASET_DIR / "invictus_enriched.csv"
OUTPUT_PATH = DATASET_DIR / "real_dataset_combined.csv"

MATCH_BUFFER = timedelta(minutes=10)
SESSION_WINDOW = timedelta(minutes=5)
BACKGROUND_GAP = timedelta(minutes=30)  # inactivity gap that starts a new benign "session"

EXPECTED_BY_TECHNIQUE = {t["id"]: (t["tactic"], t["expected_events"]) for t in TECHNIQUES}

COLUMNS = [
    "timestamp", "event_name", "event_source", "aws_region", "source_ip",
    "error_code", "label", "attack_technique", "read_only", "user_agent",
    "access_key_id", "mfa_authenticated", "target_resource", "request_params_raw",
    "principal_type", "principal_arn", "username", "session_label", "data_source",
    "session_id",
]


# ── Reused from explore.ipynb (same logic, so labeling stays comparable) ──────

def normalise_principal(identity: dict) -> dict:
    id_type = identity.get("type", "unknown")
    if id_type == "IAMUser":
        return {"principal_type": "IAMUser", "principal_arn": identity.get("arn"),
                "username": identity.get("userName")}
    elif id_type == "AssumedRole":
        session = identity.get("sessionContext", {})
        issuer = session.get("sessionIssuer", {})
        return {"principal_type": "AssumedRole", "principal_arn": identity.get("arn"),
                "username": issuer.get("userName")}
    elif id_type == "AWSService":
        return {"principal_type": "AWSService", "principal_arn": None,
                "username": identity.get("invokedBy")}
    else:
        return {"principal_type": id_type or "unknown", "principal_arn": identity.get("arn"),
                "username": identity.get("userName")}


_TARGET_KEYS = ["roleName", "userName", "groupName", "policyArn", "bucketName",
                "secretId", "instanceId", "functionName", "trailName", "keyId",
                "dbInstanceIdentifier"]


def extract_target_resource(params):
    if not params or not isinstance(params, dict):
        return None
    for key in _TARGET_KEYS:
        val = params.get(key)
        if val:
            return str(val)
    for val in params.values():
        if isinstance(val, str) and val:
            return val
    return None


def assign_background_sessions(df, gap=BACKGROUND_GAP):
    """Gap-based session segmentation for ambient benign activity: a new
    session starts whenever a collector goes >=`gap` without an event,
    the same principle web analytics session windows use (commonly a
    30-minute inactivity timeout). Replaces the placeholder
    "__PENDING_BG__<collector>" session_id with real per-burst ids, so
    benign sessions reflect actual activity bursts instead of being
    crushed into one session per calendar day regardless of volume."""
    is_pending = df["session_id"].str.startswith("__PENDING_BG__", na=False)
    if not is_pending.any():
        return df

    pending = df.loc[is_pending, ["data_source", "timestamp"]].copy()
    pending["collector"] = df.loc[is_pending, "session_id"].str.replace("__PENDING_BG__", "", regex=False)

    new_ids = pd.Series(index=pending.index, dtype=object)
    for collector, grp in pending.groupby("collector"):
        grp = grp.sort_values("timestamp")
        gap_break = grp["timestamp"].diff() >= gap
        session_num = gap_break.cumsum()  # increments at each inactivity gap
        new_ids.loc[grp.index] = [f"{collector}_bg_{n}" for n in session_num]

    df.loc[is_pending, "session_id"] = new_ids
    return df


# ── Step 1: deduped invictus data ──────────────────────────────────────────────

def load_invictus():
    df = pd.read_csv(INVICTUS_PATH)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"invictus_enriched.csv: {before} rows -> {len(df)} after dedup ({before - len(df)} removed)")
    df["data_source"] = "invictus_2023"
    # this data source is a single 55-minute capture with few real actors --
    # username already correctly identifies one coherent session here
    df["session_id"] = "invictus_" + df["username"].astype(str)
    return df[[c for c in COLUMNS if c in df.columns]]


# ── Step 2: manifest-window-based labeling of collected Stratus data ──────────

def load_manifests():
    rows = []
    for f in glob.glob(str(SCRIPT_DIR / "manifest_*.csv")):
        if "_verified" in f or "_combined" in f:
            continue
        with open(f) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def load_json_events():
    """All events, grouped by AWS account, as raw dicts (not yet enriched)."""
    events_by_account = {}
    for jf in glob.glob(str(CLOUDTRAIL_DIR / "*.json")):
        account = Path(jf).name.split("_")[0]
        with open(jf) as fh:
            data = json.load(fh)
        events_by_account.setdefault(account, []).extend(data.get("Records", []))
    return events_by_account


def build_stratus_dataset():
    manifests = load_manifests()
    events_by_account = load_json_events()

    # index manifest windows per account for fast lookup
    windows_by_account = {}
    for m in manifests:
        if not m.get("start_ts_utc") or not m.get("end_ts_utc"):
            continue
        tactic, expected = EXPECTED_BY_TECHNIQUE.get(m["technique_id"], (None, []))
        windows_by_account.setdefault(m["aws_account"], []).append({
            "start": datetime.fromisoformat(m["start_ts_utc"]) - MATCH_BUFFER,
            "end": datetime.fromisoformat(m["end_ts_utc"]) + MATCH_BUFFER,
            "expected_events": set(expected),
            "tactic": tactic,
            "collector": m["collector"],
            "run_id": m["run_id"],
        })

    rows = []
    for account, events in events_by_account.items():
        windows = windows_by_account.get(account, [])
        collector = windows[0]["collector"] if windows else account

        for ev in events:
            ts_raw = ev.get("eventTime")
            try:
                t = datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            event_name = ev.get("eventName", "unknown")
            # One IAM identity runs every detonation for a collector, sequentially,
            # over many days -- grouping by username alone would collapse dozens of
            # distinct sessions into one. Instead: an event belongs to whichever
            # manifest run's time window contains it (that run's run_id becomes the
            # session id); events outside every window are ambient background noise,
            # bucketed per collector per calendar day so they don't collapse into
            # one multi-day blob either.
            #
            # Runs happen back-to-back, so their buffered windows can overlap --
            # prioritize a window where the event actually matches THAT run's
            # expected_events over one that merely overlaps in time, otherwise a
            # genuine attack event can get silently attributed to a neighboring
            # run it doesn't belong to and lose its label.
            time_matches = [w for w in windows if w["start"] <= t <= w["end"]]
            exact_match = next((w for w in time_matches if event_name in w["expected_events"]), None)

            if exact_match is not None:
                session_id = exact_match["run_id"]
                label, attack_technique = 1, exact_match["tactic"]
            elif time_matches:
                session_id = time_matches[0]["run_id"]
                label, attack_technique = 0, None
            else:
                # Placeholder -- real session_id assigned by assign_background_sessions()
                # below, via gap-based segmentation rather than a coarse calendar bucket.
                session_id = f"__PENDING_BG__{collector}"
                label, attack_technique = 0, None

            identity = ev.get("userIdentity", {})
            principal = normalise_principal(identity)
            session = identity.get("sessionContext", {})
            attrs = session.get("attributes", {})
            params = ev.get("requestParameters")

            rows.append({
                "timestamp": ts_raw, "event_name": event_name,
                "event_source": ev.get("eventSource"), "aws_region": ev.get("awsRegion"),
                "source_ip": ev.get("sourceIPAddress"), "error_code": ev.get("errorCode"),
                "label": label, "attack_technique": attack_technique,
                "read_only": ev.get("readOnly"), "user_agent": ev.get("userAgent"),
                "access_key_id": identity.get("accessKeyId"),
                "mfa_authenticated": attrs.get("mfaAuthenticated"),
                "target_resource": extract_target_resource(params),
                "request_params_raw": json.dumps(params) if params else None,
                "data_source": f"stratus_{collector}",
                "session_id": session_id,
                **principal,
            })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = assign_background_sessions(df)

    # session-level labeling: a session is "attack" if ANY event carrying its
    # session_id (one manifest run, or one gap-segmented background burst)
    # was labeled an attack event. No time-window heuristic needed here --
    # session_id already encodes the correct boundaries.
    df["session_label"] = df.groupby("session_id")["label"].transform("max")

    n_sessions = df["session_id"].nunique()
    n_attack_sessions = df.groupby("session_id")["label"].max().sum()
    print(f"Stratus-collected data: {len(df)} events, {df['label'].sum()} labeled attack events, "
          f"{n_sessions} sessions ({n_attack_sessions} attack, {n_sessions - n_attack_sessions} benign), "
          f"across {df['data_source'].nunique()} collectors")
    return df[[c for c in COLUMNS if c in df.columns]]


def main():
    invictus_df = load_invictus()
    stratus_df = build_stratus_dataset()

    combined = pd.concat([invictus_df, stratus_df], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {OUTPUT_PATH} ({len(combined)} total rows)")
    print(f"By data_source:\n{combined['data_source'].value_counts().to_string()}")
    print(f"\nTotal attack events: {combined['label'].sum()}  |  attack sessions: {combined['session_label'].sum()}")


if __name__ == "__main__":
    main()
