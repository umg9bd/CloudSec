"""
Combines every teammate's manifest_<collector>.csv into one file, and
cross-checks that the CloudTrail JSON data those manifests refer to has
actually been pushed (not just the manifest metadata).

This does NOT re-verify against S3 -- that requires each collector's own
AWS credentials, which nobody else has. It trusts each person's own
manifest_<collector>_verified.csv (produced when *they* ran
collect_real_logs.py) and just checks that the corresponding JSON events
exist locally under stratus_own_runs/CloudTrail.

Usage:
    python combine_manifests.py
"""

import csv
import glob
import json
from datetime import timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CLOUDTRAIL_DIR = SCRIPT_DIR.parent / "stratus_own_runs" / "CloudTrail"
COMBINED_PATH = SCRIPT_DIR / "manifest_combined.csv"

MATCH_BUFFER = timedelta(minutes=10)


def load_all_manifests():
    """Every manifest_*.csv that is NOT a *_verified.csv file."""
    all_files = glob.glob(str(SCRIPT_DIR / "manifest_*.csv"))
    manifest_files = [f for f in all_files if "_verified" not in f and "_combined" not in f]

    rows = []
    fieldnames = None
    for f in manifest_files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                row["_source_file"] = Path(f).name
                rows.append(row)
    return rows, fieldnames, manifest_files


def load_available_event_names():
    """Parse every local CloudTrail JSON file once, bucketed by AWS account
    and rough time window, so we can check which manifest rows actually
    have corresponding event data on disk (not just a manifest entry)."""
    from datetime import datetime, timezone

    events_by_account = {}
    json_files = glob.glob(str(CLOUDTRAIL_DIR / "*.json"))
    for jf in json_files:
        account = Path(jf).name.split("_")[0]
        with open(jf) as fh:
            data = json.load(fh)
        for ev in data.get("Records", []):
            ts = ev.get("eventTime")
            if not ts:
                continue
            try:
                t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            events_by_account.setdefault(account, []).append((t, ev.get("eventName")))
    return events_by_account


def main():
    rows, fieldnames, manifest_files = load_all_manifests()
    print(f"Found {len(manifest_files)} manifest file(s):")
    for f in manifest_files:
        print(f"  - {Path(f).name}")
    print(f"\nTotal rows across all manifests: {len(rows)}")

    events_by_account = load_available_event_names()
    print(f"\nAccounts with CloudTrail JSON data present locally: {sorted(events_by_account.keys())}")

    from datetime import datetime, timezone

    missing_data = []
    for row in rows:
        account = row.get("aws_account", "unknown")
        if account not in events_by_account:
            missing_data.append(row)
            continue
        if not row.get("start_ts_utc") or not row.get("end_ts_utc"):
            continue
        start = datetime.fromisoformat(row["start_ts_utc"]) - MATCH_BUFFER
        end = datetime.fromisoformat(row["end_ts_utc"]) + MATCH_BUFFER
        has_any = any(start <= t <= end for t, _ in events_by_account[account])
        if not has_any:
            missing_data.append(row)

    out_fields = fieldnames + ["_source_file"]
    with open(COMBINED_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {COMBINED_PATH.name} ({len(rows)} rows, {len(rows) - len(missing_data)} with local event data)")

    if missing_data:
        missing_accounts = sorted(set(r.get("aws_account", "unknown") for r in missing_data))
        print(f"\n{len(missing_data)} row(s) have NO matching CloudTrail JSON data locally.")
        print(f"Missing data belongs to account(s): {missing_accounts}")
        print("These collectors need to push their stratus_own_runs/CloudTrail/*.json files")
        print("(the actual event data), not just their manifest_<name>.csv.")
    else:
        print("\nEvery manifest row has corresponding CloudTrail JSON data locally. Ready to build the real dataset.")


if __name__ == "__main__":
    main()
