"""
Phase 1 data collection: runs Stratus Red Team techniques repeatedly
(warmup -> detonate -> revert -> cleanup) and logs every run to a manifest
CSV, which becomes the ground-truth label source for the real dataset.

Usage:
    python run_detonations.py --reps 3
    python run_detonations.py --reps 5 --techniques aws.credential-access.ssm-retrieve-securestring-parameters
    python run_detonations.py --reps 3 --dry-run

Run this multiple times across different days/times of day rather than
all at once in a single sitting -- temporal diversity in the real dataset
comes from re-invocation, not from sleeping inside one run.
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from stratus_techniques import TECHNIQUES, TECHNIQUE_IDS

SCRIPT_DIR = Path(__file__).parent

# Resolve the stratus binary robustly rather than relying on PATH being
# correctly inherited in every shell this script gets launched from.
_FALLBACK_STRATUS_PATHS = [
    os.path.expandvars(r"%USERPROFILE%\bin\stratus.exe"),
    os.path.expandvars(r"%USERPROFILE%\tools\stratus-red-team\stratus.exe"),
]
STRATUS_BIN = shutil.which("stratus") or next(
    (p for p in _FALLBACK_STRATUS_PATHS if os.path.isfile(p)), "stratus"
)
MANIFEST_FIELDS = [
    "run_id", "collector", "aws_account", "technique_id", "tactic", "rep_index",
    "start_ts_utc", "end_ts_utc",
    "warmup_status", "detonate_status", "revert_status", "cleanup_status",
    "expected_events", "notes",
]


def detect_aws_account():
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

STEP_TIMEOUT_SEC = 300


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def run_stratus(subcommand, technique_id):
    """Run a stratus CLI subcommand against a technique. Returns (ok, output_text)."""
    try:
        env = dict(os.environ)
        env.setdefault("AWS_REGION", "us-east-1")
        result = subprocess.run(
            [STRATUS_BIN, subcommand, technique_id],
            capture_output=True, text=True, timeout=STEP_TIMEOUT_SEC, env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if subcommand == "revert" and "has no revert function" in output:
            return True, output.strip()
        ok = result.returncode == 0
        return ok, output.strip()
    except subprocess.TimeoutExpired as e:
        return False, f"TIMEOUT after {STEP_TIMEOUT_SEC}s: {e}"
    except FileNotFoundError:
        print("ERROR: 'stratus' not found on PATH. Is it installed and on PATH?", file=sys.stderr)
        sys.exit(1)


def ensure_manifest(manifest_path):
    if not manifest_path.exists():
        with open(manifest_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()


def append_manifest(manifest_path, row):
    with open(manifest_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writerow(row)


def run_one(technique, rep_index, jitter_range, dry_run, manifest_path, collector, aws_account):
    tid = technique["id"]
    run_id = str(uuid.uuid4())
    row = {
        "run_id": run_id, "collector": collector, "aws_account": aws_account,
        "technique_id": tid, "tactic": technique["tactic"],
        "rep_index": rep_index, "start_ts_utc": now_utc(), "end_ts_utc": "",
        "warmup_status": "", "detonate_status": "", "revert_status": "", "cleanup_status": "",
        "expected_events": ";".join(technique["expected_events"]), "notes": "",
    }

    print(f"\n[{rep_index}] {tid}  (run_id={run_id[:8]})")
    if technique["cost_note"]:
        print(f"    note: {technique['cost_note']}")

    if dry_run:
        print("    DRY RUN - would run warmup -> detonate -> revert -> cleanup")
        return

    notes = []
    try:
        print("    warmup...")
        ok, out = run_stratus("warmup", tid)
        row["warmup_status"] = "ok" if ok else "FAILED"
        if not ok:
            notes.append(f"warmup: {out[-500:]}")
            print(f"    warmup FAILED: {out[-300:]}")
            return  # nothing to detonate if warmup failed; still hits finally for cleanup safety

        time.sleep(random.uniform(*jitter_range))

        print("    detonate...")
        ok, out = run_stratus("detonate", tid)
        row["detonate_status"] = "ok" if ok else "FAILED"
        if not ok:
            notes.append(f"detonate: {out[-500:]}")
            print(f"    detonate FAILED: {out[-300:]}")
        else:
            print("    detonate ok")

        time.sleep(random.uniform(*jitter_range))

        print("    revert...")
        ok, out = run_stratus("revert", tid)
        row["revert_status"] = "ok" if ok else "FAILED"
        if not ok:
            notes.append(f"revert: {out[-300:]}")

    except KeyboardInterrupt:
        notes.append("interrupted by user")
        print("\n    interrupted - attempting cleanup before exit...")
        raise
    finally:
        print("    cleanup...")
        ok, out = run_stratus("cleanup", tid)
        row["cleanup_status"] = "ok" if ok else "FAILED - MANUAL CHECK NEEDED"
        if not ok:
            notes.append(f"cleanup: {out[-500:]}")
            print(f"    *** CLEANUP FAILED - check the AWS console for leftover resources: {tid} ***")
        else:
            print("    cleanup ok")
        row["end_ts_utc"] = now_utc()
        row["notes"] = " | ".join(notes)
        append_manifest(manifest_path, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collector", type=str, required=True,
                         help="Your name/handle, e.g. 'alice'. Keeps your manifest file separate from "
                              "teammates' so they merge cleanly in git (writes manifest_<collector>.csv).")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per technique in this invocation")
    parser.add_argument("--techniques", type=str, default=None,
                         help="Comma-separated technique IDs to run (default: all 11)")
    parser.add_argument("--min-jitter", type=float, default=5.0, help="Min seconds between steps")
    parser.add_argument("--max-jitter", type=float, default=30.0, help="Max seconds between steps")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing anything")
    args = parser.parse_args()

    if args.techniques:
        selected_ids = [t.strip() for t in args.techniques.split(",")]
        unknown = set(selected_ids) - set(TECHNIQUE_IDS)
        if unknown:
            print(f"ERROR: unknown technique id(s): {unknown}", file=sys.stderr)
            sys.exit(1)
        selected = [t for t in TECHNIQUES if t["id"] in selected_ids]
    else:
        selected = TECHNIQUES

    manifest_path = SCRIPT_DIR / f"manifest_{args.collector}.csv"
    ensure_manifest(manifest_path)
    aws_account = "dry-run" if args.dry_run else detect_aws_account()

    plan = [(t, r) for r in range(1, args.reps + 1) for t in selected]
    random.shuffle(plan)  # interleave techniques rather than running all reps of one back-to-back

    print(f"Collector: {args.collector}  |  AWS account: {aws_account}")
    print(f"Plan: {len(selected)} technique(s) x {args.reps} rep(s) = {len(plan)} total runs")
    print(f"Manifest: {manifest_path}")

    try:
        for technique, rep_index in plan:
            run_one(technique, rep_index, (args.min_jitter, args.max_jitter), args.dry_run,
                     manifest_path, args.collector, aws_account)
    except KeyboardInterrupt:
        print("\nStopped by user. Partial results are already in the manifest.")

    if not args.dry_run:
        print(f"\nDone. Review {manifest_path.name} for per-run status.")
        print("If any row has cleanup_status != 'ok', check the AWS console for that technique before running more.")


if __name__ == "__main__":
    main()
