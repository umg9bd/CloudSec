"""
Phase 1 data collection -- GENUINE privilege-escalation chain.

Stratus Red Team's AWS catalog has exactly ONE privilege-escalation technique
(iam-update-user-login-profile), and it is a single atomic event. It cannot
produce the multi-hop, multi-principal chain the detector is built around:

    User  --AssumeRole-->  Role  --AttachUserPolicy-->  User(grants power)

That chain is what makes privilege_gain > 0 in privilege_features.py
(AttachUserPolicy is "Permissions management", rank 4; AssumeRole is rank 3),
and it is exactly what the real capture was missing. This script emulates it
end to end on YOUR OWN sandbox account, then reverts and deletes everything.

It is the same category of authorized self-attack as run_detonations.py -- it
just composes the steps Stratus keeps separate. Run it ONLY on a dedicated
sandbox account you own. Never against anything you do not control.

What it does (warmup -> detonate -> revert -> cleanup, mirroring Stratus):

  warmup    creates an over-permissioned role (trusts your account, can only
            do iam:AttachUserPolicy) and a throwaway target user. This is the
            misconfiguration an attacker abuses -- a role scoped for one IAM
            action is enough to grant a user full admin.
  detonate  assumes the role (-> AssumeRole event), then AS THAT ROLE attaches
            a powerful managed policy to the target user (-> AttachUserPolicy
            event, performed by the assumed-role principal). This is the
            escalation.
  revert    detaches the policy from the target user.
  cleanup   deletes the target user, the role, and their inline policies.

Every resource name is prefixed `stratus-escalation-` so leftovers are trivial
to find. Cleanup runs in a finally block and there is a standalone --cleanup-only
sweep for anything a crashed run left behind.

Usage:
    python run_escalation_detonation.py --collector vansh --reps 3
    python run_escalation_detonation.py --collector vansh --dry-run
    python run_escalation_detonation.py --collector vansh --cleanup-only

Run it a few times across different days/times (like run_detonations.py) --
temporal spread comes from re-invocation, not from one long sitting.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Matches run_detonations.py's schema exactly, so collect_real_logs.py verifies
# these rows the same way it verifies Stratus rows.
MANIFEST_FIELDS = [
    "run_id", "collector", "aws_account", "technique_id", "tactic", "rep_index",
    "start_ts_utc", "end_ts_utc",
    "warmup_status", "detonate_status", "revert_status", "cleanup_status",
    "expected_events", "notes",
]

TECHNIQUE_ID = "custom.privilege-escalation.assume-role-escalate-user"
TACTIC = "privilege-escalation"
EXPECTED_EVENTS = ["AssumeRole", "AttachUserPolicy"]
RESOURCE_PREFIX = "stratus-escalation-"

# The policy the role attaches to the target user. Any managed policy produces
# the same AttachUserPolicy event and the same privilege_gain (the rank is a
# property of the ACTION, not the policy) -- IAMFullAccess is chosen only to
# make the escalation faithful. Overridable with --grant-policy if you would
# rather attach something tamer; the captured event is identical either way.
DEFAULT_GRANT_POLICY = "arn:aws:iam::aws:policy/IAMFullAccess"

STEP_TIMEOUT_SEC = 120


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def aws(args, creds=None, check=False):
    """Run an `aws` CLI command. Returns (ok, stdout, stderr).

    creds, when given, is a dict with temporary credentials injected via env
    for THIS call only -- used to act as the assumed role. Credentials are
    never written to the manifest or printed.
    """
    env = dict(os.environ)
    env.setdefault("AWS_REGION", "us-east-1")
    if creds:
        env["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
        env["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
        env["AWS_SESSION_TOKEN"] = creds["SessionToken"]
    try:
        r = subprocess.run(["aws"] + args, capture_output=True, text=True,
                            timeout=STEP_TIMEOUT_SEC, env=env)
        return r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return False, "", f"TIMEOUT after {STEP_TIMEOUT_SEC}s"
    except FileNotFoundError:
        print("ERROR: 'aws' CLI not found on PATH. Install it (README section 3) and retry.",
              file=sys.stderr)
        sys.exit(1)


def caller_account():
    ok, out, err = aws(["sts", "get-caller-identity", "--query", "Account", "--output", "text"])
    return out if ok else "unknown"


# ── resource lifecycle ──────────────────────────────────────────────────────

def role_name(suffix):
    return f"{RESOURCE_PREFIX}role-{suffix}"


def user_name(suffix):
    return f"{RESOURCE_PREFIX}target-{suffix}"


def warmup(suffix, account, notes):
    """Create the over-permissioned role + target user. Returns True on success."""
    rn, un = role_name(suffix), user_name(suffix)

    # Trust policy: any principal in THIS account may assume the role (simplest
    # same-account assume; safe in a single-owner sandbox). The account root
    # principal is the standard AWS idiom for this.
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": f"arn:aws:iam::{account}:root"},
            "Action": "sts:AssumeRole",
        }],
    }
    # The role can do exactly one thing: attach policies to users. That single
    # over-grant is the whole vulnerability being emulated.
    inline = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["iam:AttachUserPolicy"],
            "Resource": "*",
        }],
    }

    ok, _, err = aws(["iam", "create-role", "--role-name", rn,
                      "--assume-role-policy-document", json.dumps(trust),
                      "--description", "stratus escalation emulation - safe to delete"])
    if not ok:
        notes.append(f"create-role: {err[-300:]}")
        return False
    ok, _, err = aws(["iam", "put-role-policy", "--role-name", rn,
                      "--policy-name", "attach-user-policy", "--policy-document", json.dumps(inline)])
    if not ok:
        notes.append(f"put-role-policy: {err[-300:]}")
        return False
    ok, _, err = aws(["iam", "create-user", "--user-name", un])
    if not ok:
        notes.append(f"create-user: {err[-300:]}")
        return False
    return True


def detonate(suffix, account, grant_policy, notes):
    """Assume the role, then as the role attach grant_policy to the target user.

    Returns True if both the AssumeRole and the AttachUserPolicy succeeded.
    """
    rn, un = role_name(suffix), user_name(suffix)
    role_arn = f"arn:aws:iam::{account}:role/{rn}"

    # IAM is eventually consistent: a freshly created role/trust may not be
    # assumable for a few seconds. Retry briefly rather than fail spuriously.
    creds = None
    for attempt in range(6):
        ok, out, err = aws(["sts", "assume-role", "--role-arn", role_arn,
                            "--role-session-name", f"escalation-{suffix}"])
        if ok:
            creds = json.loads(out)["Credentials"]
            break
        time.sleep(5)
    if not creds:
        notes.append(f"assume-role: {err[-300:]}")
        return False

    # Act AS the assumed role. This AttachUserPolicy is emitted by the
    # assumed-role principal, so CloudTrail records it under
    # arn:aws:sts::<acct>:assumed-role/<role>/<session> -- exactly the ARN the
    # graph builder canonicalizes to Role(<role>), linking it to the AssumeRole
    # edge above (hop_count = 2, privilege_gain = +1).
    ok, _, err = aws(["iam", "attach-user-policy", "--user-name", un,
                      "--policy-arn", grant_policy], creds=creds)
    if not ok:
        notes.append(f"attach-user-policy: {err[-300:]}")
        return False
    return True


def revert(suffix, grant_policy, notes):
    un = user_name(suffix)
    ok, _, err = aws(["iam", "detach-user-policy", "--user-name", un, "--policy-arn", grant_policy])
    if not ok:
        notes.append(f"detach-user-policy: {err[-300:]}")
        return False
    return True


def cleanup(suffix, grant_policy, notes):
    """Delete everything this run created. Best-effort, order matters (a user
    with an attached policy or a role with an inline policy cannot be deleted).
    Returns True only if the account is left clean."""
    rn, un = role_name(suffix), user_name(suffix)
    clean = True

    # user: detach any managed policies (revert may not have run), then delete
    aws(["iam", "detach-user-policy", "--user-name", un, "--policy-arn", grant_policy])
    ok, out, _ = aws(["iam", "list-attached-user-policies", "--user-name", un,
                      "--query", "AttachedPolicies[].PolicyArn", "--output", "text"])
    if ok and out:
        for arn in out.split():
            aws(["iam", "detach-user-policy", "--user-name", un, "--policy-arn", arn])
    ok, _, err = aws(["iam", "delete-user", "--user-name", un])
    if not ok and "NoSuchEntity" not in err:
        notes.append(f"delete-user: {err[-200:]}")
        clean = False

    # role: delete inline policy, then the role
    aws(["iam", "delete-role-policy", "--role-name", rn, "--policy-name", "attach-user-policy"])
    ok, _, err = aws(["iam", "delete-role", "--role-name", rn])
    if not ok and "NoSuchEntity" not in err:
        notes.append(f"delete-role: {err[-200:]}")
        clean = False

    return clean


def sweep_all(notes):
    """Standalone --cleanup-only: delete every stratus-escalation-* resource
    left on the account, regardless of which run created it."""
    swept = 0
    ok, out, _ = aws(["iam", "list-users", "--query",
                      f"Users[?starts_with(UserName,'{RESOURCE_PREFIX}target-')].UserName",
                      "--output", "text"])
    for un in (out.split() if ok and out else []):
        ok2, pols, _ = aws(["iam", "list-attached-user-policies", "--user-name", un,
                            "--query", "AttachedPolicies[].PolicyArn", "--output", "text"])
        for arn in (pols.split() if ok2 and pols else []):
            aws(["iam", "detach-user-policy", "--user-name", un, "--policy-arn", arn])
        aws(["iam", "delete-user", "--user-name", un])
        swept += 1
    ok, out, _ = aws(["iam", "list-roles", "--query",
                      f"Roles[?starts_with(RoleName,'{RESOURCE_PREFIX}role-')].RoleName",
                      "--output", "text"])
    for rn in (out.split() if ok and out else []):
        ok2, pols, _ = aws(["iam", "list-role-policies", "--role-name", rn,
                            "--query", "PolicyNames", "--output", "text"])
        for pn in (pols.split() if ok2 and pols else []):
            aws(["iam", "delete-role-policy", "--role-name", rn, "--policy-name", pn])
        aws(["iam", "delete-role", "--role-name", rn])
        swept += 1
    print(f"Swept {swept} leftover stratus-escalation-* resource(s).")
    return swept


# ── manifest ────────────────────────────────────────────────────────────────

def ensure_manifest(path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()


def append_manifest(path, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writerow(row)


def run_one(rep_index, dry_run, manifest_path, collector, account, grant_policy, jitter):
    suffix = uuid.uuid4().hex[:8]
    run_id = str(uuid.uuid4())
    row = {
        "run_id": run_id, "collector": collector, "aws_account": account,
        "technique_id": TECHNIQUE_ID, "tactic": TACTIC, "rep_index": rep_index,
        "start_ts_utc": now_utc(), "end_ts_utc": "",
        "warmup_status": "", "detonate_status": "", "revert_status": "", "cleanup_status": "",
        "expected_events": ";".join(EXPECTED_EVENTS), "notes": "",
    }
    print(f"\n[{rep_index}] {TECHNIQUE_ID}  (run_id={run_id[:8]}, suffix={suffix})")

    if dry_run:
        print("    DRY RUN - would create role+user, assume role, attach policy, then revert+delete")
        return

    notes = []
    try:
        print("    warmup (create role + target user)...")
        if not warmup(suffix, account, notes):
            row["warmup_status"] = "FAILED"
            print("    warmup FAILED -", notes[-1] if notes else "")
            return
        row["warmup_status"] = "ok"
        time.sleep(random.uniform(*jitter))

        print("    detonate (assume role -> attach policy AS role)...")
        if detonate(suffix, account, grant_policy, notes):
            row["detonate_status"] = "ok"
            print("    detonate ok  (AssumeRole + AttachUserPolicy emitted)")
        else:
            row["detonate_status"] = "FAILED"
            print("    detonate FAILED -", notes[-1] if notes else "")
        time.sleep(random.uniform(*jitter))

        print("    revert (detach policy)...")
        row["revert_status"] = "ok" if revert(suffix, grant_policy, notes) else "FAILED"

    except KeyboardInterrupt:
        notes.append("interrupted by user")
        print("\n    interrupted - attempting cleanup before exit...")
        raise
    finally:
        print("    cleanup (delete user + role)...")
        clean = cleanup(suffix, grant_policy, notes)
        row["cleanup_status"] = "ok" if clean else "FAILED - MANUAL CHECK NEEDED"
        if not clean:
            print(f"    *** CLEANUP INCOMPLETE - run --cleanup-only, or check IAM for "
                  f"{RESOURCE_PREFIX}*-{suffix} ***")
        else:
            print("    cleanup ok")
        row["end_ts_utc"] = now_utc()
        row["notes"] = " | ".join(notes)
        append_manifest(manifest_path, row)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collector", required=True,
                   help="Your handle, e.g. 'vansh'. Writes manifest_<collector>.csv, the same "
                        "file run_detonations.py appends to -- rows merge cleanly.")
    p.add_argument("--reps", type=int, default=1, help="How many escalation chains to run this invocation")
    p.add_argument("--grant-policy", default=DEFAULT_GRANT_POLICY,
                   help="Managed policy ARN the role attaches to the target user (default: IAMFullAccess). "
                        "The captured event is AttachUserPolicy regardless of which policy.")
    p.add_argument("--min-jitter", type=float, default=5.0)
    p.add_argument("--max-jitter", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true", help="Print the plan, touch nothing")
    p.add_argument("--cleanup-only", action="store_true",
                   help="Delete any leftover stratus-escalation-* resources and exit (no detonation)")
    args = p.parse_args()

    account = "dry-run" if args.dry_run else caller_account()
    if account == "unknown":
        print("ERROR: could not determine AWS account. Is 'aws configure' done? "
              "(README section 4)", file=sys.stderr)
        sys.exit(1)

    if args.cleanup_only:
        sweep_all([])
        return

    manifest_path = SCRIPT_DIR / f"manifest_{args.collector}.csv"
    ensure_manifest(manifest_path)

    print(f"Collector: {args.collector}  |  AWS account: {account}")
    print(f"Plan: {args.reps} escalation chain(s)  |  grant policy: {args.grant_policy}")
    print(f"Manifest: {manifest_path}")

    try:
        for rep in range(1, args.reps + 1):
            run_one(rep, args.dry_run, manifest_path, args.collector, account,
                    args.grant_policy, (args.min_jitter, args.max_jitter))
    except KeyboardInterrupt:
        print("\nStopped by user. Partial results are in the manifest; run --cleanup-only to be safe.")

    if not args.dry_run:
        print(f"\nDone. Review {manifest_path.name}. If any cleanup_status != 'ok', run --cleanup-only.")


if __name__ == "__main__":
    main()
