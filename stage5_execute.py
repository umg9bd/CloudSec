"""
stage5_execute.py  --  Stage 5: execute an APPROVED plan in a sandbox and verify
================================================================================
Closes the agent-gate loop: take the PREDICTED CloudTrail from Stage 3
(log-generation), optionally assume a role scoped to exactly those actions
(least-privilege enforcement), run the plan in your sandbox, pull the REAL
CloudTrail it produced, and report the predicted-vs-real match rate.

That match rate is the key validation of the "hypothetical logs" premise
(the disclosure's "mapper first" check) and the enforcement demo in one.

SAFETY (this touches a real account -- read this):
  * DRY-RUN BY DEFAULT. Nothing runs without --execute.
  * ALLOWLIST ONLY. It will execute only the safe, reversible S3 secure-bucket
    lifecycle actions below. Any other predicted action is SKIPPED, even if the
    plan asks for it -- so a bad plan can't do damage through this tool.
  * UNIQUE throwaway bucket; cleanup runs in a finally block (and --cleanup-only).
  * --account asserts you're on the expected sandbox; aborts on mismatch.
  * Temporary credentials are never printed.

Uses the AWS CLI (same as stratus_collection/*), so configure your sandbox
profile first (aws configure / AWS_PROFILE). No boto3 needed.

Examples:
  python stage5_execute.py --predicted log-generation/generated_logs/s3_secure_data_lake_cloudtrail.json --account 536697249497            # dry run
  python stage5_execute.py --predicted ... --account 536697249497 --execute                                                              # really run + verify
  python stage5_execute.py --predicted ... --account 536697249497 --execute --role-arn arn:aws:iam::536697249497:role/gate-executor      # + scoped STS enforcement
  python stage5_execute.py --cleanup-only --bucket stage5-demo-abc123 --account 536697249497 --execute
"""
import argparse
import json
import os
import random
import string
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

REGION = "us-east-1"

# ── ALLOWLIST: (eventSource, eventName) -> builds the aws-cli args. Only these
#    run. Each is a low-risk, reversible S3 secure-bucket lifecycle action. ──
def _create_bucket(bkt, region):
    args = ["s3api", "create-bucket", "--bucket", bkt]
    if region != "us-east-1":
        args += ["--create-bucket-configuration", f"LocationConstraint={region}"]
    return args

ALLOW = {
    ("s3.amazonaws.com", "CreateBucket"): _create_bucket,
    ("s3.amazonaws.com", "PutBucketEncryption"): lambda b, r: [
        "s3api", "put-bucket-encryption", "--bucket", b, "--server-side-encryption-configuration",
        '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'],
    ("s3.amazonaws.com", "PutBucketPublicAccessBlock"): lambda b, r: [
        "s3api", "put-public-access-block", "--bucket", b, "--public-access-block-configuration",
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"],
    ("s3.amazonaws.com", "PutBucketVersioning"): lambda b, r: [
        "s3api", "put-bucket-versioning", "--bucket", b, "--versioning-configuration", "Status=Enabled"],
}
# IAM actions the session policy must allow, per eventName (for the STS scope)
IAM_ACTION = {
    "CreateBucket": "s3:CreateBucket", "PutBucketEncryption": "s3:PutEncryptionConfiguration",
    "PutBucketPublicAccessBlock": "s3:PutBucketPublicAccessBlock", "PutBucketVersioning": "s3:PutBucketVersioning",
}


def aws(args, creds=None, region=REGION):
    env = dict(os.environ)
    if creds:
        env["AWS_ACCESS_KEY_ID"] = creds["AccessKeyId"]
        env["AWS_SECRET_ACCESS_KEY"] = creds["SecretAccessKey"]
        env["AWS_SESSION_TOKEN"] = creds["SessionToken"]
    p = subprocess.run(["aws"] + args + ["--region", region, "--output", "json"],
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def caller_account(creds=None):
    rc, out, err = aws(["sts", "get-caller-identity"], creds)
    if rc != 0:
        sys.exit(f"cannot call AWS (is your sandbox profile configured?): {err}")
    return json.loads(out)


def load_predicted(path):
    d = json.load(open(path, encoding="utf-8"))
    evs = d["Records"] if isinstance(d, dict) and "Records" in d else d
    return [e for e in evs if isinstance(e, dict) and e.get("eventName")]


def session_policy(event_names, bucket):
    acts = sorted({IAM_ACTION[n] for n in event_names if n in IAM_ACTION})
    return json.dumps({"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Action": acts,
        "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]}]})


def assume_scoped(role_arn, policy):
    rc, out, err = aws(["sts", "assume-role", "--role-arn", role_arn,
                        "--role-session-name", "stage5-gate", "--duration-seconds", "900",
                        "--policy", policy])
    if rc != 0:
        sys.exit(f"assume-role failed: {err}")
    return json.loads(out)["Credentials"]           # never printed


def fetch_real_cloudtrail(bucket, start, want_names, creds, region, max_wait_min):
    """Poll CloudTrail lookup-events for events referencing our bucket."""
    seen = {}
    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        rc, out, err = aws(["cloudtrail", "lookup-events",
                            "--start-time", start, "--max-results", "50"], creds, region)
        if rc == 0 and out:
            for ev in json.loads(out).get("Events", []):
                raw = json.loads(ev.get("CloudTrailEvent", "{}"))
                if bucket in json.dumps(raw):
                    seen[ev["EventName"]] = raw
            if want_names <= set(seen):
                break
        print(f"    waiting for CloudTrail... have {sorted(seen)} "
              f"(CloudTrail management events lag a few minutes)", flush=True)
        time.sleep(45)
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predicted", help="Stage-3 predicted CloudTrail json (log-generation output)")
    ap.add_argument("--account", required=True, help="expected sandbox account id (asserted before anything runs)")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--role-arn", help="assume this role with a session policy scoped to the plan (enforcement demo)")
    ap.add_argument("--bucket", help="override bucket name (used by --cleanup-only)")
    ap.add_argument("--execute", action="store_true", help="actually run (default is DRY-RUN)")
    ap.add_argument("--cleanup-only", action="store_true", help="just delete the bucket and exit")
    ap.add_argument("--max-wait-min", type=float, default=12.0, help="how long to poll for real CloudTrail")
    args = ap.parse_args()

    ident = caller_account()
    print(f"caller: account {ident['Account']} as {ident['Arn']}")
    if ident["Account"] != args.account:
        sys.exit(f"ABORT: caller account {ident['Account']} != expected sandbox {args.account}")

    creds = None
    bucket = args.bucket or ("stage5-demo-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8)))

    if args.cleanup_only:
        if not args.execute:
            print(f"DRY RUN: would delete bucket {bucket}. Add --execute to really delete."); return
        rc, _, err = aws(["s3", "rb", f"s3://{bucket}", "--force"], creds, args.region)
        print(f"cleanup {bucket}: {'ok' if rc == 0 else 'FAILED ' + err}"); return

    if not args.predicted:
        sys.exit("--predicted is required (unless --cleanup-only)")
    predicted = load_predicted(args.predicted)
    pred_names = [e["eventName"] for e in predicted]
    run_names = [n for n in pred_names if (e_src(predicted, n), n) in ALLOW]
    skipped = [n for n in pred_names if n not in run_names]
    print(f"\npredicted events: {pred_names}")
    print(f"executable (allowlisted, safe): {run_names}")
    if skipped:
        print(f"SKIPPED (not in safe allowlist, will NOT run): {skipped}")

    # scoped STS enforcement (optional)
    if args.role_arn:
        pol = session_policy(run_names, bucket)
        print(f"\nleast-privilege session policy for {bucket}:\n{pol}")
        if args.execute:
            creds = assume_scoped(args.role_arn, pol)
            print("assumed scoped role (temp creds held in memory, not printed).")
    else:
        print("\n[no --role-arn] running with your default credentials -- enforcement (scoped STS) skipped.")

    cmds = [(n, ALLOW[("s3.amazonaws.com", n)](bucket, args.region)) for n in run_names]
    if not args.execute:
        print("\nDRY RUN -- would execute, against bucket", bucket + ":")
        for n, c in cmds:
            print("   aws " + " ".join(c))
        print("\nAdd --execute to run, verify against real CloudTrail, and clean up.")
        return

    start = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ran = []
    try:
        print(f"\nexecuting plan on {bucket} ...")
        for n, c in cmds:
            rc, _, err = aws(c, creds, args.region)
            print(f"   {n}: {'ok' if rc == 0 else 'FAILED ' + err}")
            if rc == 0:
                ran.append(n)
        print("\nfetching REAL CloudTrail (predicted-vs-real)...")
        real = fetch_real_cloudtrail(bucket, start, set(run_names), creds, args.region, args.max_wait_min)
        real_names = set(real)
        want = set(run_names)
        matched = want & real_names
        print("\n==================== PREDICTED vs REAL ====================")
        print(f"predicted (executable): {sorted(want)}")
        print(f"real CloudTrail seen  : {sorted(real_names)}")
        print(f"MATCH: {len(matched)}/{len(want)} predicted events appeared "
              f"({(len(matched)/len(want)*100 if want else 0):.0f}%)")
        extra = real_names - want
        if extra:
            print(f"(real events not predicted: {sorted(extra)})")
    finally:
        print(f"\ncleanup: deleting {bucket} ...")
        rc, _, err = aws(["s3", "rb", f"s3://{bucket}", "--force"], creds, args.region)
        print(f"   {'ok' if rc == 0 else 'FAILED (delete manually: aws s3 rb s3://' + bucket + ' --force) ' + err}")


def e_src(predicted, name):
    for e in predicted:
        if e["eventName"] == name:
            return e.get("eventSource", "")
    return ""


if __name__ == "__main__":
    main()
