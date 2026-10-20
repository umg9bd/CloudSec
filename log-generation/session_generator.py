"""
Session & Attack-Chain Generator for CloudTrail Log Streams.

Produces realistic, full-length CloudTrail log deliveries (15 to 50+ events per session)
combining discovery/reconnaissance, core administrative operations, and multi-step
privilege escalation attack chains matching the project's attack taxonomy.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import uuid
from typing import Any, Dict, List

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from adapter import Stage3PipelineAdapter
except ImportError:
    from stage3_intent_event_predictor.adapter import Stage3PipelineAdapter


def make_event(
    event_name: str,
    service: str,
    timestamp: datetime.datetime,
    principal_arn: str,
    account_id: str = "111122223333",
    region: str = "us-east-1",
    params: Dict[str, Any] = None,
    read_only: bool = False,
    error_code: str = None,
) -> Dict[str, Any]:
    username = principal_arn.split("/")[-1] if "/" in principal_arn else "agentic-operator"
    source_ip = f"198.51.{random.randint(1, 254)}.{random.randint(1, 254)}"
    return {
        "eventVersion": "1.08",
        "userIdentity": {
            "type": "IAMUser",
            "principalId": f"AIDA{uuid.uuid4().hex[:16].upper()}",
            "arn": principal_arn,
            "accountId": account_id,
            "userName": username,
        },
        "eventTime": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eventSource": f"{service}.amazonaws.com",
        "eventName": event_name,
        "awsRegion": region,
        "sourceIPAddress": source_ip,
        "userAgent": "aws-sdk-go/v2 aws-cli/2.15.0",
        "requestParameters": params or {},
        "responseElements": None if read_only or error_code else {"status": "SUCCESS"},
        "errorCode": error_code,
        "errorMessage": f"Access denied for {event_name}" if error_code else None,
        "requestID": str(uuid.uuid4()),
        "eventID": str(uuid.uuid4()),
        "readOnly": read_only,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": account_id,
        "eventCategory": "Management",
    }


def generate_s3_admin_session(
    bucket_name: str = "analytics-prod-data-lake",
    principal_arn: str = "arn:aws:iam::111122223333:user/agentic-operator",
    base_time: datetime.datetime = None,
) -> List[Dict[str, Any]]:
    """Generates a complete 18-event administrative S3 onboarding session."""
    t = base_time or datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)
    events = []

    # 1. Reconnaissance / Pre-flight checks
    recon = [
        ("GetCallerIdentity", "sts", {}, True),
        ("DescribeRegions", "ec2", {}, True),
        ("ListBuckets", "s3", {}, True),
        ("GetAccountSummary", "iam", {}, True),
        ("ListRoles", "iam", {}, True),
    ]
    for ev_name, svc, p, ro in recon:
        events.append(make_event(ev_name, svc, t, principal_arn, params=p, read_only=ro))
        t += datetime.timedelta(seconds=random.randint(5, 25))

    # 2. Pre-check target bucket
    events.append(make_event("HeadBucket", "s3", t, principal_arn, params={"bucketName": bucket_name}, read_only=True, error_code="404NotFound"))
    t += datetime.timedelta(seconds=8)

    # 3. Core Creation & Security Hardening (Stage 3 Actions)
    events.append(make_event("CreateBucket", "s3", t, principal_arn, params={"bucketName": bucket_name, "CreateBucketConfiguration": {}}))
    t += datetime.timedelta(seconds=6)
    events.append(make_event("PutBucketEncryption", "s3", t, principal_arn, params={"bucketName": bucket_name, "ServerSideEncryptionConfiguration": {"rules": [{"applyServerSideEncryptionByDefault": {"sseAlgorithm": "AES256"}}]}}))
    t += datetime.timedelta(seconds=5)
    events.append(make_event("PutBucketPublicAccessBlock", "s3", t, principal_arn, params={"bucketName": bucket_name, "PublicAccessBlockConfiguration": {"BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True}}))
    t += datetime.timedelta(seconds=6)
    events.append(make_event("PutBucketVersioning", "s3", t, principal_arn, params={"bucketName": bucket_name, "VersioningConfiguration": {"Status": "Enabled"}}))
    t += datetime.timedelta(seconds=7)
    events.append(make_event("PutBucketTagging", "s3", t, principal_arn, params={"bucketName": bucket_name, "Tagging": {"TagSet": [{"Key": "Environment", "Value": "Production"}]}}))
    t += datetime.timedelta(seconds=12)

    # 4. Post-verification & Initial Object Ingestion
    post_ops = [
        ("GetBucketEncryption", "s3", {"bucketName": bucket_name}, True),
        ("GetPublicAccessBlock", "s3", {"bucketName": bucket_name}, True),
        ("GetBucketVersioning", "s3", {"bucketName": bucket_name}, True),
        ("PutObject", "s3", {"bucketName": bucket_name, "key": "reports/schema.json"}, False),
        ("HeadObject", "s3", {"bucketName": bucket_name, "key": "reports/schema.json"}, True),
        ("ListObjectsV2", "s3", {"bucketName": bucket_name, "prefix": "reports/"}, True),
    ]
    for ev_name, svc, p, ro in post_ops:
        events.append(make_event(ev_name, svc, t, principal_arn, params=p, read_only=ro))
        t += datetime.timedelta(seconds=random.randint(4, 20))

    return events


def generate_privilege_escalation_chain(
    chain_type: str = "recon_then_role",
    attacker_name: str = "syn:attacker-01",
    base_time: datetime.datetime = None,
) -> List[Dict[str, Any]]:
    """Generates an authentic multi-hop privilege escalation attack chain matching project taxonomy."""
    t = base_time or datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=45)
    events = []
    attacker_arn = f"arn:aws:iam::111122223333:user/{attacker_name}"
    escalated_role_arn = f"arn:aws:iam::111122223333:role/AdminExecutionRole"

    if chain_type == "recon_then_role":
        steps = [
            ("GetCallerIdentity", "sts", {}, True, attacker_arn),
            ("DescribeVpcs", "ec2", {}, True, attacker_arn),
            ("DescribeSubnets", "ec2", {}, True, attacker_arn),
            ("DescribeSecurityGroups", "ec2", {}, True, attacker_arn),
            ("DescribeInstances", "ec2", {}, True, attacker_arn),
            ("ListRoles", "iam", {}, True, attacker_arn),
            ("GetRole", "iam", {"roleName": "existing-service-role"}, True, attacker_arn),
            ("CreateRole", "iam", {"roleName": "AdminExecutionRole"}, False, attacker_arn),
            ("AttachRolePolicy", "iam", {"roleName": "AdminExecutionRole", "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"}, False, attacker_arn),
            ("PutRolePolicy", "iam", {"roleName": "AdminExecutionRole", "policyName": "CustomInlineAccess"}, False, attacker_arn),
            ("AssumeRole", "sts", {"roleArn": escalated_role_arn}, False, attacker_arn),
            ("GetCallerIdentity", "sts", {}, True, escalated_role_arn),
            ("DescribeKey", "kms", {}, True, escalated_role_arn),
            ("ListSecrets", "secretsmanager", {}, True, escalated_role_arn),
            ("GetSecretValue", "secretsmanager", {"secretId": "prod/db/credentials"}, False, escalated_role_arn),
            ("DescribeTrails", "cloudtrail", {}, True, escalated_role_arn),
            ("StopLogging", "cloudtrail", {"name": "prod-audit-trail"}, False, escalated_role_arn),
        ]
    elif chain_type == "iam_user_persist":
        steps = [
            ("GetCallerIdentity", "sts", {}, True, attacker_arn),
            ("ListUsers", "iam", {}, True, attacker_arn),
            ("GetUser", "iam", {"userName": "admin"}, True, attacker_arn),
            ("CreateUser", "iam", {"userName": "backup-admin-user"}, False, attacker_arn),
            ("CreateLoginProfile", "iam", {"userName": "backup-admin-user"}, False, attacker_arn),
            ("CreateAccessKey", "iam", {"userName": "backup-admin-user"}, False, attacker_arn),
            ("AttachUserPolicy", "iam", {"userName": "backup-admin-user", "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"}, False, attacker_arn),
            ("ListAccessKeys", "iam", {"userName": "backup-admin-user"}, True, attacker_arn),
            ("ListBuckets", "s3", {}, True, attacker_arn),
            ("GetBucketPolicy", "s3", {"bucketName": "company-crown-jewels"}, True, attacker_arn),
        ]
    elif chain_type == "secrets_loot":
        steps = [
            ("GetCallerIdentity", "sts", {}, True, attacker_arn),
            ("ListSecrets", "secretsmanager", {}, True, attacker_arn),
            ("DescribeSecret", "secretsmanager", {"secretId": "prod/rds/master-key"}, True, attacker_arn),
            ("GetSecretValue", "secretsmanager", {"secretId": "prod/rds/master-key"}, False, attacker_arn),
            ("Decrypt", "kms", {"CiphertextBlob": "<encrypted-data>"}, False, attacker_arn),
            ("GetSecretValue", "secretsmanager", {"secretId": "prod/api/jwt-private-key"}, False, attacker_arn),
            ("ListBuckets", "s3", {}, True, attacker_arn),
            ("PutObject", "s3", {"bucketName": "exfil-staging-bucket", "key": "dump.tar.gz"}, False, attacker_arn),
        ]
    else:
        # Default role assume
        steps = [
            ("GetCallerIdentity", "sts", {}, True, attacker_arn),
            ("ListRoles", "iam", {}, True, attacker_arn),
            ("CreateRole", "iam", {"roleName": "TempPrivEscRole"}, False, attacker_arn),
            ("AttachRolePolicy", "iam", {"roleName": "TempPrivEscRole", "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"}, False, attacker_arn),
            ("AssumeRole", "sts", {"roleArn": "arn:aws:iam::111122223333:role/TempPrivEscRole"}, False, attacker_arn),
            ("GetCallerIdentity", "sts", {}, True, "arn:aws:iam::111122223333:role/TempPrivEscRole"),
            ("GetSecretValue", "secretsmanager", {"secretId": "prod/payment-service/key"}, False, "arn:aws:iam::111122223333:role/TempPrivEscRole"),
        ]

    for ev_name, svc, p, ro, p_arn in steps:
        events.append(make_event(ev_name, svc, t, p_arn, params=p, read_only=ro))
        t += datetime.timedelta(seconds=random.randint(8, 35))

    return events


def generate_enterprise_mixed_log_stream(
    total_events: int = 65,
    include_attack: bool = True,
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Generates a realistic, large enterprise CloudTrail delivery file (60+ events)
    interleaving routine benign developer background activity with a multi-step task/attack.
    """
    base_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=60)
    records = []

    # 1. Routine benign background traffic (dev, ci/cd, monitor)
    background_principals = [
        "arn:aws:iam::111122223333:user/developer-alice",
        "arn:aws:iam::111122223333:user/developer-bob",
        "arn:aws:iam::111122223333:role/github-actions-deployer",
        "arn:aws:iam::111122223333:role/aws-service-role/datadog-monitor",
    ]

    benign_ops = [
        ("DescribeInstances", "ec2", True),
        ("DescribeSecurityGroups", "ec2", True),
        ("DescribeVolumes", "ec2", True),
        ("DescribeSnapshots", "ec2", True),
        ("ListBuckets", "s3", True),
        ("GetBucketLocation", "s3", True),
        ("ListRoles", "iam", True),
        ("GetCallerIdentity", "sts", True),
        ("DescribeKey", "kms", True),
        ("ListLogGroups", "logs", True),
    ]

    t = base_time
    for _ in range(25):
        p_arn = random.choice(background_principals)
        op_name, svc, ro = random.choice(benign_ops)
        records.append(make_event(op_name, svc, t, p_arn, read_only=ro))
        t += datetime.timedelta(seconds=random.randint(10, 45))

    # 2. The core planned workflow / S3 administrative task
    s3_events = generate_s3_admin_session(base_time=t)
    records.extend(s3_events)
    t = datetime.datetime.strptime(records[-1]["eventTime"], "%Y-%m-%dT%H:%M:%SZ")

    # 3. Optional Attack Chain (e.g. recon_then_role)
    if include_attack:
        attack_events = generate_privilege_escalation_chain("recon_then_role", base_time=t + datetime.timedelta(minutes=2))
        records.extend(attack_events)

    # Sort all events chronologically
    records.sort(key=lambda r: r["eventTime"])

    log_delivery = {"Records": records}
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log_delivery, f, indent=2)
        print(f"Generated enterprise CloudTrail log ({len(records)} events): {output_path}")

    return log_delivery


def main():
    parser = argparse.ArgumentParser(description="Generate realistic multi-event CloudTrail sessions and attack streams")
    parser.add_argument("--output-dir", default=os.path.join(_current_dir, "generated_logs"))
    parser.add_argument("--events", type=int, default=65, help="Approximate total events in mixed stream")
    parser.add_argument("--include-attack", action="store_true", default=True, help="Include privilege escalation chain")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=== Generating Realistic, Large-Scale CloudTrail Sessions ===")

    # 1. S3 Full Administrative Session (18 events)
    s3_session = {"Records": generate_s3_admin_session()}
    s3_path = os.path.join(out_dir, "session_s3_admin_complete.json")
    with open(s3_path, "w", encoding="utf-8") as f:
        json.dump(s3_session, f, indent=2)
    print(f"1. S3 Complete Admin Session (18 events): {s3_path}")

    # 2. Privilege Escalation Chain: Recon -> Role -> Secrets -> StopLogging (17 events)
    pe_session = {"Records": generate_privilege_escalation_chain("recon_then_role")}
    pe_path = os.path.join(out_dir, "session_privilege_escalation_attack.json")
    with open(pe_path, "w", encoding="utf-8") as f:
        json.dump(pe_session, f, indent=2)
    print(f"2. IAM Privilege Escalation Attack Chain (17 events): {pe_path}")

    # 3. Enterprise Mixed Log Stream (60+ events) with background dev traffic + S3 onboarding + attack
    mixed_path = os.path.join(out_dir, "enterprise_multi_session_stream.json")
    generate_enterprise_mixed_log_stream(total_events=args.events, include_attack=args.include_attack, output_path=mixed_path)
    print(f"3. Enterprise Multi-Session Stream (60+ events): {mixed_path}")

    print("\nAll full session logs generated successfully in 'log-generation/generated_logs/'!")


if __name__ == "__main__":
    main()
