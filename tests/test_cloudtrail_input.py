"""
test_cloudtrail_input.py
========================
Guards feature_engine9's input layer: iter_input_rows (every format the live
pipeline accepts) and normalize_cloudtrail_row (raw CloudTrail -> internal row).

The pipeline takes raw CloudTrail JSON ({"Records": [...]}, a list, one event,
JSONL/NDJSON, optionally gzipped) or a CSV. Only the CSV path was exercised
before, and only indirectly. The riskiest untested behaviour is target
canonicalization: an AssumeRole names its role by roleArn, a CreateRole or
AttachRolePolicy by roleName -- both must become the SAME graph node, or every
multi-hop chain silently loses its middle.
"""

import csv
import gzip
import json
import os
import shutil
import tempfile
import unittest

import feature_engine9 as fe

ACCT = "111122223333"
USER_ARN = f"arn:aws:iam::{ACCT}:user/alice"
ROLE_ARN = f"arn:aws:iam::{ACCT}:role/deploy"
SESSION_ARN = f"arn:aws:sts::{ACCT}:assumed-role/deploy/s1"


def user_identity():
    return {"type": "IAMUser", "arn": USER_ARN, "userName": "alice", "accountId": ACCT,
            "accessKeyId": "AKIAEXAMPLE"}


def assumed_identity(mfa="false"):
    return {"type": "AssumedRole", "arn": SESSION_ARN, "accountId": ACCT, "accessKeyId": "ASIAEXAMPLE",
            "sessionContext": {"sessionIssuer": {"type": "Role", "arn": ROLE_ARN, "userName": "deploy"},
                               "attributes": {"mfaAuthenticated": mfa}}}


def record(name, source, identity, params=None, t="2024-01-01T10:00:00Z", **extra):
    r = {"eventTime": t, "eventName": name, "eventSource": source, "userIdentity": identity,
         "sourceIPAddress": "203.0.113.5", "userAgent": "aws-cli/2", "awsRegion": "us-east-1",
         "readOnly": False, "recipientAccountId": ACCT}
    if params is not None:
        r["requestParameters"] = params
    r.update(extra)
    return r


RECORDS = [
    record("AssumeRole", "sts.amazonaws.com", user_identity(),
           {"roleArn": ROLE_ARN, "roleSessionName": "s1"}),
    record("GetSecretValue", "secretsmanager.amazonaws.com", assumed_identity(),
           {"secretId": "db"}, t="2024-01-01T10:00:05Z"),
]


class InputSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, text, gz=False):
        path = os.path.join(self.tmp, name)
        opener = gzip.open if gz else open
        with opener(path, "wt", encoding="utf-8") as f:
            f.write(text)
        return path

    def names(self, path):
        return [r["event_name"] for r in fe.iter_input_rows(path)]


class TestInputFormats(InputSandbox):
    def test_records_envelope(self):
        path = self.write("a.json", json.dumps({"Records": RECORDS}))
        self.assertEqual(self.names(path), ["AssumeRole", "GetSecretValue"])

    def test_bare_list(self):
        self.assertEqual(self.names(self.write("a.json", json.dumps(RECORDS))),
                         ["AssumeRole", "GetSecretValue"])

    def test_single_event_object(self):
        self.assertEqual(self.names(self.write("a.json", json.dumps(RECORDS[0]))), ["AssumeRole"])

    def test_jsonl_skips_blank_lines(self):
        text = json.dumps(RECORDS[0]) + "\n\n" + json.dumps(RECORDS[1]) + "\n"
        self.assertEqual(self.names(self.write("a.jsonl", text)), ["AssumeRole", "GetSecretValue"])

    def test_gzipped_json(self):
        path = self.write("a.json.gz", json.dumps({"Records": RECORDS}), gz=True)
        self.assertEqual(self.names(path), ["AssumeRole", "GetSecretValue"])

    def test_csv(self):
        path = os.path.join(self.tmp, "a.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "event_name", "principal_arn", "label"])
            w.writeheader()
            w.writerow({"timestamp": "2024-01-01T10:00:00Z", "event_name": "ListRoles",
                        "principal_arn": USER_ARN, "label": "1"})
        (row,) = fe.iter_input_rows(path)
        self.assertEqual((row["event_name"], row["label"]), ("ListRoles", "1"))

    def test_unsupported_extension_is_refused(self):
        with self.assertRaises(ValueError):
            list(fe.iter_input_rows(self.write("a.txt", "x")))


class TestNormalizeRawCloudTrail(unittest.TestCase):
    def test_iam_user_fields(self):
        row = fe.normalize_cloudtrail_row(RECORDS[0])
        self.assertEqual(row["principal_type"], "IAMUser")
        self.assertEqual(row["principal_arn"], USER_ARN)
        self.assertEqual(row["username"], "alice")
        self.assertEqual(row["access_key_id"], "AKIAEXAMPLE")
        self.assertEqual(row["recipient_account_id"], ACCT)
        self.assertIsNone(row["mfa_authenticated"])      # absent, not "false"

    def test_assumed_role_keeps_role_and_session(self):
        row = fe.normalize_cloudtrail_row(RECORDS[1])
        self.assertEqual(row["principal_type"], "AssumedRole")
        self.assertEqual(row["principal_arn"], ROLE_ARN)   # graph node: the role
        self.assertEqual(row["session_arn"], SESSION_ARN)  # lineage: the session
        self.assertEqual(row["username"], "deploy")
        self.assertEqual(row["mfa_authenticated"], "false")

    def test_missing_fields_get_explicit_placeholders(self):
        row = fe.normalize_cloudtrail_row({"eventTime": "2024-01-01T10:00:00Z", "eventName": "X"})
        self.assertEqual(row["principal_arn"], "unknown_principal")
        self.assertEqual(row["target_resource"], "aws_service")
        self.assertIsNone(row["read_only"])

    def test_read_only_false_is_kept_not_dropped(self):
        self.assertEqual(fe.normalize_cloudtrail_row(RECORDS[0])["read_only"], False)

    def test_request_parameters_are_serialized_for_the_feature_code(self):
        row = fe.normalize_cloudtrail_row(RECORDS[0])
        self.assertEqual(json.loads(row["request_params_raw"])["roleSessionName"], "s1")


class TestTargetCanonicalization(unittest.TestCase):
    def target(self, name, params, identity=None):
        return fe.normalize_cloudtrail_row(
            record(name, "iam.amazonaws.com", identity or user_identity(), params))["target_resource"]

    def test_role_arn_and_role_name_are_the_same_node(self):
        by_arn = self.target("AssumeRole", {"roleArn": ROLE_ARN, "roleSessionName": "s"})
        by_name = self.target("AttachRolePolicy", {"roleName": "deploy",
                                                    "policyArn": "arn:aws:iam::aws:policy/ReadOnlyAccess"})
        self.assertEqual(by_arn, by_name)
        self.assertEqual(by_name, ROLE_ARN)

    def test_identity_wins_over_the_policy_it_receives(self):
        self.assertEqual(self.target("AttachUserPolicy", {"userName": "bob", "policyArn": "arn:x"}),
                         f"arn:aws:iam::{ACCT}:user/bob")

    def test_policy_only_calls_target_the_policy(self):
        arn = f"arn:aws:iam::{ACCT}:policy/p"
        self.assertEqual(self.target("CreatePolicyVersion", {"policyArn": arn}), arn)

    def test_bucket_becomes_an_s3_arn(self):
        self.assertEqual(self.target("PutBucketPolicy", {"bucketName": "b"}), "arn:aws:s3:::b")

    def test_resources_array_is_the_fallback(self):
        r = record("Decrypt", "kms.amazonaws.com", user_identity(), {},
                   resources=[{"ARN": "arn:aws:kms:us-east-1:1:key/k"}])
        self.assertEqual(fe.normalize_cloudtrail_row(r)["target_resource"], "arn:aws:kms:us-east-1:1:key/k")

    def test_enriched_target_is_not_overridden(self):
        r = dict(record("AssumeRole", "sts.amazonaws.com", user_identity(),
                        {"roleArn": ROLE_ARN, "roleSessionName": "s"}), target_resource="precomputed")
        self.assertEqual(fe.normalize_cloudtrail_row(r)["target_resource"], "precomputed")


class TestRawJsonLineage(InputSandbox):
    """Regression: on raw CloudTrail, principal_arn of an assumed-role call is
    the ROLE, so the session name was lost and no handoff was ever linked."""

    def test_handoff_is_detected_from_a_raw_cloudtrail_file(self):
        path = self.write("a.json", json.dumps({"Records": RECORDS}))
        engine = fe.FeatureEngineer()
        ctx = [engine._context(row) for row in fe.iter_input_rows(path)]
        self.assertEqual(ctx[0]["principal_handoff"], 0)
        self.assertEqual(ctx[1]["principal_handoff"], 1)

    def test_role_permissions_follow_the_session(self):
        grant = record("AttachRolePolicy", "iam.amazonaws.com", user_identity(),
                       {"roleName": "deploy", "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"},
                       t="2024-01-01T09:59:00Z")
        path = self.write("a.json", json.dumps({"Records": [grant] + RECORDS}))
        engine = fe.FeatureEngineer()
        ctx = [engine._context(row) for row in fe.iter_input_rows(path)]
        self.assertAlmostEqual(ctx[-1]["actor_permission_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
