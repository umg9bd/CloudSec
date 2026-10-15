"""
test_policy_features.py
=======================
Guards feature_engine9.parse_policy_features, which produces four LSTM inputs
(policy_statement_count_normalized, has_wildcard_action, has_wildcard_resource,
privileged_action_reach) from a request's policy document. It had no direct
tests; a silent regression here changes what every trained model sees.
"""

import json
import unittest

from feature_engine9 import parse_policy_features


def params(*statements, key="policyDocument", as_string=True, **extra):
    doc = {"Version": "2012-10-17", "Statement": list(statements)}
    return json.dumps({key: json.dumps(doc) if as_string else doc, **extra})


ALLOW_ALL = {"Effect": "Allow", "Action": "*", "Resource": "*"}


class TestPolicyFeatures(unittest.TestCase):
    def test_no_policy(self):
        self.assertEqual(parse_policy_features(json.dumps({"roleName": "r"})), (0, 0, 0, 0.0))

    def test_unparseable_input(self):
        for raw in (None, "", "not json", "[1, 2]"):
            self.assertEqual(parse_policy_features(raw), (0, 0, 0, 0.0), raw)

    def test_full_wildcard(self):
        self.assertEqual(parse_policy_features(params(ALLOW_ALL)), (1, 1, 1, 1.0))

    def test_service_wildcard_counts_as_wildcard_action(self):
        _, wa, wr, reach = parse_policy_features(
            params({"Effect": "Allow", "Action": ["s3:*"], "Resource": "arn:aws:s3:::b/*"}))
        self.assertEqual((wa, wr, reach), (1, 0, 1.0))

    def test_scoped_policy_has_no_wildcards(self):
        self.assertEqual(parse_policy_features(
            params({"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "arn:aws:s3:::b/k"})),
            (1, 0, 0, 0.0))

    def test_pass_role_and_assume_role_are_partial_reach(self):
        for action in ("iam:PassRole", "sts:AssumeRole", "IAM:PASSROLE"):
            reach = parse_policy_features(params({"Effect": "Allow", "Action": action, "Resource": "x"}))[3]
            self.assertEqual(reach, 0.6, action)

    def test_deny_statements_grant_nothing(self):
        self.assertEqual(parse_policy_features(params({"Effect": "Deny", "Action": "*", "Resource": "*"})),
                         (1, 0, 0, 0.0))

    def test_single_statement_object_not_list(self):
        raw = json.dumps({"policyDocument": {"Version": "2012-10-17", "Statement": ALLOW_ALL}})
        self.assertEqual(parse_policy_features(raw), (1, 1, 1, 1.0))

    def test_document_as_object_or_string(self):
        self.assertEqual(parse_policy_features(params(ALLOW_ALL, as_string=False)),
                         parse_policy_features(params(ALLOW_ALL)))

    def test_trust_policy_is_read_too(self):
        trust = {"Effect": "Allow", "Principal": {"AWS": "*"}, "Action": "sts:AssumeRole"}
        count, _, _, reach = parse_policy_features(params(trust, key="assumeRolePolicyDocument"))
        self.assertEqual((count, reach), (1, 0.6))

    def test_privileged_managed_policy_arn_is_full_reach(self):
        for arn in ("arn:aws:iam::aws:policy/AdministratorAccess",
                    "arn:aws:iam::aws:policy/PowerUserAccess",
                    "arn:aws:iam::aws:policy/IAMFullAccess"):
            self.assertEqual(parse_policy_features(json.dumps({"roleName": "r", "policyArn": arn}))[3],
                             1.0, arn)

    def test_scoped_managed_policy_arn_is_not(self):
        raw = json.dumps({"roleName": "r", "policyArn": "arn:aws:iam::aws:policy/ReadOnlyAccess"})
        self.assertEqual(parse_policy_features(raw)[3], 0.0)

    def test_statement_count_includes_every_statement(self):
        stmts = [{"Effect": "Allow", "Action": f"s3:Get{i}", "Resource": "x"} for i in range(3)]
        self.assertEqual(parse_policy_features(params(*stmts))[0], 3)


if __name__ == "__main__":
    unittest.main()
