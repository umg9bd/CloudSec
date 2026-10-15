"""
test_iam_permissions.py
=======================
Guards iam_permissions.py: permission state reconstructed from the event
stream's own policy content (not from event names).
"""

import json
import unittest

import iam_permissions as ip

ADMIN = "arn:aws:iam::aws:policy/AdministratorAccess"
S3_RO = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
POWER = "arn:aws:iam::aws:policy/PowerUserAccess"


def doc(*actions):
    return json.dumps({"Version": "2012-10-17",
                       "Statement": [{"Effect": "Allow", "Action": list(actions), "Resource": "*"}]})


class TestCoverage(unittest.TestCase):
    def test_allow_all_covers_everything(self):
        self.assertAlmostEqual(ip.coverage(ip.canonical_statements(ip.ALLOW_ALL)), 1.0)

    def test_nothing_covers_nothing(self):
        self.assertEqual(ip.coverage(()), 0.0)
        self.assertEqual(ip.max_access_rank(()), -1)

    def test_breadth_is_ordered(self):
        s3_ro = ip.coverage(ip.canonical_statements(ip.MANAGED_POLICIES[S3_RO]))
        power = ip.coverage(ip.canonical_statements(ip.MANAGED_POLICIES[POWER]))
        admin = ip.coverage(ip.canonical_statements(ip.MANAGED_POLICIES[ADMIN]))
        self.assertLess(0, s3_ro)
        self.assertLess(s3_ro, power)
        self.assertLess(power, admin)

    def test_notaction_excludes_the_named_service(self):
        canon = ip.canonical_statements([{"Effect": "Allow", "NotAction": ["iam:*"]}])
        full = ip.canonical_statements(ip.ALLOW_ALL)
        self.assertLess(ip.allowed_count(canon), ip.allowed_count(full))
        self.assertGreater(ip.allowed_count(canon), 0)

    def test_deny_statements_grant_nothing(self):
        self.assertEqual(ip.canonical_statements([{"Effect": "Deny", "Action": ["*"]}]), ())

    def test_read_only_policy_does_not_reach_permissions_management(self):
        rank = ip.max_access_rank(ip.canonical_statements(ip.MANAGED_POLICIES[S3_RO]))
        self.assertLess(rank, ip.ACCESS_LEVEL_RANK["Permissions management"])

    def test_admin_reaches_permissions_management(self):
        self.assertEqual(ip.max_access_rank(ip.canonical_statements(ip.ALLOW_ALL)),
                         ip.ACCESS_LEVEL_RANK["Permissions management"])


class TestIdentityKeys(unittest.TestCase):
    def test_assumed_role_session_maps_to_the_role(self):
        arn = "arn:aws:sts::111122223333:assumed-role/deploy/session-1"
        self.assertEqual(ip.actor_key("AssumedRole", arn), "role/deploy")
        self.assertEqual(ip.assumed_session(arn), ("deploy", "session-1"))

    def test_user_and_root(self):
        self.assertEqual(ip.actor_key("IAMUser", "arn:aws:iam::111122223333:user/alice"), "user/alice")
        self.assertEqual(ip.actor_key("Root", "arn:aws:iam::111122223333:root"), "root")

    def test_service_principal_has_no_key(self):
        self.assertIsNone(ip.actor_key("AWSService", None))


class TestPermissionState(unittest.TestCase):
    def setUp(self):
        self.st = ip.PermissionState()

    def test_unobserved_identity_is_unknown_not_empty(self):
        canon, known = self.st.effective("role/never-seen")
        self.assertEqual(canon, ())
        self.assertFalse(known)

    def test_attach_managed_policy_reports_before_and_after(self):
        (key, before, after), = self.st.apply("AttachRolePolicy", {"roleName": "r", "policyArn": ADMIN})
        self.assertEqual(key, "role/r")
        self.assertEqual(before, ())
        self.assertAlmostEqual(ip.coverage(after), 1.0)
        self.assertTrue(self.st.effective("role/r")[1])

    def test_inline_policy_document_is_parsed(self):
        (_, _, after), = self.st.apply("PutUserPolicy", {"userName": "u", "policyName": "p",
                                                         "policyDocument": doc("s3:GetObject")})
        self.assertEqual(after, (("Action", ("s3:getobject",)),))

    def test_detach_removes_the_grant(self):
        self.st.apply("AttachUserPolicy", {"userName": "u", "policyArn": ADMIN})
        (_, before, after), = self.st.apply("DetachUserPolicy", {"userName": "u", "policyArn": ADMIN})
        self.assertGreater(ip.coverage(before), ip.coverage(after))

    def test_group_membership_carries_group_permissions(self):
        self.st.apply("AttachGroupPolicy", {"groupName": "g", "policyArn": ADMIN})
        (key, before, after), = self.st.apply("AddUserToGroup", {"groupName": "g", "userName": "u"})
        self.assertEqual(key, "user/u")
        self.assertEqual(before, ())
        self.assertAlmostEqual(ip.coverage(after), 1.0)

    def test_group_policy_change_reaches_existing_members(self):
        self.st.apply("AddUserToGroup", {"groupName": "g", "userName": "u"})
        changes = self.st.apply("AttachGroupPolicy", {"groupName": "g", "policyArn": S3_RO})
        self.assertIn("user/u", [k for k, _, _ in changes])

    def test_unknown_customer_policy_makes_state_unknown(self):
        self.st.apply("AttachRolePolicy", {"roleName": "r",
                                           "policyArn": "arn:aws:iam::111122223333:policy/custom"})
        self.assertFalse(self.st.effective("role/r")[1])

    def test_policy_version_swap_changes_every_holder(self):
        arn = "arn:aws:iam::111122223333:policy/custom"
        self.st.apply("CreatePolicy", {"policyName": "custom", "policyDocument": doc("s3:GetObject")},
                      account_id="111122223333")
        self.st.apply("AttachUserPolicy", {"userName": "u", "policyArn": arn})
        self.st.apply("CreatePolicyVersion", {"policyArn": arn, "policyDocument": doc("*"),
                                              "setAsDefault": False})
        self.assertLess(ip.coverage(self.st.effective("user/u")[0]), 1.0)   # not default yet
        changes = self.st.apply("SetDefaultPolicyVersion", {"policyArn": arn, "versionId": "v2"})
        (key, before, after), = changes
        self.assertEqual(key, "user/u")
        self.assertAlmostEqual(ip.coverage(after), 1.0)

    def test_non_permission_events_change_nothing(self):
        self.assertEqual(self.st.apply("GetSecretValue", {"secretId": "x"}), [])
        self.assertEqual(self.st.apply("AttachRolePolicy", None), [])

    def test_json_round_trip(self):
        self.st.apply("AttachRolePolicy", {"roleName": "r", "policyArn": ADMIN})
        self.st.apply("AddUserToGroup", {"groupName": "g", "userName": "u"})
        again = ip.PermissionState.from_json(json.loads(json.dumps(self.st.to_json())))
        self.assertEqual(again.effective("role/r"), self.st.effective("role/r"))
        self.assertEqual(again.groups_of, self.st.groups_of)


if __name__ == "__main__":
    unittest.main()
