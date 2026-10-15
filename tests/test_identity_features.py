"""
test_identity_features.py
=========================
Guards the feature engine's identity / permission / privilege features and the
line between ground truth and features.

  * Ground truth never reaches a feature: adding or changing any
    GROUND_TRUTH_COLUMNS value leaves every feature identical.
  * Handoffs come from observable links (AssumeRole's roleSessionName matching
    the later assumed-role ARN; credentials issued to a user), not row order.
  * Permission features come from policy CONTENT: the same AttachRolePolicy
    call scores differently for AdministratorAccess and a read-only policy.
  * source_privilege_level / target_resource_criticality follow documented
    scales instead of the old hand tiers.
  * Label-fitted priors only learn from training rows, and shrink toward the
    fitted base rate instead of hand-assigned per-event scores.
"""

import json
import os
import tempfile
import unittest

import feature_engine9 as fe
from privilege_features import SERVICE_SENSITIVITY

ACCT = "111122223333"
USER = f"arn:aws:iam::{ACCT}:user/alice"
ADMIN = "arn:aws:iam::aws:policy/AdministratorAccess"
S3_RO = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"


def event(ts, name, source, ptype, arn, params=None, error="", **extra):
    raw = {"eventTime": f"2024-01-01T10:{ts}Z", "eventName": name, "eventSource": source,
           "principal_type": ptype, "principal_arn": arn, "error_code": error,
           "request_params_raw": json.dumps(params) if params is not None else None,
           "recipient_account_id": ACCT}
    raw.update(extra)
    return fe.normalize_cloudtrail_row(raw)


def features(engine, row):
    struct = engine.get_structural_data(row)
    temporal = dict(zip(fe.TEMPORAL_COLS, engine.get_temporal_features(row)))
    return struct, temporal


def assume(ts, role, session, arn=USER, ptype="IAMUser", error=""):
    return event(ts, "AssumeRole", "sts.amazonaws.com", ptype, arn,
                 {"roleArn": f"arn:aws:iam::{ACCT}:role/{role}", "roleSessionName": session}, error)


def as_role(ts, role, session, name="GetSecretValue", source="secretsmanager.amazonaws.com"):
    return event(ts, name, source, "AssumedRole",
                 f"arn:aws:sts::{ACCT}:assumed-role/{role}/{session}", {"secretId": "db"})


class TestGroundTruthIsolation(unittest.TestCase):
    STREAM = [
        ("00:00", "AttachRolePolicy", "iam.amazonaws.com", "IAMUser", USER,
         {"roleName": "r", "policyArn": ADMIN}),
        ("00:05", "AssumeRole", "sts.amazonaws.com", "IAMUser", USER,
         {"roleArn": f"arn:aws:iam::{ACCT}:role/r", "roleSessionName": "s"}),
        ("00:09", "GetSecretValue", "secretsmanager.amazonaws.com", "AssumedRole",
         f"arn:aws:sts::{ACCT}:assumed-role/r/s", {"secretId": "db"}),
    ]
    TRUTH = {"label": "1", "session_label": "1", "attack_tactic": "privilege-escalation",
             "attack_technique_id": "T1098.003", "attack_technique": "privilege-escalation",
             "campaign_id": "C1", "campaign_family": "role_escalation", "chain_id": "C1-01",
             "stage_id": "2", "hop_id": "1", "parent_event_id": "e1", "root_event_id": "e1",
             "causal_relation": "grants_permission", "split": "test"}

    def _run(self, with_truth):
        engine, out = fe.FeatureEngineer(), []
        for ts, *rest in self.STREAM:
            extra = self.TRUTH if with_truth else {"label": "0"}
            struct, temporal = features(engine, event(ts, *rest, **extra))
            out.append((struct, temporal))
        return out

    def test_ground_truth_columns_do_not_change_any_feature(self):
        self.assertEqual(self._run(with_truth=False), self._run(with_truth=True))

    def test_ground_truth_list_covers_the_campaign_annotations(self):
        for col in ("campaign_id", "chain_id", "stage_id", "hop_id",
                    "parent_event_id", "root_event_id", "attack_technique_id", "attack_tactic"):
            self.assertIn(col, fe.GROUND_TRUTH_COLUMNS)

    def test_no_feature_is_named_after_ground_truth(self):
        names = set(fe.TEMPORAL_COLS) | set(fe.GRAPH_ATTR_FIELDS)
        self.assertFalse(names & set(fe.GROUND_TRUTH_COLUMNS))


class TestIdentityHandoff(unittest.TestCase):
    def test_assumed_session_is_linked_to_its_assumer(self):
        e = fe.FeatureEngineer()
        features(e, assume("00:00", "r", "s"))
        _, t = features(e, as_role("00:01", "r", "s"))
        self.assertEqual(t["principal_handoff"], 1)
        self.assertAlmostEqual(t["causal_depth_normalized"], 1 / 3)

    def test_link_requires_the_matching_session_name(self):
        e = fe.FeatureEngineer()
        features(e, assume("00:00", "r", "s"))
        _, t = features(e, as_role("00:01", "r", "other-session"))
        self.assertEqual(t["principal_handoff"], 0)

    def test_denied_assume_role_creates_no_link(self):
        e = fe.FeatureEngineer()
        features(e, assume("00:00", "r", "s", error="AccessDenied"))
        _, t = features(e, as_role("00:01", "r", "s"))
        self.assertEqual(t["principal_handoff"], 0)

    def test_role_chain_depth_accumulates(self):
        e = fe.FeatureEngineer()
        features(e, assume("00:00", "r1", "s1"))
        features(e, assume("00:01", "r2", "s2", arn=f"arn:aws:sts::{ACCT}:assumed-role/r1/s1",
                           ptype="AssumedRole"))
        _, t = features(e, as_role("00:02", "r2", "s2"))
        self.assertAlmostEqual(t["causal_depth_normalized"], 2 / 3)

    def test_issued_credentials_link_the_new_user_to_the_issuer(self):
        e = fe.FeatureEngineer()
        features(e, event("00:00", "CreateAccessKey", "iam.amazonaws.com", "IAMUser", USER,
                          {"userName": "bob"}))
        _, t = features(e, event("00:03", "ListBuckets", "s3.amazonaws.com", "IAMUser",
                                 f"arn:aws:iam::{ACCT}:user/bob"))
        self.assertEqual(t["principal_handoff"], 1)

    def test_a_user_acting_on_its_own_keys_is_not_a_handoff(self):
        e = fe.FeatureEngineer()
        features(e, event("00:00", "CreateAccessKey", "iam.amazonaws.com", "IAMUser", USER,
                          {"userName": "alice"}))
        _, t = features(e, event("00:03", "ListBuckets", "s3.amazonaws.com", "IAMUser", USER))
        self.assertEqual(t["principal_handoff"], 0)

    def test_lineage_steps_include_the_ancestors_and_expire(self):
        e = fe.FeatureEngineer()
        features(e, event("00:00", "AttachRolePolicy", "iam.amazonaws.com", "IAMUser", USER,
                          {"roleName": "r", "policyArn": ADMIN}))
        features(e, assume("00:05", "r", "s"))
        _, t = features(e, as_role("00:09", "r", "s"))
        self.assertAlmostEqual(t["lineage_enabling_steps_normalized"], 2 / 5)
        late = event("00:00", "GetSecretValue", "secretsmanager.amazonaws.com", "AssumedRole",
                     f"arn:aws:sts::{ACCT}:assumed-role/r/s", {"secretId": "db"})
        late["timestamp"] = "2024-01-01T13:00:00Z"   # three hours later
        _, t = features(e, late)
        self.assertEqual(t["lineage_enabling_steps_normalized"], 0.0)


class TestPermissionFeatures(unittest.TestCase):
    def _attach(self, policy, error=""):
        e = fe.FeatureEngineer()
        return features(e, event("00:00", "AttachRolePolicy", "iam.amazonaws.com", "IAMUser", USER,
                                 {"roleName": "r", "policyArn": policy}, error))[1]

    def test_admin_grant_is_a_full_expansion(self):
        t = self._attach(ADMIN)
        self.assertAlmostEqual(t["permission_expansion_score"], 1.0)
        self.assertEqual(t["privilege_delta"], 1.0)
        self.assertAlmostEqual(t["target_permission_coverage"], 1.0)

    def test_same_event_name_with_narrow_policy_scores_lower(self):
        admin, narrow = self._attach(ADMIN), self._attach(S3_RO)
        self.assertLess(narrow["permission_expansion_score"], admin["permission_expansion_score"])
        self.assertLess(narrow["new_permission_count_log"], admin["new_permission_count_log"])
        self.assertLess(narrow["privilege_delta"], admin["privilege_delta"])

    def test_denied_grant_changes_nothing(self):
        t = self._attach(ADMIN, error="AccessDenied")
        self.assertEqual(t["permission_expansion_score"], 0.0)
        self.assertEqual(t["target_permission_coverage"], 0.0)

    def test_grant_then_use_shows_on_the_role(self):
        e = fe.FeatureEngineer()
        features(e, event("00:00", "AttachRolePolicy", "iam.amazonaws.com", "IAMUser", USER,
                          {"roleName": "r", "policyArn": ADMIN}))
        features(e, assume("00:01", "r", "s"))
        _, t = features(e, as_role("00:02", "r", "s"))
        self.assertAlmostEqual(t["actor_permission_coverage"], 1.0)

    def test_non_permission_event_has_zero_permission_features(self):
        e = fe.FeatureEngineer()
        _, t = features(e, event("00:00", "ListBuckets", "s3.amazonaws.com", "IAMUser", USER))
        for col in ("new_permission_count_log", "permission_expansion_score", "privilege_delta"):
            self.assertEqual(t[col], 0.0, col)


class TestPrivilegeAndCriticality(unittest.TestCase):
    def test_privilege_level_uses_the_documented_access_level_scale(self):
        e = fe.FeatureEngineer()
        s, _ = features(e, event("00:00", "ListRoles", "iam.amazonaws.com", "IAMUser", USER))
        self.assertEqual(s["source_privilege_level"], 1)   # List (rank 0) + 1
        s, _ = features(e, event("00:01", "GetSecretValue", "secretsmanager.amazonaws.com",
                                 "IAMUser", USER, {"secretId": "db"}))
        self.assertEqual(s["source_privilege_level"], 2)   # Read (rank 1) + 1

    def test_root_is_known_to_hold_everything(self):
        e = fe.FeatureEngineer()
        s, _ = features(e, event("00:00", "ListBuckets", "s3.amazonaws.com", "Root",
                                 f"arn:aws:iam::{ACCT}:root"))
        self.assertEqual(s["source_privilege_level"], 5)

    def test_action_missing_from_aws_catalogue_is_unknown_not_guessed(self):
        # "ListBuckets" is an API name, not an IAM action (that is s3:ListAllMyBuckets).
        e = fe.FeatureEngineer()
        s, _ = features(e, event("00:00", "ListBuckets", "s3.amazonaws.com", "IAMUser", USER))
        self.assertEqual(s["source_privilege_level"], 0)

    def test_failed_call_does_not_demonstrate_privilege(self):
        e = fe.FeatureEngineer()
        s, _ = features(e, event("00:00", "AttachUserPolicy", "iam.amazonaws.com", "IAMUser", USER,
                                 {"userName": "u", "policyArn": ADMIN}, error="AccessDenied"))
        self.assertEqual(s["source_privilege_level"], 0)

    def test_criticality_comes_from_the_shared_service_table(self):
        for source, service in [("iam.amazonaws.com", "iam"), ("s3.amazonaws.com", "s3"),
                                ("cloudtrail.amazonaws.com", "cloudtrail")]:
            self.assertEqual(fe.get_resource_criticality(source, "x"), SERVICE_SENSITIVITY[service])

    def test_criticality_ignores_resource_names(self):
        # The old table scored "prod-admin-root" higher than "x" for the same service.
        self.assertEqual(fe.get_resource_criticality("s3.amazonaws.com", "prod-admin-root"),
                         fe.get_resource_criticality("s3.amazonaws.com", "x"))


class TestPriorFitting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.paths = dict(action_prior_path=os.path.join(self.tmp, "a.json"),
                          principal_prior_path=os.path.join(self.tmp, "p.json"))

    def test_only_training_rows_are_fitted(self):
        e = fe.FeatureEngineer(**self.paths)
        for split in ("train", "val", "test", ""):
            e.observe_label(event("00:00", "CreateUser", "iam.amazonaws.com", "IAMUser", USER,
                                  label="1", split=split))
        self.assertEqual(e.labels_observed, 2)      # "train" and "no split column"
        self.assertEqual(e.labels_withheld, 2)
        self.assertEqual(e.action_risk_prior.counts["CreateUser"], (2, 2))

    def test_split_value_is_normalized(self):
        row = event("00:00", "CreateUser", "iam.amazonaws.com", "IAMUser", USER, split=" TEST ")
        self.assertEqual(row["split"], "test")

    def test_frozen_engine_fits_nothing(self):
        e = fe.FeatureEngineer(**self.paths)
        e.observe_label(event("00:00", "CreateUser", "iam.amazonaws.com", "IAMUser", USER, label="1"))
        e.save_state()
        frozen = fe.FeatureEngineer(**self.paths, freeze_priors=True)
        frozen.observe_label(event("00:00", "CreateUser", "iam.amazonaws.com", "IAMUser", USER, label="0"))
        self.assertEqual(frozen.labels_observed, 0)

    def test_unseen_event_scores_the_training_base_rate(self):
        e = fe.FeatureEngineer(**self.paths)
        for label in ("1", "0", "0", "0"):
            e.observe_label(event("00:00", "ListBuckets", "s3.amazonaws.com", "IAMUser", USER, label=label))
        self.assertAlmostEqual(e.action_risk_prior.score("NeverSeen"), 0.25)
        self.assertAlmostEqual(e.principal_risk_prior.score("FederatedUser"), 0.25)

    def test_no_hand_assigned_per_event_scores_remain(self):
        e = fe.FeatureEngineer(**self.paths)
        self.assertEqual(e.action_risk_prior.priors, {})
        self.assertEqual(e.principal_risk_prior.priors, {})
        # Before any fitting every key scores the same: no event is presumed risky.
        self.assertEqual(e.action_risk_prior.score("StopLogging"), e.action_risk_prior.score("ListBuckets"))


class TestIdentityStatePersistence(unittest.TestCase):
    def test_state_survives_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "identity.json")
        e = fe.FeatureEngineer(identity_state_path=path)
        features(e, assume("00:00", "r", "s"))
        e.save_state()
        again = fe.FeatureEngineer(identity_state_path=path)
        _, t = features(again, as_role("00:01", "r", "s"))
        self.assertEqual(t["principal_handoff"], 1)


if __name__ == "__main__":
    unittest.main()
