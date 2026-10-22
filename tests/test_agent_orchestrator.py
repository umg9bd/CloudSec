"""
test_agent_orchestrator.py
==========================
Guards the agent gate's own rules (agent_orchestrator.py) -- the parts that decide whether an
AWS action may run, without loading the classifier or the HGT/LSTM checkpoints:

  * the security section of a prompt template is never taken from a user's edit;
  * an execution is exactly the approved plan: same calls, targets, parameters and principal,
    only fresh timestamps and IDs -- and execute() refuses anything else;
  * a request executes only when every stage approved it, and every refusal says why;
  * an executed plan lands in incoming/ as one complete CloudTrail file the pipeline can read.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import agent_orchestrator as ao
import feature_engine9 as fe9

PRINCIPAL = "arn:aws:iam::111122223333:user/agentic-operator"
PREDICTED = [
    {"eventVersion": "1.08", "eventSource": "s3.amazonaws.com", "eventName": "CreateBucket", "awsRegion": "us-east-1",
     "requestParameters": {"bucketName": "audit-logs-prod"}, "eventTime": "2026-10-18T12:00:00Z",
     "userIdentity": {"type": "IAMUser", "arn": PRINCIPAL}},
    {"eventVersion": "1.08", "eventSource": "s3.amazonaws.com", "eventName": "PutBucketEncryption", "awsRegion": "us-east-1",
     "requestParameters": {"bucketName": "audit-logs-prod"}, "eventTime": "2026-10-18T12:00:00Z",
     "userIdentity": {"type": "IAMUser", "arn": PRINCIPAL}},
]
S2_OK = {"status": "HARDENED", "revised_prompt": "x", "clarification_questions": []}
S3_OK = {"stage3_contract": {"status": "GENERATED"}, "downstream_lstm_records": {"Records": PREDICTED}}


def assessment(*risks, fast=()):
    return {"threshold": 0.1849, "events": [
        {"event_name": f"E{i}", "risk": r, "risk_score": round(r * 10, 2), "alert": r >= 0.1849,
         "fast_lane": i in fast, "fast_lane_reason": "CloudTrail logging disabled" if i in fast else None}
        for i, r in enumerate(risks)]}


class TemplateTests(unittest.TestCase):
    def test_security_section_is_split_off(self):
        tpl = "Create a bucket.\n\n--- SECURITY HARDENING (Stage 2, catalogue 0.2.3) ---\n1. Encrypt.\n--- END SECURITY HARDENING ---"
        request, block = ao.split_template(tpl)
        self.assertEqual(request, "Create a bucket.")
        self.assertTrue(block.startswith("--- SECURITY HARDENING") and block.endswith("--- END SECURITY HARDENING ---"))

    def test_plain_request_has_no_section(self):
        self.assertEqual(ao.split_template("  Create a bucket.  "), ("Create a bucket.", ""))


class ExecutionMatchesPlanTests(unittest.TestCase):
    def test_execution_keeps_content_and_renews_ids(self):
        plan = ao.materialize_events(PREDICTED, PRINCIPAL)
        run = ao.materialize_events(plan, PRINCIPAL)
        self.assertEqual(ao.plan_fingerprint(plan), ao.plan_fingerprint(run))
        for p, r in zip(plan, run):
            self.assertNotEqual(p["eventID"], r["eventID"])
            self.assertEqual((p["eventName"], p["requestParameters"]), (r["eventName"], r["requestParameters"]))
        self.assertEqual({r["userIdentity"]["arn"] for r in run}, {PRINCIPAL})
        self.assertLess(run[0]["eventTime"], "9999")      # stamped, in CloudTrail's format
        self.assertNotEqual(run[0]["eventTime"], PREDICTED[0]["eventTime"])

    def test_any_change_to_what_runs_changes_the_fingerprint(self):
        plan = ao.materialize_events(PREDICTED, PRINCIPAL)
        base = ao.plan_fingerprint(plan)
        for mutate in (lambda r: r.update(eventName="DeleteBucket"),
                       lambda r: r["requestParameters"].update(bucketName="someone-elses-bucket"),
                       lambda r: r.update(userIdentity={"arn": "arn:aws:iam::111122223333:user/admin"})):
            changed = json.loads(json.dumps(plan))
            mutate(changed[0])
            self.assertNotEqual(ao.plan_fingerprint(changed), base)

    def test_materialized_events_parse_as_cloudtrail(self):
        rows = [fe9.normalize_cloudtrail_row(r) for r in ao.materialize_events(PREDICTED, PRINCIPAL)]
        self.assertEqual([r["event_name"] for r in rows], ["CreateBucket", "PutBucketEncryption"])
        self.assertEqual({r["username"] for r in rows}, {"agentic-operator"})
        self.assertEqual({r["target_resource"] for r in rows}, {"arn:aws:s3:::audit-logs-prod"})


class BlockerTests(unittest.TestCase):
    def test_all_stages_passing_means_no_blockers(self):
        self.assertEqual(ao.plan_blockers(S2_OK, S3_OK, assessment(0.07, 0.08)), [])

    def test_each_refusal_says_why(self):
        cases = [
            (dict(S2_OK, status="SECURITY_CONFLICT"), S3_OK, assessment(0.05), "SECURITY_CONFLICT"),
            (dict(S2_OK, clarification_questions=[{"question": "Which VPC?", "blocking": True}]), S3_OK,
             assessment(0.05), "Which VPC?"),
            (S2_OK, {"stage3_contract": {"status": "NEEDS_CLARIFICATION"}, "downstream_lstm_records": {"Records": []}},
             assessment(0.05), "NEEDS_CLARIFICATION"),
            (S2_OK, S3_OK, assessment(0.05, 0.9), "at or above the alert threshold"),
            (S2_OK, S3_OK, assessment(0.05, 0.05, fast=(1,)), "fast-lane"),
            (S2_OK, S3_OK, {"events": [], "error": "RuntimeError: boom", "threshold": None}, "could not score"),
            (S2_OK, S3_OK, None, "has not scored"),
        ]
        for s2, s3, a, expect in cases:
            reasons = ao.plan_blockers(s2, s3, a)
            self.assertTrue(any(expect in r for r in reasons), (expect, reasons))

    def test_non_blocking_question_does_not_block(self):
        s2 = dict(S2_OK, clarification_questions=[{"question": "Which principals?", "blocking": False}])
        self.assertEqual(ao.plan_blockers(s2, S3_OK, assessment(0.05)), [])


class ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [mock.patch.object(ao, "PIPELINE_INCOMING", ao.Path(self.tmp) / "incoming"),
                        mock.patch.object(ao, "PIPELINE_STAGING", ao.Path(self.tmp) / ".staging"),
                        mock.patch.object(ao, "AUDIT_LOG", ao.Path(self.tmp) / "audit.jsonl")]
        for p in self.patches:
            p.start()
        self.gate = ao.AgentGate.__new__(ao.AgentGate)      # no classifier / models needed
        self.gate.principal_arn = PRINCIPAL

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def approved_round(self):
        plan = ao.materialize_events(PREDICTED, PRINCIPAL)
        return ao.Round(request="r", request_id="pig-test", approved_records=plan, fingerprint=ao.plan_fingerprint(plan))

    def test_approved_plan_lands_in_incoming_as_one_cloudtrail_file(self):
        ex = self.gate.execute(self.approved_round())
        path = ao.PIPELINE_INCOMING / ex["file"]
        rows = list(fe9.iter_input_rows(str(path)))
        self.assertEqual([r["event_name"] for r in rows], ["CreateBucket", "PutBucketEncryption"])
        self.assertEqual(os.listdir(ao.PIPELINE_STAGING), [])          # no half-written leftovers
        logged = [json.loads(l) for l in open(ao.AUDIT_LOG, encoding="utf-8")]
        self.assertEqual(logged[-1]["event"], "execute")

    def test_unapproved_round_cannot_execute(self):
        rnd = self.approved_round()
        rnd.blockers = ["Stage 4: CreateRole scored 9.5/10"]
        with self.assertRaises(PermissionError):
            self.gate.execute(rnd)
        self.assertFalse(ao.PIPELINE_INCOMING.exists() and os.listdir(ao.PIPELINE_INCOMING))

    def test_tampered_plan_cannot_execute(self):
        rnd = self.approved_round()
        rnd.approved_records[0]["requestParameters"]["bucketName"] = "someone-elses-bucket"
        with self.assertRaises(PermissionError):
            self.gate.execute(rnd)


if __name__ == "__main__":
    unittest.main()
