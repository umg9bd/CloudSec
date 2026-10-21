"""
test_security_hardening_workflow.py
===================================
End to end: request dict -> HardeningService -> response dict, plus the CLI.

Covers the behaviours the module is required to get right - an approved prompt missing requirements,
an in-scope service, identity/permissions, an application prompt, an already-adequate prompt, an
ambiguous request, a conflicting request, an out-of-scope service, an unavailable source, a missing or
unapproved screening result, and a changed catalogue version - and the invariants that must hold for
EVERY response: the original request is preserved byte for byte, no infrastructure or fact is invented,
no excluded service appears, sources are traceable, output is schema-valid and deterministic, and the
request path touches neither the network nor AWS.

Fixture prompts are test inputs for this module, not CloudTrail data.
"""

import contextlib
import dataclasses
import io
import json
import os
import re
import socket
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hardening_support as hs  # noqa: E402
from security_hardening import cli  # noqa: E402
from security_hardening import integration  # noqa: E402
from security_hardening import schema_validation as sv  # noqa: E402
from security_hardening.catalogue import CatalogueError  # noqa: E402
from security_hardening.config import load_policy  # noqa: E402
from security_hardening.integration import HardeningService, to_planner_payload  # noqa: E402
from security_hardening.models import HardeningResponse, Status  # noqa: E402
from security_hardening.prompt_hardener import split_revised_prompt  # noqa: E402

PROCEED = {"HARDENED", "ALREADY_ADEQUATE"}
BATTERY = [hs.P_S3_PHOTOS, hs.P_SECURITY_GROUP, hs.P_ROLE_BUCKET, hs.P_FLASK_SQL, hs.P_ADEQUATE, hs.P_AMBIGUOUS, hs.P_ADMIN,
           hs.P_SSH_WORLD, hs.P_OUT_OF_SCOPE, "Create an S3 bucket for invoices.", "Write a function that adds two numbers.",
           "Host a static website from an S3 bucket.", "Create an RDS database for the orders app and let the web tier connect.",
           "Write a script that stores the API token and uploads reports to the S3 bucket named acme-reports in account 123456789012."]
INVENTED = [r"arn:aws:\S+", r"\b\d{12}\b", r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b", r"\bsg-[0-9a-f]{6,}\b", r"\bAKIA[0-9A-Z]{16}\b", r"\bi-[0-9a-f]{8,}\b"]
GENERIC_TOKENS = {"0.0.0.0/0"}  # appears in a catalogue clause as an example of what must NOT be opened


def valid(test, response):
    test.assertEqual(sv.validate_response(response), [], response.get("status"))
    return response


class RequiredScenarioTests(unittest.TestCase):
    def test_1_approved_prompt_missing_requirements_is_hardened_with_traceable_sources(self):
        r = valid(self, hs.harden(hs.P_S3_PHOTOS))
        self.assertEqual(r["status"], "HARDENED")
        added = hs.ids(r["added_controls"])
        self.assertTrue({"S3-PUBLIC-001", "S3-ENCRYPT-002"} <= set(added), added)
        for control in r["added_controls"]:
            ref = control["source_references"][0]
            self.assertTrue(ref["source_name"] and ref["source_version_or_section"])
            self.assertRegex(ref["source_url"], r"^(https://|repo:)")
            self.assertIn(ref["provenance_status"], {"unverified", "section_checked", "snapshot_verified"})
            self.assertTrue(control["reason"].startswith("Applies because") and control["verification_expectation"])
        self.assertTrue(r["security_gaps"])
        self.assertTrue(all(g["control_id"] in added for g in r["security_gaps"]))
        block = split_revised_prompt(r["revised_prompt"])[1]
        for control in r["added_controls"]:
            self.assertIn(f"[{control['control_id']}]", block)
        self.assertIn("do not eliminate", block, "the block must not claim to eliminate misconfiguration")

    def test_2_prompt_about_an_in_scope_cloud_service(self):
        r = valid(self, hs.harden(hs.P_SECURITY_GROUP))
        self.assertEqual((r["status"], hs.ids(r["added_controls"])), ("HARDENED", ["VPC-SGROUP-001"]))
        self.assertEqual(r["scope_findings"]["approved_services"], ["ec2"])
        self.assertIn("security group", r["added_controls"][0]["clause"])
        self.assertEqual([q["control_id"] for q in r["clarification_questions"]], ["VPC-SGROUP-001"], "no invented CIDR: ask for it")
        self.assertTrue(r["requires_user_review"], "an unresolved source range on a high-severity control needs a reviewer")
        self.assertIn("Open items", r["revised_prompt"])

    def test_3_identity_and_permission_prompt(self):
        r = valid(self, hs.harden(hs.P_ROLE_BUCKET))
        self.assertEqual((r["status"], hs.ids(r["added_controls"])), ("HARDENED", ["IAM-LEASTPRIV-001", "IAM-TRUST-003"]))
        self.assertIn("IAM role", r["added_controls"][0]["clause"])
        self.assertIn("do not remove any permission the requested functionality actually needs", r["added_controls"][0]["clause"])
        self.assertNotIn("S3-ENCRYPT-002", hs.ids(r["added_controls"]), "granting access to an existing bucket is not creating one")
        self.assertNotIn("S3-PUBLIC-001", hs.ids(r["added_controls"]), "granting access to an existing bucket is not creating one")

    def test_4_application_prompt_missing_validation(self):
        r = valid(self, hs.harden(hs.P_FLASK_SQL))
        self.assertEqual(r["status"], "HARDENED")
        added = hs.ids(r["added_controls"])
        self.assertEqual(added[0], "APP-SQLI-001", "the mandatory control leads")
        self.assertEqual(r["added_controls"][0]["enforcement"], "mandatory")
        self.assertTrue({"APP-INPUT-003", "APP-AUTHZ-006"} <= set(added))
        self.assertLessEqual(len(added), load_policy().max_added_controls)
        self.assertEqual(hs.ids(r["deferred_controls"]), ["APP-RESOURCE-009", "APP-ERRORS-010"],
                         "controls over the cap are reported, never dropped silently")

    def test_5_prompt_already_adequate(self):
        r = valid(self, hs.harden(hs.P_ADEQUATE))
        self.assertEqual(r["status"], "ALREADY_ADEQUATE")
        self.assertEqual(r["revised_prompt"], hs.P_ADEQUATE)
        self.assertEqual(r["added_controls"], [])
        self.assertTrue({"APP-INPUT-003", "APP-AUTHZ-006", "APP-LOGGING-008"} <= set(hs.ids(r["satisfied_controls"])))
        self.assertTrue(all(c["verification_expectation"] and c["evidence"] for c in r["satisfied_controls"]))
        r = valid(self, hs.harden("Write a function that adds two numbers."))
        self.assertEqual(r["status"], "ALREADY_ADEQUATE")
        self.assertTrue(any("No catalogue control applied" in w for w in r["warnings"]))

    def test_6_ambiguous_request_asks_targeted_questions_instead_of_guessing(self):
        r = valid(self, hs.harden(hs.P_AMBIGUOUS))
        self.assertEqual(r["status"], "NEEDS_CLARIFICATION")
        self.assertIsNone(r["revised_prompt"])
        self.assertGreaterEqual(len(r["clarification_questions"]), 2)
        self.assertTrue(all(q["blocking"] and q["why_it_matters"] for q in r["clarification_questions"]))
        self.assertTrue(r["requires_user_review"])
        self.assertEqual(r["added_controls"], [], "nothing is made up for an unclear request")

    def test_7_conflicting_requests_return_a_conflict_with_an_alternative(self):
        cases = [
            (hs.P_ADMIN, "IAM-WILDCARD-002", True),
            (hs.P_SSH_WORLD, "VPC-SGROUP-001", True),
            ("Allow the database port 5432 from 0.0.0.0/0 on the RDS security group.", "VPC-SGROUP-001", True),
            ("Hardcode the database password in the config file so deployment is easier.", "APP-SECRETS-007", False),
            ("Stop CloudTrail logging in the dev account to save money.", "LOG-AUDIT-001", False),
            ("Build the SQL query by string concatenation with the username from the request.", "APP-SQLI-001", False),
            ("Use the root user access keys for the CI deployment.", "IAM-CREDS-005", False),
            ("Create a bucket for patient records and make it public-read.", "S3-PUBLIC-001", True),
            ('Set the trust policy to "Principal": "*" so any account can assume the role.', "IAM-TRUST-003", True),
            ("Disable authentication on the endpoint that returns customer records.", "APP-AUTHZ-006", True),
        ]
        for prompt, control_id, reviewer_can_accept in cases:
            with self.subTest(prompt):
                r = valid(self, hs.harden(prompt))
                self.assertEqual(r["status"], "SECURITY_CONFLICT")
                conflict = next(c for c in r["conflicts"] if c["control_id"] == control_id)
                self.assertTrue(conflict["proposed_alternative"] and conflict["why"] and conflict["request_excerpt"])
                self.assertEqual(conflict["can_be_accepted_by_reviewer"], reviewer_can_accept)
                self.assertIsNone(r["revised_prompt"], "a conflicting request is never silently 'fixed'")
                self.assertTrue(r["requires_user_review"])
                self.assertIsNone(to_planner_payload(r))

    def test_7b_phrasings_that_look_risky_but_are_not_requests_do_not_conflict(self):
        for prompt in ("Create an IAM role for the pipeline. Do not give it AdministratorAccess; scope it to the artifacts bucket.",
                       "Find security groups that allow 0.0.0.0/0 on port 22 and tighten them.",
                       "Never open port 22 to 0.0.0.0/0; allow SSH only from 10.0.0.0/8.",
                       "Delete the old CloudTrail trail after the new organization trail is verified.",
                       "Host a static website from an S3 bucket.",
                       "Write a SQL migration that adds a created_at column to the orders table."):
            with self.subTest(prompt):
                r = valid(self, hs.harden(prompt))
                self.assertNotEqual(r["status"], "SECURITY_CONFLICT", r.get("conflicts"))

    def test_7c_reviewer_acceptance_applies_only_to_recommended_controls(self):
        r = valid(self, hs.harden(hs.P_ADMIN, known_constraints={"accepted_conflicts": ["IAM-WILDCARD-002"]}))
        self.assertEqual(r["status"], "HARDENED")
        self.assertTrue(r["requires_user_review"])
        self.assertIn("IAM-WILDCARD-002", hs.ids(r["added_controls"]), "the compensating requirement stays")
        self.assertTrue(any("reviewer accepted" in a for a in r["assumptions"]))
        wild = next(c for c in r["added_controls"] if c["control_id"] == "IAM-WILDCARD-002")
        self.assertTrue(wild["source_references"][0]["derived"] and wild["source_references"][0]["supporting_references"],
                        "a derived recommendation keeps its supporting references")
        r = valid(self, hs.harden("Hardcode the database password in the config file.", known_constraints={"accepted_conflicts": ["APP-SECRETS-007"]}))
        self.assertEqual(r["status"], "SECURITY_CONFLICT", "a mandatory constraint cannot be waived by the request")

    def test_8_service_outside_the_approved_scope_is_reported_not_handled(self):
        for prompt, excluded, out in ((hs.P_OUT_OF_SCOPE, ["kms", "lambda"], ["dynamodb"]),
                                      ("Use AWS Lambda to resize images stored in S3.", ["lambda"], []),
                                      ("Create a DynamoDB table and an S3 bucket for sessions.", [], ["dynamodb"]),
                                      ("Send an SNS notification when the bucket changes.", [], ["sns"])):
            with self.subTest(prompt):
                r = valid(self, hs.harden(prompt))
                self.assertEqual(r["status"], "OUT_OF_SCOPE")
                self.assertEqual((r["scope_findings"]["excluded_services"], r["scope_findings"]["out_of_scope_services"]), (excluded, out))
                self.assertIsNone(r["revised_prompt"])
                self.assertEqual(r["added_controls"], [])
                self.assertTrue(r["clarification_questions"][0]["blocking"] and r["requires_user_review"])
        self.assertEqual(hs.harden("Create a DynamoDB table and an S3 bucket.")["scope_findings"]["approved_services"], ["s3"])

    def test_8b_warn_mode_hardens_the_in_scope_part_and_says_what_it_skipped(self):
        warn = dataclasses.replace(load_policy(), out_of_scope="warn")
        svc = HardeningService(hs.hardener_for(hs.catalogue_data(), policy=warn))
        r = valid(self, svc.handle(hs.request("Create a DynamoDB table and an S3 bucket for sessions.")))
        self.assertEqual(r["status"], "HARDENED")
        self.assertTrue(any("dynamodb" in w for w in r["warnings"]))
        self.assertNotIn("dynamodb", r["revised_prompt"].split("--- SECURITY HARDENING")[1].lower())

    def test_9_unavailable_source_is_reported_or_skipped_per_policy(self):
        data = hs.catalogue_data()
        hs.control_in(data, "S3-PUBLIC-001")["source_status"] = "unavailable"
        prompt = "Create an S3 bucket for invoices."
        r = valid(self, HardeningService(hs.hardener_for(data)).handle(hs.request(prompt)))
        self.assertIn("S3-PUBLIC-001", hs.ids(r["added_controls"]))
        self.assertTrue(any("S3-PUBLIC-001" in w and "unavailable" in w for w in r["warnings"]))
        self.assertTrue(r["requires_user_review"])
        self.assertEqual(r["catalogue_freshness"]["unavailable_sources"], ["S3-PUBLIC-001"])
        skip = dataclasses.replace(load_policy(), on_unavailable_source="skip")
        r = valid(self, HardeningService(hs.hardener_for(data, policy=skip)).handle(hs.request(prompt)))
        self.assertNotIn("S3-PUBLIC-001", hs.ids(r["added_controls"]))
        self.assertIn("S3-PUBLIC-001", hs.ids(r["deferred_controls"]))

    def test_9b_nothing_adequate_is_claimed_when_applicable_controls_were_skipped(self):
        data = hs.catalogue_data()
        for cid in ("S3-PUBLIC-001", "S3-ENCRYPT-002"):
            hs.control_in(data, cid)["source_status"] = "unavailable"
        skip = dataclasses.replace(load_policy(), on_unavailable_source="skip")
        r = valid(self, HardeningService(hs.hardener_for(data, policy=skip)).handle(hs.request("Create an S3 bucket for invoices.")))
        self.assertEqual(r["status"], "ALREADY_ADEQUATE")
        self.assertTrue(r["requires_user_review"], "adequacy cannot be confirmed when applicable controls were skipped")
        self.assertEqual(sorted(hs.ids(r["deferred_controls"])), ["S3-ENCRYPT-002", "S3-PUBLIC-001"])

    def test_10_missing_or_unapproved_screening_is_explicit_and_nothing_is_processed(self):
        analyze = mock.Mock(side_effect=AssertionError("the analyser must not run without approved screening"))
        with mock.patch("security_hardening.task_analyzer.TaskAnalyzer.analyze", analyze):
            cases = {"missing key": (None, "INTEGRATION_ERROR"), "empty": ("", "INTEGRATION_ERROR"), "blank": ("   ", "INTEGRATION_ERROR"),
                     "not a string": (5, "INTEGRATION_ERROR"), "PENDING": ("PENDING", "SCREENING_PENDING"),
                     "IN_PROGRESS": ("IN_PROGRESS", "SCREENING_PENDING"), "UNSAFE": ("UNSAFE", "SCREENING_NOT_APPROVED"),
                     "BLOCKED": ("BLOCKED", "SCREENING_NOT_APPROVED"), "lower-case safe": ("safe", "SCREENING_NOT_APPROVED"),
                     "padded SAFE": ("SAFE ", "SCREENING_NOT_APPROVED")}
            for name, (status, expected) in cases.items():
                with self.subTest(name):
                    req = hs.request("Create an S3 bucket for invoices.")
                    if name == "missing key":
                        req.pop("screening_status")
                    else:
                        req["screening_status"] = status
                    r = valid(self, hs.service().handle(req))
                    self.assertEqual(r["status"], expected)
                    self.assertIsNone(r["revised_prompt"])
                    self.assertIsNone(r["original_prompt"], "no content is echoed without approval")
                    self.assertEqual((r["added_controls"], r["security_gaps"]), ([], []))
                    self.assertTrue(r["requires_user_review"])
            missing = hs.service().handle({"request_id": "m", "original_prompt": "x"})
            self.assertEqual(missing["errors"][0]["code"], "SCREENING_STATUS_MISSING")
        analyze.assert_not_called()
        with mock.patch("security_hardening.task_analyzer.TaskAnalyzer.analyze", wraps=hs.service()._hardener.analyzer.analyze) as spy:
            hs.harden("Create an S3 bucket.")
            self.assertEqual(spy.call_count, 1)

    def test_11_a_new_catalogue_version_changes_the_applicable_controls_and_is_recorded(self):
        old_cat = hs.catalogue()
        data = hs.catalogue_data()
        hs.control_in(data, "S3-ENCRYPT-002")["active_status"] = "deprecated"
        hs.control_in(data, "APP-DEPS-012")["active_status"] = "deprecated"
        data["catalogue_version"] = "9.9.0"
        new_service = HardeningService(hs.hardener_for(data))
        bucket, deps = "Create an S3 bucket for invoices.", "Install the requests library and add it to requirements.txt."
        before, after = valid(self, hs.harden(bucket)), valid(self, new_service.handle(hs.request(bucket)))
        self.assertIn("S3-ENCRYPT-002", hs.ids(before["added_controls"]))
        self.assertNotIn("S3-ENCRYPT-002", hs.ids(after["added_controls"]))
        self.assertEqual((before["source_catalogue_version"], after["source_catalogue_version"]), (old_cat.version, "9.9.0"))
        self.assertNotEqual(before["catalogue_sha256"], after["catalogue_sha256"])
        self.assertIn("catalogue 9.9.0", after["revised_prompt"])
        before_deps = valid(self, hs.harden(deps))
        self.assertEqual((before_deps["status"], hs.ids(before_deps["added_controls"])), ("HARDENED", ["APP-DEPS-012"]))
        after_deps = valid(self, new_service.handle(hs.request(deps)))
        self.assertEqual(after_deps["status"], "ALREADY_ADEQUATE")
        self.assertNotIn("APP-DEPS-012", hs.ids(after_deps["added_controls"]))
        # a caller can pin the version it expects
        pinned = new_service.handle(hs.request(bucket, source_catalogue_version=old_cat.version))
        self.assertEqual((pinned["status"], pinned["errors"][0]["code"]), ("INTEGRATION_ERROR", "CATALOGUE_VERSION_UNAVAILABLE"))
        self.assertEqual(new_service.handle(hs.request(bucket, source_catalogue_version="9.9.0"))["status"], "HARDENED")


class InvariantTests(unittest.TestCase):
    def test_the_original_request_is_preserved_byte_for_byte(self):
        originals = ["Create an S3 bucket for invoices.", "Create an S3 bucket for invoices.\n", "Create an S3 bucket for café invoices ✓.\nKeep it tidy.\n\n",
                     "  Create an S3 bucket, with leading spaces and trailing  ", 'Create an S3 bucket and add this policy: {"Statement": []}\r\n']
        for prompt in originals + BATTERY:
            r = valid(self, hs.harden(prompt))
            with self.subTest(prompt[:50]):
                if r["status"] == "HARDENED":
                    self.assertEqual(split_revised_prompt(r["revised_prompt"])[0], prompt)
                    self.assertTrue(r["revised_prompt"].startswith(prompt))
                elif r["status"] == "ALREADY_ADEQUATE":
                    self.assertEqual(r["revised_prompt"], prompt)
                self.assertEqual(r["original_prompt"], prompt)
        self.assertEqual(hs.harden(originals[2])["status"], "HARDENED")

    def test_no_infrastructure_or_facts_are_invented(self):
        for prompt in BATTERY:
            r = hs.harden(prompt)
            if r["status"] != "HARDENED":
                continue
            block = r["revised_prompt"][len(prompt):]
            found = {t for pattern in INVENTED for t in re.findall(pattern, block)}
            self.assertEqual({t for t in found if t not in prompt and t not in GENERIC_TOKENS}, set(), prompt)
        r = hs.harden(BATTERY[-1])
        self.assertEqual(r["status"], "HARDENED")
        self.assertIn("acme-reports", r["revised_prompt"], "the user's own names stay in the preserved request")
        self.assertEqual(r["revised_prompt"].count("123456789012"), 1, "an account id appears only where the user wrote it")

    def test_no_excluded_service_appears_in_any_added_text(self):
        banned = re.compile(r"(?i)\bkms\b|secrets\s*manager|\blambda\b")
        for prompt in BATTERY:
            r = hs.harden(prompt)
            if r["status"] == "HARDENED":
                self.assertIsNone(banned.search(r["revised_prompt"][len(prompt):]), prompt)

    def test_every_response_records_the_catalogue_and_configuration_used(self):
        cat = hs.catalogue()
        for prompt in BATTERY:
            r = valid(self, hs.harden(prompt))
            self.assertEqual((r["source_catalogue_version"], r["catalogue_sha256"]), (cat.version, cat.sha256))
            self.assertRegex(r["catalogue_freshness"]["analysis_config_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(r["catalogue_freshness"]["catalogue_version"], cat.version)

    def test_output_is_deterministic(self):
        for prompt in BATTERY:
            self.assertEqual(hs.harden(prompt), hs.harden(prompt))
        self.assertEqual(hs.harden(hs.P_FLASK_SQL), HardeningService().handle(hs.request(hs.P_FLASK_SQL)))

    def test_revised_prompt_stays_concise(self):
        policy = load_policy()
        for prompt in BATTERY:
            r = hs.harden(prompt)
            if r["status"] == "HARDENED":
                self.assertLessEqual(len(r["added_controls"]), policy.max_added_controls)
                self.assertLess(len(r["revised_prompt"]) - len(prompt), 4500, prompt)

    def test_request_path_uses_neither_network_nor_aws(self):
        def refuse(*a, **k):
            raise AssertionError("network access attempted on the request path")
        env_before, modules_before = dict(os.environ), set(sys.modules)
        with mock.patch.object(socket.socket, "connect", refuse), mock.patch("socket.create_connection", refuse), \
                mock.patch("socket.getaddrinfo", refuse), mock.patch("urllib.request.urlopen", refuse):
            svc = HardeningService()
            for prompt in BATTERY:
                self.assertIn("status", svc.handle(hs.request(prompt)))
        self.assertEqual(dict(os.environ), env_before, "the module must not modify the environment")
        newly_imported = {m.split(".")[0] for m in set(sys.modules) - modules_before}
        self.assertFalse(newly_imported & {"boto3", "botocore", "requests", "urllib3", "torch", "neo4j"}, newly_imported)

    def test_catalogue_file_is_not_modified_by_hardening(self):
        from security_hardening.catalogue import CATALOGUE_PATH
        before = CATALOGUE_PATH.read_bytes()
        hs.harden(hs.P_FLASK_SQL)
        self.assertEqual(CATALOGUE_PATH.read_bytes(), before)


class AdapterTests(unittest.TestCase):
    def test_malformed_requests_become_integration_errors(self):
        svc = hs.service()
        for name, raw in {"none": None, "list": [], "string": "hello", "no request id": {"screening_status": "SAFE", "original_prompt": "x"},
                          "extra field": hs.request("x", surprise=1), "prompt too long": hs.request("x" * 20001),
                          "empty prompt": hs.request(""), "bad constraints": hs.request("x", known_constraints={"accepted_conflicts": "all"})}.items():
            with self.subTest(name):
                r = valid(self, svc.handle(raw))
                self.assertEqual(r["status"], "INTEGRATION_ERROR")
                self.assertEqual(r["errors"][0]["code"], "INVALID_REQUEST")
                self.assertIsNone(r["revised_prompt"])

    def test_internal_failure_fails_closed_without_leaking_the_prompt(self):
        svc = HardeningService()
        with mock.patch.object(svc._hardener, "harden", side_effect=RuntimeError("secret prompt text")):
            r = valid(self, svc.handle(hs.request("Create an S3 bucket for invoices.")))
        self.assertEqual((r["status"], r["errors"][0]["code"]), ("INTEGRATION_ERROR", "INTERNAL_ERROR"))
        self.assertNotIn("secret prompt text", json.dumps(r))
        self.assertIsNone(r["revised_prompt"])

    def test_a_response_that_violates_its_own_schema_is_not_released(self):
        svc = HardeningService()
        broken = HardeningResponse(request_id="x", status=Status.HARDENED, original_prompt="p", revised_prompt="p")  # HARDENED with no controls
        with mock.patch.object(svc._hardener, "harden", return_value=broken):
            r = valid(self, svc.handle(hs.request("Create an S3 bucket for invoices.")))
        self.assertEqual(r["errors"][0]["code"], "RESPONSE_SCHEMA_VIOLATION")
        self.assertIsNone(r["revised_prompt"])

    def test_an_unusable_catalogue_fails_closed(self):
        with mock.patch.object(integration, "SecurityHardener", side_effect=CatalogueError(["boom: bad control"])):
            svc = HardeningService()
        self.assertFalse(svc.ready)
        r = valid(self, svc.handle(hs.request("Create an S3 bucket for invoices.")))
        self.assertEqual((r["status"], r["errors"][0]["code"]), ("INTEGRATION_ERROR", "CATALOGUE_UNAVAILABLE"))
        self.assertIn("boom", json.dumps(r["errors"]))
        self.assertIsNone(r["revised_prompt"])
        self.assertEqual(svc.handle({"request_id": "m", "original_prompt": "x"})["errors"][0]["code"], "SCREENING_STATUS_MISSING")

    def test_planner_payload_only_exists_for_proceed_statuses(self):
        r = hs.harden(hs.P_FLASK_SQL)
        payload = to_planner_payload(r)
        self.assertEqual(payload["prompt"], r["revised_prompt"])
        self.assertEqual(payload["control_ids"], hs.ids(r["added_controls"]))
        self.assertEqual(len(payload["verification"]), len(r["added_controls"]))
        self.assertEqual((payload["catalogue_version"], payload["catalogue_sha256"]), (r["source_catalogue_version"], r["catalogue_sha256"]))
        adequate = to_planner_payload(hs.harden(hs.P_ADEQUATE))
        self.assertEqual(adequate["prompt"], hs.P_ADEQUATE)
        self.assertTrue(adequate["verification"] and adequate["control_ids"] == [])
        for prompt in (hs.P_AMBIGUOUS, hs.P_ADMIN, hs.P_OUT_OF_SCOPE):
            self.assertIsNone(to_planner_payload(hs.harden(prompt)))
        self.assertIsNone(to_planner_payload(hs.service().handle(hs.request("x", status="PENDING"))))
        self.assertIsNone(to_planner_payload({"status": "HARDENED", "revised_prompt": ""}))

    def test_module_level_helper_uses_a_default_service(self):
        self.assertEqual(integration.harden_request(hs.request("Create an S3 bucket for invoices."))["status"], "HARDENED")


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_harden_prompt_json_and_exit_codes(self):
        code, out = self.run_cli("harden", "--prompt", hs.P_ROLE_BUCKET, "--screening-status", "SAFE", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "HARDENED")
        code, out = self.run_cli("harden", "--prompt", hs.P_ADMIN, "--screening-status", "SAFE")
        self.assertEqual(code, 2)
        self.assertIn("CONFLICT IAM-WILDCARD-002", out)
        code, out = self.run_cli("harden", "--prompt", "Create an S3 bucket")
        self.assertEqual(code, 2)
        self.assertIn("SCREENING_STATUS_MISSING", out)

    def test_harden_file_input_jsonl_to_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in.jsonl"), os.path.join(tmp, "out.json")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(hs.request(hs.P_ROLE_BUCKET, rid="a")) + "\n" + json.dumps(hs.request(hs.P_AMBIGUOUS, rid="b")) + "\n")
            code, out = self.run_cli("harden", "--input", src, "--output", dst)
            self.assertEqual(code, 2, "one of the two needs attention")
            with open(dst, encoding="utf-8") as fh:
                results = json.load(fh)
            self.assertEqual([(r["request_id"], r["status"]) for r in results], [("a", "HARDENED"), ("b", "NEEDS_CLARIFICATION")])
            self.assertTrue(all(sv.validate_response(r) == [] for r in results))

    def test_validate_catalogue_and_explain(self):
        code, out = self.run_cli("validate-catalogue")
        self.assertEqual(code, 0)
        self.assertIn("valid:", out)
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.yaml")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("catalogue_id: x\n")
            code, out = self.run_cli("validate-catalogue", "--catalogue", bad)
            self.assertEqual(code, 2)
            self.assertIn("INVALID", out)
        code, out = self.run_cli("explain", "--prompt", hs.P_FLASK_SQL)
        self.assertEqual(code, 0)
        self.assertIn("untrusted_input_to_sql", out)


if __name__ == "__main__":
    unittest.main()
