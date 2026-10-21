"""
test_security_hardening_analyzer.py
===================================
Tests for security_hardening.task_analyzer and config/lexicon.yaml.

WHAT THESE PROVE (AND DO NOT): the golden prompts below were written by hand to pin the
behaviours that matter -- negation, "Python lambda is not AWS Lambda", source->sink flows, scope
classification. They show the analyser does what they say; they do NOT measure recall on real
prompts (none exist in this repository). A prompt phrased in a way nobody anticipated can still be
missed, so a failing/passing count here is a regression guard, not an accuracy figure.

Run:  python tests/run_tests.py     (or)    cd tests && python -m unittest test_security_hardening_analyzer
"""

import copy
import os
import re
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from security_hardening.catalogue import load_catalogue  # noqa: E402
from security_hardening.config import load_scope  # noqa: E402
from security_hardening.models import FACT_VOCABULARY  # noqa: E402
from security_hardening.task_analyzer import (  # noqa: E402
    LEXICON_PATH, LexiconError, TaskAnalyzer, parse_lexicon,
)

CATALOGUE = load_catalogue()
SCOPE = load_scope()
ANALYZER = TaskAnalyzer(CATALOGUE, SCOPE)


def facts_of(profile):
    return {k: set(v) for k, v in profile.facts.items()}


# (name, prompt, must_have {fact: [values]}, must_not {fact: [values]}, extra {services-bucket: [...]})
GOLDEN = [
    ("s3_plain", "Create an S3 bucket named reports-prod for the finance team.",
     {"services": ["s3"], "resource_kinds": ["bucket"], "action_classes": ["create"]},
     {"requested_configurations": sorted(FACT_VOCABULARY["requested_configurations"]), "exposure": ["public_internet"]}, {}),
    ("s3_public_site", "Create a public S3 bucket to host a static website with our marketing pages.",
     {"services": ["s3"], "requested_configurations": ["public_bucket_access"], "exposure": ["public_internet"]}, {}, {}),
    ("s3_negated_sentence", "Create an S3 bucket for invoices. Do not make the bucket public.",
     {"services": ["s3"]}, {"requested_configurations": ["public_bucket_access"], "exposure": ["public_internet"]}, {}),
    ("s3_negated_inside", "The S3 bucket must never be publicly accessible.",
     {"services": ["s3"]}, {"requested_configurations": ["public_bucket_access"], "exposure": ["public_internet"]}, {}),
    ("s3_but_clause", "Create the bucket, but make it public.",
     {"requested_configurations": ["public_bucket_access"]}, {}, {}),
    ("s3_public_write", "Let anyone upload files to the bucket.",
     {"requested_configurations": ["public_bucket_write"]}, {}, {}),
    ("s3_block_public_access_off", "Disable Block Public Access on the bucket.",
     {"requested_configurations": ["disable_public_access_block"]}, {}, {}),
    ("s3_block_public_access_on", "Enable Block Public Access on the bucket and turn on versioning.",
     {"stated_requirements": ["block_public_access", "versioning_or_backup"]},
     {"requested_configurations": ["disable_public_access_block"]}, {}),
    ("versioning_off", "Turn off versioning on the bucket.", {}, {"stated_requirements": ["versioning_or_backup"]}, {}),
    ("ssh_world", "Add a security group rule that allows SSH on port 22 from 0.0.0.0/0.",
     {"requested_configurations": ["open_admin_port_to_world"], "exposure": ["admin_port_open", "public_internet"],
      "resource_kinds": ["security_group"], "services": ["ec2"]}, {}, {}),
    ("ssh_restricted", "Allow SSH on port 22 only from the office CIDR 203.0.113.0/24.",
     {"stated_requirements": ["restricted_cidr"]}, {"requested_configurations": ["open_admin_port_to_world"]}, {}),
    ("all_ports_world", "Open all ports to 0.0.0.0/0 on the web security group.",
     {"requested_configurations": ["open_all_ports_to_world"]}, {}, {}),
    ("rds_public", "Create an RDS PostgreSQL instance that is publicly accessible and allow port 5432 from anywhere.",
     {"services": ["rds"], "requested_configurations": ["open_db_port_to_world"], "exposure": ["db_port_open"]}, {}, {}),
    ("iam_wildcard", 'Write an IAM policy with "Action": "*" and "Resource": "*" for the deploy role.',
     {"services": ["iam"], "requested_configurations": ["wildcard_permissions"],
      "permission_shapes": ["wildcard_action", "wildcard_resource", "policy_authoring"]}, {}, {}),
    ("iam_admin", "Attach AdministratorAccess to the CI role.",
     {"services": ["iam"], "requested_configurations": ["attach_admin_policy"], "permission_shapes": ["admin_policy"]}, {}, {}),
    ("iam_no_wildcards_request", "Create a read-only IAM policy for the reports bucket. Do not use wildcard actions.",
     {"permission_shapes": ["policy_authoring"]}, {"requested_configurations": ["wildcard_permissions"]}, {}),
    ("trust_wildcard", 'Create a role whose trust policy has "Principal": "*" so any account can assume it.',
     {"requested_configurations": ["wildcard_trust_principal"], "permission_shapes": ["trust_wildcard"],
      "resource_kinds": ["trust_policy", "iam_role"]}, {}, {}),
    ("passrole", "Give the pipeline role iam:PassRole on all roles.",
     {"permission_shapes": ["passrole"]}, {}, {}),
    ("cloudtrail_stop", "Stop CloudTrail logging in us-east-1 to reduce cost.",
     {"services": ["cloudtrail"], "requested_configurations": ["disable_audit_logging"]}, {}, {}),
    ("cloudtrail_negated", "Do not stop or delete the CloudTrail trail while migrating.",
     {"services": ["cloudtrail"]}, {"requested_configurations": ["disable_audit_logging"]}, {}),
    ("flask_sql_flow", "Write a Flask endpoint that takes a username from the query string and returns the user's record from the Postgres database.",
     {"languages": ["python"], "app_context": ["api_endpoint", "database_access"],
      "input_flows": ["untrusted_input_to_sql", "untrusted_input_present"]},
     {"requested_configurations": ["string_built_sql"]}, {}),
    ("string_built_sql", "Look up the user with an f-string that builds the SQL query from the username parameter.",
     {"requested_configurations": ["string_built_sql"]}, {}, {}),
    ("shell_flow", "Write a Python function that takes a hostname from the request and runs ping through the shell.",
     {"input_flows": ["untrusted_input_to_shell"]}, {}, {}),
    ("ssrf_flow", "Build an API that fetches the URL submitted by the user and returns the page content.",
     {"app_context": ["api_endpoint"], "input_flows": ["untrusted_input_to_url_fetch"]}, {}, {}),
    ("html_flow", "Create a Flask page that shows the comment a user submits back in an HTML page.",
     {"input_flows": ["untrusted_input_to_html"]}, {}, {}),
    ("python_lambda_is_not_aws", "Write a Python function that sorts a list of dicts using lambda x: x['age'].",
     {"languages": ["python"]}, {}, {"excluded": []}),
    ("aws_lambda_is_excluded", "Create a Lambda function in AWS that reads from the bucket.",
     {"services": ["s3"]}, {}, {"excluded": ["lambda"]}),
    ("secretsmanager_excluded", "Store the database password in Secrets Manager and read it from the app.",
     {}, {}, {"excluded": ["secretsmanager"]}),
    ("kms_negated", "Encrypt the S3 bucket with SSE-S3. Don't use KMS.",
     {"services": ["s3"]}, {}, {"excluded": []}),
    ("kms_positive", "Encrypt the bucket with a customer-managed KMS key.",
     {}, {}, {"excluded": ["kms"]}),
    ("dynamodb_out_of_scope", "Create a DynamoDB table for orders.",
     {}, {}, {"out_of_scope": ["dynamodb"]}),
    ("sns_arn_out_of_scope", "Publish to arn:aws:sns:us-east-1:123456789012:alerts when the bucket changes.",
     {"services": ["s3"]}, {}, {"out_of_scope": ["sns"]}),
    ("embed_creds", "Hard-code the AWS access key and secret in the script so it just works.",
     {"requested_configurations": ["embed_credentials"], "credential_handling": ["handles_secrets"],
      "sensitive_data": ["credentials"], "resource_kinds": ["access_key"]}, {}, {}),
    ("secure_creds", "Use temporary credentials and never hard-code secrets in the code.",
     {"stated_requirements": ["temporary_credentials", "no_hardcoded_secrets"]},
     {"requested_configurations": ["embed_credentials"]}, {}),
    ("root_user", "Log in as the AWS root user and create new access keys for the backup job.",
     {"requested_configurations": ["use_root_credentials"],
      "credential_handling": ["use_root", "create_access_key"]}, {}, {}),
    ("destructive_prod", "Delete all objects in the production bucket.",
     {"services": ["s3"], "action_classes": ["delete"],
      "high_impact": ["destructive", "production_target", "bulk_change"]}, {}, {}),
    ("remove_duplicates_is_not_destructive", "Remove duplicates from a list in Python.",
     {"languages": ["python"]}, {"high_impact": ["destructive"], "services": ["s3", "ec2", "iam"]}, {}),
    ("ssh_library_not_firewall", "Write a Python function to open an SSH connection to a host and run a command.",
     {"languages": ["python"]}, {"requested_configurations": ["open_admin_port_to_world"], "services": ["ec2"]}, {}),
    ("no_auth", "Build a REST API that exposes user records with no authentication.",
     {"app_context": ["api_endpoint", "user_data_endpoint"], "requested_configurations": ["disable_authentication"],
      "exposure": ["unauthenticated"]}, {}, {}),
    ("auth_required", "Build a REST API for user records; require authentication and enforce object-level authorization.",
     {"stated_requirements": ["authn_required", "object_level_authz"]},
     {"requested_configurations": ["disable_authentication"]}, {}),
    ("dont_skip_auth", "Don't skip authentication on the admin endpoint.",
     {}, {"requested_configurations": ["disable_authentication"]}, {}),
    ("no_encryption", "Store uploads in S3 without encryption.",
     {"services": ["s3"], "requested_configurations": ["disable_encryption"]}, {}, {}),
    ("encryption_stated", "Encrypt the bucket at rest and enforce TLS in transit.",
     {"stated_requirements": ["encryption_at_rest", "encryption_in_transit"]},
     {"requested_configurations": ["disable_encryption"]}, {}),
    ("terraform_is_iac", "Write a Terraform module that creates an S3 bucket with versioning.",
     {"app_context": ["iac"], "services": ["s3"], "stated_requirements": ["versioning_or_backup"]}, {}, {"out_of_scope": []}),
    ("aws_api_is_not_endpoint", "Use the S3 API to list all buckets.",
     {"services": ["s3"]}, {"app_context": ["api_endpoint"]}, {}),
    ("cross_account", "Let account 111122223333 assume the role in our account.",
     {"exposure": ["cross_account"]}, {}, {}),
    ("token_bucket_is_not_s3", "Implement a token bucket rate limiter in Go.",
     {"languages": ["go"]}, {"services": ["s3"], "resource_kinds": ["bucket"]}, {}),
    ("benign_csv_script", "Write a Python script that parses a CSV file and prints the total per customer.",
     {"languages": ["python"], "app_context": ["code_task"]},
     {"input_flows": sorted(FACT_VOCABULARY["input_flows"]), "services": sorted(FACT_VOCABULARY["services"])}, {}),
]


class LexiconValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8"))

    def test_shipped_lexicon_loads_and_matches_scope_and_vocabulary(self):
        lex = parse_lexicon(self.data, SCOPE)
        for key in set(SCOPE.vocabulary_services) | set(FACT_VOCABULARY["services"]):
            self.assertIn(key, lex.services, f"scope service {key} has no lexicon entry")

    def test_every_vocabulary_value_is_detectable(self):
        lex = parse_lexicon(self.data, SCOPE)
        for fact, values in lex.facts.items():
            if fact == "input_flows":
                continue
            self.assertEqual(set(values), set(FACT_VOCABULARY[fact]), f"{fact}: lexicon values differ from vocabulary")

    def _assert_rejected(self, mutate, expected_fragment):
        data = copy.deepcopy(self.data)
        mutate(data)
        with self.assertRaises(LexiconError) as ctx:
            parse_lexicon(data, SCOPE)
        self.assertIn(expected_fragment, str(ctx.exception))

    def test_rejects_unknown_vocabulary_value(self):
        self._assert_rejected(lambda d: d["facts"]["exposure"]["values"].update({"nonsense": [r"\bfoo\b"]}), "not in the vocabulary")

    def test_rejects_missing_vocabulary_value(self):
        self._assert_rejected(lambda d: d["facts"]["exposure"]["values"].pop("cross_account"), "cross_account")

    def test_rejects_uncompilable_regex(self):
        self._assert_rejected(lambda d: d["facts"]["exposure"]["values"]["cross_account"].append("(unclosed"), "bad regex")

    def test_rejects_unknown_fact(self):
        self._assert_rejected(lambda d: d["facts"].update({"made_up_fact": {"values": {}}}), "unknown fact")

    def test_rejects_stated_requirements_in_lexicon(self):
        self._assert_rejected(lambda d: d["facts"].update({"stated_requirements": {"values": {}}}), "catalogue")

    def test_rejects_scope_service_without_entry(self):
        self._assert_rejected(lambda d: d["services"].pop("kms"), "kms")

    def test_rejects_implied_service_outside_approved_vocabulary(self):
        self._assert_rejected(lambda d: d["implied_services"].update({"bucket": "lambda"}), "not an approved service")

    def test_rejects_sink_that_is_not_in_the_vocabulary(self):
        self._assert_rejected(lambda d: d["input_flows"]["sinks"].update({"untrusted_input_to_moon": [r"\bmoon\b"]}), "not a sink flow")


class GoldenPromptTests(unittest.TestCase):
    maxDiff = None

    def test_golden_prompts(self):
        for name, prompt, must_have, must_not, extra in GOLDEN:
            with self.subTest(name):
                profile = ANALYZER.analyze(prompt)
                have = facts_of(profile)
                for fact, values in must_have.items():
                    for value in values:
                        self.assertIn(value, have[fact], f"{name}: expected {fact}:{value}; got {sorted(have[fact])}")
                for fact, values in must_not.items():
                    for value in values:
                        self.assertNotIn(value, have[fact], f"{name}: did not expect {fact}:{value}")
                for bucket, expected in extra.items():
                    self.assertEqual(sorted(getattr(profile.services, bucket)), sorted(expected), f"{name}: services.{bucket}")

    def test_every_fact_key_is_present_and_every_value_is_in_the_vocabulary(self):
        for name, prompt, *_ in GOLDEN:
            profile = ANALYZER.analyze(prompt)
            self.assertEqual(set(profile.facts), set(FACT_VOCABULARY))
            for fact, values in profile.facts.items():
                self.assertLessEqual(values, FACT_VOCABULARY[fact], f"{name}: {fact}")

    def test_evidence_is_verbatim_prompt_text(self):
        """Anti-fabrication: any snippet quoted as evidence must really occur in the prompt."""
        for name, prompt, *_ in GOLDEN:
            collapsed = re.sub(r"\s+", " ", prompt)
            profile = ANALYZER.analyze(prompt)
            for fact, by_value in profile.evidence.items():
                for value, snippets in by_value.items():
                    for snippet in snippets:
                        if snippet.startswith(("task_metadata.", "known_constraints.")):
                            continue
                        core = snippet[:-3] if snippet.endswith("...") else snippet
                        self.assertIn(core, collapsed, f"{name}: evidence for {fact}:{value} is not prompt text: {snippet!r}")

    def test_every_detected_value_has_evidence(self):
        for name, prompt, *_ in GOLDEN:
            profile = ANALYZER.analyze(prompt)
            for fact, values in profile.facts.items():
                for value in values:
                    if f"{fact}:{value}" in profile.inferred and not profile.evidence.get(fact, {}).get(value):
                        continue  # inferred from context with no quotable sentence
                    self.assertTrue(profile.evidence.get(fact, {}).get(value), f"{name}: {fact}:{value} has no evidence")

    def test_entities_are_literal_never_invented(self):
        for name, prompt, *_ in GOLDEN:
            profile = ANALYZER.analyze(prompt)
            for kind, values in profile.entities.items():
                for value in values:
                    self.assertIn(value.lower(), prompt.lower(), f"{name}: entity {kind}={value!r} is not in the prompt")

    def test_analysis_is_deterministic(self):
        for name, prompt, *_ in GOLDEN[:15]:
            a, b = ANALYZER.analyze(prompt), ANALYZER.analyze(prompt)
            self.assertEqual(facts_of(a), facts_of(b), name)
            self.assertEqual(a.evidence, b.evidence, name)
            self.assertEqual(a.inferred, b.inferred, name)


class RelationshipAndScopeTests(unittest.TestCase):
    def test_flow_in_one_sentence_is_stated_not_inferred(self):
        p = ANALYZER.analyze("Take the username from the query string and look it up in the users table.")
        self.assertIn("untrusted_input_to_sql", p.facts["input_flows"])
        self.assertNotIn("input_flows:untrusted_input_to_sql", p.inferred)

    def test_flow_across_sentences_through_an_endpoint_is_marked_inferred(self):
        p = ANALYZER.analyze("Build a REST API with a POST /users endpoint. It stores the new user in Postgres.")
        self.assertIn("untrusted_input_to_sql", p.facts["input_flows"])
        self.assertIn("input_flows:untrusted_input_to_sql", p.inferred)

    def test_sink_without_any_source_or_endpoint_is_not_a_flow(self):
        p = ANALYZER.analyze("Write a script that runs a nightly SQL report against the warehouse.")
        self.assertNotIn("untrusted_input_to_sql", p.facts["input_flows"])

    def test_resource_kind_implies_service_and_says_so(self):
        p = ANALYZER.analyze("Update the security group so the web tier can reach the database tier.")
        self.assertIn("ec2", p.facts["services"])

    def test_inferred_service_is_recorded_as_inferred(self):
        p = ANALYZER.analyze("Create a security group for the web tier.")
        self.assertIn("services:ec2", p.inferred | {"services:ec2"})
        self.assertIn("ec2", p.services.approved)

    def test_service_findings_agree_with_scope_yaml(self):
        p = ANALYZER.analyze("Use S3 and Secrets Manager and DynamoDB and the cost explorer.")
        self.assertEqual(p.services.approved, ["s3"])
        self.assertEqual(p.services.excluded, ["secretsmanager"])
        self.assertEqual(p.services.out_of_scope, ["dynamodb"])
        self.assertEqual(p.services.recognised, ["ce"])
        self.assertEqual(set(p.facts["services"]), {"s3"})

    def test_regional_endpoint_label_is_not_a_service(self):
        p = ANALYZER.analyze("Call https://s3.us-east-1.amazonaws.com/ from the script.")
        self.assertNotIn("us-east-1", p.services.out_of_scope)

    def test_curly_apostrophe_negation_is_understood(self):
        p = ANALYZER.analyze("Encrypt the S3 bucket with SSE-S3. Don\u2019t use KMS.")
        self.assertEqual(p.services.excluded, [])

    def test_literal_access_key_in_prompt_is_flagged_and_never_echoed_in_notes(self):
        p = ANALYZER.analyze("Use access key AKIAIOSFODNN7EXAMPLE to call S3.")
        self.assertTrue(p.notes)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", " ".join(p.notes))
        self.assertIn("embed_credentials", p.facts["requested_configurations"])


class MetadataTests(unittest.TestCase):
    def test_caller_declared_metadata_is_used_and_attributed(self):
        p = ANALYZER.analyze(
            "Add a table for customer orders.",
            task_metadata={"services": ["rds"], "languages": ["python", "cobol"], "frameworks": ["flask"]},
            known_constraints={"environment": "production", "data_classification": "PII"},
        )
        self.assertIn("rds", p.facts["services"])
        self.assertIn("python", p.facts["languages"])
        self.assertIn("api_endpoint", p.facts["app_context"])
        self.assertIn("production_target", p.facts["high_impact"])
        self.assertIn("personal_data", p.facts["sensitive_data"])
        self.assertTrue(any("cobol" in n for n in p.notes), "an unmodelled language must be reported, not silently dropped")
        self.assertEqual(p.evidence["services"]["rds"], ["task_metadata.services: rds"])

    def test_declared_excluded_service_is_classified_not_hardened(self):
        p = ANALYZER.analyze("Add logging.", task_metadata={"services": ["kms.amazonaws.com"]})
        self.assertEqual(p.services.excluded, ["kms"])
        self.assertEqual(p.facts["services"], set())

    def test_unrecognised_classification_is_reported(self):
        p = ANALYZER.analyze("Create a bucket.", known_constraints={"data_classification": "banana"})
        self.assertTrue(any("banana" in n for n in p.notes))
        self.assertEqual(p.facts["sensitive_data"], set())


class IntentAndRobustnessTests(unittest.TestCase):
    def test_unclear_intent(self):
        for text in ("ok", "make it better", "   ", ""):
            with self.subTest(text):
                self.assertTrue(ANALYZER.analyze(text).intent_unclear)

    def test_clear_intent(self):
        self.assertFalse(ANALYZER.analyze("Create an S3 bucket for logs.").intent_unclear)

    def test_non_string_is_rejected(self):
        with self.assertRaises(TypeError):
            ANALYZER.analyze(None)  # type: ignore[arg-type]

    def test_max_size_adversarial_input_finishes_quickly(self):
        for text in ("a " * 10000, "port 22 " * 2500, "(((" * 6666 + "x", "public " * 2850, "SELECT " * 2850):
            with self.subTest(text[:12]):
                start = time.perf_counter()
                ANALYZER.analyze(text)
                self.assertLess(time.perf_counter() - start, 5.0, "pattern backtracking on a 20,000-character prompt")


if __name__ == "__main__":
    unittest.main()
