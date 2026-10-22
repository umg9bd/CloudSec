"""
test_security_hardening_contracts.py
====================================
The module's contracts: JSON Schemas and their validator, policy/scope configuration, the
condition language, and the control catalogue (structure, semantic checks, provenance).

WHY THIS EXISTS: the catalogue is data that steers security advice, so every way it can be wrong
(typo'd tag, unknown fact, a control that touches an excluded service, a "verified" control with no
hash...) must be rejected at load time rather than silently weakening a prompt. Each case below is a
defect that was injected on purpose and must be refused.
"""

import ast
import copy
import hashlib
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import hardening_support as hs  # noqa: E402
from security_hardening import catalogue as C  # noqa: E402
from security_hardening import conditions, config, models, provenance  # noqa: E402
from security_hardening import schema_validation as sv  # noqa: E402

try:
    import jsonschema
except ImportError:  # the library is NOT a dependency; the cross-check is skipped without it
    jsonschema = None

GOOD_REQUEST = {"request_id": "r1", "screening_status": "SAFE", "original_prompt": "Create an S3 bucket"}


def _response(status, revised="text", **kw):
    return models.HardeningResponse(request_id="r1", status=status, original_prompt="p", revised_prompt=revised, **kw).to_dict()


class SchemaValidatorTests(unittest.TestCase):
    def test_valid_request_passes(self):
        self.assertEqual(sv.validate_request(GOOD_REQUEST), [])
        full = dict(GOOD_REQUEST, conversation_context={"a": 1}, task_metadata={"languages": ["python"]},
                    known_constraints={"environment": "production", "accepted_conflicts": ["IAM-WILDCARD-002"]},
                    source_catalogue_version=None)
        self.assertEqual(sv.validate_request(full), [])

    def test_request_rejects_bad_shapes(self):
        for name, doc in {
            "empty": {}, "empty id": dict(GOOD_REQUEST, request_id=""), "numeric prompt": dict(GOOD_REQUEST, original_prompt=5),
            "empty prompt": dict(GOOD_REQUEST, original_prompt=""), "unknown field": dict(GOOD_REQUEST, extra=1),
            "bad constraint type": dict(GOOD_REQUEST, known_constraints={"accepted_conflicts": "IAM-WILDCARD-002"}),
            "prompt too long": dict(GOOD_REQUEST, original_prompt="x" * 20001),
        }.items():
            with self.subTest(name):
                self.assertTrue(sv.validate_request(doc))

    def test_response_status_invariants_are_enforced_by_the_schema(self):
        S = models.Status
        self.assertEqual(sv.validate_response(_response(S.ALREADY_ADEQUATE)), [])
        cases = {
            "HARDENED without added controls": _response(S.HARDENED),
            "OUT_OF_SCOPE must not carry a revised prompt": _response(S.OUT_OF_SCOPE, revised="leak", requires_user_review=True),
            "SECURITY_CONFLICT needs a conflict": _response(S.SECURITY_CONFLICT, revised=None, requires_user_review=True),
            "NEEDS_CLARIFICATION needs a question": _response(S.NEEDS_CLARIFICATION, revised=None, requires_user_review=True),
            "INTEGRATION_ERROR needs an error": _response(S.INTEGRATION_ERROR, revised=None),
            "ALREADY_ADEQUATE needs a prompt": _response(S.ALREADY_ADEQUATE, revised=None),
        }
        for name, doc in cases.items():
            with self.subTest(name):
                self.assertTrue(sv.validate_response(doc), name)
        bad = _response(S.ALREADY_ADEQUATE)
        bad["status"] = "MAYBE"
        self.assertTrue(sv.validate_response(bad))

    def test_source_reference_must_be_https_or_repo(self):
        ref = {"source_name": "n", "source_url": "http://insecure", "source_version_or_section": "s", "retrieved_at": None,
               "content_hash": None, "provenance_status": "unverified", "derived": False}
        ctl = {"control_id": "A-B-001", "reason": "r", "source_references": [ref], "verification_expectation": "v",
               "clause": "c", "enforcement": "recommended", "severity": "high"}
        doc = _response(models.Status.HARDENED, added_controls=[ctl])
        self.assertTrue(sv.validate_response(doc))
        shown = doc["added_controls"][0]["source_references"][0]       # to_dict() copied the structure
        shown["source_url"] = "https://docs.aws.amazon.com/x"
        self.assertEqual(sv.validate_response(doc), [])
        shown["source_url"] = "repo:feature_engine9.py#CRITICAL_ACTIONS"
        self.assertEqual(sv.validate_response(doc), [])

    def test_unsupported_keyword_raises_instead_of_being_ignored(self):
        with self.assertRaises(sv.SchemaError):
            sv.validate("x", {"type": "string", "format": "email"})
        with self.assertRaises(sv.SchemaError):
            sv.validate([1], {"type": "array", "uniqueItems": True})

    def test_bundled_schemas_use_only_supported_keywords(self):
        for name in ("hardening_request", "hardening_response", "control_catalogue"):
            with self.subTest(name):
                sv.validate({}, sv.load_schema(name))  # raises SchemaError on an unsupported keyword

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed (it is not a project dependency)")
    def test_agrees_with_the_jsonschema_library(self):
        S = models.Status
        corpus = [("hardening_request", d) for d in (
            GOOD_REQUEST, {}, dict(GOOD_REQUEST, screening_status=3), dict(GOOD_REQUEST, original_prompt=""),
            dict(GOOD_REQUEST, known_constraints={"environment": "prod"}), dict(GOOD_REQUEST, extra=1))]
        for status, revised in [("ALREADY_ADEQUATE", "p"), ("ALREADY_ADEQUATE", None), ("OUT_OF_SCOPE", None),
                                ("OUT_OF_SCOPE", "x"), ("SCREENING_PENDING", None), ("BOGUS", None)]:
            doc = _response(S.ALREADY_ADEQUATE)
            doc["status"], doc["revised_prompt"] = status, revised
            corpus.append(("hardening_response", doc))
        for name in ("hardening_request", "hardening_response", "control_catalogue"):
            jsonschema.Draft202012Validator.check_schema(sv.load_schema(name))
        for name, doc in corpus:
            schema = sv.load_schema(name)
            self.assertEqual(not sv.validate(doc, schema), jsonschema.Draft202012Validator(schema).is_valid(doc), (name, doc))


class ConfigTests(unittest.TestCase):
    def _write(self, text):
        fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_policy_defaults_and_screening_dispositions(self):
        pol = config.load_policy()
        self.assertEqual(pol.approved_screening_statuses, frozenset({"SAFE"}))
        got = [pol.screening_disposition(s) for s in ("SAFE", "PENDING", "UNSAFE", "safe", None, "", "  ", 5)]
        self.assertEqual(got, ["approved", "pending", "not_approved", "not_approved", "missing", "missing", "missing", "missing"])

    def test_policy_rejects_inconsistent_configuration(self):
        base = "screening:\n  approved_statuses: [SAFE]\n  pending_statuses: [PENDING]\n"
        for name, text in {
            "status both approved and pending": "screening:\n  approved_statuses: [SAFE]\n  pending_statuses: [SAFE]\n",
            "no approved statuses": "screening:\n  approved_statuses: []\n",
            "bad out_of_scope": base + "behaviour:\n  out_of_scope: ignore\n",
            "bad source policy": base + "behaviour:\n  on_unavailable_source: shrug\n",
            "non-positive limit": base + "limits:\n  max_added_controls: 0\n",
            "bad override": base + "enforcement_overrides:\n  X-Y-001: optional\n",
        }.items():
            with self.subTest(name), self.assertRaises(config.ConfigError):
                config.load_policy(self._write(text))

    def test_scope_classification(self):
        scope = config.load_scope()
        got = {s: scope.classify(s) for s in ("s3", "iam", "ec2", "cloudtrail", "lambda", "kms", "secretsmanager", "budgets", "dynamodb")}
        self.assertEqual(got, {"s3": "approved", "iam": "approved", "ec2": "approved", "cloudtrail": "approved",
                               "lambda": "excluded", "kms": "excluded", "secretsmanager": "excluded",
                               "budgets": "recognised", "dynamodb": "out_of_scope"})

    def test_scope_rejects_a_service_listed_twice(self):
        text = ("approved_services:\n  s3: {event_source: s3.amazonaws.com}\n"
                "excluded_by_project_decision:\n  s3: {event_source: s3.amazonaws.com}\n")
        with self.assertRaises(config.ConfigError):
            config.load_scope(self._write(text))


class ConditionTests(unittest.TestCase):
    FACTS = {"services": {"s3"}, "action_classes": {"create", "write"}, "exposure": set()}

    def test_operators(self):
        ev = lambda c: conditions.evaluate_condition(c, self.FACTS)  # noqa: E731
        self.assertTrue(ev({"fact": "services", "any_of": ["s3", "iam"]}))
        self.assertFalse(ev({"fact": "services", "any_of": ["iam"]}))
        self.assertTrue(ev({"fact": "action_classes", "all_of": ["create", "write"]}))
        self.assertFalse(ev({"fact": "action_classes", "all_of": ["create", "delete"]}))
        self.assertTrue(ev({"fact": "action_classes", "none_of": ["delete"]}))
        self.assertTrue(ev({"fact": "exposure", "present": False}))
        self.assertTrue(ev({"fact": "services", "present": True}))
        self.assertTrue(ev({"all": [{"fact": "services", "any_of": ["s3"]}, {"not": {"fact": "exposure", "present": True}}]}))
        self.assertFalse(ev({"any": [{"fact": "services", "any_of": ["iam"]}, {"fact": "exposure", "present": True}]}))

    def test_validation_rejects_malformed_conditions(self):
        bad = [{"fact": "servces", "any_of": ["s3"]}, {"fact": "services", "any_of": ["s33"]}, {"fact": "services"},
               {"fact": "services", "any_of": []}, {"fact": "services", "any_of": ["s3"], "all_of": ["s3"]},
               {"all": []}, {"all": [{"fact": "services", "any_of": ["s3"]}], "any": []}, {"fact": "services", "present": "yes"}, "s3"]
        for cond in bad:
            with self.subTest(cond):
                self.assertTrue(conditions.validate_condition(cond))
        self.assertEqual(conditions.validate_condition({"any": [{"fact": "services", "any_of": ["s3"]}, {"not": {"fact": "exposure", "present": True}}]}), [])

    def test_referenced_values(self):
        cond = {"all": [{"fact": "stated_requirements", "any_of": ["least_privilege"]}, {"not": {"fact": "stated_requirements", "none_of": ["no_wildcards"]}}]}
        self.assertEqual(conditions.referenced_values(cond, "stated_requirements"), {"least_privilege", "no_wildcards"})


class CatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = config.load_scope()
        cls.cat = hs.catalogue()

    def rejected(self, mutate):
        data = hs.catalogue_data()
        mutate(data)
        return C.validate_catalogue_data(data, self.scope)

    def test_loads_with_expected_shape(self):
        self.assertRegex(self.cat.version, r"^\d+\.\d+\.\d+$")
        ids = [c.control_id for c in self.cat.controls]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(self.cat.active()), 20)
        self.assertTrue(all(c.catalogue_version <= self.cat.version for c in self.cat.controls))
        self.assertEqual(self.cat.sha256, C.load_catalogue().sha256, "loading twice must give the same content hash")

    def test_every_control_has_the_requested_fields_and_a_traceable_source(self):
        required = ["control_id", "control_category", "control_name", "control_description", "applicability_conditions",
                    "affected_services_or_components", "required_prompt_clause", "verification_expectation", "severity_or_priority",
                    "source_name", "source_url", "source_version_or_section", "retrieved_at", "content_hash", "catalogue_version",
                    "active_status", "provenance_status", "enforcement"]
        for control in self.cat.controls:
            with self.subTest(control.control_id):
                for name in required:
                    self.assertTrue(hasattr(control, name), name)
                self.assertTrue(control.source_name.strip() and control.source_version_or_section.strip())
                self.assertRegex(control.source_url, r"^(https://|repo:)")
                self.assertEqual(provenance.provenance_problems(control), [])
                if control.derived:
                    self.assertTrue(control.supporting_references, "a derived recommendation keeps its supporting references")

    def test_mandatory_is_a_policy_decision_not_every_control(self):
        mandatory = {c.control_id for c in self.cat.controls if c.enforcement == "mandatory"}
        self.assertTrue(mandatory, "some project constraints are mandatory")
        self.assertLess(len(mandatory), len(self.cat.controls) / 2, "recommendations must not all become blocking rules")

    def test_no_control_covers_an_excluded_service(self):
        for control in self.cat.controls:
            self.assertFalse(set(control.affected_services_or_components) & set(self.scope.excluded), control.control_id)
            self.assertNotRegex(control.control_id, r"(?i)^(LAMBDA|KMS|SECRETS?)-")

    def test_loader_rejects_each_class_of_defect_for_the_intended_reason(self):
        ctl = hs.control_in
        # (name, mutation, text the rejection must contain) - proves the INTENDED check fired, not an accidental one
        mutations = [
            ("typo in stated-requirement tag", lambda d: ctl(d, "IAM-LEASTPRIV-001")["satisfied_by"]["any_of"].__setitem__(0, "least_privilage"), "not valid for fact"),
            ("unknown fact in applicability", lambda d: ctl(d, "S3-PUBLIC-001")["applicability_conditions"]["all"][0].__setitem__("fact", "servces"), "unknown fact"),
            ("unknown value for a fact", lambda d: ctl(d, "S3-PUBLIC-001")["applicability_conditions"]["all"][0].__setitem__("any_of", ["s33"]), "not valid for fact"),
            ("affects an excluded service (kms)", lambda d: ctl(d, "S3-ENCRYPT-002")["affected_services_or_components"].append("kms"), "excluded by project decision"),
            ("affects an unknown service", lambda d: ctl(d, "S3-ENCRYPT-002")["affected_services_or_components"].append("dynamodb"), "not in the approved scope"),
            ("clause mentions Secrets Manager", lambda d: ctl(d, "APP-SECRETS-007").__setitem__("required_prompt_clause", ctl(d, "APP-SECRETS-007")["required_prompt_clause"] + " Store it in AWS Secrets Manager."), "excluded from this module"),
            ("clause mentions KMS", lambda d: ctl(d, "S3-ENCRYPT-002").__setitem__("required_prompt_clause", ctl(d, "S3-ENCRYPT-002")["required_prompt_clause"] + " Use a customer KMS key."), "excluded from this module"),
            ("verification mentions AWS Lambda", lambda d: ctl(d, "DEV-VERIFY-001").__setitem__("verification_expectation", ctl(d, "DEV-VERIFY-001")["verification_expectation"] + " Deploy it as an AWS Lambda."), "excluded from this module"),
            ("derived without supporting references", lambda d: ctl(d, "IAM-WILDCARD-002").__setitem__("supporting_references", []), "derived recommendation"),
            ("unverified control carrying a hash", lambda d: (ctl(d, "IAM-LEASTPRIV-001").update({"provenance_status": "unverified", "retrieved_at": None, "content_hash": "sha256:" + "0" * 64})), "unverified control must not carry"),
            ("snapshot_verified without a hash", lambda d: ctl(d, "S3-PUBLIC-001").update({"provenance_status": "snapshot_verified", "retrieved_at": "2026-10-21T00:00:00Z", "content_hash": None}), "snapshot_verified requires"),
            ("section_checked without retrieved_at", lambda d: ctl(d, "IAM-LEASTPRIV-001").update({"provenance_status": "section_checked", "retrieved_at": None, "content_hash": None}), "section_checked requires"),
            ("malformed content hash", lambda d: ctl(d, "S3-PUBLIC-001").update({"provenance_status": "snapshot_verified", "retrieved_at": "2026-10-21T00:00:00Z", "content_hash": "sha256:abc"}), "does not match pattern"),
            ("duplicate control_id", lambda d: d["controls"].append(copy.deepcopy(d["controls"][0])), "duplicate control_id"),
            ("bad regex in cues", lambda d: d["requirement_cues"]["least_privilege"].append("(unclosed"), "bad regex"),
            ("unknown cue tag", lambda d: d["requirement_cues"].__setitem__("made_up_tag", ["xx"]), "unknown tag"),
            ("tag used but without cues", lambda d: d["requirement_cues"].pop("least_privilege"), "no cue patterns"),
            ("control newer than the catalogue", lambda d: ctl(d, "DB-ACCESS-001").__setitem__("catalogue_version", "9.0.0"), "newer than the catalogue"),
            ("source url not https/repo", lambda d: ctl(d, "DB-ACCESS-001").__setitem__("source_url", "http://insecure.example"), "does not match pattern"),
            ("empty source section", lambda d: ctl(d, "DB-ACCESS-001").__setitem__("source_version_or_section", ""), "shorter than"),
            ("unknown enforcement", lambda d: ctl(d, "DB-ACCESS-001").__setitem__("enforcement", "optional"), "is not one of"),
            ("unknown control field", lambda d: ctl(d, "DB-ACCESS-001").__setitem__("severity", "high"), "unexpected property"),
            ("duplicate trigger ids", lambda d: ctl(d, "S3-PUBLIC-001")["conflict_triggers"].append(copy.deepcopy(ctl(d, "S3-PUBLIC-001")["conflict_triggers"][0])), "duplicate conflict trigger"),
            ("no controls", lambda d: d.__setitem__("controls", []), "at least 1"),
        ]
        self.assertEqual(C.validate_catalogue_data(hs.catalogue_data(), self.scope), [], "the unmodified catalogue is valid")
        for name, mutate, expected in mutations:
            with self.subTest(name):
                problems = self.rejected(mutate)
                self.assertTrue(problems, f"defect was accepted: {name}")
                self.assertIn(expected, " | ".join(problems), f"{name}: rejected, but not by the intended check")

    def test_parse_raises_one_error_listing_every_problem(self):
        data = hs.catalogue_data()
        data["requirement_cues"]["least_privilege"].append("(unclosed")      # semantic problems are reported together;
        data["requirement_cues"]["made_up_tag"] = ["xx"]                    # structural ones stop validation first (by design)
        with self.assertRaises(C.CatalogueError) as caught:
            C.parse_catalogue(data, self.scope)
        self.assertGreaterEqual(len(caught.exception.problems), 2)

    def test_unreadable_catalogue_is_a_catalogue_error(self):
        with self.assertRaises(C.CatalogueError):
            C.load_catalogue(os.path.join(tempfile.gettempdir(), "does-not-exist-catalogue.yaml"))

    def test_content_hash_changes_when_a_control_changes_and_diff_reports_it(self):
        data = hs.catalogue_data()
        draft = hs.control_in(data, "DEV-VERIFY-001")
        draft["active_status"] = "active"          # simulate explicit promotion after source review
        draft["provenance_status"] = "section_checked"
        draft["retrieved_at"] = "2026-10-21T05:00:00Z"
        draft["content_hash"] = None
        draft["source_status"] = "available"
        hs.control_in(data, "S3-ENCRYPT-002")["active_status"] = "deprecated"    # retire an active control
        hs.control_in(data, "APP-INPUT-003")["required_prompt_clause"] += " Reject unknown fields."
        data["catalogue_version"] = "0.9.0"
        newer = C.parse_catalogue(data, self.scope)
        diff = C.diff_catalogues(self.cat, newer)
        self.assertNotEqual(newer.sha256, self.cat.sha256)
        self.assertEqual(diff["activated"], ["DEV-VERIFY-001"])
        self.assertEqual(diff["deactivated"], ["S3-ENCRYPT-002"])
        self.assertIn("APP-INPUT-003", diff["changed"])
        self.assertEqual((diff["from_version"], diff["to_version"]), (self.cat.version, "0.9.0"))
        self.assertEqual(C.diff_catalogues(self.cat, self.cat)["changed"], [])

    def test_provenance_levels(self):
        base = hs.control_in(hs.catalogue_data(), "IAM-LEASTPRIV-001")
        make = lambda **kw: models.Control.from_dict({**base, **kw})  # noqa: E731
        self.assertEqual(provenance.provenance_problems(make()), [])
        self.assertTrue(provenance.provenance_problems(make(provenance_status="unverified", retrieved_at="2026-10-18T00:00:00Z", content_hash=None)))
        self.assertTrue(provenance.provenance_problems(make(provenance_status="unverified", retrieved_at=None, content_hash="sha256:" + "0" * 64)))
        self.assertEqual(provenance.provenance_problems(make(provenance_status="section_checked", retrieved_at="2026-10-18T00:00:00Z", content_hash=None)), [])
        self.assertTrue(provenance.provenance_problems(make(provenance_status="snapshot_verified", retrieved_at="2026-10-18T00:00:00Z", content_hash=None)))
        self.assertEqual(provenance.provenance_problems(make(provenance_status="snapshot_verified", retrieved_at="2026-10-18T00:00:00Z",
                                                             content_hash=provenance.sha256_bytes(b"snapshot"))), [])

    def test_source_usability_follows_policy(self):
        ctl = self.cat.get("S3-PUBLIC-001")
        warn = SimpleNamespace(require_snapshot_verified_provenance=False, on_unavailable_source="apply_with_warning")
        skip = SimpleNamespace(require_snapshot_verified_provenance=False, on_unavailable_source="skip")
        strict = SimpleNamespace(require_snapshot_verified_provenance=True, on_unavailable_source="apply_with_warning")
        self.assertEqual(provenance.control_usability(ctl, warn), (True, []))
        gone = models.Control.from_dict({**hs.control_in(hs.catalogue_data(), "S3-PUBLIC-001"), "source_status": "unavailable"})
        usable, warnings = provenance.control_usability(gone, warn)
        self.assertTrue(usable and "unavailable" in warnings[0])
        self.assertEqual(provenance.control_usability(gone, skip)[0], False)
        self.assertEqual(provenance.control_usability(ctl, strict)[0], False, "section-checked provenance is not a raw snapshot")
        verified = models.Control.from_dict({**hs.control_in(hs.catalogue_data(), "S3-PUBLIC-001"),
            "provenance_status": "snapshot_verified", "retrieved_at": "2026-10-21T00:00:00Z",
            "content_hash": provenance.sha256_bytes(b"source snapshot")})
        self.assertEqual(provenance.control_usability(verified, strict), (True, []))
        draft = self.cat.get("DEV-VERIFY-001")
        self.assertEqual(provenance.control_usability(draft, warn), (False, []))

    def test_freshness_report_is_honest_about_verification(self):
        report = provenance.freshness_report(self.cat)
        self.assertEqual(sum(report["provenance_counts"].values()), report["active_controls"])
        self.assertEqual(report["catalogue_version"], self.cat.version)
        if report["last_sync_status"] == "never_synced":
            self.assertIn("No source synchronisation", report["note"])

    def test_audit_control_has_an_external_source_in_this_standalone_export(self):
        control = self.cat.get("LOG-AUDIT-001")
        self.assertEqual(control.active_status, "active")
        self.assertEqual(control.provenance_status, "section_checked")
        self.assertEqual(control.source_url, "https://docs.aws.amazon.com/awscloudtrail/latest/userguide/best-practices-security.html")
        self.assertIsNone(control.content_hash, "raw-source hash is reserved for downloaded bytes from source_sync")


if __name__ == "__main__":
    unittest.main()
