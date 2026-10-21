"""Integrity and provenance tests that are self-contained in the supplied ZIP."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from security_hardening import integrity
from security_hardening.catalogue import load_catalogue
from security_hardening.config import load_scope


class StandaloneIntegrityTests(unittest.TestCase):
    def test_scope_config_and_lexicon_agree_and_excluded_services_are_not_supported_controls(self):
        scope = load_scope()
        lexicon = yaml.safe_load((ROOT / "security_hardening/config/lexicon.yaml").read_text())
        approved = set(scope.approved)
        lexicon_services = set(lexicon["services"])
        self.assertTrue(approved <= lexicon_services)
        excluded = set(scope.excluded)
        self.assertEqual(excluded, {"lambda", "kms", "secretsmanager"})
        self.assertTrue(excluded <= lexicon_services)
        catalogue = yaml.safe_load((ROOT / "security_hardening/catalogue/security_controls.yaml").read_text())
        for control in catalogue["controls"]:
            affected = {str(x).lower() for x in control.get("affected_services_or_components", [])}
            self.assertFalse(affected & excluded, (control["control_id"], affected & excluded))

    def test_active_external_controls_have_dated_review_records_but_no_false_raw_hash_claim(self):
        path = ROOT / "security_hardening/catalogue/security_controls.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        controls = raw["controls"]
        self.assertEqual(raw["catalogue_version"], "0.2.3")
        self.assertEqual(len(controls), 25)
        self.assertEqual(sum(c["active_status"] == "active" for c in controls), 24)
        self.assertEqual(sum(c["active_status"] == "draft" for c in controls), 1)
        self.assertFalse([c["control_id"] for c in controls if c["active_status"] == "active" and c["provenance_status"] == "unverified"])
        evidence_dir = ROOT / "security_hardening/snapshots/section_evidence"
        for c in controls:
            if c["active_status"] != "active":
                continue
            self.assertIn(c["provenance_status"], {"section_checked", "snapshot_verified"}, c["control_id"])
            if c["provenance_status"] == "section_checked":
                self.assertTrue(c.get("retrieved_at"), c["control_id"])
                self.assertIsNone(c.get("content_hash"), c["control_id"])
                evidence_path = evidence_dir / f"{c['control_id']}.json"
                self.assertTrue(evidence_path.is_file(), c["control_id"])
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                self.assertTrue(evidence.get("source_url"), c["control_id"])
                self.assertTrue(evidence.get("section") or evidence.get("source_version_or_section"), c["control_id"])
                self.assertTrue(evidence.get("reviewed_at") or evidence.get("retrieved_at"), c["control_id"])
            if c["provenance_status"] == "snapshot_verified":
                self.assertRegex(c.get("content_hash") or "", r"^sha256:[0-9a-f]{64}$")


    def test_corrected_draft_sources_and_promoted_controls_have_consistent_provenance(self):
        raw = yaml.safe_load((ROOT / "security_hardening/catalogue/security_controls.yaml").read_text(encoding="utf-8"))
        by_id = {c["control_id"]: c for c in raw["controls"]}
        self.assertEqual(by_id["APP-ERRORS-010"]["source_url"],
                         "https://cheatsheetseries.owasp.org/cheatsheets/Error_Handling_Cheat_Sheet.html")
        self.assertEqual(by_id["APP-OUTPUT-011"]["source_url"],
                         "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet")
        self.assertEqual(by_id["APP-DEPS-012"]["source_url"],
                         "https://top10.owasp.org/2025/A03_2025-Software_Supply_Chain_Failures/")
        for cid in ("APP-ERRORS-010", "APP-OUTPUT-011", "APP-DEPS-012"):
            self.assertEqual(by_id[cid]["active_status"], "active")
            self.assertEqual(by_id[cid]["provenance_status"], "section_checked")
            self.assertIsNone(by_id[cid]["content_hash"])
            self.assertTrue((ROOT / "security_hardening/snapshots/section_evidence" / f"{cid}.json").is_file())
        remaining = [c for c in raw["controls"] if c["active_status"] == "draft"]
        self.assertEqual([c["control_id"] for c in remaining], ["DEV-VERIFY-001"])
        self.assertEqual(remaining[0]["provenance_status"], "unverified")
        self.assertIsNone(remaining[0].get("retrieved_at"))
        self.assertIsNone(remaining[0].get("content_hash"))

    def test_current_standalone_manifest_is_clean(self):
        self.assertEqual(integrity.verify_manifest(), [])

    def test_manifest_detects_modified_missing_and_added_frozen_assets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "requirements.txt").write_text("PyYAML\n", encoding="utf-8")
            (root / "security_hardening/config").mkdir(parents=True)
            (root / "security_hardening/config/example.yaml").write_text("x: 1\n", encoding="utf-8")
            (root / "security_hardening/schemas").mkdir(parents=True)
            (root / "security_hardening/schemas/request.json").write_text("{}\n", encoding="utf-8")
            (root / "security_hardening/data").mkdir(parents=True)
            (root / "security_hardening/data/dataset_scope_report.json").write_text("{}\n", encoding="utf-8")
            (root / "security_hardening/snapshots/section_evidence").mkdir(parents=True)
            (root / "security_hardening/snapshots/section_evidence/a.json").write_text("{}\n", encoding="utf-8")
            manifest = integrity.build_manifest(root)
            self.assertEqual(integrity.verify_manifest(root, manifest), [])
            (root / "security_hardening/config/example.yaml").write_text("x: 2\n", encoding="utf-8")
            (root / "security_hardening/schemas/request.json").unlink()
            (root / "security_hardening/config/added.yaml").write_text("z: 3\n", encoding="utf-8")
            problems = integrity.verify_manifest(root, manifest)
            self.assertTrue(any(p.startswith("MODIFIED") for p in problems), problems)
            self.assertTrue(any(p.startswith("MISSING") for p in problems), problems)
            self.assertTrue(any(p.startswith("ADDED") for p in problems), problems)

    def test_request_path_does_not_import_network_or_cloud_execution_clients(self):
        modules = ["models.py", "config.py", "catalogue.py", "conditions.py", "provenance.py",
                   "schema_validation.py", "task_analyzer.py", "control_selector.py", "prompt_hardener.py",
                   "integration.py", "cli.py"]
        forbidden_roots = {"socket", "http", "urllib", "requests", "boto3", "botocore", "subprocess", "ftplib"}
        offenders = []
        for filename in modules:
            tree = ast.parse((ROOT / "security_hardening" / filename).read_text(encoding="utf-8"), filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                for name in names:
                    if name in forbidden_roots:
                        offenders.append((filename, name))
        self.assertEqual(offenders, [])
        init = (ROOT / "security_hardening/__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("source_sync", init)

    def test_documentation_covers_sync_installation_interfaces_and_boundaries(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        status = (ROOT / "IMPLEMENTATION_STATUS.md").read_text(encoding="utf-8").lower()
        for phrase in ("installation", "configuration", "source_sync.py", "rollback", "integration", "limitations"):
            self.assertIn(phrase, readme)
        self.assertIn("unverified", status)
        self.assertIn("source_sync.py", status)

    @unittest.skipUnless((ROOT / "feature_engine9.py").exists(),
                         "the parent pipeline source file is not included in this ZIP; end-to-end integration is unverified")
    def test_parent_pipeline_integration(self):
        # This branch runs only when testing inside the actual parent repository.
        self.fail("Parent-repository integration requires the repository's own integration test harness.")


if __name__ == "__main__":
    unittest.main()
