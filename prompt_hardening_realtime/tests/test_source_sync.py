"""Offline unit tests for source retrieval policy and review-gated catalogue synchronisation."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from security_hardening import source_sync as sync


class SourceUrlPolicyTests(unittest.TestCase):
    def test_allowlisted_official_https_is_accepted(self):
        parsed = sync._check_url("https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html")
        self.assertEqual(parsed.hostname, "docs.aws.amazon.com")

    def test_non_https_unlisted_hosts_credentials_and_nonstandard_ports_are_rejected(self):
        bad_urls = (
            "http://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html",
            "https://docs.aws.amazon.com.evil.example/file",
            "https://user:pass@docs.aws.amazon.com/file",
            "https://docs.aws.amazon.com:444/file",
        )
        for url in bad_urls:
            with self.subTest(url=url), self.assertRaises(sync.SourceSyncError):
                sync._check_url(url)


class CatalogueSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        raw = yaml.safe_load((ROOT / "security_hardening/catalogue/security_controls.yaml").read_text(encoding="utf-8"))
        # Keep one real external control so this synchroniser test is fast and deterministic.
        raw["controls"] = [c for c in raw["controls"] if c["control_id"] == "IAM-LEASTPRIV-001"]
        self.catalogue_path = self.root / "security_controls.yaml"
        self.catalogue_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        self.original_bytes = self.catalogue_path.read_bytes()
        self.snapshot_dir = self.root / "snapshots"
        self.candidate_dir = self.root / "candidates"
        self.state_path = self.root / "data" / "state.json"
        self.backup_dir = self.root / "backups"
        self.body = (
            "<html><head><title>Least privilege permissions</title></head><body>"
            "Security best practices recommend least privilege permissions scoped to only the actions "
            "and resources required by a workload. Review access policies and validate permissions."
            "</body></html>"
        ).encode("utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def successful_fetch(self, url):
        return {
            "url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "text/html",
            "retrieved_at": "2026-10-21T05:00:00Z",
            "body": self.body,
        }

    def test_success_creates_a_hashed_raw_snapshot_and_candidate_without_activation(self):
        report = sync.sync_catalogue(
            catalogue_path=self.catalogue_path,
            snapshot_dir=self.snapshot_dir,
            candidate_dir=self.candidate_dir,
            state_path=self.state_path,
            backup_dir=self.backup_dir,
            fetcher=self.successful_fetch,
        )
        self.assertFalse(report["activated"])
        self.assertEqual(self.catalogue_path.read_bytes(), self.original_bytes, "candidate mode must leave active catalogue byte-for-byte unchanged")
        self.assertFalse(report["section_not_verified"])
        self.assertFalse(report["unavailable"])
        self.assertTrue(Path(report["candidate"]).is_file())
        self.assertTrue(Path(report["report_path"]).is_file())
        candidate = yaml.safe_load(Path(report["candidate"]).read_text(encoding="utf-8"))
        control = candidate["controls"][0]
        self.assertEqual(control["provenance_status"], "snapshot_verified")
        self.assertEqual(control["content_hash"], "sha256:" + hashlib.sha256(self.body).hexdigest())
        self.assertTrue(control["retrieved_at"])
        snapshot = Path(report["available"][0]["snapshot"])
        self.assertEqual(snapshot.read_bytes(), self.body)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["last_attempt"]["status"], "candidate_created")

    def test_partial_sync_cannot_activate_or_replace_last_known_good_catalogue(self):
        def unavailable(url):
            raise sync.SourceSyncError("simulated offline source")

        report = sync.sync_catalogue(
            catalogue_path=self.catalogue_path,
            snapshot_dir=self.snapshot_dir,
            candidate_dir=self.candidate_dir,
            state_path=self.state_path,
            backup_dir=self.backup_dir,
            activate=True,
            fetcher=unavailable,
        )
        self.assertFalse(report["activated"])
        self.assertIn("activation_blocked_reason", report)
        self.assertEqual(self.catalogue_path.read_bytes(), self.original_bytes)
        data = yaml.safe_load(Path(report["candidate"]).read_text(encoding="utf-8"))
        self.assertEqual(data["freshness"]["last_sync_status"], "failed")
        self.assertTrue(report["unavailable"])
        self.assertFalse(list(self.backup_dir.glob("security_controls_*.yaml")) if self.backup_dir.exists() else [])

    def test_activation_writes_backup_and_rollback_restores_exact_prior_catalogue(self):
        report = sync.sync_catalogue(
            catalogue_path=self.catalogue_path,
            snapshot_dir=self.snapshot_dir,
            candidate_dir=self.candidate_dir,
            state_path=self.state_path,
            backup_dir=self.backup_dir,
            activate=True,
            fetcher=self.successful_fetch,
        )
        self.assertTrue(report["activated"])
        backup = Path(report["backup"])
        self.assertTrue(backup.is_file())
        self.assertNotEqual(self.catalogue_path.read_bytes(), self.original_bytes)
        restored_from = sync.rollback_catalogue(catalogue_path=self.catalogue_path, backup_dir=self.backup_dir)
        self.assertEqual(restored_from, backup)
        self.assertEqual(self.catalogue_path.read_bytes(), self.original_bytes)


if __name__ == "__main__":
    unittest.main()
