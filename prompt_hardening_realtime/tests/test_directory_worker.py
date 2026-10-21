"""Tests for the multi-request filesystem queue and request/response correlation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from security_hardening.directory_worker import DirectoryWorker
from security_hardening.schema_validation import validate_response


def request(request_id: str, prompt: str = "Create a private S3 bucket for storing application logs.") -> dict:
    return {
        "request_id": request_id,
        "screening_status": "SAFE",
        "original_prompt": prompt,
    }


class DirectoryWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.incoming = self.root / "incoming"
        self.output = self.root / "output"
        self.processed = self.root / "processed"
        self.failed = self.root / "failed"
        self.worker = DirectoryWorker(
            self.incoming, self.output, self.processed, self.failed, workers=4
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def put_request(self, filename: str, data: object) -> Path:
        path = self.incoming / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_many_requests_get_individually_correlated_responses(self) -> None:
        requests = {
            "job-alpha.json": request("req-alpha", "Create a private S3 bucket for app logs."),
            "job-beta.json": request("req-beta", "Create an IAM role with least privilege."),
            "job-gamma.json": request("req-gamma", "Create an EC2 security group for a private service."),
        }
        for filename, body in requests.items():
            self.put_request(filename, body)

        summary = self.worker.process_available()
        self.assertEqual(summary.discovered, 3)
        self.assertEqual(summary.responses_written, 3)
        self.assertEqual(summary.failures, 0)

        for filename, body in requests.items():
            response_path = self.output / f"{Path(filename).stem}.response.json"
            self.assertTrue(response_path.exists(), f"missing paired response for {filename}")
            response = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(response["request_id"], body["request_id"])
            self.assertEqual(validate_response(response), [])
            self.assertTrue((self.processed / filename).exists())
        self.assertEqual(list(self.incoming.glob("*.json")), [])
        self.assertEqual(list(self.incoming.joinpath(".processing").glob("*.claim")), [])

    def test_duplicate_request_id_is_rejected_without_overwriting_first_response(self) -> None:
        self.put_request("one.json", request("same-id"))
        self.put_request("two.json", request("same-id", "Create a private EC2 instance."))

        summary = self.worker.process_available()
        self.assertEqual(summary.responses_written, 2)
        response_one = json.loads((self.output / "one.response.json").read_text(encoding="utf-8"))
        response_two = json.loads((self.output / "two.response.json").read_text(encoding="utf-8"))
        statuses = [response_one["status"], response_two["status"]]
        self.assertEqual(statuses.count("INTEGRATION_ERROR"), 1)
        duplicate_response = response_one if response_one["errors"] else response_two
        self.assertEqual(duplicate_response["errors"][0]["code"], "DUPLICATE_REQUEST_ID")
        self.assertEqual(validate_response(response_one), [])
        self.assertEqual(validate_response(response_two), [])

    def test_invalid_json_gets_a_paired_schema_valid_error_response(self) -> None:
        (self.incoming / "broken.json").write_text('{"request_id":', encoding="utf-8")
        summary = self.worker.process_available()
        self.assertEqual(summary.responses_written, 1)
        response = json.loads((self.output / "broken.response.json").read_text(encoding="utf-8"))
        self.assertEqual(response["request_id"], "broken")
        self.assertEqual(response["status"], "INTEGRATION_ERROR")
        self.assertEqual(response["errors"][0]["code"], "INVALID_JSON")
        self.assertEqual(validate_response(response), [])

    def test_missing_request_id_gets_a_paired_error_response(self) -> None:
        self.put_request("missing-id.json", {"screening_status": "SAFE", "original_prompt": "Create an S3 bucket."})
        summary = self.worker.process_available()
        self.assertEqual(summary.responses_written, 1)
        response = json.loads((self.output / "missing-id.response.json").read_text(encoding="utf-8"))
        self.assertEqual(response["request_id"], "missing-id")
        self.assertEqual(response["errors"][0]["code"], "REQUEST_ID_MISSING")
        self.assertEqual(validate_response(response), [])

    def test_existing_response_is_never_overwritten(self) -> None:
        existing = self.output / "job.response.json"
        existing.write_text('{"existing":true}\n', encoding="utf-8")
        self.put_request("job.json", request("new-id"))
        summary = self.worker.process_available()
        self.assertEqual(summary.failures, 1)
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"existing":true}\n')
        self.assertTrue(list(self.failed.glob("*.failure.json")))
        self.assertFalse((self.processed / "job.json").exists())

    def test_interrupted_claim_is_recovered(self) -> None:
        claim_dir = self.incoming / ".processing"
        claim_dir.mkdir(parents=True, exist_ok=True)
        claim = claim_dir / f"recovered.json.{'a' * 32}.claim"
        claim.write_text(json.dumps(request("recovered-id")), encoding="utf-8")
        summary = self.worker.process_available()
        self.assertEqual(summary.responses_written, 1)
        response = json.loads((self.output / "recovered.response.json").read_text(encoding="utf-8"))
        self.assertEqual(response["request_id"], "recovered-id")
        self.assertTrue((self.processed / "recovered.json").exists())

    def test_output_mapping_is_based_on_input_filename_not_request_id_path(self) -> None:
        self.put_request("human-readable-name.json", request("event-1234"))
        self.worker.process_available()
        self.assertTrue((self.output / "human-readable-name.response.json").exists())
        response = json.loads((self.output / "human-readable-name.response.json").read_text(encoding="utf-8"))
        self.assertEqual(response["request_id"], "event-1234")


if __name__ == "__main__":
    unittest.main()
