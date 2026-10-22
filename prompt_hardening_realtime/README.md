# CloudSec Security Hardening — Directory Queue

A file-based request/response interface for the CloudSec security-hardening module. Requests are dropped into `incoming/`; the worker processes them and writes one structured response to `output/` for each request.

> **Scope:** This package hardens prompts and returns security controls, assumptions, warnings, and verification expectations. It does not deploy AWS resources, replace the upstream screening component, or prove that a generated infrastructure plan is secure.

## Contents

- [How the queue works](#how-the-queue-works)
- [Requirements and setup](#requirements-and-setup)
- [Start the worker](#start-the-worker)
- [Submit a request](#submit-a-request)
- [Match responses to requests](#match-responses-to-requests)
- [Request and response formats](#request-and-response-formats)
- [Queue safety and concurrency](#queue-safety-and-concurrency)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Current integration boundary](#current-integration-boundary)

## How the queue works

```text
Trusted upstream screening component
                 |
                 v
        incoming/<request>.json
                 |
                 v
       Directory queue worker
       - claims input atomically
       - validates request
       - hardens eligible prompt
       - records request ID/status
                 |
        +--------+---------+
        |                  |
        v                  v
output/<request>.      processed/
response.json          completed input
        |
        v
Downstream planning component

Malformed/unprocessable inputs are handled according to the worker's
error path; inspect the response and failed/ directory for details.
```

The worker polls the `incoming/` directory. Multiple request files can wait in the queue, and `--workers` controls how many requests the worker pool can process concurrently.

## Requirements and setup

Use Python 3.10 or a version supported by the project's `pyproject.toml` / `requirements.txt`.

```bash
# Extract the project ZIP, then enter the project root.
cd cloudsec_hardening_completed

# Create and activate a virtual environment (macOS/Linux).
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies.
python -m pip install -r requirements.txt

# Validate the control catalogue.
python -m security_hardening validate-catalogue
```

Run commands from the project root, where `requirements.txt`, `incoming/`, and the `security_hardening/` package are located.

## Start the worker

### Continuous mode

```bash
python -m security_hardening watch --workers 4 --poll-interval 0.5
```

- `--workers 4`: allow up to four requests to be processed concurrently.
- `--poll-interval 0.5`: check for new input files about every half second.

Keep this process running while requests are arriving.

### One-shot mode

```bash
python -m security_hardening watch --once
```

Use one-shot mode to process the files currently waiting and then exit. It is useful for local tests and batch runs.

To stop continuous mode, press `Ctrl+C`.

## Submit a request

Each request is a UTF-8 JSON file with a unique `request_id`, a screening decision supplied by the trusted upstream component, and the original prompt.

Example: save as `incoming/req-demo-s3-001.json`.

```json
{
  "request_id": "req-demo-s3-001",
  "screening_status": "SAFE",
  "original_prompt": "Create a private S3 bucket for application audit logs. Block public access, restrict access to the application's IAM role and the security team's IAM role, enforce HTTPS, enable S3-managed server-side encryption (SSE-S3) and versioning, and provide least-privilege IAM policies and security verification steps. Do not deploy any resources."
}
```

`SAFE` is only an example. In a real pipeline, the trusted upstream screening component must produce the screening decision. Do not mark arbitrary user input safe merely to bypass screening.

### Write files atomically

Do not write directly to a final `*.json` filename while the worker is polling. Write a temporary file in the same directory and rename it after writing is complete:

```bash
cat > incoming/req-demo-s3-001.tmp <<'EOF'
{
  "request_id": "req-demo-s3-001",
  "screening_status": "SAFE",
  "original_prompt": "Create a private S3 bucket for application audit logs. Block public access, enforce HTTPS, enable SSE-S3 and versioning, and provide least-privilege policies and verification steps. Do not deploy resources."
}
EOF

mv incoming/req-demo-s3-001.tmp incoming/req-demo-s3-001.json
```

The rename is atomic when source and destination are on the same filesystem. Use a distinct filename for every request.

## Match responses to requests

The response filename is derived from the input filename, and the JSON response echoes the `request_id`. Use both identifiers when correlating work.

| Input | Corresponding output |
|---|---|
| `incoming/req-001.json` | `output/req-001.response.json` |
| `incoming/req-002.json` | `output/req-002.response.json` |
| `incoming/req-003.json` | `output/req-003.response.json` |

Do not reuse request filenames or `request_id` values. The queue's SQLite index tracks processed request IDs and is intended to reject duplicates. A duplicate request ID should be treated as a submission error, not as a new independent request.

Example: if the input is `incoming/req-demo-s3-001.json`, look for `output/req-demo-s3-001.response.json`. Inspect the response's `request_id` before passing it to another component.

## Request and response formats

### Request fields

| Field | Purpose |
|---|---|
| `request_id` | Unique identifier generated by the submitting component; echoed in the response. |
| `screening_status` | Decision from the trusted upstream screening component. `SAFE` in examples is not a substitute for real screening. |
| `original_prompt` | The prompt or infrastructure request to analyse and harden. |

The precise accepted status values and any additional optional fields are defined by the request model in the installed source. Keep request JSON valid and UTF-8 encoded.

### Response

The response is JSON. Depending on the request and result, it includes the matching request ID, processing status, revised/hardened prompt, selected controls, assumptions, warnings, and verification expectations. Treat the response status as authoritative for whether the request was successfully handled; do not assume every request will be hardened.

Example shape (illustrative only; exact field names are defined by the implementation):

```json
{
  "request_id": "req-demo-s3-001",
  "status": "HARDENED",
  "hardened_prompt": "<security-enhanced prompt>",
  "controls": [],
  "assumptions": [],
  "warnings": []
}
```

Do not rely on the illustrative field names without checking the actual response schema.

## Queue safety and concurrency

- **Atomic claiming:** the worker claims a request before processing so workers do not normally process the same input at the same time.
- **Concurrent work:** `--workers` sets the worker-pool concurrency.
- **Unique IDs:** the SQLite index tracks IDs and helps prevent duplicate submissions.
- **No silent response replacement:** existing response files should not be overwritten by a later request with the same output name.
- **Malformed input:** invalid JSON or invalid request data should be inspected in the output/error path and logs; do not assume a malformed request was successfully handled.
- **Restart recovery:** the queue is designed to recover unfinished work; verify the request's index state and output before manually resubmitting.
- **Single queue owner:** run one watcher process per `incoming/` and `output/` pair unless multi-process coordination has been explicitly tested for your deployment.
- **Trusted inbox:** only trusted components should be able to write to the inbox. Validate and screen requests before placing them there.

For a production deployment, also consider filesystem permissions, disk limits, log rotation, monitoring, retention policies, and a durable shared queue if requests arrive across multiple hosts.

## Troubleshooting

### The worker does not pick up a file

- Confirm the worker is running from the project root.
- Confirm the file is under `incoming/` and ends with `.json`.
- Make sure the writer has finished and renamed the temporary file to its final `.json` name.
- Validate the file with a JSON parser and check the worker logs.

### The response is missing

- Look for `output/<input-stem>.response.json`.
- Check `processed/`, `failed/`, and worker logs.
- Check whether the `request_id` was already used in the SQLite index.
- Do not delete queue/index files or resubmit the same ID until you understand the recorded state.

### Catalogue validation fails

```bash
python -m security_hardening validate-catalogue
```

Resolve the validation error before processing requests.

### The module returns a screening-related status

The hardening adapter is not the upstream screening engine. Send only requests whose screening status has been produced by the actual screening component, and follow the response status rather than assuming a hardened prompt was produced.

## Testing

Run the automated tests from the project root using the repository's configured test command. If the project uses `pytest`, run:

```bash
python -m pytest -q
```

Some parent-pipeline integration checks may be skipped when the upstream screening and downstream planning modules are not included in the standalone package. A passing standalone suite does not establish end-to-end integration.

## Current integration boundary

This module provides a directory-based interface suitable for connecting to a larger pipeline:

1. The upstream component screens a request and writes a uniquely identified JSON file into `incoming/`.
2. The hardening worker processes it and writes a matching JSON response into `output/`.
3. The downstream planner reads the response, checks its status and request ID, and decides whether it is eligible for planning.

The standalone queue does not itself provide a network API, perform AWS deployments, or replace upstream screening. End-to-end behaviour must be tested after the real screening and planning components are connected.
