"""
security_hardening
==================
Stage 2 of the CloudSec-realtime-pipeline: security hardening of a LEGITIMATE
prompt that upstream screening has already approved.

WHAT IT DOES
------------
Takes an approved prompt, works out which security requirements the task
implies but the prompt never states, and returns a revised prompt that keeps the
original request verbatim and appends specific, source-backed security clauses,
assumptions, open items and verification expectations for the downstream
planning / verification stages.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It never classifies a prompt as safe. Upstream screening owns that; without an
  approved screening status the adapter returns an explicit non-hardened status.
* It never touches AWS: no credentials, no SDK, no network on the request path.
* It never reads, writes or regenerates the project's dataset, event taxonomy,
  checkpoints or model code (see integrity.py / tests/test_security_hardening_integrity.py).
* It never adds AWS services beyond the pipeline's own vocabulary
  (feature_engine9.FIXED_EVENT_SOURCES) and never adds Lambda, KMS or Secrets
  Manager handling (config/scope.yaml records why).

Everything here is standard library + PyYAML (already in requirements.txt), so it
runs without torch, Neo4j or Docker. Import-light by design: importing this
package does not import any submodule.
"""

__version__ = "0.2.3"
