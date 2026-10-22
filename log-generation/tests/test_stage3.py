"""
Comprehensive pytest test suite for Stage 3 Intent Extraction and Event Predictor.

Validates all 11 minimum required scenarios, plus KMS, EC2, and Lambda workflows.
"""

from __future__ import annotations

import json
import os
import sys
import pytest

pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

try:
    from engine import Stage3Engine, predict_workflow
    from extractor import (
        ExtractorException,
        MockLLMProvider,
        TimeoutException,
    )
    from schemas import (
        ActionMappingStatus,
        RequirementType,
        WorkflowStatus,
    )
except ImportError:
    from stage3_intent_event_predictor.engine import Stage3Engine, predict_workflow
    from stage3_intent_event_predictor.extractor import (
        ExtractorException,
        MockLLMProvider,
        TimeoutException,
    )
    from stage3_intent_event_predictor.schemas import (
        ActionMappingStatus,
        RequirementType,
        WorkflowStatus,
    )


@pytest.fixture
def engine():
    return Stage3Engine()


# 1. Simple supported task (GENERATED)
def test_simple_supported_task_s3_generated(engine):
    orig = "Create an S3 bucket named simple-customer-assets"
    hard = "TASK: Create S3 bucket simple-customer-assets."
    out = engine.predict(orig, hard, "req-001")

    assert out.status == WorkflowStatus.GENERATED
    assert out.request_id == "req-001"
    assert out.parsed_intent.task_category == "s3_create_bucket"
    assert out.parsed_intent.provided_parameters["bucketName"] == "simple-customer-assets"
    assert len(out.proposed_actions) == 1
    assert out.proposed_actions[0].operation == "CreateBucket"
    assert out.proposed_actions[0].mapping_rule_id == "MAP-S3-001"

    # Single expected direct event
    assert len(out.expected_direct_events) == 1
    ev = out.expected_direct_events[0]
    assert ev.prediction_id == "pred-0001"
    assert ev.eventName == "CreateBucket"
    assert ev.eventSource == "s3.amazonaws.com"
    assert ev.record_type == "PREDICTED_CLOUDTRAIL_EVENT"
    assert ev.requestParameters["bucketName"] == "simple-customer-assets"
    assert out.prediction_coverage.complete_within_scope is True


# 2. Multi-operation task with Stage 2 security constraints
def test_multi_operation_task_with_security_constraints(engine):
    orig = "Create an S3 bucket named finance-reports-bucket"
    hard = (
        "TASK: Create S3 bucket named finance-reports-bucket. "
        "SECURITY CONSTRAINTS: Require server-side encryption with AES256. "
        "Block all public access with PutPublicAccessBlock."
    )
    out = engine.predict(orig, hard, "req-002")

    assert out.status == WorkflowStatus.GENERATED
    assert len(out.security_constraints) >= 2
    assert len(out.proposed_actions) == 3

    # Operations must be ordered: CreateBucket -> PutBucketEncryption -> PutPublicAccessBlock
    ops = [a.operation for a in out.proposed_actions]
    assert ops == ["CreateBucket", "PutBucketEncryption", "PutPublicAccessBlock"]

    # Verify CloudTrail event enumeration
    assert len(out.expected_direct_events) == 3
    event_names = [e.eventName for e in out.expected_direct_events]
    # Note: API is PutPublicAccessBlock, but official CloudTrail event is PutBucketPublicAccessBlock!
    assert event_names == ["CreateBucket", "PutBucketEncryption", "PutBucketPublicAccessBlock"]

    # Check dependencies
    enc_ev = out.expected_direct_events[1]
    assert "pred-0001" in enc_ev.depends_on_predictions
    pab_ev = out.expected_direct_events[2]
    assert "pred-0001" in pab_ev.depends_on_predictions


# 3. Missing required parameters forcing NEEDS_CLARIFICATION without inventing fake names
def test_missing_required_parameters_needs_clarification(engine):
    orig = "Please create an S3 bucket for storing pictures"
    hard = "TASK: Create an S3 bucket."
    out = engine.predict(orig, hard, "req-003")

    assert out.status == WorkflowStatus.NEEDS_CLARIFICATION
    assert "bucketName" in out.missing_parameters
    assert len(out.clarification_questions) > 0
    # Crucial zero-hallucination check: no fake bucket name was invented!
    assert "bucketName" not in out.parsed_intent.provided_parameters
    assert len(out.proposed_actions) == 0
    assert len(out.expected_direct_events) == 0


# 4. Ambiguous workflows (IAM AttachRolePolicy vs PutRolePolicy) forcing clarification & conditional events
def test_ambiguous_workflows_needs_clarification_and_conditional_events(engine):
    orig = "Create an IAM role named AppWorkerRole and grant it S3 read access"
    hard = "TASK: Create IAM role AppWorkerRole with read permissions."
    out = engine.predict(orig, hard, "req-004")

    assert out.status == WorkflowStatus.NEEDS_CLARIFICATION
    assert len(out.conditional_events) == 2

    cond_names = {c.eventName for c in out.conditional_events}
    assert cond_names == {"AttachRolePolicy", "PutRolePolicy"}

    for c in out.conditional_events:
        assert len(c.condition) > 0
        assert len(c.evidence) > 0
        assert len(c.unknown_fields) > 0


# 5. Partially supported multi-step task (KMS key + unsupported Macie job)
def test_partially_supported_task_partial_status(engine):
    orig = "Create a KMS key for database encryption, and then schedule an Amazon Macie discovery job"
    hard = "TASK: Create KMS key and execute Macie discovery job."
    out = engine.predict(orig, hard, "req-005")

    assert out.status == WorkflowStatus.PARTIAL
    assert out.prediction_coverage.complete_within_scope is False
    assert any("macie" in eff.lower() for eff in out.unmodelled_effects)
    assert len(out.proposed_actions) >= 1
    assert out.proposed_actions[0].operation == "CreateKey"


# 6. Completely unsupported task forcing UNSUPPORTED
def test_completely_unsupported_task(engine):
    orig = "Deploy a decentralized blockchain node on Amazon Quantum Ledger Database"
    hard = "TASK: Deploy blockchain ledger."
    out = engine.predict(orig, hard, "req-006")

    assert out.status == WorkflowStatus.UNSUPPORTED
    assert out.prediction_coverage.complete_within_scope is False
    assert len(out.proposed_actions) == 0
    assert len(out.expected_direct_events) == 0
    assert len(out.unmodelled_effects) > 0


# 7. Conflicting requirements between user prompt and security constraints
def test_conflicting_requirements(engine):
    orig = "Create a public S3 bucket named static-public-assets-123"
    hard = "TASK: Create S3 bucket static-public-assets-123. SECURITY REQUIREMENTS: Block all public access."
    out = engine.predict(orig, hard, "req-007")

    assert out.status == WorkflowStatus.NEEDS_CLARIFICATION
    assert any("conflict" in amb.lower() for amb in out.parsed_intent.ambiguities)
    assert len(out.clarification_questions) > 0


# 8. Invalid/malformed LLM JSON response forcing EXTRACTION_FAILED
def test_invalid_malformed_llm_json_extraction_failed():
    mock_provider = MockLLMProvider(mode="malformed_json")
    engine = Stage3Engine(llm_provider=mock_provider)
    
    # Use prompt that bypasses fast-path to trigger LLM provider
    orig = "Process custom request that bypasses regex rules with special tokens"
    hard = "TASK: Custom complex request."
    out = engine.predict(orig, hard, "req-008")

    assert out.status == WorkflowStatus.EXTRACTION_FAILED
    assert out.prediction_coverage.complete_within_scope is False


# 9. LLM timeout / unavailable model handling without guessing events
def test_llm_timeout_handling():
    mock_provider = MockLLMProvider(mode="timeout")
    engine = Stage3Engine(llm_provider=mock_provider)

    orig = "Run complex batch analytics job with external orchestrator"
    hard = "TASK: External analytics job."
    out = engine.predict(orig, hard, "req-009")

    assert out.status == WorkflowStatus.EXTRACTION_FAILED
    assert len(out.expected_direct_events) == 0
    assert len(out.proposed_actions) == 0


# 10. Anti-hallucination guard: LLM attempting to inject unverified API or event names
def test_anti_hallucination_guard_rejects_injected_api_or_event_names():
    mock_provider = MockLLMProvider(mode="hallucinated_injection")
    engine = Stage3Engine(llm_provider=mock_provider)

    orig = "Perform advanced administrative operations"
    hard = "TASK: Perform administrative actions."
    out = engine.predict(orig, hard, "req-010")

    # The engine must strip 'FakeAWSDoEverything' and 'FakeEventNameLogged'
    for action in out.proposed_actions:
        assert action.operation != "FakeAWSDoEverything"
    for ev in out.expected_direct_events:
        assert ev.eventName != "FakeEventNameLogged"
        assert ev.eventName in ["CreateBucket", "PutBucketEncryption", "PutBucketPublicAccessBlock"]


# 11. Verification that every emitted event has record_type, valid provenance URLs, and non-empty unknown_fields
def test_predicted_event_record_type_and_unknown_fields(engine):
    orig = "Create an S3 bucket named security-log-archive"
    hard = "TASK: Create S3 bucket security-log-archive. SECURITY CONSTRAINTS: Require encryption."
    out = engine.predict(orig, hard, "req-011")

    assert len(out.expected_direct_events) > 0
    for ev in out.expected_direct_events:
        assert ev.record_type == "PREDICTED_CLOUDTRAIL_EVENT"
        assert len(ev.source_refs) > 0
        for url in ev.source_refs:
            assert url.startswith("https://docs.aws.amazon.com/")
        
        # Verify execution-only fields are NOT faked
        assert "eventID" in ev.unknown_fields
        assert "requestID" in ev.unknown_fields
        assert "eventTime" in ev.unknown_fields
        assert "responseElements" in ev.unknown_fields


# 12. Graph node linkage and PROPOSED state verification
def test_graph_node_linkage_and_proposed_state(engine):
    orig = "Create an S3 bucket named telemetry-storage-prod"
    hard = "TASK: Create S3 bucket telemetry-storage-prod."
    out = engine.predict(orig, hard, "req-012")

    graph = out.action_dependency_graph
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

    # Verify all edges are PROPOSED
    for edge in graph.edges:
        assert edge.state == "PROPOSED"

    # Verify PredictedEvent node links to prediction_id
    pred_event_nodes = [n for n in graph.nodes if n.type == "PredictedEvent"]
    assert len(pred_event_nodes) == 1
    assert pred_event_nodes[0].properties["prediction_id"] == "pred-0001"


# 13. KMS Key creation with rotation
def test_kms_key_workflow_with_rotation(engine):
    orig = "Create a KMS customer managed key for encrypting RDS tables"
    hard = "TASK: Create KMS key. SECURITY REQUIREMENTS: Enable key rotation."
    out = engine.predict(orig, hard, "req-013")

    assert out.status == WorkflowStatus.GENERATED
    assert len(out.proposed_actions) == 2
    assert out.proposed_actions[0].operation == "CreateKey"
    assert out.proposed_actions[1].operation == "EnableKeyRotation"

    ev_names = [e.eventName for e in out.expected_direct_events]
    assert ev_names == ["CreateKey", "EnableKeyRotation"]
    assert out.expected_direct_events[0].eventSource == "kms.amazonaws.com"


# 14. EC2 Security Group workflow
def test_ec2_security_group_workflow(engine):
    orig = "Create an EC2 security group named web-dmz-sg in vpc-0123456789abcdef0"
    hard = "TASK: Create security group web-dmz-sg in vpc-0123456789abcdef0."
    out = engine.predict(orig, hard, "req-014")

    assert out.status == WorkflowStatus.GENERATED
    assert len(out.proposed_actions) == 1
    assert out.proposed_actions[0].operation == "CreateSecurityGroup"
    assert out.expected_direct_events[0].eventSource == "ec2.amazonaws.com"
    assert out.expected_direct_events[0].requestParameters["vpcId"] == "vpc-0123456789abcdef0"


# 15. Lambda Create Function workflow
def test_lambda_create_function_workflow(engine):
    orig = "Create a Lambda function named data-sync-worker with execution role arn:aws:iam::111122223333:role/lambda-role"
    hard = "TASK: Deploy Lambda function data-sync-worker."
    out = engine.predict(orig, hard, "req-015")

    assert out.status == WorkflowStatus.GENERATED
    assert len(out.proposed_actions) == 1
    assert out.proposed_actions[0].operation == "CreateFunction"
    # Verify exact CloudTrail event name for Lambda CreateFunction!
    assert out.expected_direct_events[0].eventName == "CreateFunction20150331"
    assert out.expected_direct_events[0].eventSource == "lambda.amazonaws.com"


# 16. Pipeline Adapter Integration test
def test_pipeline_adapter():
    try:
        from adapter import Stage3PipelineAdapter
    except ImportError:
        from stage3_intent_event_predictor.adapter import Stage3PipelineAdapter

    adapter = Stage3PipelineAdapter()
    orig = "Create an S3 bucket named customer-invoices-2026"
    hard = "TASK: Create S3 bucket. SECURITY CONSTRAINTS: Require encryption. Block all public access."

    adapted = adapter.process_stage2_output(
        original_prompt=orig,
        hardened_prompt=hard,
        principal_arn="arn:aws:iam::111122223333:user/agentic-operator",
        request_id="adapter-req-001"
    )

    assert "stage3_contract" in adapted
    assert "downstream_gnn_structural_rows" in adapted
    assert "downstream_lstm_records" in adapted
    assert "downstream_event_sequence" in adapted

    # GNN structural rows
    gnn_rows = adapted["downstream_gnn_structural_rows"]
    assert len(gnn_rows) == 3
    for row in gnn_rows:
        assert row["source_node"] == "arn:aws:iam::111122223333:user/agentic-operator"
        assert row["target_node"] == "arn:aws:s3:::customer-invoices-2026"
        assert row["state"] == "PROPOSED"

    # LSTM event sequence
    seq = adapted["downstream_event_sequence"]
    assert seq == ["CreateBucket", "PutBucketEncryption", "PutBucketPublicAccessBlock"]

    # CloudTrail JSON simulation records
    records = adapted["downstream_lstm_records"]["Records"]
    assert len(records) == 3
    assert records[0]["eventName"] == "CreateBucket"
    assert records[0]["record_type"] == "PREDICTED_CLOUDTRAIL_EVENT"

