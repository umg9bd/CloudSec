# Stage 3: Evidence-Grounded Hybrid Intent Extraction and Exact Predicted CloudTrail Event Generation

## 1. Module Overview

This module implements **Stage 3** of the 6-stage agentic AWS cloud-security pipeline. It sits directly between **Stage 2 (Security Prompt Hardening)** and **Stage 4 (GNN / Temporal Sequence Privilege-Escalation Analysis)**.

### The Problem It Solves
When autonomous AI agents are tasked with managing AWS cloud infrastructure, assessing their security risks before execution requires simulating the exact audit trail they will produce. Traditional approaches that ask LLMs to generate predicted AWS API calls or CloudTrail logs suffer from severe hallucination:
- Fabricating non-existent API operations or event names.
- Outputting vague claims (e.g., *"some S3 logs will occur"*).
- Fabricating runtime identifiers (`eventID`, `requestID`, `eventTime`, `responseElements`).

### The Solution: Constrained Extraction + Deterministic Planning
Stage 3 enforces a **strict Zero-Hallucination Policy**:
1. **Constrained Extraction**: An LLM (or sub-millisecond regex fast-path) extracts *only* the user's high-level task category, explicit parameters, and upstream security constraints into a strict Pydantic schema. The LLM is strictly prohibited from generating API or event names.
2. **Deterministic Action Planning**: A local, versioned catalogue maps the extracted task category to verified AWS API operations and evaluates security constraints.
3. **Exhaustive Event Enumeration**: Exact, official CloudTrail `eventName` and `eventSource` records are enumerated one-by-one with official AWS documentation URLs and representative fixtures.
4. **Action Dependency Graph**: A multigraph with `PROPOSED` edge states is constructed, providing explicit linkages from proposed actions and target resources to predicted events for consumption by Stage 4 GNN and LSTM models.

---

## 2. Architecture & Pipeline Flow

```text
[Original Prompt + Hardened Prompt (Stage 2)]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Hybrid Intent Extractor                                  │
│    • Deterministic Fast-Path (<0.1ms, zero external calls)  │
│    • Constrained LLM Extractor (gemini-2.5-flash / Mock)    │
│    • Strict Pydantic Schema Validation & Sanitization       │
└───────────────────────────┬─────────────────────────────────┘
                            │ (ParsedIntent, SecurityConstraints)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Deterministic Action Planner                             │
│    • Local Action Catalogue Lookup (S3, IAM, KMS, EC2, Lambda)│
│    • Security Constraint Action Mapping (e.g., PutBucketEncryption)│
│    • Parameter Validation & Ambiguity Detection             │
│    • Topological Sort for Dependency Ordering              │
└───────────────────────────┬─────────────────────────────────┘
                            │ (ProposedActions, ConditionalEvents)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. CloudTrail Event Enumerator                              │
│    • Exhaustive Event-by-Event Generation                   │
│    • Traceable Provenance (Official AWS Docs & Fixtures)    │
│    • Non-empty unknown_fields (Zero Runtime Fabrication)   │
└───────────────────────────┬─────────────────────────────────┘
                            │ (ExpectedDirectEvents)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Action Dependency Graph Builder                          │
│    • Nodes: Task, ProposedAction, Service, Resource, Event  │
│    • Edges: PROPOSES, INVOKES, TARGETS, MAPS_TO_EVENT, ...   │
│    • All State: "PROPOSED" (Cleanly separated from observed)│
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 16-Field Output Contract & Stage 4 Pipeline Adapter      │
│    • Structural Rows (log_id, source, target, edge_type)    │
│    • CloudTrail JSON Simulation Stream (Records: [...])     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Layout

```text
stage3_intent_event_predictor/
├── __init__.py                     # Package export of engine and schemas
├── schemas.py                      # Strict Pydantic v2 schemas and 16-field Output Contract
├── loader.py                       # Thread-safe singleton catalogue loader and schema validator
├── extractor.py                    # Hybrid fast-path regex + Gemini/Mock LLM extractor + sanitization
├── planner.py                      # Deterministic action selector, dependency resolver & parameter validator
├── enumerator.py                   # CloudTrail event generator with documentation provenance
├── graph_builder.py                # Graph builder constructing PROPOSED state nodes and edges
├── engine.py                       # Main Stage3Engine orchestration entrypoint
├── adapter.py                      # Stage 2 -> Stage 3 -> Stage 4 pipeline adapter
├── benchmark.py                    # Micro-benchmarks for latency, memory, and concurrency
├── catalogue/
│   ├── aws_action_catalogue.json   # Validated catalogue with S3, IAM, KMS, EC2, Lambda mappings
│   └── fixtures/                   # Sanitized real-world CloudTrail JSON event fixtures
│       ├── s3_create_bucket.json
│       ├── s3_put_bucket_encryption.json
│       ├── s3_put_bucket_public_access_block.json
│       ├── iam_create_role.json
│       ├── iam_attach_role_policy.json
│       ├── kms_create_key.json
│       ├── kms_enable_key_rotation.json
│       ├── ec2_create_security_group.json
│       └── lambda_create_function.json
└── tests/
    └── test_stage3.py              # 16 comprehensive unit, integration, and anti-hallucination tests
```

---

## 4. The 16-Field Output Contract

Every response produced by `Stage3Engine.predict()` strictly satisfies the 16-field Pydantic schema:

| Field Key | Type | Description |
|---|---|---|
| `status` | `WorkflowStatus` | `"GENERATED"`, `"PARTIAL"`, `"NEEDS_CLARIFICATION"`, `"UNSUPPORTED"`, or `"EXTRACTION_FAILED"` |
| `request_id` | `str` | Unique request identifier |
| `parsed_intent` | `ParsedIntent` | Extracted task category, desired outcome, provided parameters, and provenance |
| `security_constraints` | `List[SecurityConstraint]` | Constraints extracted from Stage 2 prompt hardening |
| `proposed_actions` | `List[ProposedAction]` | Ordered list of planned AWS API operations with rationale and rule IDs |
| `expected_direct_events` | `List[PredictedCloudTrailEvent]` | Itemized list of predicted CloudTrail events with docs provenance |
| `conditional_events` | `List[ConditionalEvent]` | Separately itemized events with explicit condition strings and evidence |
| `action_dependency_graph` | `ActionDependencyGraph` | Graph containing nodes and edges in `PROPOSED` state |
| `unmodelled_effects` | `List[str]` | Asynchronous side-effects or requested actions outside catalogue scope |
| `prediction_coverage` | `PredictionCoverage` | Indicates whether all requested operations are completely modeled |
| `missing_parameters` | `List[str]` | Parameters required by the catalogue but absent in the request |
| `clarification_questions` | `List[str]` | Actionable questions to disambiguate the request |
| `assumptions` | `List[str]` | Explicit assumptions made during planning |
| `source_provenance` | `List[str]` | Deduplicated official AWS documentation URLs |
| `catalogue_version` | `str` | Version of the local action catalogue (e.g. `1.2.0`) |
| `intent_model_version` | `str` | Version string of the extraction engine |

---

## 5. Official AWS Documentation Provenance

Every catalogue mapping is grounded in official AWS documentation and representative fixtures:

* **S3 `CreateBucket`**: [Amazon S3 CreateBucket API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CreateBucket.html)
* **S3 `PutBucketEncryption`**: [Amazon S3 PutBucketEncryption API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutBucketEncryption.html)
* **S3 `PutPublicAccessBlock`** *(CloudTrail eventName: `PutBucketPublicAccessBlock`)*: [Amazon S3 PutPublicAccessBlock API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutPublicAccessBlock.html)
* **S3 `PutBucketVersioning`**: [Amazon S3 PutBucketVersioning API](https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutBucketVersioning.html)
* **IAM `CreateRole`**: [AWS IAM CreateRole API](https://docs.aws.amazon.com/IAM/latest/APIReference/API_CreateRole.html)
* **IAM `AttachRolePolicy`**: [AWS IAM AttachRolePolicy API](https://docs.aws.amazon.com/IAM/latest/APIReference/API_AttachRolePolicy.html)
* **IAM `PutRolePolicy`**: [AWS IAM PutRolePolicy API](https://docs.aws.amazon.com/IAM/latest/APIReference/API_PutRolePolicy.html)
* **KMS `CreateKey`**: [AWS KMS CreateKey API](https://docs.aws.amazon.com/kms/latest/APIReference/API_CreateKey.html)
* **KMS `EnableKeyRotation`**: [AWS KMS EnableKeyRotation API](https://docs.aws.amazon.com/kms/latest/APIReference/API_EnableKeyRotation.html)
* **EC2 `CreateSecurityGroup`**: [Amazon EC2 CreateSecurityGroup API](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateSecurityGroup.html)
* **Lambda `CreateFunction`** *(CloudTrail eventName: `CreateFunction20150331`)*: [AWS Lambda CreateFunction API](https://docs.aws.amazon.com/lambda/latest/dg/API_CreateFunction.html)

---

## 6. Downstream Integration Adapter (Stage 4 Bridge)

To connect Stage 3 with the existing repository without modifying pre-existing files, use `Stage3PipelineAdapter`:

```python
from stage3_intent_event_predictor.adapter import Stage3PipelineAdapter

adapter = Stage3PipelineAdapter()

adapted = adapter.process_stage2_output(
    original_prompt="Create an S3 bucket named telemetry-data-lake",
    hardened_prompt="TASK: Create S3 bucket. SECURITY CONSTRAINTS: Require encryption. Block all public access.",
    principal_arn="arn:aws:iam::111122223333:user/deployer-bot",
)

# 1. Feed into FeatureEngine / Neo4j Graph Builder
structural_rows = adapted["downstream_gnn_structural_rows"]
# Example row: {'log_id': 'pred_stream:pred-0001', 'source_node': '...', 'target_node': 'arn:aws:s3:::telemetry-data-lake', 'edge_type': 'CreateBucket', 'label': 0, 'state': 'PROPOSED'}

# 2. Feed into real-time pipeline streaming input
cloudtrail_records = adapted["downstream_lstm_records"]

# 3. Feed into LSTM / Transformer sequence tracker
event_sequence = adapted["downstream_event_sequence"]
# ['CreateBucket', 'PutBucketEncryption', 'PutBucketPublicAccessBlock']
```

---

## 7. Performance & Benchmarking

Measured on local hardware (Intel/AMD x86_64, Windows, Python 3.13):

* **Cold Start Latency**: `7.81 ms`
* **Warm End-to-End Latency**:
  - **p50**: `0.110 ms` (110 microseconds)
  - **p95**: `0.299 ms` (299 microseconds)
  - **p99**: `0.460 ms`
* **Stage Latency Breakdown (Mean)**:
  - Intent Extraction (Fast-Path): `0.0323 ms`
  - Deterministic Planning: `0.0220 ms`
  - Event Enumeration: `0.0275 ms`
  - Graph Construction: `0.0528 ms`
* **Throughput**: `3,012.86 requests/second` (10 concurrent worker threads)
* **Peak Memory**: `203.77 KB` (cold start), `83.67 KB` (warm operation)
