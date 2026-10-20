"""
Hybrid Intent Extractor: Deterministic Fast-Path + Constrained LLM Extractor + Mock Adapter.

Strictly extracts user intent and parameters into a validated Pydantic schema without allowing
the LLM to invent AWS API operations or CloudTrail event records.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

try:
    from .loader import get_catalogue
    from .schemas import (
        ParsedIntent,
        SecurityConstraint,
        WorkflowStatus,
    )
except (ImportError, ValueError):
    from loader import get_catalogue
    from schemas import (
        ParsedIntent,
        SecurityConstraint,
        WorkflowStatus,
    )

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "for", "to", "with", "in", "from", "and", "or", "is", "a", "an", "the",
    "that", "this", "storing", "pictures", "assets", "files", "data", "reports", "my"
}


class ExtractorException(Exception):
    """Base exception for intent extraction failures."""
    def __init__(self, message: str, status: WorkflowStatus = WorkflowStatus.EXTRACTION_FAILED):
        super().__init__(message)
        self.status = status


class TimeoutException(ExtractorException):
    pass


class LLMProvider:
    """Base interface for LLM extraction providers."""
    def extract_intent_raw(self, original_prompt: str, hardened_prompt: str) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for deterministic testing of edge cases:
    malformed JSON, timeouts, hallucinated operations, unknown tasks, etc.
    """
    def __init__(self, mode: str = "valid", custom_payload: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.custom_payload = custom_payload

    def extract_intent_raw(self, original_prompt: str, hardened_prompt: str) -> str:
        if self.mode == "timeout":
            raise TimeoutException("Model call timed out after 5000ms", status=WorkflowStatus.EXTRACTION_FAILED)
        if self.mode == "malformed_json":
            return "{ task_category: 's3_create_bucket', INVALID_JSON ..."
        if self.mode == "hallucinated_injection":
            return json.dumps({
                "task_category": "s3_create_bucket",
                "desired_outcome": "Create bucket and run fake API",
                "injected_api_operation": "FakeAWSDoEverything",
                "injected_cloudtrail_event": "FakeEventNameLogged",
                "provided_parameters": {"bucketName": "my-secure-bucket"},
                "explicit_requirements": ["create bucket"],
                "missing_parameters": [],
                "ambiguities": [],
                "needs_clarification": False,
                "clarification_questions": []
            })
        if self.mode == "unsupported_task":
            return json.dumps({
                "task_category": "unsupported_quantum_ledger_blockchain",
                "desired_outcome": "Deploy hyperledger quantum node",
                "provided_parameters": {},
                "explicit_requirements": [],
                "missing_parameters": [],
                "ambiguities": [],
                "needs_clarification": False,
                "clarification_questions": []
            })
        if self.mode == "custom" and self.custom_payload is not None:
            return json.dumps(self.custom_payload)

        # Default valid response
        return json.dumps({
            "task_category": "s3_create_bucket",
            "desired_outcome": "Create an S3 bucket",
            "provided_parameters": {"bucketName": "mock-bucket-123"},
            "explicit_requirements": ["create bucket"],
            "missing_parameters": [],
            "ambiguities": [],
            "needs_clarification": False,
            "clarification_questions": []
        })


class GoogleGenAIProvider(LLMProvider):
    """
    Google Gemini API provider using official google-genai SDK.
    Uses gemini-2.5-flash with structured JSON output and temperature=0.0.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def extract_intent_raw(self, original_prompt: str, hardened_prompt: str) -> str:
        if not self.api_key:
            raise ExtractorException("GEMINI_API_KEY not configured", status=WorkflowStatus.EXTRACTION_FAILED)

        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)
            system_instruction = (
                "You are an intent extraction parser. Your task is to extract user intent "
                "into a strict JSON object. NEVER generate AWS API operation names or CloudTrail "
                "event names. Only extract task_category, desired_outcome, explicit_requirements, "
                "provided_parameters, missing_parameters, ambiguities, needs_clarification, "
                "and clarification_questions."
            )
            prompt = (
                f"Original Prompt: {original_prompt}\n\n"
                f"Security Hardened Prompt: {hardened_prompt}\n\n"
                "Extract the structured intent JSON:"
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.0,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"GoogleGenAIProvider failed: {e}")
            raise ExtractorException(f"LLM extraction failed: {str(e)}", status=WorkflowStatus.EXTRACTION_FAILED)


class DeterministicFastPath:
    """
    Sub-millisecond regex/keyword rule engine for unambiguous supported requests.
    Zero external calls, zero hallucination risk.
    """
    def __init__(self):
        self.catalogue = get_catalogue()

    def try_extract(
        self,
        original_prompt: str,
        hardened_prompt: str
    ) -> Optional[Tuple[ParsedIntent, List[SecurityConstraint]]]:
        orig = original_prompt.strip()
        hard = hardened_prompt.strip()
        combined = f"{orig} {hard}".lower()

        # Extract Security Constraints from hardened prompt
        security_constraints = self._extract_security_constraints(hardened_prompt)

        # Check for explicit unsupported tasks
        unsupported_keywords = [
            "neptune", "quantum", "blockchain", "macie", "lightsail",
            "honeycode", "chime", "rekognition video face stream"
        ]
        for kw in unsupported_keywords:
            if kw in combined:
                # If unsupported keyword found alone, fast-path can return unsupported
                if not any(sup in combined for sup in ["s3", "bucket", "iam", "role", "kms", "security group", "lambda"]):
                    return ParsedIntent(
                        task_category="unsupported",
                        desired_outcome=orig,
                        explicit_requirements=[orig],
                        provided_parameters={},
                        missing_parameters=[],
                        ambiguities=[],
                        needs_clarification=False,
                        clarification_questions=[]
                    ), security_constraints

        # 1. Lambda Create Function (check before IAM role to prevent matching execution role in Lambda creation)
        if re.search(r"\b(lambda|function)\b", combined) and ("create" in combined or "deploy" in combined):
            fn_name_match = re.search(r"(?:function\s+(?:named|name|called)?\s*[:=]?\s*['\"]?([a-zA-Z0-9-_]{1,64})['\"]?)", orig, re.IGNORECASE)
            role_match = re.search(r"(arn:aws:iam::[0-9]{12}:role/[a-zA-Z0-9+=,.@-]{1,64})", orig)

            provided_params = {
                "handler": "index.handler",
                "runtime": "python3.11"
            }
            param_prov = {
                "handler": "DEFAULT",
                "runtime": "DEFAULT"
            }
            missing_params = []

            if fn_name_match and fn_name_match.group(1).lower() not in STOP_WORDS:
                provided_params["functionName"] = fn_name_match.group(1)
                param_prov["functionName"] = "USER_PROMPT"
            else:
                missing_params.append("functionName")

            if role_match:
                provided_params["role"] = role_match.group(1)
                param_prov["role"] = "USER_PROMPT"
            else:
                missing_params.append("role")

            needs_clarification = len(missing_params) > 0
            questions = []
            if "functionName" in missing_params:
                questions.append("What is the function name for the Lambda?")
            if "role" in missing_params:
                questions.append("What is the IAM execution role ARN for the Lambda function?")

            return ParsedIntent(
                task_category="lambda_create_function",
                desired_outcome=orig,
                explicit_requirements=["Create Lambda serverless function"],
                provided_parameters=provided_params,
                parameter_provenance=param_prov,
                missing_parameters=missing_params,
                ambiguities=[],
                needs_clarification=needs_clarification,
                clarification_questions=questions,
            ), security_constraints

        # 2. EC2 Security Group (check before IAM role)
        if re.search(r"\b(security\s*group|secgroup)\b", combined) and ("create" in combined or "provision" in combined):
            sg_name_match = re.search(r"(?:security\s*group\s+(?:named|name|called)?\s*[:=]?\s*['\"]?([a-zA-Z0-9-_]{1,64})['\"]?)", orig, re.IGNORECASE)
            vpc_match = re.search(r"(vpc-[0-9a-fA-F]{8,17})", orig)

            provided_params = {
                "description": orig
            }
            param_prov = {
                "description": "USER_PROMPT"
            }
            missing_params = []

            if sg_name_match and sg_name_match.group(1).lower() not in STOP_WORDS:
                provided_params["groupName"] = sg_name_match.group(1)
                param_prov["groupName"] = "USER_PROMPT"
            else:
                missing_params.append("groupName")

            if vpc_match:
                provided_params["vpcId"] = vpc_match.group(1)
                param_prov["vpcId"] = "USER_PROMPT"
            else:
                missing_params.append("vpcId")

            needs_clarification = len(missing_params) > 0
            questions = []
            if "groupName" in missing_params:
                questions.append("What is the name for the new Security Group?")
            if "vpcId" in missing_params:
                questions.append("Which VPC ID (e.g., vpc-xxxx) should this Security Group belong to?")

            return ParsedIntent(
                task_category="ec2_create_security_group",
                desired_outcome=orig,
                explicit_requirements=["Create EC2 VPC Security Group"],
                provided_parameters=provided_params,
                parameter_provenance=param_prov,
                missing_parameters=missing_params,
                ambiguities=[],
                needs_clarification=needs_clarification,
                clarification_questions=questions,
            ), security_constraints

        # 3. IAM Create Role (check before S3 create bucket so "role with s3 access" doesn't become s3_create_bucket)
        if re.search(r"\b(iam\s+role|role)\b", combined) and ("create" in combined or "provision" in combined or "grant" in combined):
            match = re.search(r"(?:role\s+(?:named|name|called)?\s*[:=]?\s*['\"]?([a-zA-Z0-9+=,.@-]{1,64})['\"]?)", orig, re.IGNORECASE)
            role_name = None
            if match and match.group(1).lower() not in STOP_WORDS:
                role_name = match.group(1)

            provided_params = {}
            missing_params = []
            param_prov = {}

            if role_name:
                provided_params["roleName"] = role_name
                param_prov["roleName"] = "USER_PROMPT"
            else:
                missing_params.append("roleName")

            # Check trust policy / assumeRolePolicyDocument
            if "lambda" in combined:
                provided_params["assumeRolePolicyDocument"] = {
                    "Version": "2012-10-17",
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]
                }
                param_prov["assumeRolePolicyDocument"] = "DEFAULT"
            else:
                provided_params["assumeRolePolicyDocument"] = {
                    "Version": "2012-10-17",
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
                }
                param_prov["assumeRolePolicyDocument"] = "DEFAULT"

            ambiguities = []
            clarification_questions = []
            needs_clarification = False

            if missing_params:
                needs_clarification = True
                clarification_questions.append("What is the name of the IAM role to create?")

            # Ambiguity: Did the user ask to attach a managed policy or create an inline policy?
            if "read" in combined or "policy" in combined or "access" in combined:
                if not re.search(r"arn:aws:iam::[0-9a-zA-Z-]*:policy/[a-zA-Z0-9+=,.@-]{1,128}", orig):
                    ambiguities.append("Policy attachment method ambiguous: managed policy ARN not specified vs inline PutRolePolicy.")
                    needs_clarification = True
                    clarification_questions.append("Should permissions be granted via managed policy ARN (AttachRolePolicy) or inline policy document (PutRolePolicy)?")

            return ParsedIntent(
                task_category="iam_create_role",
                desired_outcome=orig,
                explicit_requirements=["Create IAM execution role"],
                provided_parameters=provided_params,
                parameter_provenance=param_prov,
                missing_parameters=missing_params,
                ambiguities=ambiguities,
                needs_clarification=needs_clarification,
                clarification_questions=clarification_questions,
            ), security_constraints

        # 4. KMS Create Key
        if re.search(r"\b(kms|key)\b", combined) and ("create" in combined or "provision" in combined or "rotation" in combined):
            provided_params = {
                "description": orig,
                "keyUsage": "ENCRYPT_DECRYPT"
            }
            param_prov = {
                "description": "USER_PROMPT",
                "keyUsage": "DEFAULT"
            }
            return ParsedIntent(
                task_category="kms_create_key",
                desired_outcome=orig,
                explicit_requirements=["Create KMS customer managed key"],
                provided_parameters=provided_params,
                parameter_provenance=param_prov,
                missing_parameters=[],
                ambiguities=[],
                needs_clarification=False,
                clarification_questions=[],
            ), security_constraints

        # 5. S3 Create Bucket
        if re.search(r"\b(s3|bucket)\b", combined) and ("create" in combined or "host" in combined or "provision" in combined):
            # Explicit naming pattern
            match = re.search(r"(?:bucket\s+(?:named|name|called)\s*[:=]?\s*['\"]?([a-z0-9][a-z0-9.-]{2,62})['\"]?)", orig, re.IGNORECASE)
            bucket_name = None
            if match and match.group(1).lower() not in STOP_WORDS:
                bucket_name = match.group(1).lower()
            else:
                # Quoted pattern: bucket 'my-bucket'
                match_quoted = re.search(r"bucket\s+['\"]([a-z0-9][a-z0-9.-]{2,62})['\"]", orig, re.IGNORECASE)
                if match_quoted and match_quoted.group(1).lower() not in STOP_WORDS:
                    bucket_name = match_quoted.group(1).lower()
                else:
                    # Specific pattern: "create s3 bucket <name>"
                    match2 = re.search(r"create\s+(?:an?\s+)?s3\s+bucket\s+['\"]?([a-z0-9][a-z0-9.-]{2,62})['\"]?", orig, re.IGNORECASE)
                    if match2 and match2.group(1).lower() not in STOP_WORDS:
                        bucket_name = match2.group(1).lower()

            provided_params = {}
            missing_params = []
            param_prov = {}

            if bucket_name:
                provided_params["bucketName"] = bucket_name
                param_prov["bucketName"] = "USER_PROMPT"
            else:
                missing_params.append("bucketName")

            ambiguities = []
            clarification_questions = []
            needs_clarification = False

            if missing_params:
                needs_clarification = True
                clarification_questions.append("What is the desired name for the S3 bucket?")

            # Check for conflict between prompt and security requirements
            if "public" in orig.lower() and any("block" in c.description.lower() for c in security_constraints):
                ambiguities.append("Conflicting requirements: User requested public access, but security policy requires blocking public access.")
                needs_clarification = True
                clarification_questions.append("User requested public access, but organizational policy enforces PutPublicAccessBlock. Confirm if public access is strictly authorized.")

            return ParsedIntent(
                task_category="s3_create_bucket",
                desired_outcome=orig,
                explicit_requirements=["Create S3 bucket"],
                provided_parameters=provided_params,
                parameter_provenance=param_prov,
                missing_parameters=missing_params,
                ambiguities=ambiguities,
                needs_clarification=needs_clarification,
                clarification_questions=clarification_questions,
            ), security_constraints

        return None

    def _extract_security_constraints(self, hardened_prompt: str) -> List[SecurityConstraint]:
        constraints = []
        lower = hardened_prompt.lower()

        if "encryption" in lower or "encrypt" in lower or "kms" in lower:
            constraints.append(SecurityConstraint(
                constraint_id="SEC-001",
                description="Require server-side encryption at rest",
                triggered_action_ids=["act-s3-put-bucket-encryption"],
                source="STAGE_2_HARDENING"
            ))

        if re.search(r"block\s+(?:all\s+)?public\s*access", lower) or "publicaccessblock" in lower or "private" in lower:
            constraints.append(SecurityConstraint(
                constraint_id="SEC-002",
                description="Require PutBucketPublicAccessBlock on S3 bucket",
                triggered_action_ids=["act-s3-put-public-access-block"],
                source="STAGE_2_HARDENING"
            ))

        if "versioning" in lower:
            constraints.append(SecurityConstraint(
                constraint_id="SEC-003",
                description="Require S3 bucket versioning enabled",
                triggered_action_ids=["act-s3-put-bucket-versioning"],
                source="STAGE_2_HARDENING"
            ))

        if "rotation" in lower or "key rotation" in lower:
            constraints.append(SecurityConstraint(
                constraint_id="SEC-004",
                description="Require KMS customer managed key rotation enabled",
                triggered_action_ids=["act-kms-enable-key-rotation"],
                source="STAGE_2_HARDENING"
            ))

        return constraints


class HybridIntentExtractor:
    """
    Main extraction manager:
    1. Runs fast path first (<1ms)
    2. Falls back to LLM provider (or Mock) if fast path cannot classify
    3. Strictly schema-validates the output in ordinary Python/Pydantic
    4. Strips any hallucinated API/event names injected by the LLM
    """
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.fast_path = DeterministicFastPath()
        self.catalogue = get_catalogue()
        self.llm_provider = llm_provider or MockLLMProvider()

    def extract(
        self,
        original_prompt: str,
        hardened_prompt: str
    ) -> Tuple[ParsedIntent, List[SecurityConstraint], str]:
        """
        Returns (parsed_intent, security_constraints, model_version_string)
        """
        # 1. Deterministic Fast-Path
        fast_result = self.fast_path.try_extract(original_prompt, hardened_prompt)
        if fast_result is not None:
            intent, constraints = fast_result
            return intent, constraints, "fast-path-regex-v1"

        # 2. LLM Extractor
        raw_output = self.llm_provider.extract_intent_raw(original_prompt, hardened_prompt)
        
        # 3. Schema validation and anti-hallucination guard
        intent, constraints = self._validate_and_sanitize(raw_output, original_prompt, hardened_prompt)
        return intent, constraints, f"llm-{self.llm_provider.__class__.__name__}"

    def _validate_and_sanitize(
        self,
        raw_output: str,
        original_prompt: str,
        hardened_prompt: str
    ) -> Tuple[ParsedIntent, List[SecurityConstraint]]:
        try:
            data = json.loads(raw_output)
        except Exception as e:
            raise ExtractorException(f"LLM returned malformed JSON: {e}", status=WorkflowStatus.EXTRACTION_FAILED)

        if not isinstance(data, dict):
            raise ExtractorException("LLM output is not a JSON dictionary", status=WorkflowStatus.EXTRACTION_FAILED)

        # Anti-hallucination guard: strip any injected API or CloudTrail names
        data.pop("injected_api_operation", None)
        data.pop("injected_cloudtrail_event", None)
        data.pop("api_operation", None)
        data.pop("eventName", None)
        data.pop("eventSource", None)

        task_cat = data.get("task_category")
        if not task_cat:
            raise ExtractorException("Missing task_category in LLM response", status=WorkflowStatus.EXTRACTION_FAILED)

        # Check if task is recognized by our catalogue
        if task_cat != "unsupported" and not self.catalogue.is_task_supported(task_cat):
            # Unknown category is marked unsupported
            data["task_category"] = "unsupported"

        # Parameter provenance
        param_prov = data.get("parameter_provenance", {})
        for p in data.get("provided_parameters", {}):
            if p not in param_prov:
                param_prov[p] = "USER_PROMPT"
        data["parameter_provenance"] = param_prov

        # Extract security constraints from hardened prompt
        constraints = self.fast_path._extract_security_constraints(hardened_prompt)

        try:
            intent = ParsedIntent(**data)
            return intent, constraints
        except ValidationError as ve:
            raise ExtractorException(f"Schema validation failed: {ve}", status=WorkflowStatus.EXTRACTION_FAILED)
