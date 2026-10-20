"""
Stage 2 -> Stage 3 -> Stage 4 Integration Adapter.

Converts Stage 3 output into downstream formats required by:
1. Feature Engine & Neo4j Graph Builder (structural rows: log_id, source_node, target_node, edge_type, label)
2. Real-time Pipeline & LSTM Sequence Model (CloudTrail JSON Records stream)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from .engine import Stage3Engine
    from .schemas import Stage3Output
except (ImportError, ValueError):
    from engine import Stage3Engine
    from schemas import Stage3Output


class Stage3PipelineAdapter:
    def __init__(self, engine: Optional[Stage3Engine] = None):
        self.engine = engine or Stage3Engine()

    def process_stage2_output(
        self,
        original_prompt: str,
        hardened_prompt: str,
        principal_arn: str = "arn:aws:iam::111122223333:user/agentic-operator",
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes Stage 3 prediction and adapts outputs for Stage 4 consumption.
        """
        stage3_out: Stage3Output = self.engine.predict(
            original_prompt=original_prompt,
            hardened_prompt=hardened_prompt,
            request_id=request_id,
        )

        # 1. Structural rows for GNN / Neo4j (log_id, source_node, target_node, edge_type, label)
        structural_rows = []
        for ev in stage3_out.expected_direct_events:
            target_node = self._resolve_target_node(ev)
            structural_rows.append({
                "log_id": f"pred_stream:{ev.prediction_id}",
                "source_node": principal_arn,
                "target_node": target_node,
                "edge_type": ev.eventName,
                "label": 0,  # Unlabeled prediction (to be evaluated by Stage 4)
                "state": "PROPOSED",
            })

        # 2. Simulated CloudTrail records stream for FeatureEngine / LSTM
        cloudtrail_records = []
        for ev in stage3_out.expected_direct_events:
            record = {
                "eventVersion": "1.08",
                "userIdentity": {
                    "type": "IAMUser",
                    "arn": principal_arn,
                    "accountId": "111122223333",
                    "userName": principal_arn.split("/")[-1] if "/" in principal_arn else "agentic-operator"
                },
                "eventTime": "2026-10-18T12:00:00Z",  # Simulation baseline timestamp
                "eventSource": ev.eventSource,
                "eventName": ev.eventName,
                "awsRegion": "us-east-1",
                "requestParameters": ev.requestParameters,
                "responseElements": None,
                "readOnly": False,
                "eventType": "AwsApiCall",
                "managementEvent": True,
                "recipientAccountId": "111122223333",
                "eventCategory": "Management",
                "prediction_id": ev.prediction_id,
                "record_type": ev.record_type,
            }
            cloudtrail_records.append(record)

        # 3. Ordered event names for LSTM sequence tracker
        event_names_sequence = [ev.eventName for ev in stage3_out.expected_direct_events]

        return {
            "stage3_contract": stage3_out.model_dump(),
            "downstream_gnn_structural_rows": structural_rows,
            "downstream_lstm_records": {"Records": cloudtrail_records},
            "downstream_event_sequence": event_names_sequence,
        }

    def _resolve_target_node(self, event) -> str:
        params = event.requestParameters
        if event.service == "s3":
            bucket = params.get("bucketName")
            return f"arn:aws:s3:::{bucket}" if bucket else "arn:aws:s3:::unknown-bucket"
        elif event.service == "iam":
            role = params.get("roleName")
            return f"arn:aws:iam::111122223333:role/{role}" if role else "arn:aws:iam::111122223333:role/unknown"
        elif event.service == "kms":
            key_id = params.get("keyId")
            return f"arn:aws:kms:us-east-1:111122223333:key/{key_id}" if key_id else "arn:aws:kms:::proposed-cmk"
        elif event.service == "ec2":
            sg = params.get("groupName")
            return f"arn:aws:ec2:::security-group/{sg}" if sg else "arn:aws:ec2:::security-group/unknown"
        elif event.service == "lambda":
            fn = params.get("functionName")
            return f"arn:aws:lambda:us-east-1:111122223333:function:{fn}" if fn else "arn:aws:lambda:::function/unknown"
        return f"{event.service}.amazonaws.com"
