import json
import logging
import ipaddress
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Set up logging for production visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Generates structural data for Graph Neural Networks (GNNs) and 
    temporal feature vectors for Sequential Models (LSTM/GRU).
    """

    def __init__(self):
        # Action Map: Normalized risk weights (1-10) based on AWS security benchmarks
        self.action_map = {
            "GetCallerIdentity": 1, "ListBuckets": 2, "DescribeInstances": 2,
            "CreateUser": 5, "CreateRole": 5, "CreateAccessKey": 6,
            "PutUserPolicy": 9, "AttachUserPolicy": 9, "UpdateAssumeRolePolicy": 9,
            "DeleteTrail": 10, "StopLogging": 10
        }
        
        # Principal risk tiers for ordinal encoding
        self.p_type_map = {
            "IAMUser": 0.2, 
            "AssumedRole": 0.6, 
            "Root": 1.0, 
            "FederatedUser": 0.4
        }
        
        self.default_action_weight = 0.0
        self.primary_region = "us-east-1"

    def _validate_log(self, log: Dict[str, Any]) -> bool:
        required_fields = ['event_name', 'timestamp', 'principal_type']
        for field in required_fields:
            if field not in log or log[field] is None:
                logger.warning(f"Missing critical field '{field}' in log. Skipping.")
                return False
        return True

    def get_structural_data(self, log: Dict[str, Any]) -> Dict[str, str]:
        """
        Extracts Nodes and Edges for GNN graph construction.
        Maps the relationship: Principal (Actor) -> Action -> Resource (Target).
        """
        if not self._validate_log(log):
            return {}

        target = log.get('target_resource') or log.get('event_source', 'unknown_service')
        
        return {
            "source_node": log.get('principal_arn', 'unknown_principal'),
            "target_node": target,
            "edge_type": log.get('event_name', 'unknown_action'),
            "is_error": "true" if log.get('error_code') else "false"
        }

    def _get_identity_features(self, log: Dict[str, Any]) -> List[float]:
        """PILLAR 1: IDENTITY & AUTH (5 features)"""
        f = []
        # 1. MFA Missing (1 = Risk, 0 = Safe)
        mfa = str(log.get('mfa_authenticated', '')).lower()
        f.append(1.0 if mfa in ['false', 'no', ''] else 0.0)
        
        # 2. Is Root User
        f.append(1.0 if log.get('principal_type') == "Root" else 0.0)
        
        # 3. Is AssumedRole (Lateral movement proxy)
        f.append(1.0 if log.get('principal_type') == "AssumedRole" else 0.0)
        
        # 4. Access Key Presence
        f.append(1.0 if log.get('access_key_id') else 0.0)
        
        # 5. Principal Risk Ordinal
        f.append(self.p_type_map.get(log.get('principal_type', ''), 0.0))
        return f

    def _get_temporal_features(self, log: Dict[str, Any]) -> List[float]:
        """PILLAR 2: LEARNED TEMPORAL CONTEXT (4 features)"""
        f = []
        try:
            ts = log.get('timestamp')
            # Handle varied timestamp formats if necessary
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S%z")
            
            # 6-7. Circular Hour Encoding (Sine/Cosine)
            hour_norm = 2 * np.pi * dt.hour / 24.0
            f.append(float(np.sin(hour_norm)))
            f.append(float(np.cos(hour_norm)))
            
            # 8. Weekend Indicator
            f.append(1.0 if dt.weekday() >= 5 else 0.0)
            
            # 9. Business Hour Proxy (9-5)
            f.append(0.0 if 9 <= dt.hour <= 17 else 1.0)
        except (ValueError, TypeError) as e:
            logger.error(f"Timestamp parsing error: {e}")
            f.extend([0.0, 0.0, 0.0, 0.0])
        return f

    def _get_action_intent_features(self, log: Dict[str, Any]) -> List[float]:
        """PILLAR 3: ACTION & INTENT RISK (8 features)"""
        f = []
        event_name = log.get('event_name', '')
        
        # 10. Event Action Weight (Normalized)
        weight = self.action_map.get(event_name, 0) / 10.0
        f.append(float(weight))
        
        # 11. Is Write Action
        f.append(1.0 if str(log.get('read_only')).lower() == 'false' else 0.0)
        
        # 12. Has Error
        f.append(1.0 if log.get('error_code') else 0.0)
        
        # 13. Is Access Denied
        f.append(1.0 if log.get('error_code') == "AccessDenied" else 0.0)
        
        # 14. Is Identity Service
        is_id = 1.0 if log.get('event_source') in ["iam.amazonaws.com", "sts.amazonaws.com"] else 0.0
        f.append(is_id)
        
        # 15. Recon/Discovery Flag
        recon = 1.0 if any(event_name.startswith(x) for x in ['Describe', 'List', 'Get']) else 0.0
        f.append(recon)
        
        # 16. Defense Evasion Flag
        evasion = 1.0 if event_name in ["DeleteTrail", "StopLogging"] else 0.0
        f.append(evasion)
        
        # 17. Caller ID Recon
        f.append(1.0 if event_name == "GetCallerIdentity" else 0.0)
        return f

    def _get_metadata_features(self, log: Dict[str, Any]) -> List[float]:
        """PILLAR 4: METADATA & CONTEXT (8 features)"""
        f = []
        
        # 18. Scripted Agent (CLI/Automation Detection)
        ua = str(log.get('user_agent', '')).lower()
        automation_tools = ['aws-cli', 'boto', 'python', 'terraform', 'cloudformation']
        f.append(1.0 if any(x in ua for x in automation_tools) else 0.0)
        
        # 19. Public IP Check
        ip = str(log.get('source_ip', ''))
        try:
            f.append(1.0 if not ipaddress.ip_address(ip).is_private else 0.0)
        except ValueError:
            f.append(1.0) # Assume risk on malformed/unknown IP
            
        # 20. Parameter Complexity
        params_str = str(log.get('request_params_raw', '{}'))
        f.append(min(len(params_str) / 1000.0, 1.0))
        
        # 21. Identity Resource Target
        target_res = str(log.get('target_resource') or '').lower()
        f.append(1.0 if any(x in target_res or x in params_str.lower() for x in ['policy', 'role', 'user']) else 0.0)
        
        # 22. Region Mismatch
        f.append(1.0 if log.get('aws_region') != self.primary_region else 0.0)
        
        # 23. Persistence Signal
        persist = 1.0 if log.get('event_name') in ["CreateUser", "CreateAccessKey", "CreateRole"] else 0.0
        f.append(persist)
        
        # 24. Credential Access Signal
        cred_src = ["secrets", "kms", "ssm"]
        f.append(1.0 if any(x in log.get('event_source', '') for x in cred_src) else 0.0)
        
        # 25. Escalation Pivot
        escalate = 1.0 if log.get('event_name') in ["AttachUserPolicy", "PutUserPolicy"] else 0.0
        f.append(escalate)
        
        return f

    def extract_sequence_features(self, log: Dict[str, Any]) -> np.ndarray:
        """
        Orchestrates all pillars into a single 25-feature vector.
        Returns a NumPy array for direct model consumption.
        """
        if not self._validate_log(log):
            return np.zeros(25)

        # Concatenate all feature pillars
        vector = (
            self._get_identity_features(log) +
            self._get_temporal_features(log) +
            self._get_action_intent_features(log) +
            self._get_metadata_features(log)
        )
        
        # Ensure strict length of 25
        if len(vector) != 25:
            logger.error(f"Feature mismatch: Expected 25, got {len(vector)}")
            return np.zeros(25)
            
        return np.array(vector, dtype=np.float32)

    def process_batch(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        """Processes a list of logs and returns a feature matrix (N, 25)."""
        logger.info(f"Processing batch of {len(logs)} logs...")
        matrix = [self.extract_sequence_features(log) for log in logs]
        return np.vstack(matrix)

# --- EXAMPLE USAGE & MOCK DATA ---
if __name__ == "__main__":
    engineer = FeatureEngineer()

    # Mock CloudTrail Event
    sample_log = {
        "timestamp": "2026-03-31 23:15:00+0000",
        "event_name": "StopLogging",
        "event_source": "cloudtrail.amazonaws.com",
        "principal_type": "IAMUser",
        "principal_arn": "arn:aws:iam::123456789012:user/attacker",
        "source_ip": "192.168.1.1", # Private IP
        "user_agent": "aws-cli/2.0.0 Python/3.8.5",
        "read_only": "false",
        "aws_region": "us-west-2",
        "mfa_authenticated": "false",
        "request_params_raw": '{"trailName": "management-events"}'
    }

    # 1. Structural Data (GNN)
    graph_data = engineer.get_structural_data(sample_log)
    print("\n--- GNN STRUCTURAL DATA ---")
    print(json.dumps(graph_data, indent=4))

    # 2. Sequence Features (LSTM/GRU)
    feature_vector = engineer.extract_sequence_features(sample_log)
    print("\n--- SEQUENCE FEATURE VECTOR (25) ---")
    print(feature_vector)
    print(f"Shape: {feature_vector.shape}")
