import json
import numpy as np
import ipaddress
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        # Action Map: Maps API verbs to risk weights based on AWS User Guide
        self.action_map = {
            "GetCallerIdentity": 1, "ListBuckets": 2, "DescribeInstances": 2,
            "CreateUser": 5, "CreateRole": 5, "CreateAccessKey": 6,
            "PutUserPolicy": 9, "AttachUserPolicy": 9, "UpdateAssumeRolePolicy": 9,
            "DeleteTrail": 10, "StopLogging": 10
        }
        self.default_action = 0

    def get_structural_data(self, log):
        """Extracts Nodes and Edges for GNN (Akshaya's Task)"""
        # Uses target_resource from CSV; falls back to service if empty
        target = log.get('target_resource') or 'aws_internal_service'
        
        return {
            "source_node": log.get('principal_arn', 'unknown_principal'),
            "target_node": target,
            "edge_type": log.get('event_name', 'unknown_action')
        }

    def get_temporal_features(self, log):
        """Generates the optimized 25-feature vector for Nandan's Sequence Model"""
        features = []
        
        # --- PILLAR 1: IDENTITY & AUTH (5 features) ---
        # 1. MFA Missing (1 = High Risk/Missing, 0 = Used)
        mfa_val = str(log.get('mfa_authenticated', '')).lower()
        features.append(1 if mfa_val in ['false', 'no', ''] else 0)
        # 2. Is Root User (Immediate 1.0 signal)
        features.append(1 if log.get('principal_type') == "Root" else 0)
        # 3. Is AssumedRole (Lateral movement proxy)
        features.append(1 if log.get('principal_type') == "AssumedRole" else 0)
        # 4. Access Key Presence (1 = Key, 0 = Session/Password)
        features.append(1 if log.get('access_key_id') else 0)
        # 5. Principal Risk Ordinal (Ranked identity power)
        p_type_map = {"IAMUser": 0.2, "AssumedRole": 0.6, "Root": 1.0}
        features.append(p_type_map.get(log.get('principal_type'), 0.0))

        # --- PILLAR 2: LEARNED TEMPORAL CONTEXT (4 features) ---
        # 6-7. Circular Hour Encoding (Sine/Cosine) 
        # Prevents "23:00" and "00:00" from appearing far apart in math
        dt = datetime.strptime(log.get('timestamp'), "%Y-%m-%d %H:%M:%S%z")
        hour_norm = 2 * np.pi * dt.hour / 24.0
        features.append(np.sin(hour_norm)) # Feature 6
        features.append(np.cos(hour_norm)) # Feature 7
        
        # 8. Weekend Indicator (1 = Sat/Sun, 0 = Weekday)
        features.append(1 if dt.weekday() >= 5 else 0)
        # 9. Business Hour Proxy (1 = Outside 9-5, 0 = Inside)
        features.append(0 if 9 <= dt.hour <= 17 else 1)

        # --- PILLAR 3: ACTION & INTENT RISK (8 features) ---
        # 10. Event Action Weight (Normalized 0-1)
        features.append(self.action_map.get(log.get('event_name'), self.default_action) / 10.0)
        # 11. Is Write Action (Mutates permissions/state)
        features.append(1 if str(log.get('read_only')).lower() == 'false' else 0)
        # 12. Has Error (Anomaly indicator)
        features.append(1 if log.get('error_code') else 0)
        # 13. Is Access Denied (Direct 'testing the fences' signal)
        features.append(1 if log.get('error_code') == "AccessDenied" else 0)
        # 14. Is Identity Service (IAM/STS = 1, Other = 0)
        features.append(1 if log.get('event_source') in ["iam.amazonaws.com", "sts.amazonaws.com"] else 0)
        # 15. Recon/Discovery Flag (Describe/List/Get)
        features.append(1 if any(log.get('event_name', '').startswith(x) for x in ['Describe', 'List', 'Get']) else 0)
        # 16. Defense Evasion Flag (Turning off logging)
        features.append(1 if log.get('event_name') in ["DeleteTrail", "StopLogging"] else 0)
        # 17. Caller ID Recon (GetCallerIdentity)
        features.append(1 if log.get('event_name') == "GetCallerIdentity" else 0)

        # --- PILLAR 4: METADATA & CONTEXT (8 features) ---
        # 18. Scripted Agent (1 = CLI/Boto3/Terraform, 0 = Console)
        ua = str(log.get('user_agent', '')).lower()
        features.append(1 if any(x in ua for x in ['aws-cli', 'boto', 'python', 'terraform']) else 0)
        # 19. Public IP (1 = External, 0 = Internal/Private)
        ip = str(log.get('source_ip', ''))
        try:
            features.append(1 if not ipaddress.ip_address(ip).is_private else 0)
        except: features.append(1) # Unknown/Public
        # 20. Parameter Complexity (Normalized length of params)
        params_str = str(log.get('request_params_raw', '{}'))
        features.append(min(len(params_str) / 1000.0, 1.0))
        # 21. Identity Resource Target (Is the resource a policy/role?)
        target_res = str(log.get('target_resource') or '').lower()
        params_res = params_str.lower()
        features.append(1 if any(x in target_res or x in params_res for x in ['policy', 'role', 'user']) else 0)
        # 22. Region Mismatch (1 = Not in primary us-east-1)
        features.append(1 if log.get('aws_region') != "us-east-1" else 0)
        # 23. Persistence Signal (CreateUser/Key)
        features.append(1 if log.get('event_name') in ["CreateUser", "CreateAccessKey", "CreateRole"] else 0)
        # 24. Credential Access Signal (KMS/SecretsManager)
        features.append(1 if any(x in log.get('event_source', '') for x in ['secrets', 'kms']) else 0)
        # 25. Escalation Pivot (Direct permission attachment)
        features.append(1 if log.get('event_name') in ["AttachUserPolicy", "PutUserPolicy"] else 0)

        return features