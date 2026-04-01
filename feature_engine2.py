import json
import numpy as np
import ipaddress
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        # Action Map: Maps API verbs to risk weights based on AWS User Guide
       self.action_map = {
        # --- 1. RECONNAISSANCE (Low Weight, but Essential for GNN Edges) ---
        "GetCallerIdentity": 2,      # Who am I? (First thing an attacker runs)
        "ListBuckets": 2,            # What data is here?
        "DescribeInstances": 2,      # What servers are here?
        "ListUsers": 2,              # Mapping the IAM landscape
        "GetAccountAuthorizationDetails": 4, # Goldmine for privilege mapping

        # --- 2. PERSISTENCE (Creating Backdoors) ---
        "CreateUser": 7, 
        "CreateRole": 7, 
        "CreateAccessKey": 8,        # High risk: Long-term backdoor
        "CreateLoginProfile": 8,     # Adding a password to a console-less user

        # --- 3. PRIVILEGE ESCALATION (The "Crown Jewels" of Creep) ---
        "PutUserPolicy": 10, 
        "AttachUserPolicy": 10, 
        "UpdateAssumeRolePolicy": 10,
        "PassRole": 9,               # CRITICAL: Used to give a role to a service (e.g., EC2)
        "CreatePolicyVersion": 9,    # Sneaky: Changing an existing policy to "Allow *"
        "SetDefaultPolicyVersion": 9,

        # --- 4. CREDENTIAL ACCESS & EXFILTRATION ---
        "GetSecretValue": 8,         # Secrets Manager: Directly stealing passwords
        "Decrypt": 7,                # KMS: Decrypting sensitive data
        "AssumeRole": 6,             # Lateral movement proxy

        # --- 5. DEFENSE EVASION (Blindfolding the Admin) ---
        "DeleteTrail": 10, 
        "StopLogging": 10,
        "UpdateDetector": 9,         # GuardDuty: Disabling the very thing that catches them
        "DeleteFlowLogs": 8,         # Wiping network evidence

        # --- 6. COMMAND & CONTROL / EXECUTION ---
        "SendCommand": 8,            # SSM: Running remote shell commands on EC2
        "InvokeFunction": 7          # Lambda: Triggering malicious code
}
Self.default_action = 0

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
    
sample_log = {
        "timestamp": "2026-03-31 23:15:00+0000",
        "event_name": "StopLogging",
        "event_source": "cloudtrail.amazonaws.com",
        "principal_type": "IAMUser",
        "principal_arn": "arn:aws:iam::123456789012:user/attacker",
        "source_ip": "192.168.1.1", #pivate ip 
        "user_agent": "aws-cli/2.0.0 Python/3.8.5",
        "read_only": "false",
        "aws_region": "us-west-2",
        "mfa_authenticated": "false",
        "request_params_raw": '{"trailName": "management-events"}'
    }

engine = FeatureEngineer()
print("Structural:", engine.get_structural_data(sample_log))
print("Numerical Features:", engine.get_temporal_features(sample_log))