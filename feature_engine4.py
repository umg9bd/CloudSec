import json
import numpy as np
import ipaddress
from datetime import datetime

class StateTracker:
    """Tracks historical behavior to detect Privilege Escalation and Velocity."""
    def __init__(self):
        self.user_registry = {}

    def get_metrics(self, p_arn, event_name, current_ts):
        if p_arn not in self.user_registry:
            self.user_registry[p_arn] = {
                'last_ts': current_ts,
                'actions': {event_name}
            }
            return 0.0, 1 # No previous history = No velocity, but is NEW action
        
        user_data = self.user_registry[p_arn]
        
        # Calculate Velocity (Time delta normalized)
        delta = (current_ts - user_data['last_ts']).total_seconds()
        velocity = max(0, 1 - (delta / 3600)) 
        
        # Check for Privilege Creep
        is_new_action = 1 if event_name not in user_data['actions'] else 0
        
        # Update State for next log
        user_data['last_ts'] = current_ts
        user_data['actions'].add(event_name)
        
        return velocity, is_new_action

class FeatureEngineer:
    def __init__(self):
        self.tracker = StateTracker()
        self.action_map = {
            # --- 1. RECONNAISSANCE ---
        "GetCallerIdentity": 2,      # Who am I?
        "ListBuckets": 2,            # What data is here?
        "DescribeInstances": 2,      # What servers are here?
        "ListUsers": 2,              # Mapping the IAM landscape
        "GetAccountAuthorizationDetails": 4, # Goldmine for privilege mapping

        # --- 2. PERSISTENCE (Creating Backdoors) ---
        "CreateUser": 7, 
        "CreateRole": 7, 
        "CreateAccessKey": 8,        # (High risk) Long-term backdoor
        "CreateLoginProfile": 8,     

        # --- 3. PRIVILEGE ESCALATION---
        "PutUserPolicy": 10, 
        "AttachUserPolicy": 10, 
        "UpdateAssumeRolePolicy": 10,
        "PassRole": 9,               
        "CreatePolicyVersion": 9,    # Changing an existing policy to "Allow *"
        "SetDefaultPolicyVersion": 9,

        # --- 4. CREDENTIAL ACCESS & EXFILTRATION ---
        "GetSecretValue": 8,         # Secrets Manager: Directly stealing passwords
        "Decrypt": 7,                # KMS: Decrypting sensitive data
        "AssumeRole": 6,             # Lateral movement proxy

        # --- 5. DEFENSE EVASION ---
        "DeleteTrail": 10, 
        "StopLogging": 10,
        "UpdateDetector": 9,         # GuardDuty: Disabling the very thing that catches them
        "DeleteFlowLogs": 8,         # Wiping network evidence

        # --- 6. COMMAND & CONTROL / EXECUTION ---
        "SendCommand": 8,            # SSM: Running remote shell commands on EC2
        "InvokeFunction": 7          # Lambda: Triggering malicious code
        }
        self.default_risk = 1

    def get_structural_data(self, log):
        """Generates GNN Triples"""
        p_arn = log.get('principal_arn', 'unknown_principal')
        event = log.get('event_name', 'unknown_action')
        target = log.get('target_resource') or 'aws_service'
            
        return {
            'source_node': p_arn,
            'target_node': target,
            'edge_type': event
        }

    def get_temporal_features(self, log):
        """Generates 25D Vector."""
        f = []
        p_arn = log.get('principal_arn', 'unknown')
        
        # Timestamp parsing - handle both field names and formats
        timestamp_str = log.get('timestamp') or log.get('eventTime')
        
        if 'T' in timestamp_str:  
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:  # Standard format (2026-03-31 10:00:00+0000)
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S%z")

        # PILLAR 1: IDENTITY & DYNAMICS (5 Features)
        mfa = str(log.get('mfa_authenticated', 'false')).lower()
        f.append(1 if mfa in ['false', 'no', 'none', ''] else 0) # 1
        
        p_type = log.get('principal_type', '')
        p_risk = {"Root": 1.0, "IAMUser": 0.8, "AssumedRole": 0.5}.get(p_type, 0.3)
        f.append(p_risk) # 2
        
        f.append(1 if log.get('access_key_id') else 0) # 3
        
        velocity, is_new_action = self.tracker.get_metrics(p_arn, log.get('event_name'), dt)
        f.append(velocity) # 4
        f.append(is_new_action) # 5

        # PILLAR 2: TEMPORAL CONTEXT (4 Features)
        h_rad = 2 * np.pi * dt.hour / 24.0
        f.append(np.sin(h_rad)) # 6
        f.append(np.cos(h_rad)) # 7
        f.append(1 if dt.weekday() >= 5 else 0) # 8
        f.append(1 if dt.hour < 9 or dt.hour > 18 else 0) # 9

        # PILLAR 3: ACTION INTENT (8 Features)
        f.append(self.action_map.get(log.get('event_name'), self.default_risk) / 10.0) # 10
        f.append(1 if str(log.get('read_only', 'true')).lower() == 'false' else 0) # 11
        
        err = log.get('error_code', '')
        f.append(1 if err else 0) # 12
        f.append(1 if err == "AccessDenied" else 0) # 13
        
        f.append(1 if "iam" in log.get('event_source', '') else 0) # 14
        f.append(1 if any(x in log.get('event_name', '') for x in ['Describe', 'List', 'Get']) else 0) # 15
        f.append(1 if log.get('event_name') in ["DeleteTrail", "StopLogging"] else 0) # 16
        f.append(1 if log.get('event_name') == "GetCallerIdentity" else 0) # 17

        # PILLAR 4: METADATA ANOMALIES (8 Features)
        ua = str(log.get('user_agent', '')).lower()
        f.append(1 if any(x in ua for x in ['kali', 'pacu', 'metasploit', 'requests']) else 0) # 18
        
        try:
            ip = log.get('source_ip', '0.0.0.0')
            f.append(1 if not ipaddress.ip_address(ip).is_private else 0) # 19
        except: f.append(1)
        
        params = str(log.get('request_params_raw', '{}'))
        f.append(min(len(params) / 750.0, 1.0)) # 20
        
        target_str = str(log.get('target_resource', '')).lower()
        f.append(1 if any(x in target_str for x in ['admin', 'vault', 'prod']) else 0) # 21
        f.append(1 if log.get('aws_region') != "us-east-1" else 0) # 22
        f.append(1 if "Create" in log.get('event_name', '') and "Key" in log.get('event_name', '') else 0) # 23
        f.append(1 if any(x in log.get('event_source', '') for x in ['secrets', 'kms']) else 0) # 24
        f.append(1 if "Attach" in log.get('event_name', '') or "Put" in log.get('event_name', '') else 0) # 25

        return f

if __name__ == "__main__":
    engine = FeatureEngineer()

    # Log 1:
    log1 = {
        "timestamp": "2026-03-31 10:00:00+0000",
        "event_name": "GetCallerIdentity",
        "event_source": "sts.amazonaws.com",
        "principal_type": "IAMUser",
        "principal_arn": "arn:aws:iam::123:user/Attacker",
        "source_ip": "10.0.0.5",
        "user_agent": "aws-cli/2.0",
        "read_only": "true",
        "aws_region": "us-east-1",
        "mfa_authenticated": "true"
    }

    # Log 2: 
    log2 = {
        "timestamp": "2026-03-31 10:00:02+0000", # 2s later
        "event_name": "PutUserPolicy",
        "event_source": "iam.amazonaws.com",
        "principal_type": "IAMUser",
        "principal_arn": "arn:aws:iam::123:user/Attacker",
        "source_ip": "192.175.1.1",
        "user_agent": "pacu/aws-exploitation-framework",
        "read_only": "false",
        "aws_region": "us-east-1",
        "mfa_authenticated": "false"
    }
    log3= {
        "eventVersion": "1.08",
        "userIdentity": {
            "type": "IAMUser",
            "principalId": "AIDASCYV6JEXAMPLE",
            "arn": "arn:aws:iam::123456789012:user/Udi",
            "accountId": "123456789012",
            "accessKeyId": "AKIAIOSFODNN7EXAMPLE"
        },
        "eventTime": "2026-04-01T10:00:00Z",
        "eventSource": "ec2.amazonaws.com",
        "eventName": "DescribeInstances",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "10.0.0.15",
        "userAgent": "aws-cli/2.15.0 Python/3.11.6",
        "requestParameters": None,
        "responseElements": None,
        "readOnly": True,
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": "123456789012",
        "eventCategory": "Management"
    }

    # Log 1:
    print(f"Structural 1: {engine.get_structural_data(log1)}")
    print(f"Temporal 1: {engine.get_temporal_features(log1)}")

    print("\n")

    # Output for Log 2:

    struct = engine.get_structural_data(log2)
    vector = engine.get_temporal_features(log2)

    print(f"Structural 2: {struct}")
    print(f"Temporal 2: {vector}\n")

    #output for log 3
    struct3 = engine.get_structural_data(log3)
    vector3 = engine.get_temporal_features(log3)
    print(f"Structural 3: {struct3}")
    print(f"Temporal 3: {vector3}\n")