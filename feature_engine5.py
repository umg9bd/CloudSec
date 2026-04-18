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

def generate_mock_cloudtrail_log():
    """Generates realistic synthetic CloudTrail events for real-time testing."""
    import random
    from datetime import datetime, timezone

    actions = [
        # Recon
        ("GetCallerIdentity", "sts.amazonaws.com", "true"),
        ("ListBuckets", "s3.amazonaws.com", "true"),
        ("DescribeInstances", "ec2.amazonaws.com", "true"),
        ("ListUsers", "iam.amazonaws.com", "true"),
        ("GetAccountAuthorizationDetails", "iam.amazonaws.com", "true"),
        # Persistence
        ("CreateUser", "iam.amazonaws.com", "false"),
        ("CreateRole", "iam.amazonaws.com", "false"),
        ("CreateAccessKey", "iam.amazonaws.com", "false"),
        ("CreateLoginProfile", "iam.amazonaws.com", "false"),
        # Privilege Escalation
        ("PutUserPolicy", "iam.amazonaws.com", "false"),
        ("AttachUserPolicy", "iam.amazonaws.com", "false"),
        ("UpdateAssumeRolePolicy", "iam.amazonaws.com", "false"),
        ("PassRole", "iam.amazonaws.com", "false"),
        ("CreatePolicyVersion", "iam.amazonaws.com", "false"),
        # Creds
        ("GetSecretValue", "secretsmanager.amazonaws.com", "true"),
        ("Decrypt", "kms.amazonaws.com", "true"),
        ("AssumeRole", "sts.amazonaws.com", "false"),
        # Evasion
        ("DeleteTrail", "cloudtrail.amazonaws.com", "false"),
        ("StopLogging", "cloudtrail.amazonaws.com", "false"),
        ("UpdateDetector", "guardduty.amazonaws.com", "false"),
        # C2
        ("SendCommand", "ssm.amazonaws.com", "false")
    ]
    
    event_name, event_source, read_only = random.choice(actions)
    
    principals = [
        {"arn": "arn:aws:iam::123456789012:user/DevUser", "type": "IAMUser"},
        {"arn": "arn:aws:iam::123456789012:user/AdminUser", "type": "IAMUser"},
        {"arn": "arn:aws:iam::123456789012:role/EC2_Service_Role", "type": "AssumedRole"},
        {"arn": "arn:aws:iam::123456789012:user/Attacker", "type": "IAMUser"}
    ]
    principal = random.choice(principals)
    
    ips = ["10.0.0.15", "10.0.1.200", "192.168.1.100", "203.0.113.45", "198.51.100.22"]
    user_agents = [
        "aws-cli/2.0 Python/3.8",
        "console.amazonaws.com",
        "Boto3/1.17",
        "pacu/aws-exploitation-framework",
        "python-requests/2.25.1"
    ]
    
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    ua = random.choice(user_agents)
    ip = random.choice(ips)
    
    # Determine if it's considered malicious
    is_evil = (principal["arn"].endswith("Attacker") or "pacu" in ua or ip.startswith("198.51."))
    
    log = {
        "timestamp": now_iso,
        "event_name": event_name,
        "event_source": event_source,
        "principal_type": principal["type"],
        "principal_arn": principal["arn"],
        "source_ip": ip,
        "user_agent": ua,
        "read_only": read_only,
        "aws_region": random.choice(["us-east-1", "us-east-1", "us-west-2"]),
        "mfa_authenticated": str(random.choice([True, True, False])).lower(),
        "error_code": "AccessDenied" if random.random() < 0.05 else "",
        "target_resource": f"resource-{random.randint(1,50)}",
        "label": "1" if is_evil else "0",
        "request_params_raw": "{}" 
    }
    
    if random.random() > 0.5:
        log["access_key_id"] = f"AKIAIOSFODNN7{random.randint(1000,9999)}"
        
    return log

def ingest_realtime_stream(engine, queue_url=None):
    """
    Ingests CloudTrail logs in real-time and continuously appends to the structural and temporal files.
    """
    import time
    import csv
    import os
    import json
    import uuid
    import random
    
    output_struct_file = "datasets/privilege-escalation/invictus_structural.csv"
    output_temporal_file = "datasets/privilege-escalation/invictus_temporal.csv"
    
    temporal_cols = [
        "no_mfa", "principal_risk", "has_access_key", "action_velocity", "is_new_action",
        "time_sin", "time_cos", "is_weekend", "is_off_hours",
        "mitre_action_risk", "is_write_action", "has_error", "is_access_denied",
        "is_iam_event", "is_recon_action", "is_defense_evasion", "is_get_caller_identity",
        "is_malicious_user_agent", "is_public_ip", "params_length_normalized",
        "targets_sensitive_resource", "is_non_default_region", "is_create_key",
        "is_secrets_or_kms", "is_permission_modification"
    ]
    
    write_headers = not os.path.exists(output_struct_file)
    
    print(f"Starting real-time log ingestion ({'SQS queue' if queue_url else 'Synthetic Mock Data Generator'} mode)...")
    print("Press Ctrl+C to stop.")
    
    if queue_url:
        import boto3
        sqs = boto3.client('sqs')
    else:
        sqs = None
        
    while True:
        try:
            logs_to_process = []
            
            # --- 1. FETCH FROM QUEUE OR GENERATE MOCK ---
            if sqs and queue_url:
                response = sqs.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=5 # Long polling
                )
                if 'Messages' in response:
                    for msg in response['Messages']:
                        try:
                            log = json.loads(msg['Body'])
                            logs_to_process.append(log)
                        except Exception:
                            pass
                        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
            else:
                # Generate realistic random logs dynamically!
                time.sleep(random.uniform(0.5, 2.5))
                log = generate_mock_cloudtrail_log()
                logs_to_process.append(log)
                
            # --- 2. PROCESS IN REAL TIME ---
            if logs_to_process:
                with open(output_struct_file, mode='a', newline='', encoding='utf-8') as struct_out, \
                     open(output_temporal_file, mode='a', newline='', encoding='utf-8') as temp_out:
                    
                    struct_fieldnames = ["log_id", "source_node", "target_node", "edge_type", "label"]
                    struct_writer = csv.DictWriter(struct_out, fieldnames=struct_fieldnames)
                    
                    temp_fieldnames = ["log_id"] + temporal_cols + ["label"]
                    temp_writer = csv.DictWriter(temp_out, fieldnames=temp_fieldnames)
                    
                    if write_headers:
                        struct_writer.writeheader()
                        temp_writer.writeheader()
                        write_headers = False
                        
                    for log in logs_to_process:
                        struct = engine.get_structural_data(log)
                        temporal = engine.get_temporal_features(log)
                        label = log.get("label", "0")
                        log_id = f"rt_{str(uuid.uuid4())[:8]}" # Unique ID marking real-time origin
                        
                        struct_writer.writerow({
                            "log_id": log_id,
                            "source_node": struct["source_node"],
                            "target_node": struct["target_node"],
                            "edge_type": struct["edge_type"],
                            "label": label
                        })
                        
                        temp_row = {"log_id": log_id, "label": label}
                        for i, val in enumerate(temporal):
                            temp_row[temporal_cols[i]] = val
                        temp_writer.writerow(temp_row)
                        
                        malicious_tag = "[🚨 MALICIOUS]" if label == "1" else "[✓ BENIGN]"
                        print(f"[{time.strftime('%H:%M:%S')}] Streamed {malicious_tag} event '{struct['edge_type']}' from {struct['source_node']}")
                        
        except KeyboardInterrupt:
            print("\nStopped real-time ingestion.")
            break
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(5)
                        
        except KeyboardInterrupt:
            print("\nStopped real-time ingestion.")
            break
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    import csv
    import os
    import sys
    
    engine = FeatureEngineer()
    
    # Simple CLI parser
    if len(sys.argv) > 1 and sys.argv[1] == "--realtime":
        queue_url = sys.argv[2] if len(sys.argv) > 2 else None
        ingest_realtime_stream(engine, queue_url)
        sys.exit(0)
    
    # Original Batch Logic
    input_file = "datasets/privilege-escalation/invictus_enriched.csv"
    output_struct_file = "datasets/privilege-escalation/invictus_structural.csv"
    output_temporal_file = "datasets/privilege-escalation/invictus_temporal.csv"
    
    print(f"Reading logs in BATCH mode from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        exit(1)
    
    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_struct_file, mode='w', newline='', encoding='utf-8') as struct_out, \
         open(output_temporal_file, mode='w', newline='', encoding='utf-8') as temp_out:
        
        reader = csv.DictReader(infile)
        
        struct_fieldnames = ["log_id", "source_node", "target_node", "edge_type", "label"]
        struct_writer = csv.DictWriter(struct_out, fieldnames=struct_fieldnames)
        struct_writer.writeheader()
        
        temporal_cols = [
            "no_mfa", "principal_risk", "has_access_key", "action_velocity", "is_new_action",
            "time_sin", "time_cos", "is_weekend", "is_off_hours",
            "mitre_action_risk", "is_write_action", "has_error", "is_access_denied",
            "is_iam_event", "is_recon_action", "is_defense_evasion", "is_get_caller_identity",
            "is_malicious_user_agent", "is_public_ip", "params_length_normalized",
            "targets_sensitive_resource", "is_non_default_region", "is_create_key",
            "is_secrets_or_kms", "is_permission_modification"
        ]
        temp_fieldnames = ["log_id"] + temporal_cols + ["label"]
        temp_writer = csv.DictWriter(temp_out, fieldnames=temp_fieldnames)
        temp_writer.writeheader()
        
        count = 0
        for row in reader:
            struct = engine.get_structural_data(row)
            temporal = engine.get_temporal_features(row)
            label = row.get("label", "0")
            
            struct_writer.writerow({
                "log_id": count,
                "source_node": struct["source_node"],
                "target_node": struct["target_node"],
                "edge_type": struct["edge_type"],
                "label": label
            })
            
            temp_row = {"log_id": count, "label": label}
            for i, val in enumerate(temporal):
                temp_row[temporal_cols[i]] = val
                
            temp_writer.writerow(temp_row)
            count += 1
            
    print(f"Successfully processed {count} logs.")
    print(f"Saved structural triplets to {output_struct_file}")
    print(f"Saved temporal vectors to {output_temporal_file}")
    print("Run tracking in realtime: python feature_engine5.py --realtime [SQS_URL]")