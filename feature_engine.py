import json
from datetime import datetime
import ipaddress

class FeatureEngineer:
    def __init__(self):
        # Dictionary for Action Encoding 
        self.action_map = {"AssumeRole": 1, "PutUserPolicy": 2, "RunInstances": 3}
        self.default_action = 0

    def get_structural_data(self, log):
        """Extracts Nodes and Edges for the GNN"""
        return {
            "source_node": log.get('userIdentity', {}).get('arn', 'unknown_user'),
            "target_node": log.get('requestParameters', {}).get('resourceName', 'aws_service'),
            "edge_type": log.get('eventName', 'unknown_action')
        }

    def get_temporal_features(self, log):
        """Extracts the 19 numerical features for the Ensemble"""
        features = []
        
        # 1. MFA Check (Binary)
        mfa = 1 if log.get('additionalEventData', {}).get('MFAUsed') == "Yes" else 0
        features.append(mfa)
        
        # 2. Event Action Encoding
        action_val = self.action_map.get(log.get('eventName'), self.default_action)
        features.append(action_val)
        
        # 3. Hour of Day (Temporal)
        event_time = log.get('eventTime')
        dt = datetime.strptime(event_time, "%Y-%m-%dT%H:%M:%SZ")
        features.append(dt.hour / 23.0) # Normalized 0-1
        
        # 4. Error Status (Success=0, Error=1)
        error = 1 if 'errorCode' in log else 0
        features.append(error)

        # Have to add more features here to reach 19
        return features

# TEST BLOCK
sample_invictus_log = {
    "eventTime": "2026-03-24T10:00:00Z",
    "eventName": "PutUserPolicy",
    "userIdentity": {"arn": "arn:aws:iam::123:user/Attacker"},
    "additionalEventData": {"MFAUsed": "No"}
}

engine = FeatureEngineer()
print("Structural:", engine.get_structural_data(sample_invictus_log))
print("Numerical Features:", engine.get_temporal_features(sample_invictus_log))