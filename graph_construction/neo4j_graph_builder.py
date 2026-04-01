"""
Neo4j Graph Builder — invictus_labeled_final.csv
=================================================

Graph Schema (per project spec Section 8):
───────────────────────────────────────────
NODES
  (:Principal  {name, principal_type, is_attacker, is_service})
      IAM principals — IAMUser / AssumedRole / AWSService
  (:AWSService {name})
      AWS service endpoints that are acted upon (event_source)
  (:IPAddress  {address})
      Source IPs — useful for lateral movement / exfil detection

EDGES
  (Principal)-[:INVOKED {props}]->(AWSService)
      Actor called this AWS service's API
      Props carry ALL security-relevant features needed by the GNN & sequence model

FEATURE ENGINEERING (aligned with FeatureEngineer class)
  Structural (get_structural_data equivalent):
    source_node   → principal ARN / username
    target_node   → AWS service (event_source)
    edge_type     → event_name (API call)

  Temporal / numerical features stored on each edge:
    mfa_authenticated     → binary (0/1)
    action_encoded        → integer encoding of event_name
    hour_normalized       → hour-of-day / 23.0  (0.0–1.0)
    is_error              → 0 = success, 1 = error
    is_read_only          → 0 = mutating, 1 = read-only
    label                 → event-level ground truth (0/1)
    session_label         → session-level ground truth (0/1)
    attack_technique      → MITRE technique string (empty = benign)
    privilege_score       → 1.0 for privilege-escalation events, else 0.0
    is_attack_user        → 1 if username is a known attacker identity
    event_source_category → short service name (iam, s3, sts, kms …)

Run:
    pip install neo4j pandas
    python3 neo4j_graph_builder.py
"""

import pandas as pd
from datetime import datetime, timezone
from neo4j import GraphDatabase

# ── Connection ────────────────────────────────────────────────────────────────
URI      = "bolt://localhost:7687"          
USER     = "neo4j"
PASSWORD = "test1234"                      
CSV_PATH = "./invictus_synthetic_1000.csv"

# ── Known attacker identities (from IR report) ────────────────────────────────
ATTACKER_IDENTITIES = {
    "bert-jan",
    "stratus-red-team-ec2-get-password-data-role",
    "stratus-red-team-ec2-steal-credentials-role",
    "stratus-red-team-leave-org-role",
    "stratus-red-team-get-usr-data-role",
    "stratus-red-team-ec2-enumerate-role",
    "stratus-red-team-ec2lui-role-pcccexdthk",
    "stratus-red-team-ec2lui-role-wuzemnoeqa",
    "stratus-red-team-nmfalu-gfjyeaypjt",
}

# ── Privilege-escalation API calls (MITRE ATT&CK for Cloud) ──────────────────
PRIV_ESC_ACTIONS = {
    "PutRolePolicy", "PutUserPolicy", "PutGroupPolicy",
    "AttachRolePolicy", "AttachUserPolicy", "AttachGroupPolicy",
    "CreateAccessKey", "CreateLoginProfile", "UpdateLoginProfile",
    "AddUserToGroup", "UpdateAssumeRolePolicy", "CreateRole",
    "PassRole", "SetDefaultPolicyVersion",
}

# ── Read-only API prefixes ────────────────────────────────────────────────────
READ_ONLY_PREFIXES = (
    "Get", "List", "Describe", "Head", "Lookup",
    "Scan", "Query", "Search", "Check", "Validate",
)

# ── Action encoding map ───────────────────────────────────
ACTION_MAP = {
    "AssumeRole": 1,
    "PutUserPolicy": 2, "PutRolePolicy": 3, "PutGroupPolicy": 4,
    "AttachRolePolicy": 5, "AttachUserPolicy": 6,
    "CreateRole": 7, "CreateUser": 8, "CreateAccessKey": 9,
    "GetSecretValue": 10, "GetPasswordData": 11,
    "DeleteTrail": 12, "StopLogging": 13, "PutEventSelectors": 14,
    "PutBucketPolicy": 15, "DeleteBucketPolicy": 16,
    "UpdateAssumeRolePolicy": 17, "AddUserToGroup": 18,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def classify_principal(username: str, principal_type: str) -> str:
    """Return User, Service, or Role."""
    if pd.isna(username):
        return "Service"
    if "amazonaws.com" in username:
        return "Service"
    if principal_type == "AssumedRole":
        return "Role"
    return "User"


def event_source_category(event_source: str) -> str:
    """Extract short service name: iam.amazonaws.com → iam"""
    if pd.isna(event_source):
        return "unknown"
    return event_source.replace(".amazonaws.com", "")


def parse_hour_normalized(timestamp: str) -> float:
    """Parse ISO timestamp and return hour / 23.0 (0.0–1.0)."""
    try:
        dt = datetime.fromisoformat(str(timestamp))
        return round(dt.hour / 23.0, 4)
    except Exception:
        return 0.0


def is_read_only(event_name: str) -> bool:
    return any(event_name.startswith(p) for p in READ_ONLY_PREFIXES)


def edge_features(row) -> dict:
    """
    Build the full feature dict for a INVOKED edge.
    These are the features consumed by the GNN and sequence model.
    """
    event_name = str(row.get("event_name", ""))
    username   = str(row.get("username", "")) if not pd.isna(row.get("username")) else ""

    return {
        # ── Structural (get_structural_data equivalent) ──────────────────
        "edge_type":              event_name,
        "event_source":           str(row.get("event_source", "")),
        "aws_region":             str(row.get("aws_region", "")),
        "source_ip":              str(row.get("source_ip", "")) if not pd.isna(row.get("source_ip")) else "",

        # ── Temporal / numerical (get_temporal_features equivalent) ──────
        "hour_normalized":        parse_hour_normalized(row.get("timestamp")),
        "action_encoded":         ACTION_MAP.get(event_name, 0),
        "is_error":               0 if pd.isna(row.get("error_code")) else 1,
        "is_read_only":           1 if is_read_only(event_name) else 0,

        # ── Security signals ──────────────────────────────────────────────
        "privilege_score":        1.0 if event_name in PRIV_ESC_ACTIONS else 0.0,
        "is_attack_user":         1 if username in ATTACKER_IDENTITIES else 0,
        "event_source_category":  event_source_category(row.get("event_source", "")),

        # ── Ground truth labels (for supervised GNN training) ─────────────
        "label":                  int(row.get("label", 0)),
        "session_label":          int(row.get("session_label", 0)),
        "attack_technique":       "" if pd.isna(row.get("attack_technique")) else str(row.get("attack_technique")),
    }


# ── Cypher statements ─────────────────────────────────────────────────────────

CONSTRAINTS = [
    # Principal uniqueness on ARN (most specific identifier)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Principal) REQUIRE p.arn IS UNIQUE",
    # AWSService uniqueness on name
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:AWSService) REQUIRE s.name IS UNIQUE",
    # IPAddress uniqueness
    "CREATE CONSTRAINT IF NOT EXISTS FOR (ip:IPAddress) REQUIRE ip.address IS UNIQUE",
]

MERGE_PRINCIPAL = """
MERGE (p:Principal {arn: $arn})
SET
  p.username       = $username,
  p.principal_type = $principal_type,
  p.actor_class    = $actor_class,
  p.is_attacker    = $is_attacker,
  p.is_service     = $is_service
"""

MERGE_SERVICE = """
MERGE (s:AWSService {name: $name})
SET s.category = $category
"""

MERGE_IP = """
MERGE (ip:IPAddress {address: $address})
"""

# Core INVOKED edge — aggregates per (principal, service, event_name)
# ON CREATE initialises counters; ON MATCH increments them.
# SET always refreshes the latest security signal values.
MERGE_INVOKED = """
MATCH (p:Principal  {arn: $arn})
MATCH (s:AWSService {name: $service_name})
MERGE (p)-[r:INVOKED {edge_type: $edge_type}]->(s)
  ON CREATE SET
    r.count          = 1,
    r.first_seen     = $timestamp,
    r.last_seen      = $timestamp,
    r.attack_count   = $label,
    r.session_hit    = $session_label
  ON MATCH SET
    r.count          = r.count + 1,
    r.last_seen      = $timestamp,
    r.attack_count   = r.attack_count + $label,
    r.session_hit    = CASE WHEN $session_label = 1 THEN 1 ELSE r.session_hit END
SET
  r.hour_normalized       = $hour_normalized,
  r.action_encoded        = $action_encoded,
  r.is_error              = $is_error,
  r.is_read_only          = $is_read_only,
  r.privilege_score       = $privilege_score,
  r.is_attack_user        = $is_attack_user,
  r.event_source_category = $event_source_category,
  r.label                 = $label,
  r.session_label         = $session_label,
  r.attack_technique      = $attack_technique,
  r.aws_region            = $aws_region,
  r.source_ip             = $source_ip
"""

# FROM_IP edge — links the INVOKED action back to originating IP
MERGE_FROM_IP = """
MATCH (p:Principal  {arn: $arn})
MATCH (ip:IPAddress {address: $address})
MERGE (p)-[:ORIGINATED_FROM]->(ip)
"""


# ── Main builder ──────────────────────────────────────────────────────────────

def build_graph():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)
    df["timestamp"]   = df["timestamp"].astype(str)
    df["username"]    = df["username"].fillna("unknown")
    df["principal_arn"] = df["principal_arn"].fillna(
        df["username"].apply(lambda u: f"arn:synthetic::{u}")
    )
    print(f"  {len(df):,} events | {df['label'].sum()} attack | {df['session_label'].sum()} in attack sessions")

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    with driver.session() as session:

        # 1. Constraints
        print("Creating constraints …")
        for cql in CONSTRAINTS:
            session.run(cql)

        # 2. Wipe existing graph for clean rebuild
        print("Clearing existing graph …")
        session.run("MATCH (n) DETACH DELETE n")

        # 3. Ingest rows
        print(f"Ingesting {len(df):,} events …")
        for i, row in df.iterrows():

            arn            = str(row["principal_arn"])
            username       = str(row["username"])
            principal_type = str(row.get("principal_type", "unknown"))
            actor_class    = classify_principal(username, principal_type)
            is_attacker    = username in ATTACKER_IDENTITIES
            is_service     = principal_type == "AWSService" or "amazonaws.com" in username
            service_name   = str(row.get("event_source", "unknown"))
            category       = event_source_category(service_name)
            source_ip      = str(row["source_ip"]) if not pd.isna(row.get("source_ip")) else None
            timestamp      = str(row["timestamp"])
            feats          = edge_features(row)

            # Principal node
            session.run(
                MERGE_PRINCIPAL,
                arn=arn,
                username=username,
                principal_type=principal_type,
                actor_class=actor_class,
                is_attacker=is_attacker,
                is_service=is_service,
            )

            # AWSService node
            session.run(MERGE_SERVICE, name=service_name, category=category)

            # INVOKED edge
            session.run(
                MERGE_INVOKED,
                arn=arn,
                service_name=service_name,
                timestamp=timestamp,
                **feats,
            )

            # IPAddress node + ORIGINATED_FROM edge
            if source_ip:
                session.run(MERGE_IP, address=source_ip)
                session.run(MERGE_FROM_IP, arn=arn, address=source_ip)

            if (i + 1) % 500 == 0:
                print(f"  … {i+1:,} rows processed")

    driver.close()
    print("\n✅ Graph built successfully.")
    print("\nUseful queries to run in Neo4j Browser (localhost:7474):")
    print("─" * 60)
    print("// All attacker actions")
    print("MATCH (p:Principal {is_attacker: true})-[r:INVOKED]->(s)")
    print("RETURN p.username, r.edge_type, s.name, r.attack_technique")
    print("ORDER BY r.privilege_score DESC\n")
    print("// Privilege escalation edges only")
    print("MATCH (p)-[r:INVOKED]->(s)")
    print("WHERE r.privilege_score = 1.0")
    print("RETURN p, r, s\n")
    print("// Sessions containing attack events")
    print("MATCH (p)-[r:INVOKED]->(s)")
    print("WHERE r.session_label = 1")
    print("RETURN p, r, s LIMIT 100")


if __name__ == "__main__":
    build_graph()
