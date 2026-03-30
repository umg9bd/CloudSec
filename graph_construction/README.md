# Graph Construction — Privilege Escalation Detection

Builds a Neo4j knowledge graph from AWS CloudTrail logs for GNN-based anomaly detection.

---

## Setup

### 1. Start Neo4j with Docker

```bash
docker run --name neo4j-local -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test1234 -v neo4j-data:/data neo4j:5
```

- Neo4j Browser → http://localhost:7474
- Username: `neo4j`
- Password: `test1234`

To stop and restart without losing data:
```bash
docker stop neo4j-local
docker start neo4j-local
```

To wipe and start fresh:
```bash
docker stop neo4j-local && docker rm neo4j-local
docker run --name neo4j-local -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/test1234 -v neo4j-data:/data neo4j:5
```

---

### 2. Install Python Dependencies

```bash
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Run the Graph Builder

```bash
python3 invictus_neo4j_builder.py
```

To rebuild from scratch (wipes existing graph first):
```bash
python3 invictus_neo4j_builder.py
```
The script always runs `MATCH (n) DETACH DELETE n` before ingesting, so re-running is always safe.

---

## Graph Schema

| Element | Label | Key Property |
|---|---|---|
| Node | `Principal` | `arn` |
| Node | `AWSService` | `name` |
| Node | `IPAddress` | `address` |
| Edge | `INVOKED` | principal → service |
| Edge | `ORIGINATED_FROM` | principal → IP |

---

## Extracted Features (on `INVOKED` edge)

| Feature | Column | Description |
|---|---|---|
| `mfa_authenticated` | — | MFA status (default 0, not in CSV) |
| `action_encoded` | `event_name` | Integer encoding of API call |
| `hour_normalized` | `timestamp` | Hour of day / 23.0 (0.0–1.0) |
| `is_error` | `error_code` | 0 = success, 1 = error |
| `label` | `label` | Event-level ground truth |
| `session_label` | `session_label` | Session-level ground truth |
| `privilege_score` | `event_name` | 1.0 if privilege escalation action |
| `is_attack_user` | `username` | 1 if known attacker identity |

---

## Useful Cypher Queries

```cypher
-- View full graph
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100

-- All attacker actions
MATCH (p:Principal {is_attacker: true})-[r:INVOKED]->(s)
RETURN p.username, r.edge_type, s.name, r.attack_technique
ORDER BY r.privilege_score DESC

-- Privilege escalation edges only
MATCH (p)-[r:INVOKED]->(s)
WHERE r.privilege_score = 1.0
RETURN p, r, s

-- Sessions containing attack events
MATCH (p)-[r:INVOKED]->(s)
WHERE r.session_label = 1
RETURN p, r, s LIMIT 100
```

---

## Files

| File | Description |
|---|---|
| `invictus_neo4j_builder.py` | Main graph builder script |
| `invictus_labeled_final.csv` | Real Invictus IR CloudTrail dataset (2,900 events) |
| `invictus_synthetic_1000.csv` | Synthetic dataset with 4 features extracted (1,000 events) |
| `requirements.txt` | Python dependencies |
