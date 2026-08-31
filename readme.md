# CloudSec Feature-GNN Pipeline

This repository implements the graph construction and GNN pipeline for cloud attack detection using AWS CloudTrail logs.

## Prerequisites

- Python 3.10+
- Neo4j Desktop / Neo4j Server running
- Git

---

# 1. Clone the Repository

```bash
git clone <repository_url>
cd CloudSec-feature-Gnn
```

---

# 2. Create a Virtual Environment

Create an isolated Python environment.

```bash
python3 -m venv venv
```

Activate it:

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

# 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# 4. Build the Neo4j Graph

Navigate to the graph construction module.

```bash
cd graph_construction
```

Run the graph builder:

```bash
python3 neo4j_graph_builder.py
```

### What this does

- Reads the processed CloudTrail feature data
- Constructs the heterogeneous graph in Neo4j
- Creates nodes and relationships representing cloud entities and interactions
- Exports the graph into PyTorch Geometric's `HeteroData` format for GNN training

---

# 5. Return to the Project Root

```bash
cd ..
```

---

# 6. Train the GraphSAGE Model

```bash
python3 train.py --model sage --epochs 100 --save_dir ./checkpoints
```

### What this does

- Loads the generated `HeteroData`
- Trains the GraphSAGE model
- Saves the best-performing checkpoint in the `checkpoints/` directory

---

# 7. Run Inference

```bash
python infer.py \
    --checkpoint checkpoints/best_GraphSAGE_wrapped.pt \
    --input incoming/synthetic_attack_chain.json \
    --threshold 0.5
```

### What this does

- Loads the trained GraphSAGE model
- Processes a new CloudTrail event sequence
- Predicts suspicious activities
- Produces inference results based on the specified confidence threshold

---

# Project Workflow

```
Raw CloudTrail Logs
        │
        ▼
Feature Engineering
        │
        ▼
Neo4j Graph Construction
        │
        ▼
Heterogeneous Graph (HeteroData)
        │
        ▼
GraphSAGE Training
        │
        ▼
Model Checkpoints
        │
        ▼
Inference on New Attack Chains
```

---

# Output

- **Neo4j Graph** representing cloud entities and relationships
- **HeteroData** graph for GNN processing
- **Trained GraphSAGE model**
- **Inference predictions** on incoming attack chains
