# Real-Time GraphSAGE Privilege Escalation Detection

## Overview

This project detects **AWS privilege-escalation attacks** from CloudTrail logs using a heterogeneous **Graph Neural Network (GraphSAGE)**, **Neo4j**, and **incremental streaming inference**.

The system continuously updates an IAM privilege graph from incoming CloudTrail events and performs real-time prediction without retraining the model.

---

## Architecture

```
CloudTrail Logs
       │
       ▼
Feature Engineering
       │
       ▼
Incremental Graph Update
 (Neo4j + In-Memory Graph)
       │
       ▼
K-Hop Neighbourhood Extraction
       │
       ▼
PyTorch Geometric HeteroData
       │
       ▼
GraphSAGE
       │
 ┌─────┴─────┐
 │           │
Benign   Malicious
               │
               ▼
      Blast Radius Analysis
               │
               ▼
          JSON Alert
```

---

## Why GraphSAGE?

- Inductive learning for previously unseen AWS entities
- Efficient neighborhood sampling
- Real-time streaming inference
- No retraining required
- Scales to continuously evolving IAM graphs
- Supports heterogeneous graph structures naturally

---

## Project Structure

```
.
├── train.py                     # Model training
├── infer.py                     # Real-time inference
├── model_graphsage.py           # GraphSAGE implementation
├── data_loader.py               # Neo4j → PyTorch Geometric loader
├── privilege_features.py        # Graph feature generation
├── incremental_updater.py       # Incremental Neo4j updates
├── feature_engine.py            # Feature engineering
├── blast_radius.py              # Blast radius analysis
├── graph_construction/
│   ├── neo4j_graph_builder.py
│   └── cloudtrail_structural.csv
├── checkpoints/
├── incoming/
└── alerts/
```

---

# Installation

Create a virtual environment

```bash
python3 -m venv venv
```

Activate it

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Step 1: Build the Neo4j Privilege Graph

Navigate to the graph construction folder

```bash
cd graph_construction
```

Build the graph

```bash
python3 neo4j_graph_builder.py
```

This imports CloudTrail events into Neo4j and creates the privilege propagation graph.

---

# Step 2: Train GraphSAGE

Return to the project root

```bash
cd ..
```

Train the model

```bash
python3 train.py --model sage --epochs 100 --save_dir ./checkpoints
```

The best model checkpoint will be saved inside

```
checkpoints/
```

---

# Step 3: Wrap the Checkpoint

Convert the training checkpoint into an inference-compatible checkpoint.

```bash
python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt
```

This generates

```
checkpoints/best_GraphSAGE_wrapped.pt
```

---

# Step 4: Run Real-Time Streaming Inference

```bash
python infer.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --watch incoming --alert-dir alerts --threshold 0.5 --seed-from-neo4j
```

The inference engine will

- Monitor the `incoming/` directory
- Process newly added CloudTrail JSON logs
- Update the Neo4j graph incrementally
- Extract the affected k-hop neighborhood
- Generate node embeddings
- Perform GraphSAGE inference
- Trigger blast-radius analysis if malicious
- Save alerts as JSON

---

# Input

Place CloudTrail JSON log files inside

```
incoming/
```

The watcher automatically detects new files and starts inference.

---

# Output

Detected attacks are written to

```
alerts/
```

Each alert contains

- Prediction score
- Attack probability
- Privilege escalation decision
- Blast radius analysis
- Timestamp
- Event metadata

---

# Training Pipeline

1. Build Neo4j privilege graph
2. Convert graph into PyTorch Geometric `HeteroData`
3. Fit feature scalers and encoders
4. Train GraphSAGE
5. Save the best checkpoint

---

# Streaming Inference Pipeline

1. Watch `incoming/`
2. Read new CloudTrail log
3. Engineer features
4. Incrementally update Neo4j
5. Extract affected k-hop neighborhood
6. Build `HeteroData`
7. Apply saved preprocessing
8. Run GraphSAGE inference
9. Perform blast-radius analysis
10. Save JSON alert

---

# Design Principles

- Incremental graph updates
- Streaming inference
- No retraining
- Inductive Graph Neural Network
- Explainable blast radius analysis
- Consistent preprocessing between training and inference
- Scalable heterogeneous IAM graph representation

---

# Requirements

- Python 3.10+
- Neo4j 5.x
- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- Pandas
- NumPy

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Example Workflow

```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

cd graph_construction

python3 neo4j_graph_builder.py

cd ..

python3 train.py --model sage --epochs 100 --save_dir ./checkpoints

python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt

python infer.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --watch incoming --alert-dir alerts --threshold 0.5 --seed-from-neo4j
```

---

## Results

The system performs continuous detection of AWS privilege-escalation attacks by combining:

- Neo4j graph storage
- Incremental graph updates
- Heterogeneous GraphSAGE
- Streaming inference
- Explainable blast-radius analysis

This enables scalable, real-time security monitoring without requiring model retraining as new CloudTrail events arrive.
# Real-Time GraphSAGE Privilege Escalation Detection

## Overview

This project detects **AWS privilege-escalation attacks** from CloudTrail logs using a heterogeneous **Graph Neural Network (GraphSAGE)**, **Neo4j**, and **incremental streaming inference**.

The system continuously updates an IAM privilege graph from incoming CloudTrail events and performs real-time prediction without retraining the model.

---

## Architecture

```
CloudTrail Logs
       │
       ▼
Feature Engineering
       │
       ▼
Incremental Graph Update
 (Neo4j + In-Memory Graph)
       │
       ▼
K-Hop Neighbourhood Extraction
       │
       ▼
PyTorch Geometric HeteroData
       │
       ▼
GraphSAGE
       │
 ┌─────┴─────┐
 │           │
Benign   Malicious
               │
               ▼
      Blast Radius Analysis
               │
               ▼
          JSON Alert
```

---

## Why GraphSAGE?

- Inductive learning for previously unseen AWS entities
- Efficient neighborhood sampling
- Real-time streaming inference
- No retraining required
- Scales to continuously evolving IAM graphs
- Supports heterogeneous graph structures naturally

---

## Project Structure

```
.
├── train.py                     # Model training
├── infer.py                     # Real-time inference
├── model_graphsage.py           # GraphSAGE implementation
├── data_loader.py               # Neo4j → PyTorch Geometric loader
├── privilege_features.py        # Graph feature generation
├── incremental_updater.py       # Incremental Neo4j updates
├── feature_engine.py            # Feature engineering
├── blast_radius.py              # Blast radius analysis
├── graph_construction/
│   ├── neo4j_graph_builder.py
│   └── cloudtrail_structural.csv
├── checkpoints/
├── incoming/
└── alerts/
```

---

# Installation

Create a virtual environment

```bash
python3 -m venv venv
```

Activate it

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Step 1: Build the Neo4j Privilege Graph

Navigate to the graph construction folder

```bash
cd graph_construction
```

Build the graph

```bash
python3 neo4j_graph_builder.py
```

This imports CloudTrail events into Neo4j and creates the privilege propagation graph.

---

# Step 2: Train GraphSAGE

Return to the project root

```bash
cd ..
```

Train the model

```bash
python3 train.py --model sage --epochs 100 --save_dir ./checkpoints
```

The best model checkpoint will be saved inside

```
checkpoints/
```

---

# Step 3: Wrap the Checkpoint

Convert the training checkpoint into an inference-compatible checkpoint.

```bash
python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt
```

This generates

```
checkpoints/best_GraphSAGE_wrapped.pt
```

---

# Step 4: Run Real-Time Streaming Inference

```bash
python infer.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --watch incoming --alert-dir alerts --threshold 0.5 --seed-from-neo4j
```

The inference engine will

- Monitor the `incoming/` directory
- Process newly added CloudTrail JSON logs
- Update the Neo4j graph incrementally
- Extract the affected k-hop neighborhood
- Generate node embeddings
- Perform GraphSAGE inference
- Trigger blast-radius analysis if malicious
- Save alerts as JSON

---

# Input

Place CloudTrail JSON log files inside

```
incoming/
```

The watcher automatically detects new files and starts inference.

---

# Output

Detected attacks are written to

```
alerts/
```

Each alert contains

- Prediction score
- Attack probability
- Privilege escalation decision
- Blast radius analysis
- Timestamp
- Event metadata

---

# Training Pipeline

1. Build Neo4j privilege graph
2. Convert graph into PyTorch Geometric `HeteroData`
3. Fit feature scalers and encoders
4. Train GraphSAGE
5. Save the best checkpoint

---

# Streaming Inference Pipeline

1. Watch `incoming/`
2. Read new CloudTrail log
3. Engineer features
4. Incrementally update Neo4j
5. Extract affected k-hop neighborhood
6. Build `HeteroData`
7. Apply saved preprocessing
8. Run GraphSAGE inference
9. Perform blast-radius analysis
10. Save JSON alert

---

# Design Principles

- Incremental graph updates
- Streaming inference
- No retraining
- Inductive Graph Neural Network
- Explainable blast radius analysis
- Consistent preprocessing between training and inference
- Scalable heterogeneous IAM graph representation

---

# Requirements

- Python 3.10+
- Neo4j 5.x
- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- Pandas
- NumPy

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Example Workflow

```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

cd graph_construction

python3 neo4j_graph_builder.py

cd ..

python3 train.py --model sage --epochs 100 --save_dir ./checkpoints

python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt

python infer.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --watch incoming --alert-dir alerts --threshold 0.5 --seed-from-neo4j
```

---

## Results

The system performs continuous detection of AWS privilege-escalation attacks by combining:

- Neo4j graph storage
- Incremental graph updates
- Heterogeneous GraphSAGE
- Streaming inference
- Explainable blast-radius analysis

This enables scalable, real-time security monitoring without requiring model retraining as new CloudTrail events arrive.
