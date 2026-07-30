# Real-Time GraphSAGE Privilege Escalation Detection

## Overview

This project detects AWS privilege-escalation attacks from CloudTrail
logs using a heterogeneous Graph Neural Network (GraphSAGE), Neo4j, and
incremental streaming inference.

## Architecture

``` text
CloudTrail
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
    ├── Benign
    └── Malicious
            │
            ▼
     Blast Radius Analysis
            │
            ▼
        JSON Alert
```

## Why GraphSAGE?

-   Inductive learning for unseen AWS entities
-   Efficient neighborhood sampling
-   Streaming inference without retraining
-   Scales to continuously growing graphs
-   Works naturally with heterogeneous IAM graphs

## Training Pipeline

1.  Parse CloudTrail logs.
2.  Engineer temporal, structural and privilege features.
3.  Build Neo4j graph.
4.  Convert to PyTorch Geometric HeteroData.
5.  Fit scalers and label encoders.
6.  Train GraphSAGE.
7.  Save checkpoint.

## Streaming Inference

1.  Watch incoming directory.
2.  Feature engineer new event.
3.  Incrementally update Neo4j.
4.  Extract affected k-hop neighborhood.
5.  Build HeteroData.
6.  Apply training scalers.
7.  Run GraphSAGE.
8.  Trigger blast radius if malicious.
9.  Save JSON alert.

## Repository

-   train.py --- training
-   infer.py --- streaming inference
-   model_graphsage.py --- model
-   data_loader.py --- graph loading
-   privilege_features.py --- graph creation
-   incremental_updater.py --- graph updates
-   feature_engine9.py --- feature engineering
-   blast_radius.py --- blast radius

## Training

``` bash
python3 train.py \         
    --model sage \
    --epochs 100 \
    --save_dir ./checkpoints
```

## Wrap checkpoint

``` bash
python infer.py --wrap-checkpoint checkpoints/best_sage.pt
```

## Run inference

``` bash
python infer.py   --checkpoint checkpoints/best_sage_wrapped.pt   --watch incoming   --alert-dir alerts   --threshold 0.5   --seed-from-neo4j


```

## Outputs

Alerts are written into the alerts directory as JSON.

## Design Principles

-   Incremental graph updates
-   No retraining
-   Inductive GNN
-   Explainable blast radius
-   Consistent preprocessing between training and inference

