# Real-Time GraphSAGE Privilege Escalation Detection

## Overview

This project detects AWS privilege-escalation attacks from CloudTrail
logs using a heterogeneous Graph Neural Network (GraphSAGE), Neo4j, and
incremental streaming inference. Trained on procedurally-generated
synthetic CloudTrail sessions, validated against real attack data collected
with [Stratus Red Team](https://stratus-red-team.cloud/) across 4
independent AWS accounts.

## Results

| | Precision | Recall | F1 |
|---|---|---|---|
| GraphSAGE, session-level, real held-out test data | 0.859 | 0.790 | **0.823** [95% CI: 0.766, 0.875] |
| Rule-based baseline (GuardDuty-style, 11 rules) | 0.889 | 0.623 | 0.732 [95% CI: 0.672, 0.790] |

The GNN clears the rule-based baseline on real, previously-unseen attack
sessions — verified with a dev-set-only selected threshold, checked once on
held-out test data, threshold-stability checked, and bootstrap-CI checked.
Getting here required diagnosing and fixing a real synthetic-to-real
generalization gap (two structural bugs, one ruled-out hypothesis, and the
actual fix — a rank-normalization feature transform). Full evidence trail,
caveats, and what's still open: **`PROJECT_STATUS_REPORT.md`**. Full runnable
commands with expected output at each step: **`DEMO_GUIDE.md`**.

One honest caveat up front: the win is a *session-level* effect (the model
correctly flags at least one edge per attack session) — edge-level accuracy
on individual actions is still weak (AUC≈0.54). See the report for the full
picture before citing the headline number alone.

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
-   Efficient neighborhood sampling by prioritising edges with higher probability as attack in the sample
-   Streaming inference without retraining
-   Scales to continuously growing graphs
-   Works naturally with heterogeneous IAM graphs

## Training Pipeline
 Build Neo4j graph.
 Convert to PyTorch Geometric HeteroData.
 Fit scalers and label encoders.
 Train GraphSAGE.
 Save checkpoint.

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

-   train.py --- training (GraphSAGE and GAT)
-   infer.py --- streaming inference + checkpoint wrapping (see note below)
-   model_graphsage.py / model_gat.py --- models
-   data_loader.py --- graph loading, feature normalization
-   privilege_features.py --- node/edge identity, relation classification
-   graph_construction/neo4j_graph_builder.py --- batch graph construction
-   incremental_updater.py --- streaming graph updates
-   feature_engine9.py --- feature engineering (raw CloudTrail -> structural/temporal CSVs)
-   datasets/privilege-escalation/generate_synthetic_data.py --- synthetic training data generator
-   build_graph.py --- CLI wrapper to load a structural CSV into Neo4j
-   evaluate_on_real.py --- edge-level real-data evaluation
-   evaluate_session_level.py --- session-level real-data evaluation (comparable to the rule baseline)
-   datasets/privilege-escalation/evaluate_baselines.py --- rule-based baselines
-   blast_radius.py --- downstream reachability/impact analysis (not yet exercised)
-   explainability.py --- prediction explanations (not yet exercised)
-   PROJECT_STATUS_REPORT.md --- full evaluation history, evidence, and publication roadmap
-   DEMO_GUIDE.md --- runnable demo script with expected output at each step

## Training

``` bash
python train.py \
    --model sage \
    --epochs 100 \
    --save_dir ./checkpoints
```

## Wrap checkpoint

``` bash
python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt --wrapped-output checkpoints/best_GraphSAGE_wrapped.pt
```

## Evaluate against real data

``` bash
python evaluate_on_real.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage
python evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage --raw-csv datasets/privilege-escalation/real_dataset_test.csv --threshold 0.35
```

Full setup (Neo4j, environment variables, expected output) in `DEMO_GUIDE.md`.

## Run live streaming inference

``` bash
python infer.py   --checkpoint checkpoints/best_GraphSAGE_wrapped.pt   --watch incoming   --alert-dir alerts   --threshold 0.5   --seed-from-neo4j
```
Insert json logs into incoming directory to get real time prediction of the action performed.

**Known issue**: `infer.py`'s live single-event feature builder constructs
edge features independently of `data_loader.py` and has not yet been
updated for the rank-normalized feature schema behind the current best
checkpoint (see `PROJECT_STATUS_REPORT.md` section 6.16) — it will run
without erroring but produce incorrect scores until this is fixed. Use the
batch evaluation commands above for anything that needs to be trusted right
now.

## Outputs

Alerts are written into the alerts directory as JSON.

## Design Principles

-   Incremental graph updates
-   No retraining
-   Inductive GNN
-   Explainable blast radius
-   Consistent preprocessing between training and inference
