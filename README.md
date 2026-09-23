1. Project Overview

Cloud environments generate large volumes of CloudTrail activity involving users, assumed roles, resources, policies, and AWS API actions.

Instead of treating every CloudTrail record independently, this project transforms CloudTrail activity into a heterogeneous graph so that relationships between cloud entities can be modeled.

text
AWS CloudTrail
      |
      v
Stateful Feature Engineering
      |
      v
Neo4j Privilege Propagation Graph
      |
      v
PyTorch Geometric HeteroData
      |
      +------------------+
      |                  |
      v                  v
     HGT            GraphSAGE / GAT
   PRIMARY             BASELINES
      |                  |
      +---------+--------+
                |
                v
      Edge/Event Classification
                |
                v
      Explainability + Alerts
2. Architecture
2.1 High-Level Pipeline
text
                +---------------------+
                |   AWS CloudTrail    |
                |      JSON Logs      |
                +----------+----------+
                           |
                           v
              +------------------------+
              | Feature Engine         |
              | feature_engine9.py     |
              +----------+-------------+
                         |
                         v
              +------------------------+
              | Structural CSV         |
              | cloudtrail_structural  |
              +----------+-------------+
                         |
                         v
              +------------------------+
              | Neo4j Graph Builder    |
              | neo4j_graph_builder.py |
              +----------+-------------+
                         |
                         v
              +------------------------+
              | Privilege Propagation  |
              | Graph (Neo4j)          |
              +----------+-------------+
                         |
                         v
              +------------------------+
              | data_loader.py         |
              | Neo4j -> HeteroData    |
              +----------+-------------+
                         |
                         v
             +--------------------------+
             | Heterogeneous GNN Models |
             |                          |
             | HGT       <- Primary      |
             | GraphSAGE <- Baseline    |
             | GAT       <- Baseline    |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | Edge/Event Classifier    |
             | Attack Probability       |
             +------------+-------------+
                          |
                    +-----+-----+
                    |           |
                    v           v
             Explainability   Alerts
3. Heterogeneous Graph Representation

The graph is heterogeneous because different cloud entities have different semantic meanings.

Node Types

The current loader supports:

User
Role
UnresolvedPrincipal
Service
Resource
Policy

Example:

text
User
  |
  | ASSUMES
  v
Role
  |
  | READ
  v
Resource

A policy relationship can look like:

text
User
  |
  | PERMISSIONS_MANAGEMENT
  v
Policy
4. Edge Types

The graph supports relationships including:

text
ASSUMES
LIST
READ
WRITE
TAGGING
PERMISSIONS_MANAGEMENT
UNKNOWN_ACTION

Edges are represented using heterogeneous triples:

text
(SourceType, Relation, DestinationType)

Examples:

text
(User, ASSUMES, Role)
(User, READ, Resource)
(Role, READ, Resource)
(User, WRITE, Resource)
(User, PERMISSIONS_MANAGEMENT, Policy)

This distinction matters because:

text
User -> READ -> Resource

and:

text
Role -> READ -> Resource

are different heterogeneous edge types.

The loader therefore groups data using:

python
(src_type, relation, dst_type)

rather than grouping by relation alone.

5. Edge/Event Classification

The primary prediction target is the CloudTrail event represented by a graph edge.

Each edge contains:

text
edge_index
edge_attr
y

where:

text
edge_index -> graph topology
edge_attr  -> edge-level features
y          -> attack / benign label

The classifier produces an attack probability for each edge.

Conceptually:

text
CloudTrail Event
      |
      v
Graph Edge
      |
      v
HGT Contextual Representation
      |
      v
Edge Classifier
      |
      v
P(attack)

The system therefore performs event-level classification, rather than assigning only one overall risk score to a principal.

6. Node Features
Principal-side Nodes

User, Role, and UnresolvedPrincipal use:

text
out_degree
unique_targets
unique_actions
role_transition_count
Target-side Nodes

Service, Resource, and Policy use:

text
in_degree
unique_principals
resource_sensitivity
distance_to_sensitive_resource

Resource additionally contains:

text
resource_type

Node type itself is represented by the heterogeneous model architecture.

7. Edge Features

The current edge feature schema contains:

text
hop_count
privilege_gain
privilege_gain_defined
abnormal_path_frequency
action_global_frequency_log
is_privilege_escalation_technique
is_read_only
edge_type

The current numeric edge feature dimension is:

text
8

A shared edge-feature scaler is fitted across all relations so that different relation buckets use the same numeric scale.

8. Privilege Propagation Features

privilege_features.py provides structural security features.

Hop Count

Measures graph distance associated with privilege propagation.

Privilege Gain

Represents changes in privilege level when the required graph information is available.

Abnormal Path Frequency

Captures unusual structural behavior in graph paths.

Resource Sensitivity

Represents the sensitivity associated with a resource.

Distance to Sensitive Resource

Measures graph distance to sensitive resources, with an explicit sentinel value when no relevant resource is reachable within the configured cutoff.

9. HGT

The primary architecture is a Heterogeneous Graph Transformer (HGT).

The graph contains multiple node and edge types, making relation-aware heterogeneous message passing important.

Current HGT configuration:

text
Model       = HGT
Hidden dim  = 128
Layers      = 2
Heads       = 4
Edge dim    = 8

The architecture contains separate input projections for different node types and relation-aware HGT convolution layers.

The checkpoint contains HGT-specific parameters such as:

text
encoder.input_proj.User
encoder.input_proj.Role
encoder.input_proj.Resource
encoder.edge_projs...
encoder.convs.0
encoder.convs.1
10. GraphSAGE, GAT, HGT, and Baseline Ensemble

The repository supports three heterogeneous GNN architectures:

text
GraphSAGE
GAT
HGT

Model selection in train.py remains:

bash
--model sage
--model gat
--model hgt
--model both
--model all

HGT is the primary architecture.

GraphSAGE and GAT are comparative baseline architectures. In addition to evaluating them individually, the repository now contains a GraphSAGE + GAT weighted-logit ensemble used specifically for comparative evaluation against the primary HGT model.

The ensemble is a separate evaluation path. It does not replace HGT as the primary deployed architecture.

The ensemble comparison script:

text
compare_hgt_vs_sage_gat_ensemble.py

uses the same offline graph representation for all three individual models and then compares:

text
HGT
GraphSAGE
GAT
GraphSAGE + GAT ensemble

This distinction is important because the earlier checkpoints for GraphSAGE/GAT were trained against an older graph schema. The current comparison therefore uses freshly retrained offline baseline checkpoints from train_offline_baselines.py rather than forcing incompatible old checkpoints to load.

10.1 Corrected Offline Baseline Training

The current baseline checkpoints were retrained against the exact offline graph loaded by:

text
offline_pipeline.py

The dedicated training script is:

text
train_offline_baselines.py

Run:

bash
python3 train_offline_baselines.py

This produces:

text
checkpoints_offline_current/
├── best_GraphSAGE.pt
└── best_GAT.pt

The training configuration used for the corrected baseline run is:

text
device      = CPU
hidden      = 128
layers      = 2
heads       = 4
dropout     = 0.3
learning rate = 1e-3
loss        = focal
epochs      = 100
threshold   = 0.5
patience    = 15
split       = stratified
seed        = 42

The important reproducibility property is that these two checkpoints use the same offline graph schema later used by the comparison script.

10.2 HGT vs GraphSAGE/GAT Ensemble Comparison

Run the full comparison with:

bash
python3 compare_hgt_vs_sage_gat_ensemble.py   --csv ./graph_construction/cloudtrail_structural.csv   --device cpu   --seed 42   --hgt_checkpoint ./checkpoints_hgt_corrected/best_HGT.pt   --sage_checkpoint ./checkpoints_offline_current/best_GraphSAGE.pt   --gat_checkpoint ./checkpoints_offline_current/best_GAT.pt

The script evaluates the three individual models and calibrates a two-model GraphSAGE + GAT logit ensemble on the validation split before evaluating it on the test split.

Latest corrected offline comparison:

Model	Accuracy	Precision	Recall	F1	AUROC	AUPR
GraphSAGE	0.9966	1.0000	0.9306	0.9640	0.9991	0.9896
GAT	0.9973	1.0000	0.9444	0.9714	0.9991	0.9873
HGT	0.9952	0.9221	0.9861	0.9530	0.9999	0.9975
GraphSAGE + GAT ensemble	0.9966	1.0000	0.9306	0.9640	0.9991	0.9896

Evaluation details:

text
Seed        = 42
Split       = stratified
Test edges  = 1458
Threshold   = 0.5

The metrics should be interpreted together rather than as a single scalar ranking. The HGT run has the highest AUROC/AUPR in this particular corrected test run, while the baseline/ensemble path has a higher F1 at the evaluated threshold. These are different evaluation criteria and are not interchangeable.

The comparison is an offline experiment. It does not need to be rerun whenever infer.py is used.

10.3 Inspect the Corrected Baseline Checkpoints

Inspect GraphSAGE:

bash
python3 -c "
import torch
c=torch.load(
    'checkpoints_offline_current/best_GraphSAGE.pt',
    map_location='cpu',
    weights_only=False
)
print(type(c))
print(list(c.keys())[:20] if isinstance(c, dict) else 'not a dict')
"

Inspect GAT:

bash
python3 -c "
import torch
c=torch.load(
    'checkpoints_offline_current/best_GAT.pt',
    map_location='cpu',
    weights_only=False
)
print(type(c))
print(list(c.keys())[:20] if isinstance(c, dict) else 'not a dict')
"

Do not substitute the older checkpoints under:

text
checkpoints/

for the corrected offline comparison unless they are retrained against the same current graph schema.

11. AWS Target Extraction

feature_engine9.py contains AWS-specific target extraction logic.

Operations currently handled include:

text
AssumeRole
CreateRole
UpdateAssumeRolePolicy
AttachRolePolicy
PutRolePolicy
CreatePolicyVersion
SetDefaultPolicyVersion
CreateAccessKey
PutBucketPolicy
GetSecretValue
StopLogging
DeleteTrail

Examples:

text
AssumeRole
    |
    v
arn:aws:iam::<account>:role/<role>

and:

text
PutBucketPolicy
    |
    v
arn:aws:s3:::<bucket>

This allows AWS targets stored inside requestParameters to be represented correctly in the graph.

12. Neo4j

Neo4j stores the Privilege Propagation Graph.

Docker Setup

Stop and remove an existing container:

bash
docker stop neo4j-local && docker rm neo4j-local

Create the Neo4j container:

bash
docker run \
  --name neo4j-local \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  -v neo4j-data:/data \
  neo4j

Neo4j Browser:

text
http://localhost:7474

Bolt endpoint:

text
bolt://localhost:7687

Credentials:

text
username: neo4j
password: test1234

Start an existing container:

bash
docker start neo4j-local
13. Rebuild the Neo4j Graph

The graph builder is:

text
graph_construction/neo4j_graph_builder.py

Rebuild the graph from the structural CSV:

bash
python -c "import graph_construction.neo4j_graph_builder as g; g.CSV_PATH='graph_construction/cloudtrail_structural.csv'; g.build_graph()"
14. Neo4j Sanity Checks

Check the number of roles:

bash
docker exec -it neo4j-local \
  cypher-shell -u neo4j -p test1234 \
  "MATCH (r:Role) RETURN count(r) AS role_count;"

Check User -> Role assumptions:

bash
docker exec -it neo4j-local \
  cypher-shell -u neo4j -p test1234 \
  "MATCH (u:User)-[:ASSUMES]->(r:Role) RETURN u.key, r.key LIMIT 20;"

Check whether those assumptions are followed by role activity:

bash
docker exec -it neo4j-local \
  cypher-shell -u neo4j -p test1234 \
  "MATCH (u:User)-[:ASSUMES]->(r:Role)-[e]->(x) RETURN count(*) AS linked_role_activity;"
15. Python Environment

Create the virtual environment:

bash
python3 -m venv venv

Activate it:

bash
source venv/bin/activate

Install dependencies:

bash
pip install -r requirements.txt

Check Python:

bash
python --version
16. Feature Engineering

Run the feature engine:

bash
python feature_engine9.py \
  --input incoming/synthetic_attack_chain.json

This processes CloudTrail JSON and produces structural information used to construct the graph.

17. Training HGT

Train the primary HGT model:

bash
python train.py \
  --model hgt \
  --epochs 100 \
  --save_dir ./checkpoints_hgt_corrected

The best checkpoint is written to:

text
checkpoints_hgt_corrected/best_HGT.pt
18. Training CLI

View all training arguments:

bash
python train.py --help

Important options include:

text
--model
--epochs
--hidden
--layers
--heads
--attn_dropout
--hgt_group
--lr
--dropout
--loss
--compare
--sampling
--max_neighbors
--num_hops
--num_samples_per_relation
--sampling_seed
--explain
--explain_method
--ablation
--threshold
--patience
--split
--seed
--neo4j_uri
--neo4j_user
--neo4j_pass
--device
--save_dir
19. Loss Function

The default training configuration uses focal loss:

text
alpha = 0.25
gamma = 2.0

Alternative BCE loss:

bash
--loss bce

The attack class is relatively rare, so class imbalance needs to be considered when evaluating the model.

20. Dataset Splits

The default split is stratified:

bash
--split stratified

The split is performed over a deterministic global edge order.

Conceptually:

text
All heterogeneous edge tensors
             |
             v
    Deterministic global order
             |
             v
       Stratified split
        /      |      \
       v       v       v
     train    val     test

An alternative is:

bash
--split principal_disjoint

which attempts to keep principal identities separated between splits.

The current dataset has few distinct labelled attacker identities, so principal-disjoint splitting can become degenerate depending on the random seed.

21. Global Edge Ordering

PyTorch Geometric stores every heterogeneous edge type separately.

For example:

text
(User, READ, Resource)
(User, WRITE, Resource)
(Role, READ, Resource)

each has its own tensors.

Training and evaluation still require a single flattened label vector.

The project therefore uses:

python
sorted(data.edge_types)

as the canonical global order.

The same order is used for:

text
model outputs
labels
train masks
validation masks
test masks

This prevents prediction/label index misalignment.

22. Checkpoint Inspection

The current corrected training checkpoints are:

text
checkpoints_hgt_corrected/best_HGT.pt
checkpoints_offline_current/best_GraphSAGE.pt
checkpoints_offline_current/best_GAT.pt

Inspect a training checkpoint:

bash
python -c "
import torch

c=torch.load(
    'checkpoints_hgt_corrected/best_HGT.pt',
    map_location='cpu',
    weights_only=False
)

print(type(c))
print(list(c.keys())[:20])
"
23. Wrapping the HGT Checkpoint

The training checkpoint is a bare state dictionary.

Wrap it with model metadata for inference:

bash
python infer.py \
  --wrap-checkpoint ./checkpoints_hgt_corrected/best_HGT.pt \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass test1234 \
  --model-type hgt \
  --heads 4 \
  --hidden-dim 128 \
  --num-layers 2 \
  --wrapped-output ./checkpoints_hgt_corrected/best_HGT_wrapped.pt
24. Verify the Wrapped Checkpoint
bash
python -c "
import torch

c=torch.load(
    'checkpoints_hgt_corrected/best_HGT_wrapped.pt',
    map_location='cpu',
    weights_only=False
)

a=c['model_args']

print('model_type:', a['model_type'])
print('hidden_dim:', a['hidden_dim'])
print('num_sage_layers:', a['num_sage_layers'])
print('heads:', a['heads'])
print('edge_feat_dim:', a['edge_feat_dim'])
print(
    'ASSUMES:',
    [e for e in a['edge_types'] if e[1]=='ASSUMES']
)
"

Expected configuration:

text
model_type: hgt
hidden_dim: 128
num_sage_layers: 2
heads: 4
edge_feat_dim: 8
24.1 Current Corrected Checkpoint Set

For the current offline experiments, use:

text
HGT:
checkpoints_hgt_corrected/best_HGT.pt

GraphSAGE:
checkpoints_offline_current/best_GraphSAGE.pt

GAT:
checkpoints_offline_current/best_GAT.pt

The HGT checkpoint is the corrected HGT checkpoint used by the current comparison and explainability runs.

The GraphSAGE and GAT checkpoints were freshly retrained against the same current offline graph schema used by compare_hgt_vs_sage_gat_ensemble.py.

These files are distinct from older checkpoints that may still exist under:

text
checkpoints/

Do not mix checkpoints trained against different graph schemas.

25. Inference

Run inference on a single input:

bash
python infer.py \
  --checkpoint ./checkpoints_hgt_corrected/best_HGT_wrapped.pt \
  --input ./incoming/synthetic_attack_chain.json \
  --seed-from-neo4j \
  --threshold 0.5 \
  --alert-dir ./alerts_hgt_corrected
26. Watch Mode

Run real-time directory watching:

bash
python infer.py \
  --checkpoint ./checkpoints_hgt_corrected/best_HGT_wrapped.pt \
  --watch incoming \
  --alert-dir alerts_hgt_corrected \
  --threshold 0.5 \
  --seed-from-neo4j

Other inference options include:

text
--poll-interval
--state-file
--hop-radius
--hidden-dim
--num-layers
--heads
--device

View inference options:

bash
python infer.py --help
27. Synthetic Attack Testing

Synthetic CloudTrail input can be processed using:

bash
python feature_engine9.py \
  --input incoming/synthetic_attack_chain.json

This is useful for:

text
pipeline testing
target extraction testing
qualitative inference
alert generation

Synthetic inputs should not be treated as a supervised evaluation benchmark unless they contain explicit ground-truth labels and are kept separate from training.

28. Evaluation Metrics

The system reports:

text
Accuracy
Precision
Recall
F1
AUROC
AUPR

Because attack events are much less frequent than benign events, accuracy should not be considered sufficient by itself.

Important metrics include:

text
Precision
Recall
F1
AUROC
AUPR

AUPR is particularly useful for imbalanced attack detection.

29. Explainability

The repository contains a single main explainability module:

text
explainability.py

hgt_attention_explainability.py is an optional HGT-internal attention inspection utility and is not required by the main explainability workflow.

The main module provides:

text
EdgeExplainer
FeatureAblation

The edge explainer supports:

text
gradient
gnnexplainer

Gradient-based explanation is the default dependable path:

bash
python3 explainability.py

The standalone runner loads:

text
graph_construction/cloudtrail_structural.csv
checkpoints_hgt_corrected/best_HGT.pt

and then:

text
offline graph
      |
      v
stratified test split
      |
      v
corrected HGT checkpoint
      |
      v
EdgeExplainer
      |
      +--> Top-K high-confidence test attack edges
      |
      +--> Feature importance
      |
      v
FeatureAblation
      |
      v
ΔF1 for each edge feature

Run only the edge-level explanation:

bash
python3 explainability.py --no-ablation

Run the real PyG GNNExplainer path:

bash
python3 explainability.py --method gnnexplainer

The explainability runner currently uses the HGT architecture:

text
hidden dim  = 128
layers      = 2
heads       = 4
edge dim    = 8

The feature names used by the explainer are:

text
hop_count
privilege_gain
privilege_gain_defined
abnormal_path_frequency
action_global_frequency_log
is_privilege_escalation_technique
is_read_only
edge_type

The top-k routine selects the highest-confidence attack predictions inside the supplied test mask and explains the selected edges.

The feature-ablation analysis zeroes one feature column at a time across the populated edge types, evaluates the model again, and reports the resulting F1 change.

For a syntax-only check:

bash
python3 -m py_compile explainability.py
29.1 Explainability Interpretation

The explanation is intended to answer:

text
Why did this particular edge/event receive a high attack probability?

rather than:

text
Which principal is globally risky?

The explanation can be traced back to the source log_id where that identifier is available in the loaded graph.

The project should distinguish:

text
local explanation:
one alert / one edge

global ablation:
overall model sensitivity to an edge feature

The two analyses answer different questions and should not be presented as interchangeable.

30. Blast Radius

The repository also contains:

text
blast_radius.py

The purpose is to analyze the graph consequences of suspicious activity and identify reachable resources or privilege paths associated with an event.

31. Current Dataset Structure

The current graph contains node types including:

text
User
Role
UnresolvedPrincipal
Resource
Policy

with populated heterogeneous edge triples derived from Neo4j.

The exact counts can change when the database is rebuilt from another dataset version.

32. Known Dataset Limitation

The current dataset contains real:

text
User -> ASSUMES -> Role

relationships.

However, the current full graph does not contain complete paths of the form:

text
User
  |
  | ASSUMES
  v
Role
  |
  | ACTION
  v
Resource

for the observed dataset.

Therefore, some multi-hop privilege-propagation features cannot represent the intended complete attack-chain semantics on the current dataset.

This is a dataset/linkage limitation, not necessarily a graph-construction code failure.

The current research implementation evaluates HGT without modifying the underlying dataset to artificially create those missing links.

33. External Evaluation

A stronger generalization experiment can use a completely separate benchmark:

text
Dataset A
   |
   v
Train HGT
   |
   v
Freeze model
   |
   v
Dataset B / Novel scenarios
   |
   v
Evaluate

Potential external evaluation sources include:

text
held-out internal data
novel synthetic attack chains
controlled AWS lab CloudTrail logs
external public CloudTrail/security datasets

The external benchmark should not be used to retrain the model before evaluation.

34. Tests

Run all tests:

bash
pytest -q

Verbose mode:

bash
pytest -v
35. Python Compilation Checks

Compile important project files:

bash
python -m py_compile \
  data_loader.py \
  train.py \
  infer.py \
  model_hgt.py \
  model_graphsage.py \
  model_gat.py \
  privilege_features.py \
  feature_engine9.py

No output means compilation succeeded.

36. Warning Checks

The DataFrame pipeline uses explicit copies and .loc assignments to avoid pandas chained-assignment warnings.

Run the strict loader warning check:

bash
python -W error::FutureWarning:data_loader \
  train.py \
  --model hgt \
  --epochs 1 \
  --save_dir ./warning_check_strict

This makes FutureWarnings originating from data_loader.py fail immediately.

The development environment may still show a dependency warning from PyTorch/PyG:

text
torch.jit.script is not supported in Python 3.14+

This originates inside the installed dependency stack rather than the project's data_loader.py.

37. HGT/PyG Compatibility Note

The currently installed PyTorch Geometric version does not accept:

text
group=
dropout=

in the same way expected by the project's HGT compatibility layer.

The implementation detects this and constructs HGTConv without unsupported parameters.

You may therefore see messages indicating that:

text
group

and:

text
dropout

are not being passed to the installed HGTConv.

This is a compatibility issue rather than a training failure and should be considered when reproducing experiments.

38. Neo4j TAGGING Note

The loader currently includes TAGGING in the supported relation list.

If the database does not contain any TAGGING relationships, querying:

cypher
MATCH (src)-[r:TAGGING]->(dst)

can produce a Neo4j UnknownRelationshipTypeWarning.

The loader then receives zero TAGGING edges.

This does not prevent other populated relations from being loaded.

39. End-to-End Workflow
Step 1: Activate environment
bash
cd /Users/akshaya/Downloads/hgt
source venv/bin/activate
Step 2: Start Neo4j
bash
docker start neo4j-local
Step 3: Process CloudTrail
bash
python feature_engine9.py \
  --input incoming/synthetic_attack_chain.json
Step 4: Rebuild graph
bash
python -c "import graph_construction.neo4j_graph_builder as g; g.CSV_PATH='graph_construction/cloudtrail_structural.csv'; g.build_graph()"
Step 5: Compile
bash
python -m py_compile \
  data_loader.py \
  train.py \
  infer.py \
  model_hgt.py \
  model_graphsage.py \
  model_gat.py \
  privilege_features.py \
  feature_engine9.py
Step 6: Train HGT
bash
python train.py \
  --model hgt \
  --epochs 100 \
  --save_dir ./checkpoints_hgt_corrected
Step 7: Wrap checkpoint
bash
python infer.py \
  --wrap-checkpoint ./checkpoints_hgt_corrected/best_HGT.pt \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass test1234 \
  --model-type hgt \
  --heads 4 \
  --hidden-dim 128 \
  --num-layers 2 \
  --wrapped-output ./checkpoints_hgt_corrected/best_HGT_wrapped.pt
Step 8: Verify checkpoint
bash
python -c "
import torch
c=torch.load(
    'checkpoints_hgt_corrected/best_HGT_wrapped.pt',
    map_location='cpu',
    weights_only=False
)
a=c['model_args']
print('model_type:', a['model_type'])
print('hidden_dim:', a['hidden_dim'])
print('num_sage_layers:', a['num_sage_layers'])
print('heads:', a['heads'])
print('edge_feat_dim:', a['edge_feat_dim'])
print('ASSUMES:', [e for e in a['edge_types'] if e[1]=='ASSUMES'])
"
Step 9: Run inference
bash
python infer.py \
  --checkpoint ./checkpoints_hgt_corrected/best_HGT_wrapped.pt \
  --input ./incoming/synthetic_attack_chain.json \
  --seed-from-neo4j \
  --threshold 0.5 \
  --alert-dir ./alerts_hgt_corrected
40. Pre-Push Checklist

Compile:

bash
python -m py_compile \
  data_loader.py \
  train.py \
  infer.py \
  model_hgt.py \
  model_graphsage.py \
  model_gat.py \
  privilege_features.py \
  feature_engine9.py

Run tests:

bash
pytest -q

Run HGT smoke test:

bash
python -W error::FutureWarning:data_loader \
  train.py \
  --model hgt \
  --epochs 1 \
  --save_dir ./warning_check_strict

Inspect repository state:

bash
git status

Review changes:

bash
git diff

Review staged changes:

bash
git diff --cached

Stage everything:

bash
git add .

Commit:

bash
git commit -m "Update heterogeneous HGT pipeline"

Push:

bash
git push
41. Recommended Repository Structure
text
hgt/
|
├── train.py
├── infer.py
├── data_loader.py
|
├── model_hgt.py
├── model_graphsage.py
├── model_gat.py
|
├── feature_engine9.py
├── privilege_features.py
├── explainability.py
├── blast_radius.py
|
├── train_offline_baselines.py
├── compare_hgt_vs_sage_gat_ensemble.py
|
├── graph_construction/
│   ├── neo4j_graph_builder.py
│   └── cloudtrail_structural.csv
|
├── incoming/
│   └── .gitkeep
|
├── alerts_hgt_corrected/
│   └── .gitkeep
|
├── checkpoints_hgt_corrected/
│   ├── best_HGT.pt
│   └── best_HGT_wrapped.pt
|
├── checkpoints_offline_current/
│   ├── best_GraphSAGE.pt
│   └── best_GAT.pt
|
├── datasets/
|
├── tests/
│   └── ...
|
├── requirements.txt
├── README.md
└── .gitignore

Runtime-generated inputs, alerts, and large checkpoints should generally be controlled through .gitignore rather than committing every generated artifact.

hgt_attention_explainability.py is not required by the main explainability workflow and can remain outside the minimal runtime path.

42. Research Contribution

The current implementation combines:

text
AWS CloudTrail
        +
Stateful Feature Engineering
        +
Heterogeneous Privilege Propagation Graph
        +
Relation-aware Graph Learning
        +
Edge-level Attack Classification
        +
Explainability
        +
Real-time Inference

The primary modeling contribution is the use of a heterogeneous graph architecture where cloud identities, roles, resources, policies, and other entities are represented as different node types while CloudTrail actions are represented as typed edges.

HGT is the primary model, while GraphSAGE and GAT provide comparison architectures.

The intended detection target is cloud attack activity at the event/edge level rather than only assigning an aggregate risk score to a principal.

43. Reproducibility

Record the following for every experiment:

text
Python version
PyTorch version
PyTorch Geometric version
Neo4j version
Dataset version
Random seed
Model type
Hidden dimension
Number of layers
Number of heads
Loss function
Learning rate
Weight decay
Dropout
Split strategy
Threshold
Comparison script version
Checkpoint path
Explainability method (if used)

Example configuration:

text
Model: HGT
Hidden: 128
Layers: 2
Heads: 4
Loss: Focal
Seed: 42
Split: Stratified
Optimizer: AdamW
Learning rate: 1e-3
Weight decay: 1e-4
44. Quick Command Reference
Activate environment
bash
source venv/bin/activate
Start Neo4j
bash
docker start neo4j-local
Run feature engine
bash
python feature_engine9.py --input incoming/synthetic_attack_chain.json
Rebuild Neo4j graph
bash
python -c "import graph_construction.neo4j_graph_builder as g; g.CSV_PATH='graph_construction/cloudtrail_structural.csv'; g.build_graph()"
Train HGT
bash
python train.py --model hgt --epochs 100 --save_dir ./checkpoints_hgt_corrected
Train corrected offline baselines
bash
python3 train_offline_baselines.py
Compare HGT with GraphSAGE + GAT ensemble
bash
python3 compare_hgt_vs_sage_gat_ensemble.py   --csv ./graph_construction/cloudtrail_structural.csv   --device cpu   --seed 42   --hgt_checkpoint ./checkpoints_hgt_corrected/best_HGT.pt   --sage_checkpoint ./checkpoints_offline_current/best_GraphSAGE.pt   --gat_checkpoint ./checkpoints_offline_current/best_GAT.pt
Run explainability
bash
python3 explainability.py
Run edge explanations without ablation
bash
python3 explainability.py --no-ablation
Test PyG GNNExplainer
bash
python3 explainability.py --method gnnexplainer
Compile explainability
bash
python3 -m py_compile explainability.py
Wrap HGT checkpoint
bash
python infer.py   --wrap-checkpoint ./checkpoints_hgt_corrected/best_HGT.pt   --neo4j-uri bolt://localhost:7687   --neo4j-user neo4j   --neo4j-pass test1234   --model-type hgt   --heads 4   --hidden-dim 128   --num-layers 2   --wrapped-output ./checkpoints_hgt_corrected/best_HGT_wrapped.pt
Run inference
bash
python infer.py   --checkpoint ./checkpoints_hgt_corrected/best_HGT_wrapped.pt   --input ./incoming/synthetic_attack_chain.json   --seed-from-neo4j   --threshold 0.5   --alert-dir ./alerts_hgt_corrected
Watch mode
bash
python infer.py   --checkpoint ./checkpoints_hgt_corrected/best_HGT_wrapped.pt   --watch incoming   --alert-dir alerts_hgt_corrected   --threshold 0.5   --seed-from-neo4j
Tests
bash
pytest -q
Syntax check
bash
python -m py_compile data_loader.py train.py infer.py model_hgt.py model_graphsage.py model_gat.py privilege_features.py feature_engine9.py explainability.py train_offline_baselines.py compare_hgt_vs_sage_gat_ensemble.py
Strict loader warning check
bash
python -W error::FutureWarning:data_loader   train.py   --model hgt   --epochs 1   --save_dir ./warning_check_strict
Git
bash
git status
git diff
git add .
git commit -m "Update heterogeneous HGT pipeline"
git push
Architecture at a Glance
text
                       AWS CloudTrail
                              |
                              v
                  +----------------------+
                  | Stateful Feature     |
                  | Engineering          |
                  | feature_engine9.py   |
                  +----------+-----------+
                             |
                             v
                  Structural CloudTrail CSV
                             |
                             v
                  +----------------------+
                  | Neo4j Graph Builder   |
                  +----------+-----------+
                             |
                             v
                  Privilege Propagation Graph
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
          User             Role           Resource
            |                |                |
            +----------------+----------------+
                             |
                             v
                      PyG HeteroData
                             |
             +---------------+---------------+
             |               |               |
             v               v               v
            HGT         GraphSAGE            GAT
         PRIMARY         BASELINE          BASELINE
             |               |               |
             +---------------+---------------+
                             |
                             v
                     Edge Classifier
                             |
                             v
                    Attack Probability
                             |
                    +--------+--------+
                    |                 |
                    v                 v
              Explainability       Alerts
45. Current Reproducibility Notes (2026-09-23)

The current experimental state includes the following changes:

Corrected offline pipeline

offline_pipeline.py is the current offline graph-loading path used by the corrected baseline training and HGT/baseline comparison.

offline_pipeline_fixed.py is a saved copy of the corrected implementation.

Corrected baseline checkpoints

GraphSAGE and GAT were retrained using the same offline graph schema used during comparison:

text
checkpoints_offline_current/best_GraphSAGE.pt
checkpoints_offline_current/best_GAT.pt
Corrected HGT checkpoint

The current HGT checkpoint is:

text
checkpoints_hgt_corrected/best_HGT.pt
Baseline ensemble comparison

The new comparison experiment is:

text
HGT
vs
GraphSAGE
vs
GAT
vs
GraphSAGE + GAT ensemble

The comparison is implemented in:

text
compare_hgt_vs_sage_gat_ensemble.py
Explainability

The main explainability entry point is now:

text
explainability.py

Run the complete explainability workflow with:

bash
python3 explainability.py

This includes edge-level explanation and feature ablation.

The optional hgt_attention_explainability.py utility is not required by the main workflow.

Deployment separation

Training and offline comparison are separate from live inference.

Do not retrain the baselines merely because infer.py is being run. Inference should load the existing wrapped checkpoint and process the new event/input stream.

Model's predictions on a synthetic attack chain along with blast radius and alerts
<img width="1451" height="440" alt="image" src="https://github.com/user-attachments/assets/7800bb6f-b569-4e0d-8a24-d1e1f2d34032" />
