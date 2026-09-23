from types import SimpleNamespace
import torch

from offline_pipeline import load_offline
from data_loader import (
    stratified_edge_split,
    compute_class_weights,
)
from train import build_model, train_model

CSV_PATH = "./graph_construction/cloudtrail_structural.csv"
SAVE_DIR = "./checkpoints_offline_current"

args = SimpleNamespace(
    device="cpu",
    hidden=128,
    layers=2,
    heads=4,
    dropout=0.3,
    lr=1e-3,
    loss="focal",
    epochs=100,
    threshold=0.5,
    patience=15,
    split="stratified",
    seed=42,
    save_dir=SAVE_DIR,
)

print("=" * 70)
print("LOADING CURRENT OFFLINE GRAPH")
print("=" * 70)

data, meta = load_offline(CSV_PATH, device=args.device)

print("\nNode counts:")
for ntype, x in data.node_items():
    print(f"  {ntype}: {x.x.shape[0]}")

print("\nPopulated edge types:")
for triple in meta["populated_triples"]:
    print(f"  {triple}")

print(f"\nTotal edge triples: {len(meta['populated_triples'])}")
print(f"Edge feature dimension: {meta['edge_feat_dim']}")

print("\n" + "=" * 70)
print("CREATING STRATIFIED SPLIT")
print("=" * 70)

train_masks, val_masks, test_masks = stratified_edge_split(
    data,
    seed=args.seed,
)

device = torch.device(args.device)

train_masks = {t: m.to(device) for t, m in train_masks.items()}
val_masks   = {t: m.to(device) for t, m in val_masks.items()}
test_masks  = {t: m.to(device) for t, m in test_masks.items()}

pos_weight = compute_class_weights(data, train_masks).to(device)

print(f"\nSeed: {args.seed}")
print(f"Positive class weight: {pos_weight.item():.6f}")

print("\n" + "=" * 70)
print("TRAINING GRAPHSAGE")
print("=" * 70)

sage_model = build_model("sage", meta, args)

sage_metrics = train_model(
    "GraphSAGE",
    sage_model,
    data,
    train_masks,
    val_masks,
    test_masks,
    args,
    pos_weight,
)

print("\nGraphSAGE test metrics:")
for k, v in sage_metrics.items():
    if k != "confusion":
        print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("TRAINING GAT")
print("=" * 70)

gat_model = build_model("gat", meta, args)

gat_metrics = train_model(
    "GAT",
    gat_model,
    data,
    train_masks,
    val_masks,
    test_masks,
    args,
    pos_weight,
)

print("\nGAT test metrics:")
for k, v in gat_metrics.items():
    if k != "confusion":
        print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nSaved checkpoints:")
print(f"  {SAVE_DIR}/best_GraphSAGE.pt")
print(f"  {SAVE_DIR}/best_GAT.pt")
