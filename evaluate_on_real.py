"""
Evaluates an already-trained, wrapped checkpoint against whatever graph
currently exists in Neo4j (real_dataset_test_structural.csv, in our case),
reusing the training-fitted scalers/encoders baked into the checkpoint by
infer.py's wrap_checkpoint -- never re-fitting on the evaluation graph.

This is the train-on-synthetic / test-on-real number, comparable to the
rule-based baselines in evaluate_baselines.py (GuardDuty-style F1=0.732).

Usage:
    python evaluate_on_real.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage
"""

import argparse

import torch

from data_loader import PrivilegePropagationGraphLoader
from utils import evaluate


def build_model_from_args(name: str, model_args: dict):
    node_feat_dims = model_args["node_feat_dims"]
    edge_types = model_args["edge_types"]
    edge_feat_dim = model_args["edge_feat_dim"]
    hidden_dim = model_args["hidden_dim"]
    dropout = model_args["dropout"]

    if name == "sage":
        from model_graphsage import GraphSAGEAnomalyDetector
        return GraphSAGEAnomalyDetector(
            node_feat_dims=node_feat_dims, edge_types=edge_types, edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim, num_sage_layers=model_args["num_sage_layers"], dropout=dropout,
        )
    elif name == "gat":
        from model_gat import GATAnomalyDetector
        return GATAnomalyDetector(
            node_feat_dims=node_feat_dims, edge_types=edge_types, edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim, heads=4, num_gat_layers=model_args["num_sage_layers"], dropout=dropout,
        )
    raise ValueError(f"Unknown model: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["sage", "gat"], required=True)
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="test1234")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    print(f"Loading checkpoint {args.checkpoint} ...")
    # weights_only=False: this checkpoint carries fitted sklearn scalers/
    # encoders (not just tensor weights), which PyTorch 2.6+'s default
    # weights_only=True refuses to unpickle. Safe here -- this is our own
    # checkpoint from this session, not a downloaded/untrusted file.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = ckpt["model_args"]
    fit_artifacts = ckpt["fit_artifacts"]

    print("Loading evaluation graph from Neo4j, reusing training-fitted scalers/encoders (no re-fit)...")
    loader = PrivilegePropagationGraphLoader(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_pass,
        fit_artifacts=fit_artifacts,
    )
    data, meta = loader.load()

    # The model's HeteroConv layers have a distinct weight matrix per
    # (src_type, relation, dst_type) triple, fixed at training time
    # (model_args["edge_types"], from the synthetic graph). A real graph
    # can and does contain triples never seen in training -- e.g. this
    # dataset has actual :Service nodes, which never appeared in synthetic
    # data -- and the model has no parameters to score those edges with.
    # forward() silently drops them, producing fewer logits than total
    # edges. We evaluate only on the triples the model was trained for,
    # and report how much real-world coverage that excludes -- an honest
    # generalization-gap number, not something to paper over.
    trained_triples = set(tuple(t) for t in model_args["edge_types"])
    real_triples = set(data.edge_types)
    untrained_triples = real_triples - trained_triples
    total_real_edges = sum(data[t].y.shape[0] for t in real_triples)
    excluded_edges = sum(data[t].y.shape[0] for t in untrained_triples)

    print(f"\nModel trained on {len(trained_triples)} edge-type triples; "
          f"real graph has {len(real_triples)} triples.")
    if untrained_triples:
        print(f"EXCLUDING {excluded_edges}/{total_real_edges} real edges "
              f"({excluded_edges/total_real_edges:.1%}) outside the trained schema:")
        for t in sorted(untrained_triples):
            print(f"    {t}  ({data[t].y.shape[0]} edges)")
        for t in untrained_triples:
            del data[t]

    model = build_model_from_args(args.model, model_args)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Evaluate on every remaining (in-schema) edge -- there's no train/val
    # split to make on a held-out evaluation graph, every edge is "test".
    all_true_masks = {t: torch.ones(data[t].y.shape[0], dtype=torch.bool) for t in data.edge_types}

    print("\n" + "=" * 60)
    print(f"REAL-DATA EVALUATION -- {args.model.upper()} (trained on synthetic)")
    print("=" * 60)
    metrics = evaluate(model, data, all_true_masks, threshold=args.threshold)

    print("\nConfusion Matrix:")
    print(f"{'':>15} {'Pred_normal':>15} {'Pred_attack':>15}")
    cm = metrics["confusion"]
    print(f"{'True_normal':>15} {cm[0][0]:>15} {cm[0][1]:>15}")
    print(f"{'True_attack':>15} {cm[1][0]:>15} {cm[1][1]:>15}")

    print(f"\nSUMMARY: P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
          f"F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}")
    print("Compare against GuardDuty-style rule baseline: F1=0.732 [95% CI: 0.672, 0.790]")


if __name__ == "__main__":
    main()
