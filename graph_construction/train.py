"""
train.py  (v3 — Privilege Propagation Graph)
===============================================
Full training pipeline for:
  - GraphSAGE anomaly detector (PRIMARY)
  - GAT anomaly detector        (COMPARISON)
over the heterogeneous Privilege Propagation Graph (6 node types, up to
20 populated (src_type, relation, dst_type) edge triples).

WHAT CHANGED FROM v2 AND WHY
─────────────────────────────────────────────────────────────────────────
- Loader: CloudTrailGraphLoader -> PrivilegePropagationGraphLoader (the
  class was renamed when the graph became heterogeneous; this import was
  actually broken in the codebase until this file was updated — verified
  via grep before starting this rewrite).
- Masks are now dicts ({triple: BoolTensor}), not single tensors, because
  a heterogeneous graph has no single edge_index to mask. All mask
  handling here goes through data_loader.py's global_labels /
  flatten_mask_dict so training/evaluation still operate on one flat
  vector under the hood.
- build_model() now passes `node_feat_dims` (a dict, one entry per node
  type) and `edge_types` (the list of populated triples) instead of two
  fixed feature-dimension integers — required because HeteroConv needs
  every relation declared at construction time (see
  model_graphsage.py's docstring).
- Explainability now uses explainability.py's EdgeExplainer (real PyG
  GNNExplainer path + a verified gradient fallback) instead of utils.py's
  old GNNExplainerWrapper placeholder.
- The session-level --hybrid path is gone (see utils.py's docstring for
  why — no verified session/temporal signal exists in this dataset).

Run:
    python3 train.py --model sage --epochs 100 --hidden 128 --loss focal
    python3 train.py --model gat  --epochs 100 --hidden 128 --loss focal
    python3 train.py --model both --epochs 100 --compare
    python3 train.py --model both --split principal_disjoint --seed 7
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import (
    PrivilegePropagationGraphLoader,
    compute_class_weights,
    flatten_mask_dict,
    global_labels,
    principal_disjoint_split,
    stratified_edge_split,
)
from model_gat import GATAnomalyDetector
from model_graphsage import GraphSAGEAnomalyDetector
from explainability import EdgeExplainer, FeatureAblation, TargetEdge
from utils import (
    FocalLoss,
    evaluate,
    print_comparison_table,
    print_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Privilege Propagation Graph GNN Trainer")
    p.add_argument("--model",    choices=["sage", "gat", "both"], default="both")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--layers",   type=int,   default=2)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--dropout",  type=float, default=0.3)
    p.add_argument("--loss",     choices=["focal", "bce"], default="focal")
    p.add_argument("--compare",  action="store_true", help="Print comparison table")
    p.add_argument("--explain",  action="store_true", help="Run explainability")
    p.add_argument("--explain_method", choices=["gradient", "gnnexplainer"], default="gradient",
                   help="gradient = verified autograd method (default). gnnexplainer = real PyG "
                        "Explainer/GNNExplainer path — verify against your installed PyG version "
                        "first, see explainability.py module docstring.")
    p.add_argument("--ablation", action="store_true", help="Run feature ablation")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int,   default=15,
                   help="Early stopping patience (epochs without val F1 improvement)")
    p.add_argument("--split",    choices=["stratified", "principal_disjoint"],
                   default="stratified",
                   help="stratified = random edge split preserving label ratio (default, "
                        "no ordering assumption). principal_disjoint = entity-disjoint split "
                        "for testing inductive generalisation; HIGH VARIANCE on this dataset "
                        "(only 13 principal-side identities, 2 with attack edges) — see "
                        "data_loader.py's principal_disjoint_split docstring.")
    p.add_argument("--seed",     type=int,   default=42, help="Split random seed")
    p.add_argument("--neo4j_uri",  default="bolt://localhost:7687")
    p.add_argument("--neo4j_user", default="neo4j")
    p.add_argument("--neo4j_pass", default="test1234")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default="./checkpoints")
    return p.parse_args()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(name: str, meta: dict, args) -> nn.Module:
    node_feat_dims = meta["node_feat_dim"]        # {ntype: dim}
    edge_types     = meta["populated_triples"]    # [(src,rel,dst), ...]
    e_feat         = meta["edge_feat_dim"]

    if name == "sage":
        return GraphSAGEAnomalyDetector(
            node_feat_dims=node_feat_dims,
            edge_types=edge_types,
            edge_feat_dim=e_feat,
            hidden_dim=args.hidden,
            num_sage_layers=args.layers,
            dropout=args.dropout,
        )
    elif name == "gat":
        return GATAnomalyDetector(
            node_feat_dims=node_feat_dims,
            edge_types=edge_types,
            edge_feat_dim=e_feat,
            hidden_dim=args.hidden,
            heads=4,
            num_gat_layers=args.layers,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Loss factory ──────────────────────────────────────────────────────────────

def build_loss(loss_name: str, pos_weight: torch.Tensor) -> nn.Module:
    if loss_name == "focal":
        log.info("Using Focal Loss (α=0.25, γ=2.0)")
        return FocalLoss(alpha=0.25, gamma=2.0)
    else:
        log.info("Using BCEWithLogitsLoss with pos_weight=%.2f", pos_weight.item())
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ── Single model training loop ────────────────────────────────────────────────

def train_model(
    name:        str,
    model:       nn.Module,
    data,
    train_masks: dict,
    val_masks:   dict,
    test_masks:  dict,
    args,
    pos_weight:  torch.Tensor,
) -> dict:
    device    = torch.device(args.device)
    model     = model.to(device)
    criterion = build_loss(args.loss, pos_weight.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Precompute the flat label vector and the flat train mask ONCE — both
    # are pure functions of `data`/`train_masks` and don't change per epoch.
    y_full          = torch.tensor(global_labels(data), dtype=torch.long, device=device)
    train_flat_mask = flatten_mask_dict(data, train_masks).to(device)

    best_val_f1  = -1.0
    best_state   = None
    patience_ctr = 0

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"best_{name}.pt")

    log.info("=" * 60)
    log.info("Training %s | device=%s | loss=%s | split=%s | seed=%d",
              name.upper(), args.device, args.loss, args.split, args.seed)
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        logits = model(data)  # flat, ordered by sorted(data.edge_types)

        y_train = y_full[train_flat_mask].float()
        loss = criterion(logits[train_flat_mask], y_train)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, data, val_masks, threshold=args.threshold)
            val_f1 = val_metrics["f1"]

            log.info(
                "Epoch %3d/%d | loss=%.4f | val_F1=%.4f | val_AUC=%.4f | %.1fs",
                epoch, args.epochs, loss.item(), val_f1,
                val_metrics["roc_auc"], time.time() - t0,
            )

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_state   = deepcopy(model.state_dict())
                patience_ctr = 0
                torch.save(best_state, ckpt_path)
                log.info("  ↑ New best val F1=%.4f — checkpoint saved.", best_val_f1)
            else:
                patience_ctr += 5
                if patience_ctr >= args.patience:
                    log.info("Early stopping at epoch %d (patience=%d).", epoch, args.patience)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Loaded best checkpoint (val F1=%.4f)", best_val_f1)

    log.info("\nFinal TEST evaluation — %s", name.upper())
    test_metrics = evaluate(model, data, test_masks, threshold=args.threshold)
    print_confusion_matrix(test_metrics["confusion"])

    return test_metrics


# ── Explainability runner ─────────────────────────────────────────────────────

def run_explainability(model, data, test_masks, args):
    log.info("\n── Explainability (%s) ─────────────────────────────────────", args.explain_method)

    explainer = EdgeExplainer(model, method=args.explain_method)
    log.info("Top-5 highest-confidence attack predictions on test set:")
    top_k = explainer.explain_top_k(data, test_masks, k=5)

    for target, feat_map in top_k.items():
        # data[triple].log_id is a plain list of opaque strings (see
        # data_loader.py) — indexing it already gives a str, so no
        # .item() (that's a torch.Tensor/numpy-scalar method, and would
        # raise AttributeError on a str).
        log_id = data[target.triple].log_id[target.local_index]
        print(f"\n  {target.triple} #{target.local_index}  (log_id={log_id})")
        for feat_name, importance in list(feat_map.items())[:5]:
            bar = "█" * int(importance * 40)
            print(f"    {feat_name:<35} {importance:.3f}  {bar}")

    if args.ablation:
        log.info("\n── Feature Ablation ────────────────────────────────────────")
        ablation = FeatureAblation(model)
        results  = ablation.run(data, test_masks, evaluate)
        print("\nFeature Ablation (F1 drop when zeroed, across all relations at once):")
        for feat, drop in list(results.items())[:10]:
            bar = "█" * max(0, int(drop * 200))
            sign = "+" if drop > 0 else ""
            print(f"  {feat:<35} {sign}{drop:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    log.info("Device: %s", device)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    loader = PrivilegePropagationGraphLoader(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_pass,
        device=args.device,
    )
    data, meta = loader.load()

    log.info("Node counts: %s", meta["node_counts"])
    log.info("Populated (src,rel,dst) triples: %d | edge_feat_dim: %d",
             len(meta["populated_triples"]), meta["edge_feat_dim"])
    log.info("%d principals flagged as known-attacker identities (metadata only, "
             "NOT a model feature — see data_loader.py docstring): %s",
             sum(meta["attacker_identity_by_key"].values()),
             [k for k, v in meta["attacker_identity_by_key"].items() if v])

    # ── 2. Train/val/test split ───────────────────────────────────────────────
    if args.split == "stratified":
        train_masks, val_masks, test_masks = stratified_edge_split(data, seed=args.seed)
    else:
        train_masks, val_masks, test_masks = principal_disjoint_split(data, seed=args.seed)

    train_masks = {t: m.to(device) for t, m in train_masks.items()}
    val_masks   = {t: m.to(device) for t, m in val_masks.items()}
    test_masks  = {t: m.to(device) for t, m in test_masks.items()}

    # ── 3. Class imbalance weight ─────────────────────────────────────────────
    pos_weight = compute_class_weights(data, train_masks).to(device)

    # ── 4. Train models ───────────────────────────────────────────────────────
    results = {}

    if args.model in ("sage", "both"):
        sage_model = build_model("sage", meta, args)
        sage_params = sum(p.numel() for p in sage_model.parameters())
        log.info("GraphSAGE parameters: %d", sage_params)

        sage_metrics = train_model(
            "GraphSAGE", sage_model, data,
            train_masks, val_masks, test_masks, args, pos_weight
        )
        results["GraphSAGE"] = sage_metrics

        if args.explain:
            run_explainability(sage_model, data, test_masks, args)

    if args.model in ("gat", "both"):
        gat_model  = build_model("gat", meta, args)
        gat_params = sum(p.numel() for p in gat_model.parameters())
        log.info("GAT parameters: %d", gat_params)

        gat_metrics = train_model(
            "GAT", gat_model, data,
            train_masks, val_masks, test_masks, args, pos_weight
        )
        results["GAT"] = gat_metrics

    # ── 5. Comparison table ───────────────────────────────────────────────────
    if args.compare and "GraphSAGE" in results and "GAT" in results:
        print_comparison_table(results["GraphSAGE"], results["GAT"])
        _print_recommendation(results["GraphSAGE"], results["GAT"])

    return results


def _print_recommendation(sage: dict, gat: dict):
    print("  Recommendation:")
    if sage["f1"] >= gat["f1"]:
        print("  ✅ GraphSAGE achieves equal or better F1.")
        print("     Preferred for production: inductive, scalable, faster.")
    else:
        gap = gat["f1"] - sage["f1"]
        print(f"  ⚠️  GAT has higher F1 (+{gap:.4f}).")
        print("     Consider GAT if the graph is bounded and latency allows.")
        print("     GraphSAGE still preferred for new-principal robustness.")
    print()


if __name__ == "__main__":
    main()
