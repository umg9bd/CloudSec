"""
train.py
========
Full training pipeline for:
  - GraphSAGE anomaly detector (PRIMARY)
  - GAT anomaly detector        (COMPARISON)

Includes:
  - Time-aware train/val/test split
  - Focal loss + weighted BCE (configurable)
  - Early stopping
  - Evaluation at each epoch (F1, AUC, confusion)
  - Model comparison table
  - Explainability via GNNExplainerWrapper + FeatureAblation
  - Session-level LSTM hybrid demo
  - Mini-batch path (NeighborLoader) for 100K+ edge graphs

Run:
    python3 train.py --model sage --epochs 100 --hidden 128 --loss focal
    python3 train.py --model gat  --epochs 100 --hidden 128 --loss focal
    python3 train.py --model both --epochs 100 --compare
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import (
    CloudTrailGraphLoader,
    compute_class_weights,
    time_aware_split,
)
from model_gat import GATAnomalyDetector
from model_graphsage import GraphSAGEAnomalyDetector
from utils import (
    FeatureAblation,
    FocalLoss,
    GNNExplainerWrapper,
    HybridSAGELSTM,
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
    p = argparse.ArgumentParser(description="CloudTrail GNN Trainer")
    p.add_argument("--model",    choices=["sage", "gat", "both"], default="both")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--layers",   type=int,   default=2)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--dropout",  type=float, default=0.3)
    p.add_argument("--loss",     choices=["focal", "bce"], default="focal")
    p.add_argument("--compare",  action="store_true", help="Print comparison table")
    p.add_argument("--explain",  action="store_true", help="Run explainability")
    p.add_argument("--ablation", action="store_true", help="Run feature ablation")
    p.add_argument("--hybrid",   action="store_true", help="Train SAGE+LSTM hybrid")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int,   default=15,
                   help="Early stopping patience (epochs without val F1 improvement)")
    p.add_argument("--neo4j_uri",  default="bolt://localhost:7687")
    p.add_argument("--neo4j_user", default="neo4j")
    p.add_argument("--neo4j_pass", default="test1234")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", default="./checkpoints")
    return p.parse_args()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(name: str, meta: dict, args) -> nn.Module:
    p_feat   = meta["n_principal_feat"]
    s_feat   = meta["n_service_feat"]
    e_feat   = meta["edge_feat_dim"]

    if name == "sage":
        return GraphSAGEAnomalyDetector(
            principal_feat_dim=p_feat,
            service_feat_dim=s_feat,
            edge_feat_dim=e_feat,
            hidden_dim=args.hidden,
            num_sage_layers=args.layers,
            dropout=args.dropout,
        )
    elif name == "gat":
        return GATAnomalyDetector(
            principal_feat_dim=p_feat,
            service_feat_dim=s_feat,
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
    name:       str,
    model:      nn.Module,
    data,
    train_mask: torch.Tensor,
    val_mask:   torch.Tensor,
    test_mask:  torch.Tensor,
    args,
    pos_weight: torch.Tensor,
) -> dict:
    """
    Train one model. Returns test metrics dict.
    """
    device   = torch.device(args.device)
    model    = model.to(device)
    criterion = build_loss(args.loss, pos_weight.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1   = -1.0
    best_state    = None
    patience_ctr  = 0

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"best_{name}.pt")

    log.info("=" * 60)
    log.info("Training %s | device=%s | loss=%s", name.upper(), args.device, args.loss)
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        # Forward pass (full-graph; see mini-batch note at bottom for 100K+)
        logits = model(data)

        # Compute loss on training edges only
        y_train = data["principal", "invoked", "awsservice"].y[train_mask].float()
        loss = criterion(logits[train_mask], y_train)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, data, val_mask, threshold=args.threshold)
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

    # ── Load best checkpoint & evaluate on test ───────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Loaded best checkpoint (val F1=%.4f)", best_val_f1)

    log.info("\nFinal TEST evaluation — %s", name.upper())
    test_metrics = evaluate(model, data, test_mask, threshold=args.threshold)
    print_confusion_matrix(test_metrics["confusion"])

    return test_metrics


# ── Explainability runner ─────────────────────────────────────────────────────

def run_explainability(model, data, test_mask, args):
    log.info("\n── Explainability ──────────────────────────────────────────")

    explainer = GNNExplainerWrapper(model)
    log.info("Top-5 highest-confidence attack predictions on test set:")
    top_k = explainer.explain_top_k(data, test_mask, k=5)

    for edge_idx, feat_map in top_k.items():
        prob = torch.sigmoid(model(data)[edge_idx]).item()
        print(f"\n  Edge {edge_idx}  |  pred_prob={prob:.4f}")
        for feat_name, importance in list(feat_map.items())[:5]:
            bar = "█" * int(importance * 40)
            print(f"    {feat_name:<30} {importance:.3f}  {bar}")

    if args.ablation:
        log.info("\n── Feature Ablation ────────────────────────────────────────")
        ablation = FeatureAblation(model)
        results  = ablation.run(data, test_mask)
        print("\nFeature Ablation (F1 drop when zeroed):")
        for feat, drop in list(results.items())[:10]:
            bar = "█" * max(0, int(drop * 200))
            sign = "+" if drop > 0 else ""
            print(f"  {feat:<30} {sign}{drop:.4f}  {bar}")


# ── Hybrid SAGE + LSTM runner ─────────────────────────────────────────────────

def run_hybrid(sage_model, data, train_mask, test_mask, args, pos_weight):
    log.info("\n── Hybrid GraphSAGE + LSTM (session-level) ─────────────────")
    device = torch.device(args.device)

    hidden_dim   = args.hidden
    edge_emb_dim = hidden_dim * 3    # from get_edge_embeddings

    hybrid = HybridSAGELSTM(
        sage_model=sage_model,
        edge_embed_dim=edge_emb_dim,
        lstm_hidden=64,
        num_lstm_layers=2,
        dropout=args.dropout,
    ).to(device)

    # Group edges into per-principal sessions
    sessions = HybridSAGELSTM.group_edges_by_principal(data)
    # Filter to training sessions (sessions that have at least one training edge)
    train_idx_set = set(train_mask.nonzero(as_tuple=True)[0].tolist())

    session_edge_seqs = []
    session_labels    = []

    edge_attr = data["principal", "invoked", "awsservice"]
    y_session = edge_attr.y_session

    for principal_idx, edge_indices in sessions.items():
        # Keep only edges in train_mask for training sessions
        train_edges = [e for e in edge_indices if e in train_idx_set]
        if not train_edges:
            continue
        session_edge_seqs.append(torch.tensor(train_edges, device=device))
        # Session label = 1 if ANY edge is session_label=1
        sess_lbl = int(y_session[train_edges].max().item())
        session_labels.append(sess_lbl)

    if not session_edge_seqs:
        log.warning("No training sessions found — skipping hybrid training.")
        return

    sess_label_tensor = torch.tensor(session_labels, dtype=torch.float, device=device)
    pos_w_sess = (sess_label_tensor == 0).sum() / (sess_label_tensor == 1).sum().clamp(min=1)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_w_sess)
    opt        = optim.AdamW(hybrid.parameters(), lr=args.lr, weight_decay=1e-4)

    log.info(
        "Hybrid training | sessions=%d | attack_sessions=%d",
        len(session_labels), sum(session_labels)
    )

    for epoch in range(1, min(args.epochs, 50) + 1):
        hybrid.train()
        opt.zero_grad()
        logits = hybrid(data, session_edge_seqs)
        loss   = criterion(logits, sess_label_tensor)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            preds = (torch.sigmoid(logits) >= 0.5).long()
            acc   = (preds == sess_label_tensor.long()).float().mean()
            log.info("Hybrid epoch %3d | loss=%.4f | session_acc=%.4f", epoch, loss.item(), acc.item())

    log.info("Hybrid training complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    log.info("Device: %s", device)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    loader = CloudTrailGraphLoader(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_pass,
        device=args.device,
    )
    data, meta = loader.load()

    # Store feature dims in meta for model construction
    meta["n_principal_feat"] = data["principal"].x.shape[1]
    meta["n_service_feat"]   = data["awsservice"].x.shape[1]

    log.info("Edge feature dim: %d", meta["edge_feat_dim"])
    log.info("Principal feat dim: %d | Service feat dim: %d",
             meta["n_principal_feat"], meta["n_service_feat"])

    # ── 2. Train/val/test split ───────────────────────────────────────────────
    train_mask, val_mask, test_mask = time_aware_split(data)
    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)
    test_mask  = test_mask.to(device)

    # ── 3. Class imbalance weight ─────────────────────────────────────────────
    y_train    = data["principal", "invoked", "awsservice"].y[train_mask]
    pos_weight = compute_class_weights(y_train).to(device)

    # ── 4. Train models ───────────────────────────────────────────────────────
    results = {}

    if args.model in ("sage", "both"):
        sage_model = build_model("sage", meta, args)
        sage_params = sum(p.numel() for p in sage_model.parameters())
        log.info("GraphSAGE parameters: %d", sage_params)

        sage_metrics = train_model(
            "GraphSAGE", sage_model, data,
            train_mask, val_mask, test_mask, args, pos_weight
        )
        results["GraphSAGE"] = sage_metrics

        if args.explain:
            run_explainability(sage_model, data, test_mask, args)

        if args.hybrid:
            run_hybrid(sage_model, data, train_mask, test_mask, args, pos_weight)

    if args.model in ("gat", "both"):
        gat_model  = build_model("gat", meta, args)
        gat_params = sum(p.numel() for p in gat_model.parameters())
        log.info("GAT parameters: %d", gat_params)

        gat_metrics = train_model(
            "GAT", gat_model, data,
            train_mask, val_mask, test_mask, args, pos_weight
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


# ── Mini-batch scaling note ───────────────────────────────────────────────────
"""
SCALING TO 100K+ EDGES
────────────────────────
The full-graph forward pass above works well up to ~50K edges.
For larger graphs, switch to NeighborLoader:

    from torch_geometric.loader import NeighborLoader

    loader = NeighborLoader(
        data,
        num_neighbors={
            ("principal", "invoked", "awsservice"): [15, 10],
        },
        batch_size=512,
        input_nodes=("principal", train_principal_mask),
    )

    for batch in loader:
        batch   = batch.to(device)
        logits  = model(batch)
        # batch contains the sampled subgraph; labels/masks are preserved
        mask    = batch["principal","invoked","awsservice"].input_id
        y_batch = batch["principal","invoked","awsservice"].y
        loss    = criterion(logits, y_batch.float())
        ...

Neighbourhood sampling:
  num_neighbors=[15, 10] means:
    Layer 1: sample up to 15 neighbours for each node
    Layer 2: sample up to 10 neighbours for each of THOSE nodes
  Memory: O(batch_size × 15 × 10) instead of O(N²)
  This is the core scalability advantage of GraphSAGE.
"""


if __name__ == "__main__":
    main()
