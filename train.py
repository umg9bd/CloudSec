"""
train.py  (v4 — Privilege Propagation Graph + HGT + neighbor sampling)
===========================================================================
Full training pipeline for:
  - GraphSAGE anomaly detector (PRIMARY baseline)
  - GAT anomaly detector        (COMPARISON baseline)
  - HGT anomaly detector         (candidate PRIMARY structural model —
                                   see model_hgt.py; kept a candidate, not
                                   auto-promoted, until run_experiments.py's
                                   ablation actually shows it wins — see
                                   FINAL_REPORT.md "Is HGT justified")
over the heterogeneous Privilege Propagation Graph (6 node types, up to
20 populated (src_type, relation, dst_type) edge triples).

WHAT CHANGED IN v4 AND WHY
─────────────────────────────────────────────────────────────────────────
- --model now accepts "hgt" alongside the existing "sage"/"gat", and
  "all" (sage+gat+hgt together) in addition to the existing "both"
  (sage+gat only, kept for backward compatibility with any existing
  scripts/muscle memory calling --model both).
- --sampling {none,relation_aware,uniform,full} (default "none" —
  training behaviour is BYTE-FOR-BYTE unchanged unless this is passed
  explicitly; see neighbor_sampling.py). When not "none", ONE sampled
  training view is built (seeded from train-split edges only — val/test
  are always evaluated on the full, unsampled graph, matching
  train_scalable.py's existing train_hgt() convention) and shared across
  every model trained in this run, so a sampling ablation (task
  requirement 10/11: "same dataset... comparable training conditions")
  compares models under literally the same sampled subgraph, not
  independently-resampled ones.
- --max_neighbors / --num_hops / --num_samples_per_relation /
  --sampling_seed / --sampling_strategy expose neighbor_sampling.py's
  SamplingConfig (task requirement 7 — nothing hardcoded).
- --heads (GAT/HGT attention heads) and --hgt_layers / --attn_dropout
  are now CLI flags instead of GAT's previous hardcoded heads=4.
- GraphSAGE and GAT remain full-batch by default and are UNCHANGED when
  --sampling none (the default) — this file does not silently ensemble
  or auto-promote HGT; --model chooses exactly what gets trained, same
  as before.

WHAT CHANGED IN v3 (kept for history)
─────────────────────────────────────────────────────────────────────────
- Loader: CloudTrailGraphLoader -> PrivilegePropagationGraphLoader.
- Masks are dicts ({triple: BoolTensor}), going through data_loader.py's
  global_labels / flatten_mask_dict.
- build_model() takes `node_feat_dims` (dict) and `edge_types` (list).
- Explainability uses explainability.py's EdgeExplainer.
- The session-level --hybrid path is gone (no verified temporal signal).

Run:
    python3 train.py --model sage --epochs 100 --hidden 128 --loss focal
    python3 train.py --model gat  --epochs 100 --hidden 128 --loss focal
    python3 train.py --model hgt  --epochs 100 --hidden 128 --heads 4
    python3 train.py --model all  --epochs 100 --compare
    python3 train.py --model all  --sampling relation_aware --max_neighbors 50 --compare
    python3 train.py --model both --split principal_disjoint --seed 7

For the full ablation matrix (task requirement 11: A-E, sweeping K), use
run_experiments.py instead of calling this file directly per cell.
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
from model_hgt import HGTAnomalyDetector
from explainability import EdgeExplainer, FeatureAblation, TargetEdge
from neighbor_sampling import SamplingConfig, build_sampled_training_view
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
    p.add_argument("--model",    choices=["sage", "gat", "hgt", "both", "all"], default="both",
                   help="both = sage+gat (backward compatible). all = sage+gat+hgt.")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--hidden",   type=int,   default=128, help="HIDDEN_DIM")
    p.add_argument("--layers",   type=int,   default=2,   help="NUM_LAYERS (sage/gat/hgt encoder depth)")
    p.add_argument("--heads",    type=int,   default=4,   help="NUM_HEADS (gat/hgt attention heads)")
    p.add_argument("--attn_dropout", type=float, default=0.1, help="HGT attention dropout (see model_hgt.py's honest caveat: not every installed PyG version honours this)")
    p.add_argument("--hgt_group", default="sum",
                   help="HGT relation-aggregation group ('sum'/'mean'/'max'). Passed through to "
                        "HGTConv; NOTE some PyG versions no longer accept this kwarg at all — see "
                        "model_hgt.py's _make_hgt_conv, which detects that and logs a warning "
                        "rather than silently ignoring the flag.")
    p.add_argument("--lr",       type=float, default=1e-3, help="LEARNING_RATE")
    p.add_argument("--dropout",  type=float, default=0.3, help="DROPOUT")
    p.add_argument("--loss",     choices=["focal", "bce"], default="focal")
    p.add_argument("--compare",  action="store_true", help="Print comparison table")
    # ── Neighbor sampling (task requirements 4/5/6/7) — see neighbor_sampling.py.
    # Default "none" preserves this file's pre-existing full-batch behaviour
    # exactly; sampling is opt-in, never automatic.
    p.add_argument("--sampling", choices=["none", "relation_aware", "uniform", "full"], default="none",
                   help="none (default) = full-batch, unchanged behaviour. relation_aware = task "
                        "requirement 5's floor-then-proportional allocator. uniform = the naive "
                        "baseline it's compared against. full = sampler runs but never caps "
                        "anything (isolates hop-limiting from degree-capping in the ablation).")
    p.add_argument("--max_neighbors", type=int, default=50, help="MAX_NEIGHBORS")
    p.add_argument("--num_hops", type=int, default=2, help="NUM_HOPS")
    p.add_argument("--num_samples_per_relation", type=int, default=5, help="NUM_SAMPLES_PER_RELATION")
    p.add_argument("--sampling_seed", type=int, default=42)
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
            heads=args.heads,
            num_gat_layers=args.layers,
            dropout=args.dropout,
        )
    elif name == "hgt":
        return HGTAnomalyDetector(
            node_feat_dims=node_feat_dims,
            edge_types=edge_types,
            edge_feat_dim=e_feat,
            hidden_dim=args.hidden,
            heads=args.heads,
            num_hgt_layers=args.layers,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            group=args.hgt_group,
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
    train_view:  tuple = None,
) -> dict:
    """
    `train_view`: optional (sampled_train_data, sampled_train_mask) from
    neighbor_sampling.build_sampled_training_view — when given, the epoch
    loop's forward/backward runs on THAT (smaller) graph instead of the
    full `data`/`train_masks`, while validation/test evaluation always
    still runs on the full, unsampled `data` (same convention
    train_scalable.py's train_hgt() already uses: measure generalisation
    beyond whatever region training was restricted to). When None
    (default — i.e. --sampling none), behaviour is identical to before
    this parameter existed.
    """
    device    = torch.device(args.device)
    model     = model.to(device)
    criterion = build_loss(args.loss, pos_weight.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Precompute the flat label vector and the flat train mask ONCE — both
    # are pure functions of `train_data`/`train_masks` and don't change per
    # epoch. `train_data` is the (possibly sampled) graph the forward pass
    # actually runs on; `data` (used below for val/test) is always the
    # full graph regardless of `train_view`.
    train_data = data
    if train_view is not None:
        train_data, train_masks = train_view
        log.info(
            "[%s] training on SAMPLED view: %s nodes, %s edges (vs full graph %s nodes, %d edges) "
            "— see --sampling %s",
            name, {k: v.x.shape[0] for k, v in train_data.node_items()},
            {k: v.edge_index.shape[1] for k, v in train_data.edge_items()},
            {k: v.x.shape[0] for k, v in data.node_items()},
            sum(data[t].edge_index.shape[1] for t in data.edge_types),
            args.sampling,
        )
    y_full          = torch.tensor(global_labels(train_data), dtype=torch.long, device=device)
    train_flat_mask = flatten_mask_dict(train_data, train_masks).to(device)

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

        logits = model(train_data)  # flat, ordered by sorted(train_data.edge_types)

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
             sum(bool(v) for v in meta["attacker_identity_by_key"].values()),
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

    # ── 3b. Optional shared sampled training view (task requirement 4/5/6/7)
    # Built ONCE, from train-split edges only, and reused for every model
    # trained in this run — so a sampling ablation compares models under
    # literally the same sampled subgraph rather than independently
    # resampled ones (task requirement 10's "comparable training
    # conditions"). None of this runs, and nothing about training changes,
    # unless --sampling is explicitly passed.
    train_view = None
    if args.sampling != "none":
        sampling_cfg = SamplingConfig(
            max_neighbors=args.max_neighbors,
            num_hops=args.num_hops,
            num_samples_per_relation=args.num_samples_per_relation,
            strategy=args.sampling,
            seed=args.sampling_seed,
        )
        log.info("Building shared sampled training view: %s", sampling_cfg)
        train_view = build_sampled_training_view(
            data, meta["populated_triples"], train_masks, sampling_cfg, device=args.device,
        )

    # ── 4. Train models ───────────────────────────────────────────────────────
    results = {}

    if args.model in ("sage", "both", "all"):
        sage_model = build_model("sage", meta, args)
        sage_params = sum(p.numel() for p in sage_model.parameters())
        log.info("GraphSAGE parameters: %d", sage_params)

        sage_metrics = train_model(
            "GraphSAGE", sage_model, data,
            train_masks, val_masks, test_masks, args, pos_weight, train_view=train_view,
        )
        results["GraphSAGE"] = sage_metrics

        if args.explain:
            run_explainability(sage_model, data, test_masks, args)

    if args.model in ("gat", "both", "all"):
        gat_model  = build_model("gat", meta, args)
        gat_params = sum(p.numel() for p in gat_model.parameters())
        log.info("GAT parameters: %d", gat_params)

        gat_metrics = train_model(
            "GAT", gat_model, data,
            train_masks, val_masks, test_masks, args, pos_weight, train_view=train_view,
        )
        results["GAT"] = gat_metrics

    if args.model in ("hgt", "all"):
        hgt_model  = build_model("hgt", meta, args)
        hgt_params = sum(p.numel() for p in hgt_model.parameters())
        log.info("HGT parameters: %d", hgt_params)

        hgt_metrics = train_model(
            "HGT", hgt_model, data,
            train_masks, val_masks, test_masks, args, pos_weight, train_view=train_view,
        )
        results["HGT"] = hgt_metrics

        if args.explain:
            run_explainability(hgt_model, data, test_masks, args)

    # ── 5. Comparison table ───────────────────────────────────────────────────
    if args.compare:
        if len(results) >= 3:
            from train_scalable import compare_n_way
            compare_n_way(results)
        elif "GraphSAGE" in results and "GAT" in results:
            print_comparison_table(results["GraphSAGE"], results["GAT"])
            _print_recommendation(results["GraphSAGE"], results["GAT"])
        elif len(results) == 1:
            log.info("Only one model was trained (--model %s) — nothing to compare.", args.model)

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
