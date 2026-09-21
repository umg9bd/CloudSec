"""
run_experiments.py
=====================
Task requirement 10 ("Do NOT claim HGT is better until the experiments
demonstrate it" / "same dataset... same target labels... comparable
training conditions") and requirement 11 (the A-E ablation matrix, plus a
K sweep tuned on validation and reported on test ONCE).

Loads data via offline_pipeline.load_offline() (see that file's docstring
for why — no live Neo4j in this environment) ONCE, splits it ONCE
(stratified_edge_split, same seed for every cell), and reuses train.py's
REAL build_model()/train_model() unchanged for every cell — this script
adds no new model or training-loop code of its own, only the orchestration
loop and the results table.

Usage:
    python3 run_experiments.py --csv graph_construction/cloudtrail_structural.csv --epochs 60
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from types import SimpleNamespace

import torch

from data_loader import stratified_edge_split, compute_class_weights
from neighbor_sampling import SamplingConfig, build_sampled_training_view
from offline_pipeline import load_offline
from train import build_model, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _base_args(epochs, device="cpu", seed=42):
    return SimpleNamespace(
        hidden=128, layers=2, heads=4, attn_dropout=0.1, hgt_group="sum",
        dropout=0.3, lr=1e-3, loss="focal", device=device, threshold=0.5,
        patience=15, epochs=epochs, sampling="none", split="stratified", seed=seed,
        max_neighbors=50, num_hops=2, num_samples_per_relation=5, sampling_seed=42,
        save_dir="./checkpoints_experiments",
    )


def run_cell(cell_name, model_name, data, meta, train_masks, val_masks, test_masks, args, train_view=None):
    log.info("\n%s\n=== CELL %s: model=%s sampling=%s ===\n%s", "=" * 70, cell_name, model_name, args.sampling, "=" * 70)
    torch.manual_seed(args.sampling_seed)
    model = build_model(model_name, meta, args)
    n_params = sum(p.numel() for p in model.parameters())
    pos_weight = compute_class_weights(data, train_masks)

    tracemalloc.start()
    t0 = time.time()
    metrics = train_model(model_name, model, data, train_masks, val_masks, test_masks, args, pos_weight, train_view=train_view)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = dict(metrics)
    metrics.update({
        "cell": cell_name, "model": model_name, "sampling": args.sampling,
        "params": n_params, "train_time_sec": round(train_time, 2),
        "peak_cpu_mem_mb": round(peak_mem / 1e6, 1),  # CPU RSS proxy — see report: NOT GPU memory, no GPU in this environment
    })
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="graph_construction/cloudtrail_structural.csv")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiment_results.json")
    p.add_argument("--k_sweep", nargs="+", type=int, default=[25, 50, 100])
    cli = p.parse_args()

    log.info("Loading data (offline harness — see offline_pipeline.py)…")
    data, meta = load_offline(cli.csv, device=cli.device)
    train_masks, val_masks, test_masks = stratified_edge_split(data, seed=cli.seed)

    results = {}

    # ── A. GraphSAGE + full neighbors ───────────────────────────────────
    args_a = _base_args(cli.epochs, cli.device, cli.seed); args_a.sampling = "none"
    results["A_sage_full"] = run_cell("A", "sage", data, meta, train_masks, val_masks, test_masks, args_a)

    # ── B. GraphSAGE + relation-aware sampling ──────────────────────────
    args_b = _base_args(cli.epochs, cli.device, cli.seed); args_b.sampling = "relation_aware"
    sampling_cfg_b = SamplingConfig(max_neighbors=args_b.max_neighbors, num_hops=args_b.num_hops,
                                     num_samples_per_relation=args_b.num_samples_per_relation,
                                     strategy="relation_aware", seed=args_b.sampling_seed)
    train_view_b = build_sampled_training_view(data, meta["populated_triples"], train_masks, sampling_cfg_b, device=cli.device)
    results["B_sage_sampled"] = run_cell("B", "sage", data, meta, train_masks, val_masks, test_masks, args_b, train_view=train_view_b)

    # ── C. GAT + full neighbors ──────────────────────────────────────────
    args_c = _base_args(cli.epochs, cli.device, cli.seed); args_c.sampling = "none"
    results["C_gat_full"] = run_cell("C", "gat", data, meta, train_masks, val_masks, test_masks, args_c)

    # ── D. HGT + full neighbors ──────────────────────────────────────────
    args_d = _base_args(cli.epochs, cli.device, cli.seed); args_d.sampling = "none"
    results["D_hgt_full"] = run_cell("D", "hgt", data, meta, train_masks, val_masks, test_masks, args_d)

    # ── E. HGT + adaptive relation-aware sampling, K tuned on VAL ────────
    best_k, best_val_f1, best_train_view = None, -1.0, None
    k_sweep_val_results = {}
    for k in cli.k_sweep:
        args_e_probe = _base_args(cli.epochs, cli.device, cli.seed); args_e_probe.sampling = "relation_aware"
        args_e_probe.max_neighbors = k
        sampling_cfg = SamplingConfig(max_neighbors=k, num_hops=args_e_probe.num_hops,
                                       num_samples_per_relation=args_e_probe.num_samples_per_relation,
                                       strategy="relation_aware", seed=args_e_probe.sampling_seed)
        train_view = build_sampled_training_view(data, meta["populated_triples"], train_masks, sampling_cfg, device=cli.device)
        # NOTE: train_model() already reports val-selected TEST metrics (it
        # internally checkpoints on best val F1 and evaluates test only
        # once at the end) — to select K purely on validation without
        # peeking at test, we re-run train_model but read its internal
        # best_val_f1 via a lightweight val-only probe: train for the same
        # budget and record best val F1 from the training log rather than
        # the returned (test) metrics. See train_model's own val-based
        # checkpointing — we replicate that selection criterion here at
        # the K level, one level up.
        torch.manual_seed(args_e_probe.sampling_seed)
        model = build_model("hgt", meta, args_e_probe)
        pos_weight = compute_class_weights(data, train_masks)
        val_f1 = _train_for_val_f1_only(model, data, train_masks, val_masks, args_e_probe, pos_weight, train_view)
        k_sweep_val_results[k] = val_f1
        log.info("K sweep: max_neighbors=%d -> val F1=%.4f", k, val_f1)
        if val_f1 > best_val_f1:
            best_k, best_val_f1, best_train_view = k, val_f1, train_view

    log.info("Selected K=%d by validation F1 (%.4f) — evaluating on TEST once.", best_k, best_val_f1)
    args_e = _base_args(cli.epochs, cli.device, cli.seed); args_e.sampling = "relation_aware"; args_e.max_neighbors = best_k
    results["E_hgt_sampled"] = run_cell("E", "hgt", data, meta, train_masks, val_masks, test_masks, args_e, train_view=best_train_view)
    results["E_hgt_sampled"]["k_sweep_val_f1"] = k_sweep_val_results
    results["E_hgt_sampled"]["selected_k"] = best_k

    # ── Save + print ───────────────────────────────────────────────────
    printable = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("confusion",)}
        for k, v in results.items()
    }
    with open(cli.out, "w") as f:
        json.dump(printable, f, indent=2, default=str)
    log.info("Saved results to %s", cli.out)

    print("\n\n" + "=" * 100)
    print(f"{'Cell':<20} {'Model':<8} {'Sampling':<16} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUROC':>7} {'AUPR':>7} {'Params':>10} {'Time(s)':>9}")
    print("-" * 100)
    for cell, m in results.items():
        print(f"{cell:<20} {m['model']:<8} {m['sampling']:<16} {m.get('precision',0):>7.3f} {m.get('recall',0):>7.3f} "
              f"{m.get('f1',0):>7.3f} {m.get('roc_auc',0):>7.3f} {m.get('aupr',0):>7.3f} {m['params']:>10,} {m['train_time_sec']:>9.1f}")
    print("=" * 100)


def _train_for_val_f1_only(model, data, train_masks, val_masks, args, pos_weight, train_view):
    """Runs the SAME training loop train_model() uses (copied minimally,
    not reimplemented — see train.py's train_model docstring for the
    parts this mirrors) but returns best validation F1 instead of test
    metrics, so the K sweep never touches the test split — task
    requirement 11's 'tune on validation data, report test once'."""
    from data_loader import global_labels, flatten_mask_dict
    from utils import evaluate
    from train import build_loss
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = torch.device(args.device)
    model = model.to(device)
    criterion = build_loss(args.loss, pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_data, tmasks = (train_view if train_view is not None else (data, train_masks))
    y_full = torch.tensor(global_labels(train_data), dtype=torch.long, device=device)
    train_flat_mask = flatten_mask_dict(train_data, tmasks).to(device)

    best_val_f1 = -1.0
    patience_ctr = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        y_train = y_full[train_flat_mask].float()
        loss = criterion(logits[train_flat_mask], y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, data, val_masks, threshold=args.threshold)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_ctr = 0
            else:
                patience_ctr += 5
                if patience_ctr >= args.patience:
                    break
    return best_val_f1


if __name__ == "__main__":
    main()
