"""
train_scalable.py
====================
Mini-batch training infrastructure for the HGT + ensemble extension —
new file; does not modify train.py's existing full-batch train_model()
loop for GraphSAGE/GAT at all. Provides:

  - train_graphsage_minibatch(): LinkNeighborLoader-driven mini-batch
    training for GraphSAGEAnomalyDetector — opt-in scalability
    infrastructure for a graph that keeps growing via
    IncrementalGraphUpdater. GraphSAGEAnomalyDetector's own architecture/
    forward()/loss/checkpoint format are untouched; this file only adds
    a different way to FEED it during training. On the current ~9,700
    edge scale this is not memory-load-bearing (train.py's existing
    full-batch loop already handles it comfortably) — it's headroom.

  - train_hgt(): HGTLoader-driven training over the node_importance.py-
    selected "important region" subgraph.

  - calibrate_ensemble_alpha(): grid search over the validation split for
    the alpha maximising validation F1 (matches utils.evaluate()'s own
    metric choice). Deliberately not a learned parameter, so the final
    ensemble stays exactly the simple, auditable
    "FinalScore = alpha*HGT + (1-alpha)*GraphSAGE" the extension brief
    asked for, rather than something opaque.

  - compare_n_way(): utils.py's print_comparison_table() only compares
    two named models; this is the same idea generalised to however many
    {name: metrics} pairs are passed, without touching utils.py.

WHY LinkNeighborLoader, NOT NeighborLoader, FOR GRAPHSAGE
─────────────────────────────────────────────────────────────────────────
This is edge classification (attack / benign per edge), not node
classification. LinkNeighborLoader samples seed EDGES plus their
neighbourhoods directly; NeighborLoader samples starting from seed NODES
and would need a manual edge-labelling step bolted on afterwards — the
extension brief explicitly left this choice open ("whichever is more
appropriate for edge prediction"); this is that choice, made and
justified rather than left as a TODO.

HOW LABELS ARE RECOVERED FROM A SAMPLED MINI-BATCH
─────────────────────────────────────────────────────────────────────────
Deliberately does NOT rely on LinkNeighborLoader's edge_label /
edge_label_index kwargs — the exact correlation between those and the
sampled edge_index for a multi-relation heterogeneous graph is the one
part of the PyG API most likely to vary by version and hardest to verify
without execution (see the honest caveat below). Instead, `.y` is read
straight off each sampled batch's per-triple tensors via
data_loader.py's own global_labels() — the SAME function utils.py's
evaluate() already uses on a full graph. PyG's samplers subset every
edge-level attribute (edge_attr, y, ...) consistently with the sampled
edge_index, so this sidesteps the edge_label/edge_label_index question
entirely rather than depending on it.

AN HONEST CAVEAT (same spirit as explainability.py's module docstring)
─────────────────────────────────────────────────────────────────────────
No torch / torch_geometric install in this environment — everything
below is written carefully against PyG's documented NeighborLoader /
LinkNeighborLoader / HGTLoader APIs but UNEXECUTED. The spot most likely
to need adjustment against your installed version: HGTLoader's exact
multi-node-type `input_nodes` format (single-node-type examples are the
textbook case; this graph has up to 6 populated node types). Verify in
the notebook before trusting results here over train.py's existing
full-batch numbers for anything you'd put in a paper.
"""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, HGTLoader

from utils import FocalLoss, evaluate
from data_loader import global_labels
from node_importance import select_important_nodes

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]


# ══════════════════════════════════════════════════════════════════════════
# Shared training knobs — one config object rather than a long parameter
# list repeated across training functions, matching how train.py's
# argparse Namespace already plays the same role for the full-batch loop.
# ══════════════════════════════════════════════════════════════════════════

class ScalableTrainConfig:
    def __init__(
        self,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        num_neighbors: Optional[List[int]] = None,   # fanout per hop, e.g. [15, 10]
        num_workers: int = 2,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        grad_clip_norm: float = 1.0,
        early_stop_patience: int = 15,
        eval_every: int = 5,
        device: Optional[str] = None,
        ddp: bool = False,                            # only meaningful if >1 GPU visible
        save_dir: str = "./checkpoints",
        top_frac: float = 0.2,                         # HGT node-importance selection
        min_per_type: int = 1,
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [15, 10]
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip_norm = grad_clip_norm
        self.early_stop_patience = early_stop_patience
        self.eval_every = eval_every
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ddp = ddp and torch.cuda.device_count() > 1
        self.save_dir = save_dir
        self.top_frac = top_frac
        self.min_per_type = min_per_type


def _loader_kwargs(cfg: ScalableTrainConfig) -> dict:
    kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
    )
    # persistent_workers=True with num_workers=0 raises in PyTorch — guard it.
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _amp_autocast(cfg: ScalableTrainConfig):
    return torch.autocast(device_type=cfg.device.type, enabled=(cfg.use_amp and cfg.device.type == "cuda"))


# ══════════════════════════════════════════════════════════════════════════
# GraphSAGE — mini-batch training (architecture/forward/loss untouched)
# ══════════════════════════════════════════════════════════════════════════

def train_graphsage_minibatch(
    model: nn.Module,
    train_data: HeteroData,
    val_data: HeteroData,
    train_mask_dict: Dict[EdgeTriple, torch.Tensor],
    val_mask_dict: Dict[EdgeTriple, torch.Tensor],
    edge_types: List[EdgeTriple],
    cfg: ScalableTrainConfig,
    model_name: str = "GraphSAGE_minibatch",
) -> dict:
    """
    Mirrors train.py's train_model() conventions — AdamW, CosineAnnealingLR,
    FocalLoss default, grad clipping, early stopping on val F1, bare
    state_dict checkpoint at f"{save_dir}/best_{model_name}.pt" (same
    naming pattern train.py actually uses — capitalised model name and
    all; see this file's module-level notes) — adapted to iterate
    LinkNeighborLoader mini-batches per epoch instead of one full-graph
    forward/backward pass. GraphSAGEAnomalyDetector itself needs no
    changes: HeteroConv/SAGEConv message-passing is loader-agnostic, it
    just processes whatever HeteroData it's handed, mini-batch or full.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    model = model.to(cfg.device)
    if cfg.ddp:
        model = nn.parallel.DistributedDataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = FocalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device.type == "cuda"))

    # One LinkNeighborLoader per populated triple, cycled round-robin per
    # epoch (see module docstring's honest caveat on this choice).
    loaders = {}
    for t in edge_types:
        if t not in train_data.edge_types:
            continue
        if train_data[t].edge_index.shape[1] == 0:
            continue
        loaders[t] = LinkNeighborLoader(
            train_data,
            num_neighbors=cfg.num_neighbors,
            edge_label_index=(t, train_data[t].edge_index),
            **_loader_kwargs(cfg),
        )

    best_val_f1 = -1.0
    best_state = None
    patience_ctr = 0
    ckpt_path = os.path.join(cfg.save_dir, f"best_{model_name}.pt")
    history = {"epoch": [], "train_loss": [], "val_f1": [], "val_auc": []}

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        epoch_loss, n_batches = 0.0, 0
        opt.zero_grad()

        iters = {t: iter(l) for t, l in loaders.items()}
        active = set(iters.keys())
        step = 0
        while active:
            for t in list(active):
                try:
                    batch = next(iters[t])
                except StopIteration:
                    active.discard(t)
                    continue
                batch = batch.to(cfg.device)
                with _amp_autocast(cfg):
                    logits = model(batch)
                    y = torch.as_tensor(global_labels(batch), dtype=torch.float, device=cfg.device)
                    if y.shape[0] != logits.shape[0]:
                        log.warning(
                            "[%s] label/logit count mismatch (%d vs %d) on a %s batch — "
                            "skipping this batch rather than misaligning loss to the wrong "
                            "edges. Likely cause: an empty triple sampled into the batch.",
                            model_name, y.shape[0], logits.shape[0], t,
                        )
                        continue
                    loss = loss_fn(logits, y) / cfg.grad_accum_steps
                scaler.scale(loss).backward()
                step += 1
                if step % cfg.grad_accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                epoch_loss += loss.item() * cfg.grad_accum_steps
                n_batches += 1

        sched.step()
        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        log.info("[%s] epoch %d/%d  loss=%.4f  (%d batches, %.1fs)",
                  model_name, epoch + 1, cfg.epochs, avg_loss, n_batches, elapsed)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            metrics = evaluate(model, val_data, val_mask_dict)
            log.info("[%s] val F1=%.4f AUC=%.4f", model_name, metrics["f1"], metrics["roc_auc"])
            history["val_f1"].append(metrics["f1"])
            history["val_auc"].append(metrics["roc_auc"])
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
                torch.save(best_state, ckpt_path)
            else:
                patience_ctr += cfg.eval_every
                if patience_ctr >= cfg.early_stop_patience:
                    log.info("[%s] early stopping at epoch %d (best val F1=%.4f)",
                              model_name, epoch + 1, best_val_f1)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_f1": best_val_f1, "checkpoint_path": ckpt_path, "history": history}


# ══════════════════════════════════════════════════════════════════════════
# HGT — HGTLoader over the node_importance-selected important region
# ══════════════════════════════════════════════════════════════════════════

def train_hgt(
    model: nn.Module,
    train_data: HeteroData,
    val_data: HeteroData,
    train_mask_dict: Dict[EdgeTriple, torch.Tensor],
    val_mask_dict: Dict[EdgeTriple, torch.Tensor],
    edge_types: List[EdgeTriple],
    node_feature_schema: dict,
    cfg: ScalableTrainConfig,
    model_name: str = "HGT",
) -> dict:
    """
    Same overall shape as train_graphsage_minibatch, with two
    differences: (1) input_nodes for the loader come from
    node_importance.select_important_nodes rather than the whole graph,
    and (2) HGTLoader (not LinkNeighborLoader) does the sampling, since
    it's built specifically for heterogeneous, multi-node-type
    neighbourhood expansion from a set of seed nodes — see the extension
    brief's "Entire Graph -> Rank nodes -> Select top X% -> HGTLoader ->
    expand to heterogeneous neighbourhood" diagram, implemented directly.
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    model = model.to(cfg.device)
    if cfg.ddp:
        model = nn.parallel.DistributedDataParallel(model)

    important_nodes = select_important_nodes(
        train_data, edge_types, node_feature_schema,
        top_frac=cfg.top_frac, min_per_type=cfg.min_per_type,
    )
    log.info("[%s] important nodes selected: %s", model_name,
              {nt: idx.shape[0] for nt, idx in important_nodes.items()})

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = FocalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device.type == "cuda"))

    # HGTLoader takes ONE dict of {node_type: seed_indices} for
    # input_nodes — unlike LinkNeighborLoader, it expands node
    # neighbourhoods rather than sampling around seed edges, so labelled
    # supervision comes from whatever edges land inside the expanded
    # subgraph (read via global_labels(batch), same as the GraphSAGE
    # loop above), not from an edge-specific seed list.
    loader = HGTLoader(
        train_data,
        num_samples=cfg.num_neighbors,
        input_nodes=important_nodes,
        **_loader_kwargs(cfg),
    )

    best_val_f1 = -1.0
    best_state = None
    patience_ctr = 0
    ckpt_path = os.path.join(cfg.save_dir, f"best_{model_name}.pt")
    history = {"epoch": [], "train_loss": [], "val_f1": [], "val_auc": []}

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        epoch_loss, n_batches = 0.0, 0
        opt.zero_grad()

        for step, batch in enumerate(loader, start=1):
            batch = batch.to(cfg.device)
            if not batch.edge_types:
                continue
            with _amp_autocast(cfg):
                logits = model(batch)
                y = torch.as_tensor(global_labels(batch), dtype=torch.float, device=cfg.device)
                if y.shape[0] != logits.shape[0] or logits.shape[0] == 0:
                    continue
                loss = loss_fn(logits, y) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            epoch_loss += loss.item() * cfg.grad_accum_steps
            n_batches += 1

        sched.step()
        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        log.info("[%s] epoch %d/%d  loss=%.4f  (%d batches, %.1fs)",
                  model_name, epoch + 1, cfg.epochs, avg_loss, n_batches, elapsed)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            # Evaluated on the FULL val_data, not just the important-node
            # subgraph — HGTAnomalyDetector has no internal sampling (see
            # its module docstring), so this measures how well it
            # generalises beyond the region it was trained on, which is
            # exactly what we want to know before deploying it.
            metrics = evaluate(model, val_data, val_mask_dict)
            log.info("[%s] val F1=%.4f AUC=%.4f", model_name, metrics["f1"], metrics["roc_auc"])
            history["val_f1"].append(metrics["f1"])
            history["val_auc"].append(metrics["roc_auc"])
            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_state = deepcopy(model.state_dict())
                patience_ctr = 0
                torch.save(best_state, ckpt_path)
            else:
                patience_ctr += cfg.eval_every
                if patience_ctr >= cfg.early_stop_patience:
                    log.info("[%s] early stopping at epoch %d (best val F1=%.4f)",
                              model_name, epoch + 1, best_val_f1)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_f1": best_val_f1, "checkpoint_path": ckpt_path, "history": history}


# ══════════════════════════════════════════════════════════════════════════
# Ensemble alpha calibration
# ══════════════════════════════════════════════════════════════════════════

def calibrate_ensemble_alpha(
    sage_model: nn.Module,
    hgt_model: nn.Module,
    val_data: HeteroData,
    val_mask_dict: Dict[EdgeTriple, torch.Tensor],
    alphas: Optional[List[float]] = None,
) -> Tuple[float, dict]:
    """
    Grid search: builds an EnsembleModel at each candidate alpha, scores
    it on val_data via the same utils.evaluate() every other model uses,
    and returns (best_alpha, {alpha: metrics}). Deliberately a grid
    search over a handful of values, not a learned/optimised parameter —
    keeps "FinalScore = alpha*HGT + (1-alpha)*GraphSAGE" exactly as
    literal and auditable as the extension brief specified.
    """
    from model_ensemble import EnsembleModel

    alphas = alphas if alphas is not None else [round(a, 2) for a in np.linspace(0.0, 1.0, 11)]
    results = {}
    best_alpha, best_f1 = 0.5, -1.0
    for alpha in alphas:
        ensemble = EnsembleModel([("graphsage", sage_model, 1.0 - alpha), ("hgt", hgt_model, alpha)])
        ensemble.eval()
        metrics = evaluate(ensemble, val_data, val_mask_dict)
        results[alpha] = metrics
        log.info("[ensemble calibration] alpha=%.2f  F1=%.4f  AUC=%.4f", alpha, metrics["f1"], metrics["roc_auc"])
        if metrics["f1"] > best_f1:
            best_f1, best_alpha = metrics["f1"], alpha

    log.info("[ensemble calibration] best alpha=%.2f (val F1=%.4f)", best_alpha, best_f1)
    return best_alpha, results


# ══════════════════════════════════════════════════════════════════════════
# N-way comparison table — generalises utils.py's print_comparison_table
# ══════════════════════════════════════════════════════════════════════════

def compare_n_way(named_metrics: Dict[str, dict]) -> None:
    """
    named_metrics: {model_name: metrics_dict} where each metrics_dict is
    whatever utils.evaluate() returns. Prints a table with one column per
    model — same keys/format utils.py's two-model print_comparison_table
    already uses, generalised to N without touching that file.
    """
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = list(named_metrics.keys())
    col_w = max(12, max(len(n) for n in names) + 2)
    header = f"{'Metric':<15}" + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + "=" * len(header))
    print("  Model Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"  {k:<13}"
        for n in names:
            row += f"{named_metrics[n].get(k, 0.0):>{col_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")
