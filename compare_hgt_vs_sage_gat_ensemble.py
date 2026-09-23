from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from data_loader import stratified_edge_split
from offline_pipeline import load_offline
from train import build_model
from model_ensemble import EnsembleModel


def calibrate_two_model_ensemble(
    sage_model,
    gat_model,
    val_data,
    val_mask_dict,
):
    """Select GraphSAGE/GAT fusion weight using validation F1 only."""
    alphas = [i / 10.0 for i in range(11)]

    results = {}
    best_alpha = 0.5
    best_f1 = -1.0

    for alpha in alphas:
        ensemble = EnsembleModel([
            ("graphsage", sage_model, 1.0 - alpha),
            ("gat", gat_model, alpha),
        ])
        ensemble.eval()

        metrics = evaluate(
            ensemble,
            val_data,
            val_mask_dict,
        )

        results[alpha] = metrics

        print(
            f"[ensemble calibration] "
            f"alpha(GAT)={alpha:.1f} "
            f"F1={metrics['f1']:.4f} "
            f"AUC={metrics['roc_auc']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_alpha = alpha

    print(
        f"[ensemble calibration] "
        f"best alpha={best_alpha:.1f} "
        f"(val F1={best_f1:.4f})"
    )

    return best_alpha, results

from utils import evaluate, print_comparison_table


def make_args(device: str, seed: int):
    return SimpleNamespace(
        hidden=128,
        layers=2,
        heads=4,
        attn_dropout=0.1,
        hgt_group="sum",
        dropout=0.3,
        lr=1e-3,
        loss="focal",
        device=device,
        threshold=0.5,
        patience=15,
        epochs=100,
        sampling="none",
        split="stratified",
        seed=seed,
        max_neighbors=50,
        num_hops=2,
        num_samples_per_relation=5,
        sampling_seed=seed,
        save_dir="./checkpoints_sage_gat_vs_hgt",
    )


def load_model(name, checkpoint, meta, args, device):
    model = build_model(name, meta, args)

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    # Support either a bare state_dict or a wrapped checkpoint.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"Loaded {name} from {checkpoint}")
    return model


def clean(metrics):
    return {
        k: v
        for k, v in metrics.items()
        if k not in {"probs", "labels", "report", "confusion"}
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--csv",
        default="./graph_construction/cloudtrail_structural.csv",
    )
    p.add_argument(
        "--device",
        default="cpu",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--hgt_checkpoint",
        default="./checkpoints_hgt_corrected/best_HGT.pt",
    )
    p.add_argument(
        "--sage_checkpoint",
        default="./checkpoints/best_GraphSAGE.pt",
    )
    p.add_argument(
        "--gat_checkpoint",
        default="./checkpoints/best_GAT.pt",
    )
    p.add_argument(
        "--out",
        default="hgt_vs_sage_gat_ensemble.json",
    )

    cli = p.parse_args()

    device = torch.device(cli.device)

    print("\nLoading data...")
    data, meta = load_offline(cli.csv, device=cli.device)

    print("Creating the same stratified split...")
    train_masks, val_masks, test_masks = stratified_edge_split(
        data,
        seed=cli.seed,
    )

    args = make_args(cli.device, cli.seed)

    print("\nLoading pretrained models...")

    sage = load_model(
        "sage",
        cli.sage_checkpoint,
        meta,
        args,
        device,
    )

    gat = load_model(
        "gat",
        cli.gat_checkpoint,
        meta,
        args,
        device,
    )

    hgt = load_model(
        "hgt",
        cli.hgt_checkpoint,
        meta,
        args,
        device,
    )

    print("\nEvaluating individual models on TEST...")

    sage_metrics = evaluate(
        sage,
        data,
        test_masks,
        threshold=args.threshold,
        return_probs=True,
    )

    gat_metrics = evaluate(
        gat,
        data,
        test_masks,
        threshold=args.threshold,
        return_probs=True,
    )

    hgt_metrics = evaluate(
        hgt,
        data,
        test_masks,
        threshold=args.threshold,
        return_probs=True,
    )

    print("\nCalibrating GraphSAGE + GAT weight using VALIDATION only...")

    best_alpha, val_sweep = calibrate_two_model_ensemble(
        sage,
        gat,
        data,
        val_masks,
    )

    print(f"\nBest GAT alpha = {best_alpha:.2f}")
    print(
        f"GraphSAGE weight = {1.0 - best_alpha:.2f}, "
        f"GAT weight = {best_alpha:.2f}"
    )

    ensemble = EnsembleModel(
        [
            ("graphsage", sage, 1.0 - best_alpha),
            ("gat", gat, best_alpha),
        ]
    ).to(device)

    ensemble.eval()

    print("\nEvaluating GraphSAGE + GAT ensemble on TEST...")

    ensemble_metrics = evaluate(
        ensemble,
        data,
        test_masks,
        threshold=args.threshold,
        return_probs=True,
    )

    named = {
        "GraphSAGE": sage_metrics,
        "GAT": gat_metrics,
        "GraphSAGE+GAT": ensemble_metrics,
        "HGT": hgt_metrics,
    }

    print_comparison_table(
        sage_metrics,
        gat_metrics,
    )

    print("\n==============================")
    print("HGT vs GraphSAGE+GAT")
    print("==============================")

    print(
        f"HGT             F1={hgt_metrics['f1']:.4f} "
        f"AUC={hgt_metrics['roc_auc']:.4f} "
        f"AUPR={hgt_metrics['aupr']:.4f}"
    )

    print(
        f"GraphSAGE+GAT   F1={ensemble_metrics['f1']:.4f} "
        f"AUC={ensemble_metrics['roc_auc']:.4f} "
        f"AUPR={ensemble_metrics['aupr']:.4f}"
    )

    payload = {
        "protocol": {
            "csv": cli.csv,
            "seed": cli.seed,
            "split": "stratified",
            "hgt_checkpoint": cli.hgt_checkpoint,
            "graphsage_checkpoint": cli.sage_checkpoint,
            "gat_checkpoint": cli.gat_checkpoint,
            "ensemble_weight_selection": "validation F1 only",
            "ensemble_formula":
                "(1-alpha)*GraphSAGE_logit + alpha*GAT_logit",
            "selected_gat_alpha": best_alpha,
        },
        "validation_alpha_sweep": {
            str(alpha): clean(metrics)
            for alpha, metrics in val_sweep.items()
        },
        "test": {
            name: clean(metrics)
            for name, metrics in named.items()
        },
    }

    with open(cli.out, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved results to: {cli.out}")


if __name__ == "__main__":
    main()
