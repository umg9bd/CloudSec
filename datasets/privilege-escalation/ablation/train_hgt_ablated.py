"""
train_hgt_ablated.py
====================
Retrains the HGT exactly as graph_construction/train.py does (same defaults the live
checkpoints/best_HGT_seed{1,2,3}_wrapped.pt carry: hidden 128, 2 layers, 4 heads, dropout 0.3,
focal loss, stratified split, cloudtrail_structural.csv), but with one ablation group removed
from the graph (ablation_groups.graph_ablation). Writes <out-dir>/best_HGT_seed<seed>_wrapped.pt.

    python datasets/privilege-escalation/ablation/train_hgt_ablated.py --group privilege --seed 1
    python datasets/privilege-escalation/ablation/train_hgt_ablated.py --group none --seed 1   # full-feature retrain control
"""
import argparse
import os
import shutil
import sys
import time

from ablation_groups import GRAPH_GROUPS, REPO_ROOT, graph_ablation

STRUCTURAL_CSV = os.path.join(REPO_ROOT, "datasets", "privilege-escalation", "cloudtrail_structural.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=["none"] + list(GRAPH_GROUPS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", default=None, help="default: results/ablation/hgt_<group>")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "ablation", f"hgt_{args.group}")
    work = os.path.join(out_dir, f"_seed{args.seed}")
    os.makedirs(work, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)  # train.py seeds only its split; seed init too so reruns repeat
    import train as gtrain
    sys.argv = ["train.py", "--model", "hgt", "--offline-csv", STRUCTURAL_CSV, "--seed", str(args.seed),
                "--device", "cpu", "--save_dir", work]
    t0 = time.time()
    with graph_ablation(None if args.group == "none" else args.group):
        gtrain.main()
    dst = os.path.join(out_dir, f"best_HGT_seed{args.seed}_wrapped.pt")
    shutil.move(os.path.join(work, "best_HGT_wrapped.pt"), dst)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[ablation] group={args.group} seed={args.seed} -> {dst} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
