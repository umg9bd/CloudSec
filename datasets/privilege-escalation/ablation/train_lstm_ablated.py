"""
train_lstm_ablated.py
=====================
Experiment F: the live LSTM recipe (v6.3 trainer -> head fine-tune on real dev, L2-SP 1, as served
in pipeline_config.json) rerun with one temporal feature group dropped. Uses the scripts unchanged:
the trainer's own --drop/--out-dir, and the fine-tune's module-level BASE_*/OUT_DIR redirected here.

    python datasets/privilege-escalation/ablation/train_lstm_ablated.py --group timing
    python datasets/privilege-escalation/ablation/train_lstm_ablated.py --group control   # nothing extra dropped

Writes results/ablation/lstm_<group>/ (v6.3) and results/ablation/lstm_<group>_ft/ (served checkpoint).
Never scores the real test split.
"""
import argparse
import os
import pathlib
import subprocess
import sys
import time

from ablation_groups import LSTM_BASE_DROP, REPO_ROOT, TEMPORAL_GROUPS

OUT_ROOT = pathlib.Path(REPO_ROOT) / "results" / "ablation"
L2SP = "1"  # the served v6.3-ft's choice (leave-one-tactic-out CV), kept fixed so only the features change


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=["control"] + list(TEMPORAL_GROUPS))
    args = ap.parse_args()
    drop = LSTM_BASE_DROP + ([] if args.group == "control" else TEMPORAL_GROUPS[args.group])
    base_dir, ft_dir = OUT_ROOT / f"lstm_{args.group}", OUT_ROOT / f"lstm_{args.group}_ft"
    t0 = time.time()

    trainer = os.path.join(REPO_ROOT, "temporal-analysis", "train_lstm_transformer_v6_3.py")
    subprocess.run([sys.executable, trainer, "--drop", ",".join(drop), "--out-dir", str(base_dir)],
                   check=True, cwd=REPO_ROOT)
    print(f"[ablation] v6.3 without {args.group}: {time.time() - t0:.0f}s", flush=True)

    sys.path.insert(0, os.path.join(REPO_ROOT, "temporal-analysis"))
    import finetune_lstm_v6_3 as ft
    ft.BASE_DIR, ft.BASE_CKPT, ft.OUT_DIR = base_dir, base_dir / "temporal_lstm_transformer.pt", ft_dir
    sys.argv = ["finetune_lstm_v6_3.py", "--l2sp", L2SP]
    ft.main()
    print(f"[ablation] group={args.group} dropped={drop} -> {ft_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
