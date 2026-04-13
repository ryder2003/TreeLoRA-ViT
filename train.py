"""
train.py
--------
Top-level training script for TreeLoRA + ViT-B/16 continual learning.

Before running for imagenet_r or cub200, download datasets with:
    python download_datasets.py

Usage examples:
    # Full Split CIFAR-100 (10 tasks, 5 epochs each)
    python train.py --dataset cifar100 --n_tasks 10 --epochs 5

    # Quick smoke test (2 tasks, 1 epoch, small batch, no pretrained weights)
    python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --batch_size 16 --no_pretrained

    # Split ImageNet-R (20 tasks × 10 classes)  — requires download_datasets.py
    python train.py --dataset imagenet_r --n_tasks 20 --epochs 5

    # Split CUB-200  (10 tasks × 20 classes)    — requires download_datasets.py
    python train.py --dataset cub200 --n_tasks 10 --epochs 5

    # Save checkpoints and logs to output directory
    python train.py --dataset cifar100 --output_dir ./runs/cifar100_run1
"""

import argparse
import os
import time
import torch

from datasets import get_split_cifar100, get_split_imagenet_r, get_split_cub200
from continual_learner import TreeLoRALearner


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TreeLoRA ViT Continual Learning (ViT-B/16, paper reproduction)"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar100", "imagenet_r", "cub200"],
        help="CL benchmark to run (default: cifar100)",
    )
    p.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for all datasets (default: ./data). "
             "Sub-folders (imagenet-r/, CUB_200_2011/) are created by download_datasets.py.",
    )
    p.add_argument(
        "--n_tasks", type=int, default=None,
        help="Number of continual-learning tasks. "
             "Defaults: cifar100=10, imagenet_r=20, cub200=10.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--no_pretrained", action="store_true",
        help="Do NOT load ImageNet-21k pretrained weights (for fast smoke tests)",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--lora_rank",  type=int,   default=4,
        help="LoRA rank r (paper default: 4)",
    )
    p.add_argument(
        "--lora_alpha", type=float, default=8.0,
        help="LoRA alpha (paper default: 8.0 -> scaling = 2.0)",
    )
    p.add_argument(
        "--lora_depth", type=int, default=5,
        help="KD-tree depth = number of LoRA layers used (paper: 5)",
    )

    # ── Regularisation ────────────────────────────────────────────────────────
    p.add_argument(
        "--reg", type=float, default=0.5,
        help="Regularisation strength (paper default: 0.5-2.0; paper experiments use 0.5-1.5)",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=10,    help="Epochs per task (paper: 8-10 for better retention)")
    p.add_argument("--batch_size",  type=int,   default=64,   help="Batch size (paper: 64 for CIFAR-100, 32 for others)")
    p.add_argument("--lr",          type=float, default=3e-3,
                   help="Learning rate (paper recommended: 2e-3 to 5e-3, default 3e-3)")
    p.add_argument("--num_workers", type=int,   default=4,
                   help="DataLoader worker processes (set 0 on Windows if you get multiprocessing errors)")

    # ── Device ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda', 'cuda:0', 'cpu', etc. (default: auto-detect)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save checkpoints, accuracy matrix, and training logs. "
             "If not set, defaults to ./runs/<dataset>_<timestamp>/",
    )

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset defaults per benchmark
# ──────────────────────────────────────────────────────────────────────────────

DATASET_DEFAULTS = {
    "cifar100":    {"n_tasks": 10, "total_classes": 100, "epochs": 10, "batch_size": 64},
    "imagenet_r":  {"n_tasks": 20, "total_classes": 200, "epochs": 8, "batch_size": 32},
    "cub200":      {"n_tasks": 10, "total_classes": 200, "epochs": 10, "batch_size": 32},
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Fill in defaults
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.n_tasks is None:
        args.n_tasks = defaults["n_tasks"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    total_classes    = defaults["total_classes"]
    classes_per_task = total_classes // args.n_tasks

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory (auto-generate if not specified)
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs", f"{args.dataset}_{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TreeLoRA ViT-B/16 -- {args.dataset.upper()}")
    print(f"  Device       : {device}")
    print(f"  Data root    : {os.path.abspath(args.data_root)}")
    print(f"  Tasks        : {args.n_tasks}  ({classes_per_task} classes each)")
    print(f"  Epochs/task  : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  LoRA         : rank={args.lora_rank}  alpha={args.lora_alpha}  "
          f"scaling={args.lora_alpha/args.lora_rank:.1f}")
    print(f"  Tree         : depth={args.lora_depth}  reg={args.reg}")
    print(f"  Pretrained   : {not args.no_pretrained}")
    print(f"  Output dir   : {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")

    # ── Save config ────────────────────────────────────────────────────────────
    import json
    config = vars(args).copy()
    config["device"] = str(device)
    config["total_classes"] = total_classes
    config["classes_per_task"] = classes_per_task
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Build dataset ──────────────────────────────────────────────────────────
    # All loaders accept `data_root` and manage sub-paths internally.
    loader_kwargs = dict(
        data_root=args.data_root,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.dataset == "cifar100":
        task_dataloaders, class_splits = get_split_cifar100(**loader_kwargs)

    elif args.dataset == "imagenet_r":
        task_dataloaders, class_splits = get_split_imagenet_r(**loader_kwargs)

    elif args.dataset == "cub200":
        task_dataloaders, class_splits = get_split_cub200(**loader_kwargs)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Quick sanity check
    n_tr = len(task_dataloaders[0][0].dataset)
    n_te = len(task_dataloaders[0][1].dataset)
    print(f"Dataset ready: {args.n_tasks} tasks | "
          f"Task-0 has {n_tr} train / {n_te} eval samples\n")

    # ── Build learner ──────────────────────────────────────────────────────────
    learner = TreeLoRALearner(
        num_tasks=args.n_tasks,
        classes_per_task=classes_per_task,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_depth=args.lora_depth,
        reg=args.reg,
        lr=args.lr,
        device=device,
        pretrained=not args.no_pretrained,
        output_dir=args.output_dir,
    )

    learner.model.print_trainable_summary()

    # ── Run continual learning ─────────────────────────────────────────────────
    t0 = time.time()
    acc_matrix, final_acc, bwt = learner.run(
        task_dataloaders=task_dataloaders,
        epochs=args.epochs,
    )
    elapsed = time.time() - t0

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\nAccuracy Matrix (rows = after training task T, cols = task evaluated):")
    header = "         " + "  ".join(f"T{j:02d}" for j in range(args.n_tasks))
    print(header)
    for i, row in enumerate(acc_matrix):
        cells   = "  ".join(f"{v:5.1f}" for v in row)
        padding = "  ".join("  ---" for _ in range(args.n_tasks - len(row)))
        print(f"  T{i:02d}:  {cells}  {padding}")

    print(f"\n  Average Accuracy (Acc) : {final_acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Total training time    : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    print(f"  Results saved to       : {os.path.abspath(args.output_dir)}\n")


if __name__ == "__main__":
    main()
