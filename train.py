"""
train.py
--------
Top-level training script for TreeLoRA + ViT-B/16 continual learning.

Usage examples:
    # Full Split CIFAR-100 run (10 tasks, 5 epochs each)
    python train.py --dataset cifar100 --n_tasks 10 --epochs 5

    # Quick smoke test (2 tasks, 1 epoch, small batch)
    python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --batch_size 16 --no_pretrained

    # Split ImageNet-R (requires manual download)
    python train.py --dataset imagenet_r --n_tasks 20 --epochs 5

    # Split CUB-200 (requires manual download)
    python train.py --dataset cub200 --n_tasks 10 --epochs 5
"""

import argparse
import time
import torch

from datasets import get_split_cifar100, get_split_imagenet_r, get_split_cub200
from continual_learner import TreeLoRALearner


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TreeLoRA ViT Continual Learning (ViT-B/16, paper reproduction)"
    )

    # Dataset
    p.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar100", "imagenet_r", "cub200"],
        help="CL benchmark to run (default: cifar100)",
    )
    p.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for datasets",
    )
    p.add_argument(
        "--n_tasks",  type=int, default=10,
        help="Number of continual learning tasks (default: 10)",
    )

    # Model
    p.add_argument(
        "--no_pretrained", action="store_true",
        help="Do NOT load ImageNet-21k pretrained weights (for fast smoke tests)",
    )

    # LoRA
    p.add_argument(
        "--lora_rank",  type=int,   default=4,
        help="LoRA rank r (paper default: 4)",
    )
    p.add_argument(
        "--lora_alpha", type=float, default=8.0,
        help="LoRA alpha α (paper default: 8.0 → scaling = 2.0)",
    )
    p.add_argument(
        "--lora_depth", type=int, default=5,
        help="KD-tree depth = number of LoRA layers used in the tree (paper: 5)",
    )

    # Regularisation
    p.add_argument(
        "--reg", type=float, default=0.5,
        help="Regularisation strength λ (paper default: 0.5, 0 = no TreeLoRA reg)",
    )

    # Training
    p.add_argument("--epochs",     type=int,   default=5,    help="Epochs per task")
    p.add_argument("--batch_size", type=int,   default=64,   help="Batch size")
    p.add_argument("--lr",         type=float, default=5e-3, help="Learning rate (paper range: [0.003, 0.007])")
    p.add_argument("--num_workers",type=int,   default=4,    help="DataLoader workers")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  TreeLoRA ViT-B/16  ─  {args.dataset.upper()}")
    print(f"  Device : {device}")
    print(f"  Tasks  : {args.n_tasks}")
    print(f"  Epochs : {args.epochs}")
    print(f"  LoRA   : rank={args.lora_rank}  alpha={args.lora_alpha}  "
          f"scaling={args.lora_alpha/args.lora_rank:.1f}")
    print(f"  Tree   : depth={args.lora_depth}  reg={args.reg}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    if args.dataset == "cifar100":
        classes_per_task = 100 // args.n_tasks
        task_dataloaders, class_splits = get_split_cifar100(
            data_root=args.data_root,
            n_tasks=args.n_tasks,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.dataset == "imagenet_r":
        classes_per_task = 200 // args.n_tasks
        task_dataloaders, class_splits = get_split_imagenet_r(
            data_root=args.data_root,
            n_tasks=args.n_tasks,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.dataset == "cub200":
        classes_per_task = 200 // args.n_tasks
        task_dataloaders, class_splits = get_split_cub200(
            data_root=args.data_root,
            n_tasks=args.n_tasks,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset ready: {args.n_tasks} tasks, "
          f"{classes_per_task} classes/task\n")

    # ------------------------------------------------------------------
    # Build learner
    # ------------------------------------------------------------------
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
    )

    learner.model.print_trainable_summary()

    # ------------------------------------------------------------------
    # Run continual learning
    # ------------------------------------------------------------------
    t0 = time.time()
    acc_matrix, final_acc, bwt = learner.run(
        task_dataloaders=task_dataloaders,
        epochs=args.epochs,
    )
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\nAccuracy Matrix (rows=after task T, cols=task evaluated):")
    header = "       " + "  ".join(f"T{j:02d}" for j in range(args.n_tasks))
    print(header)
    for i, row in enumerate(acc_matrix):
        cells = "  ".join(f"{v:5.1f}" for v in row)
        padding = "  ".join("  ---" for _ in range(args.n_tasks - len(row)))
        print(f"  T{i:02d}: {cells}  {padding}")

    print(f"\n  Average Accuracy (Acc) : {final_acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Total training time    : {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
