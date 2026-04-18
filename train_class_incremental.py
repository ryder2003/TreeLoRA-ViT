"""
train_class_incremental.py
---------------------------
Training script for CLASS-INCREMENTAL TreeLoRA (matches paper's evaluation).

Key differences from task-incremental:
- Single unified head with ALL classes
- No label remapping
- Model must distinguish all classes simultaneously
- Expected accuracy: 60-70% (matches paper)

Usage:
    python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 10
"""

import argparse
import os
import random
import time
import torch
import numpy as np

from datasets_class_incremental import (
    get_split_cifar100_class_incremental,
    get_split_imagenet_r_class_incremental,
    get_split_cub200_class_incremental
)
from continual_learner_class_incremental import ClassIncrementalTreeLoRALearner


DATASET_DEFAULTS = {
    "cifar100":    {"n_tasks": 10, "total_classes": 100},
    "imagenet_r":  {"n_tasks": 20, "total_classes": 200},
    "cub200":      {"n_tasks": 10, "total_classes": 200},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="TreeLoRA CLASS-INCREMENTAL Learning (Paper's Evaluation Protocol)"
    )

    # Dataset
    p.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=["cifar100", "imagenet_r", "cub200"],
        help="Dataset to use"
    )
    p.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for datasets"
    )
    p.add_argument(
        "--n_tasks", type=int, default=None,
        help="Number of tasks (default: 10 for CIFAR/CUB, 20 for ImageNet-R)"
    )

    # Model
    p.add_argument(
        "--no_pretrained", action="store_true",
        help="Don't load pretrained weights"
    )

    # LoRA
    p.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    p.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha")
    p.add_argument("--lora_depth", type=int, default=5, help="KD-tree depth")

    # Training
    p.add_argument("--reg", type=float, default=0.1,
                   help="Regularization strength (paper default recommendation: 0.1)")
    p.add_argument("--epochs", type=int, default=20, help="Epochs per task (paper: 20 for CIFAR-100)")
    p.add_argument("--batch_size", type=int, default=192, help="Batch size (paper setting: 192)")
    p.add_argument("--lr", type=float, default=5e-3, help="Learning rate (paper default 0.005)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--deterministic", action="store_true",
        help="Enable deterministic mode for reproducibility (slower)."
    )
    p.add_argument(
        "--mask_seen_classes", action="store_true",
        help="Optional: mask unseen classes during train/eval (not paper default)."
    )
    p.add_argument(
        "--freeze_old_head_rows", action="store_true",
        help="Optional: freeze previous classifier rows each task (not paper default)."
    )

    # Device
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for results")

    return p.parse_args()


def set_reproducibility(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def assert_strict_protocol(task_dataloaders, class_splits, total_classes):
    # Check unified global label space (no remapping) on first few tasks.
    for task_id in range(min(3, len(task_dataloaders))):
        train_loader, _ = task_dataloaders[task_id]
        _, y = next(iter(train_loader))
        expected_min = min(class_splits[task_id])
        expected_max = max(class_splits[task_id])
        y_min = int(y.min().item())
        y_max = int(y.max().item())
        if y_min < expected_min or y_max > expected_max:
            raise RuntimeError(
                "Protocol violation: labels appear remapped. "
                f"Task {task_id} expected [{expected_min}, {expected_max}] got [{y_min}, {y_max}]"
            )

    # Check labels do not exceed global class count.
    _, y0 = next(iter(task_dataloaders[0][0]))
    if int(y0.max().item()) >= total_classes:
        raise RuntimeError("Protocol violation: labels exceed total_classes range.")


def main():
    args = parse_args()
    set_reproducibility(args.seed, args.deterministic)

    # Fill defaults
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.n_tasks is None:
        args.n_tasks = defaults["n_tasks"]
    total_classes = defaults["total_classes"]
    classes_per_task = total_classes // args.n_tasks

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("runs_class_incremental", 
                                       f"{args.dataset}_{timestamp}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TreeLoRA ViT-B/16 -- CLASS-INCREMENTAL -- {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"  Evaluation Protocol : CLASS-INCREMENTAL")
    print(f"  Device              : {device}")
    print(f"  Data root           : {os.path.abspath(args.data_root)}")
    print(f"  Tasks               : {args.n_tasks}")
    print(f"  Total classes       : {total_classes}")
    print(f"  Classes per task    : {classes_per_task}")
    print(f"  Epochs/task         : {args.epochs}")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  LR                  : {args.lr}")
    print(f"  LoRA                : rank={args.lora_rank}  alpha={args.lora_alpha}  "
          f"scaling={args.lora_alpha/args.lora_rank:.1f}")
    print(f"  Tree                : depth={args.lora_depth}  reg={args.reg}")
    print(f"  Seed                : {args.seed}")
    print(f"  Deterministic       : {args.deterministic}")
    print(f"  Mask seen classes   : {args.mask_seen_classes}")
    print(f"  Freeze old head rows: {args.freeze_old_head_rows}")
    print(f"  Pretrained          : {not args.no_pretrained}")
    print(f"  Output dir          : {os.path.abspath(args.output_dir)}")
    print(f"{'='*70}\n")

    # Save config
    import json
    config = vars(args).copy()
    config["device"] = str(device)
    config["total_classes"] = total_classes
    config["classes_per_task"] = classes_per_task
    config["evaluation_protocol"] = "class_incremental"
    config["seed"] = args.seed
    config["deterministic"] = args.deterministic
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load dataset (class-incremental version)
    loader_kwargs = dict(
        data_root=args.data_root,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.dataset == "cifar100":
        task_dataloaders, class_splits = get_split_cifar100_class_incremental(**loader_kwargs)
    elif args.dataset == "imagenet_r":
        task_dataloaders, class_splits = get_split_imagenet_r_class_incremental(**loader_kwargs)
    elif args.dataset == "cub200":
        task_dataloaders, class_splits = get_split_cub200_class_incremental(**loader_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n_tr = len(task_dataloaders[0][0].dataset)
    n_te = len(task_dataloaders[0][1].dataset)
    print(f"Dataset ready: {args.n_tasks} tasks | "
          f"Task-0 has {n_tr} train / {n_te} eval samples")
    
    # Verify labels are NOT remapped
    x, y = next(iter(task_dataloaders[0][0]))
    print(f"Task 0 label range: {y.min().item()} to {y.max().item()} "
          f"(expected: {class_splits[0][0]} to {class_splits[0][-1]})")
    print()

    assert_strict_protocol(task_dataloaders, class_splits, total_classes)

    # Build learner
    learner = ClassIncrementalTreeLoRALearner(
        num_tasks=args.n_tasks,
        total_classes=total_classes,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_depth=args.lora_depth,
        reg=args.reg,
        lr=args.lr,
        device=device,
        pretrained=not args.no_pretrained,
        output_dir=args.output_dir,
        mask_seen_classes=args.mask_seen_classes,
        freeze_old_head_rows=args.freeze_old_head_rows,
    )

    if learner.model.head.out_features != total_classes:
        raise RuntimeError(
            "Protocol violation: learner is not using a unified global head."
        )

    learner.model.print_trainable_summary()

    # Run continual learning
    t0 = time.time()
    acc_matrix, final_acc, bwt = learner.run(
        task_dataloaders=task_dataloaders,
        epochs=args.epochs,
    )
    elapsed = time.time() - t0

    # Summary
    print("\nAccuracy Matrix (rows = after training task T, cols = task evaluated):")
    header = "         " + "  ".join(f"T{j:02d}" for j in range(args.n_tasks))
    print(header)
    for i, row in enumerate(acc_matrix):
        cells = "  ".join(f"{v:5.1f}" for v in row)
        padding = "  ".join("  ---" for _ in range(args.n_tasks - len(row)))
        print(f"  T{i:02d}:  {cells}  {padding}")

    print(f"\n  Average Accuracy (Acc) : {final_acc:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt:.2f}%")
    print(f"  Total training time    : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    print(f"  Results saved to       : {os.path.abspath(args.output_dir)}\n")

    # Expected performance comparison
    expected = {
        "cifar100": {"acc": "~88.54%", "bwt": "~-4.37%"},
        "imagenet_r": {"acc": "~71.94%", "bwt": "~-4.06%"},
        "cub200": {"acc": "~73.66%", "bwt": "~-4.87%"},
    }
    exp = expected[args.dataset]
    print(f"  Expected (from paper): Acc={exp['acc']}, BWT={exp['bwt']}\n")


if __name__ == "__main__":
    main()
