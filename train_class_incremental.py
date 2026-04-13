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
import time
import torch

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
    p.add_argument("--reg", type=float, default=1.5,
                   help="Regularization strength (1.0-2.0 recommended)")
    p.add_argument("--epochs", type=int, default=10, help="Epochs per task")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Device
    p.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for results")

    return p.parse_args()


def main():
    args = parse_args()

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
        "cifar100": {"acc": "65-70%", "bwt": "-8% to -12%"},
        "imagenet_r": {"acc": "55-60%", "bwt": "-12% to -18%"},
        "cub200": {"acc": "60-65%", "bwt": "-10% to -15%"},
    }
    exp = expected[args.dataset]
    print(f"  Expected (from paper): Acc={exp['acc']}, BWT={exp['bwt']}\n")


if __name__ == "__main__":
    main()
