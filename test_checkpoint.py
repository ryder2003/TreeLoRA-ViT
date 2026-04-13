"""
test_checkpoint.py
------------------
Test a saved TreeLoRA checkpoint on all tasks.

Usage:
    python test_checkpoint.py --checkpoint ./runs/cifar100_20260413_023153/task_9
"""

import argparse
import torch
from datasets import get_split_cifar100, get_split_imagenet_r, get_split_cub200
from continual_learner import ContinualLearner

def test_checkpoint(checkpoint_path, dataset_name, n_tasks, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    if dataset_name == "cifar100":
        task_loaders, class_splits = get_split_cifar100(
            data_root="./data", n_tasks=n_tasks, batch_size=batch_size
        )
        num_classes_per_task = 10
    elif dataset_name == "imagenet_r":
        task_loaders, class_splits = get_split_imagenet_r(
            data_root="./data", n_tasks=n_tasks, batch_size=batch_size
        )
        num_classes_per_task = 200 // n_tasks
    elif dataset_name == "cub200":
        task_loaders, class_splits = get_split_cub200(
            data_root="./data", n_tasks=n_tasks, batch_size=batch_size
        )
        num_classes_per_task = 20
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create learner
    learner = ContinualLearner(
        device=device,
        lora_rank=4,
        lora_alpha=8.0,
        lora_depth=5,
        reg_strength=1.5,
        num_classes_per_task=num_classes_per_task
    )
    
    # Restore state
    learner.model.load_state_dict(checkpoint["model_state"])
    learner.task_heads = checkpoint["task_heads"]
    learner.kd_tree = checkpoint["kd_tree"]
    learner.seen_tasks = checkpoint["seen_tasks"]
    
    print(f"Checkpoint loaded: {len(learner.seen_tasks)} tasks trained\n")
    
    # Test on all seen tasks
    print("=" * 60)
    print("  Testing on all tasks")
    print("=" * 60)
    
    for task_id in learner.seen_tasks:
        _, eval_loader = task_loaders[task_id]
        acc = learner.evaluate_task(task_id, eval_loader)
        print(f"  Task {task_id}: {acc:.2f}%")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--dataset", type=str, default="cifar100",
                       choices=["cifar100", "imagenet_r", "cub200"])
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    test_checkpoint(args.checkpoint, args.dataset, args.n_tasks, args.batch_size)

if __name__ == "__main__":
    main()
