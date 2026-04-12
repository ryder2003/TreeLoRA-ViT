"""
validate_fixes.py
-----------------
Quick validation that the TreeLoRA fixes are working correctly.

This runs a minimal 2-task experiment and checks that:
1. Task 0 accuracy remains high after training task 1 (> 70%)
2. Checkpoints are saved correctly
3. Results are documented properly

Usage:
    python validate_fixes.py
"""

import os
import sys
import json
import torch
from datasets import get_split_cifar100
from continual_learner import TreeLoRALearner


def validate():
    print("\n" + "="*60)
    print("TreeLoRA Fix Validation")
    print("="*60 + "\n")
    
    # Minimal config for quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./runs/validation_test"
    
    print(f"Device: {device}")
    print(f"Output: {output_dir}\n")
    
    # Get 2-task CIFAR-100
    task_dataloaders, _ = get_split_cifar100(
        data_root="./data",
        n_tasks=2,
        batch_size=64,
        num_workers=2,
    )
    
    # Create learner
    learner = TreeLoRALearner(
        num_tasks=2,
        classes_per_task=50,  # 2 tasks × 50 classes
        lora_rank=4,
        lora_alpha=8.0,
        lora_depth=5,
        reg=1.0,
        lr=0.003,
        device=device,
        pretrained=True,
        output_dir=output_dir,
    )
    
    print("Training 2 tasks (3 epochs each)...\n")
    
    # Train
    acc_matrix, final_acc, bwt = learner.run(
        task_dataloaders=task_dataloaders,
        epochs=3,
    )
    
    # Validate results
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60 + "\n")
    
    task0_after_task0 = acc_matrix[0][0]
    task0_after_task1 = acc_matrix[1][0]
    task1_after_task1 = acc_matrix[1][1]
    
    print(f"Task 0 accuracy after task 0: {task0_after_task0:.2f}%")
    print(f"Task 0 accuracy after task 1: {task0_after_task1:.2f}%")
    print(f"Task 1 accuracy after task 1: {task1_after_task1:.2f}%")
    print(f"\nFinal Acc: {final_acc:.2f}%")
    print(f"BWT: {bwt:.2f}%\n")
    
    # Check criteria
    checks = []
    
    # 1. Task 0 should maintain > 70% accuracy
    if task0_after_task1 > 70.0:
        checks.append(("✓", f"Task 0 retention: {task0_after_task1:.1f}% > 70%"))
    else:
        checks.append(("✗", f"Task 0 retention: {task0_after_task1:.1f}% < 70% (FAIL)"))
    
    # 2. Task 1 should learn well (> 60%)
    if task1_after_task1 > 60.0:
        checks.append(("✓", f"Task 1 learning: {task1_after_task1:.1f}% > 60%"))
    else:
        checks.append(("✗", f"Task 1 learning: {task1_after_task1:.1f}% < 60% (FAIL)"))
    
    # 3. Checkpoints exist
    task0_ckpt = os.path.join(output_dir, "task_0", "model_state.pt")
    task1_ckpt = os.path.join(output_dir, "task_1", "model_state.pt")
    if os.path.exists(task0_ckpt) and os.path.exists(task1_ckpt):
        checks.append(("✓", "Checkpoints saved"))
    else:
        checks.append(("✗", "Checkpoints missing (FAIL)"))
    
    # 4. Results JSON exists
    results_path = os.path.join(output_dir, "final_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        checks.append(("✓", f"Results saved: {results_path}"))
    else:
        checks.append(("✗", "Results JSON missing (FAIL)"))
    
    # Print checks
    print("Validation Checks:")
    print("-" * 60)
    for symbol, msg in checks:
        print(f"  {symbol} {msg}")
    print()
    
    # Overall result
    all_passed = all(symbol == "✓" for symbol, _ in checks)
    
    if all_passed:
        print("="*60)
        print("✓ ALL CHECKS PASSED - Fixes are working correctly!")
        print("="*60)
        print("\nYou can now run full training with confidence:")
        print("  python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5")
        return 0
    else:
        print("="*60)
        print("✗ SOME CHECKS FAILED - Please review the issues above")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(validate())
