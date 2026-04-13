"""
test_class_incremental.py
--------------------------
Quick test to verify class-incremental implementation works correctly.

This runs 2 tasks for 2 epochs to ensure:
1. Labels are NOT remapped
2. Single unified head is used
3. Model can learn and evaluate correctly
"""

import torch
from datasets_class_incremental import get_split_cifar100_class_incremental
from continual_learner_class_incremental import ClassIncrementalTreeLoRALearner


def test_class_incremental():
    print("\n" + "="*70)
    print("  Testing Class-Incremental Implementation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\n1. Loading CIFAR-100 (class-incremental)...")
    task_loaders, class_splits = get_split_cifar100_class_incremental(
        data_root="./data",
        n_tasks=2,  # Only 2 tasks for quick test
        batch_size=32,
        num_workers=2
    )
    
    # Verify labels are NOT remapped
    print("\n2. Verifying label ranges...")
    for task_id in range(2):
        train_loader, _ = task_loaders[task_id]
        x, y = next(iter(train_loader))
        print(f"   Task {task_id}: classes {class_splits[task_id]}")
        print(f"            labels range: {y.min().item()} to {y.max().item()}")
        
        expected_min = class_splits[task_id][0]
        expected_max = class_splits[task_id][-1]
        assert y.min().item() >= expected_min, "Labels incorrectly remapped!"
        assert y.max().item() <= expected_max, "Labels incorrectly remapped!"
    
    print("   ✓ Labels are NOT remapped (correct!)")
    
    # Create learner
    print("\n3. Creating class-incremental learner...")
    learner = ClassIncrementalTreeLoRALearner(
        num_tasks=2,
        total_classes=100,  # Single head with ALL 100 classes
        lora_rank=4,
        lora_alpha=8.0,
        lora_depth=5,
        reg=1.5,
        lr=0.003,
        device=device,
        pretrained=True,
        output_dir="./test_class_inc_output"
    )
    
    # Verify single unified head
    print(f"   Head output dimension: {learner.model.head.out_features}")
    assert learner.model.head.out_features == 100, "Should have 100-class head!"
    print("   ✓ Single unified head with 100 classes (correct!)")
    
    # Train
    print("\n4. Training 2 tasks (2 epochs each)...")
    acc_matrix, final_acc, bwt = learner.run(
        task_dataloaders=task_loaders,
        epochs=2  # Quick test
    )
    
    # Verify results
    print("\n5. Verifying results...")
    print(f"   Task 0 accuracy: {acc_matrix[0][0]:.2f}%")
    print(f"   Task 1 accuracy: {acc_matrix[1][1]:.2f}%")
    print(f"   Task 0 after Task 1: {acc_matrix[1][0]:.2f}%")
    print(f"   Final Acc: {final_acc:.2f}%")
    print(f"   BWT: {bwt:.2f}%")
    
    # Sanity checks
    assert acc_matrix[0][0] > 30, "Task 0 should learn something!"
    assert acc_matrix[1][1] > 30, "Task 1 should learn something!"
    assert final_acc > 20, "Final accuracy too low!"
    
    print("\n" + "="*70)
    print("  ✓ All tests passed!")
    print("="*70)
    print("""
Expected behavior:
  - Accuracy will be LOWER than task-incremental (30-60% vs 90%+)
  - This is CORRECT - class-incremental is much harder
  - Model must distinguish ALL 100 classes, not just 10
  
To run full training:
  python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 10
""")


if __name__ == "__main__":
    test_class_incremental()
