"""Quick smoke test — runs a 2-task, 1-epoch CIFAR-100 training."""
import traceback
try:
    import torch
    from datasets import get_split_cifar100
    from continual_learner import TreeLoRALearner

    print("Import OK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dataset (2 tasks only)
    task_dataloaders, class_splits = get_split_cifar100(
        data_root="./data", n_tasks=2, batch_size=16, num_workers=0,
    )
    print(f"Dataset OK: {len(task_dataloaders)} tasks")

    # Build learner
    learner = TreeLoRALearner(
        num_tasks=2, classes_per_task=50,
        lora_rank=4, lora_alpha=8.0, lora_depth=5, reg=0.5,
        lr=5e-3, device=device, pretrained=False,
    )
    print("Learner OK")
    learner.model.print_trainable_summary()

    # Run training
    acc_matrix, final_acc, bwt = learner.run(
        task_dataloaders=task_dataloaders, epochs=1,
    )

    print(f"\nFinal Acc: {final_acc:.2f}%")
    print(f"BWT: {bwt:.2f}%")
    print("\n[OK] SMOKE TEST PASSED")

except Exception as e:
    traceback.print_exc()
    print("\n[FAIL] SMOKE TEST FAILED")
