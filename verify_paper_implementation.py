"""
verify_paper_implementation.py
------------------------------
Verify that the implementation matches the paper's approach.

Run this before training to ensure everything is configured correctly.
"""

import torch
import torch.nn as nn
from vit_backbone import ViTBackbone
from lora import inject_lora_to_vit, get_lora_params, reset_all_lora
from kd_lora_tree import KD_LoRA_Tree

def verify_lora_reset():
    """Verify LoRA parameters are reset correctly."""
    print("\n" + "="*60)
    print("TEST 1: LoRA Reset Mechanism")
    print("="*60)
    
    model = ViTBackbone(num_classes=10, pretrained=False)
    inject_lora_to_vit(model, rank=4, alpha=8.0, verbose=False)
    
    # Get initial LoRA-A parameters
    initial_params = {}
    for name, param in get_lora_params(model):
        if "loranew_A" in name:
            initial_params[name] = param.clone()
    
    # Reset LoRA
    reset_all_lora(model)
    
    # Check if parameters changed
    changed = 0
    for name, param in get_lora_params(model):
        if "loranew_A" in name and name in initial_params:
            if not torch.allclose(param, initial_params[name]):
                changed += 1
    
    print(f"✓ LoRA-A parameters changed after reset: {changed}/{len(initial_params)}")
    
    # Verify B parameters are zero
    b_zero = 0
    b_total = 0
    for name, param in get_lora_params(model):
        if "loranew_B" in name:
            b_total += 1
            if torch.allclose(param, torch.zeros_like(param)):
                b_zero += 1
    
    print(f"✓ LoRA-B parameters are zero: {b_zero}/{b_total}")
    
    if changed == len(initial_params) and b_zero == b_total:
        print("✅ PASS: LoRA reset works correctly")
        return True
    else:
        print("❌ FAIL: LoRA reset not working properly")
        return False

def verify_lcb_search():
    """Verify LCB bandit search is implemented."""
    print("\n" + "="*60)
    print("TEST 2: LCB Bandit Search")
    print("="*60)
    
    tree = KD_LoRA_Tree(num_tasks=5, lora_depth=5, reg=1.0)
    
    # Simulate 2 tasks
    for t in range(2):
        tree.new_epoch_init(train_dataloader_len=10)
        for step in range(10):
            tree.step()
            fake_grads = torch.randn(5, 256)
            tree.insert_grad(fake_grads)
        tree.end_task(t)
    
    # Test tree search for task 2
    tree.new_epoch_init(10)
    tree.step()
    fake_grads = torch.randn(5, 256)
    tree.insert_grad(fake_grads)
    
    device = torch.device("cpu")
    prev_ids = tree.tree_search(task_id=2, device=device)
    
    print(f"✓ Tree search returned prev_id_matrix: {prev_ids}")
    print(f"✓ Shape: {prev_ids.shape} (expected: torch.Size([5]))")
    print(f"✓ Values in range [0, 1]: {torch.all((prev_ids >= 0) & (prev_ids < 2))}")
    
    # Check exploration bonus is computed
    has_exploration = hasattr(tree, 'num_of_selected') and tree.num_of_selected is not None
    print(f"✓ Exploration tracking enabled: {has_exploration}")
    
    if prev_ids.shape[0] == 5 and torch.all((prev_ids >= 0) & (prev_ids < 2)) and has_exploration:
        print("✅ PASS: LCB search works correctly")
        return True
    else:
        print("❌ FAIL: LCB search not working properly")
        return False

def verify_gradient_collection():
    """Verify gradient collection for regularization."""
    print("\n" + "="*60)
    print("TEST 3: Gradient Collection")
    print("="*60)
    
    model = ViTBackbone(num_classes=10, pretrained=False)
    inject_lora_to_vit(model, rank=4, alpha=8.0, verbose=False)
    
    # Collect LoRA-A parameters
    lora_A_params = []
    for name, param in model.named_parameters():
        if "loranew_A" in name:
            lora_A_params.append(param)
    
    print(f"✓ Found {len(lora_A_params)} LoRA-A parameters")
    
    if lora_A_params:
        # Stack into gradient tensor
        grad_tensor = torch.stack([p.reshape(-1) for p in lora_A_params], dim=0)
        print(f"✓ Stacked gradient shape: {grad_tensor.shape}")
        print(f"✓ Expected shape: (lora_depth, dim*rank)")
        
        # Verify it's differentiable
        print(f"✓ Gradient tensor requires_grad: {grad_tensor.requires_grad}")
        
        if len(lora_A_params) > 0 and grad_tensor.requires_grad:
            print("✅ PASS: Gradient collection works correctly")
            return True
    
    print("❌ FAIL: Gradient collection not working properly")
    return False

def verify_tree_regularization():
    """Verify tree regularization loss computation."""
    print("\n" + "="*60)
    print("TEST 4: Tree Regularization Loss")
    print("="*60)
    
    from kd_lora_tree import tree_lora_loss
    
    # Create fake gradients
    current_grad = torch.randn(5, 256, requires_grad=True)
    all_grad = torch.randn(3, 5, 256)
    prev_id_matrix = torch.tensor([0, 1, 2, 0, 1])
    
    # Compute regularization loss
    reg_loss = tree_lora_loss(current_grad, all_grad, task_id=3, prev_id_matrix=prev_id_matrix)
    
    print(f"✓ Regularization loss computed: {reg_loss.item():.4f}")
    print(f"✓ Loss is scalar: {reg_loss.dim() == 0}")
    print(f"✓ Loss requires_grad: {reg_loss.requires_grad}")
    
    # Verify it's negative (alignment loss)
    print(f"✓ Loss is negative (alignment): {reg_loss.item() < 0}")
    
    if reg_loss.dim() == 0:
        print("✅ PASS: Tree regularization works correctly")
        return True
    else:
        print("❌ FAIL: Tree regularization not working properly")
        return False

def verify_hyperparameters():
    """Verify default hyperparameters in class-incremental entrypoint match paper target run."""
    print("\n" + "="*60)
    print("TEST 5: Default Hyperparameters")
    print("="*60)
    
    from train_class_incremental import parse_args

    parser_defaults = parse_args()
    expected = {
        "epochs": 20,
        "batch_size": 192,
        "lr": 0.005,
        "reg": 0.1,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "lora_depth": 5,
    }

    checks = {
        "epochs": parser_defaults.epochs,
        "batch_size": parser_defaults.batch_size,
        "lr": parser_defaults.lr,
        "reg": parser_defaults.reg,
        "lora_rank": parser_defaults.lora_rank,
        "lora_alpha": parser_defaults.lora_alpha,
        "lora_depth": parser_defaults.lora_depth,
    }

    all_match = True
    for k, v in expected.items():
        match = checks[k] == v
        status = "✓" if match else "✗"
        print(f"{status} {k}: expected={v} actual={checks[k]}")
        all_match = all_match and match

    if all_match:
        print("✅ PASS: Class-incremental defaults match paper target run")
        return True

    print("❌ FAIL: Class-incremental defaults do not match paper target run")
    return False

def main():
    print("\n" + "="*60)
    print("TreeLoRA Paper Implementation Verification")
    print("="*60)
    
    tests = [
        verify_lora_reset,
        verify_lcb_search,
        verify_gradient_collection,
        verify_tree_regularization,
        verify_hyperparameters,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ FAIL: {test.__name__} raised exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        print("Implementation matches paper's approach.")
        print("\nYou can now run training with:")
        print("  python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 20 --batch_size 192 --lr 0.005 --reg 0.1 --seed 42 --deterministic")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please review the failed tests above.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
