"""
verify_fixes.py
---------------
Fast unit tests verifying all 5 critical fixes without a full training run.
Uses tiny random tensors - no GPU or dataset needed, completes in <10 seconds.
"""
import math, sys, os, json, tempfile
import torch
import torch.nn as nn

print("=" * 60)
print("  TreeLoRA Fix Verification Suite")
print("=" * 60)

errors = []

# ──────────────────────────────────────────────────────────────
# Fix 1: LoRA reset_parameters() exists and zeros B
# ──────────────────────────────────────────────────────────────
print("\n[1] LoRA re-initialisation (reset_all_lora) ...")
try:
    from lora import LoRAQKV, reset_all_lora, inject_lora_to_vit
    from vit_backbone import ViTBackbone

    # Build a tiny model without pretrained weights
    model = ViTBackbone(num_classes=10, pretrained=False)
    inject_lora_to_vit(model, rank=4, alpha=8.0, verbose=False)

    # Grab one LoRA layer
    block = model.vit.blocks[0]
    qkv = block.attn.qkv

    # Manually set B to non-zero
    with torch.no_grad():
        qkv.loranew_B.fill_(9.9)
        qkv.loranew_B_v.fill_(9.9)

    assert qkv.loranew_B.abs().max().item() > 0, "B not set to 9.9?"

    # Reset
    reset_all_lora(model)

    assert qkv.loranew_B.abs().max().item() == 0.0, \
        f"B not zeroed after reset: max={qkv.loranew_B.abs().max().item()}"
    assert qkv.loranew_B_v.abs().max().item() == 0.0, \
        f"B_v not zeroed after reset: max={qkv.loranew_B_v.abs().max().item()}"
    assert qkv.loranew_A.abs().max().item() > 0, \
        "A is all zeros after reset (should be Kaiming)"

    print("  [OK] LoRA B zeroed, A re-initialised with Kaiming")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"Fix 1: {e}")

# ──────────────────────────────────────────────────────────────
# Fix 2: insert_grad accumulates lora_depth times per call
# ──────────────────────────────────────────────────────────────
print("\n[2] insert_grad loop (accumulates lora_depth x per call) ...")
try:
    from kd_lora_tree import KD_LoRA_Tree

    lora_depth = 5
    feat_dim   = 100
    n_steps    = 10

    tree = KD_LoRA_Tree(num_tasks=5, lora_depth=lora_depth, reg=0.5)
    tree.new_epoch_init(n_steps)

    grads = torch.ones(lora_depth, feat_dim)   # all-ones tensor
    tree.insert_grad(grads)                    # single call

    # After 1 call: current_grad should equal grads * lora_depth / n_steps
    expected = grads * lora_depth / n_steps
    diff = (tree.current_grad - expected).abs().max().item()
    assert diff < 1e-5, \
        f"insert_grad wrong after 1 call. Expected {expected[0,0]:.6f}, " \
        f"got {tree.current_grad[0,0]:.6f}, diff={diff}"

    print(f"  [OK] After 1 call: current_grad[0,0] = {tree.current_grad[0,0]:.6f} "
          f"(expected {expected[0,0]:.6f})")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"Fix 2: {e}")

# ──────────────────────────────────────────────────────────────
# Fix 3: get_loss uses .clone() not .abs() — sign preserved
# ──────────────────────────────────────────────────────────────
print("\n[3] get_loss sign behavior (.clone() not .abs()) ...")
try:
    from kd_lora_tree import KD_LoRA_Tree, tree_lora_loss
    import inspect

    src = inspect.getsource(KD_LoRA_Tree.get_loss)
    assert ".abs()" not in src, \
        "get_loss still contains .abs() — fix was not applied!"
    assert ".clone()" in src, \
        "get_loss does not contain .clone() — fix was not applied!"
    print("  [OK] get_loss uses .detach().clone() (not .abs())")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"Fix 3: {e}")

# ──────────────────────────────────────────────────────────────
# Fix 4: Checkpoint saving works end-to-end
# ──────────────────────────────────────────────────────────────
print("\n[4] Checkpoint saving (save_checkpoint / save_final_results) ...")
try:
    from continual_learner import TreeLoRALearner

    with tempfile.TemporaryDirectory() as tmpdir:
        learner = TreeLoRALearner(
            num_tasks=2, classes_per_task=10,
            lora_rank=4, lora_alpha=8.0, lora_depth=5, reg=0.5,
            lr=1e-3, pretrained=False, output_dir=tmpdir,
        )

        # Fake a completed task 0
        learner._set_task_head(0)
        learner.task_heads[0] = {
            k: v.clone() for k, v in learner.model.head.state_dict().items()
        }
        learner.acc_matrix = [[87.5]]
        learner.training_log = [{"task_id": 0, "epochs": [{"epoch": 1, "avg_loss": 0.5, "train_acc": 87.5}]}]

        # Fake a tree gradient so tree_state won't be all None
        learner.tree.all_accumulate_grads[0] = torch.randn(24, 3072)

        # Save checkpoint
        learner.save_checkpoint(task_id=0)

        ckpt_dir = os.path.join(tmpdir, "task_0")
        assert os.path.isfile(os.path.join(ckpt_dir, "lora_weights.pt")), \
            "lora_weights.pt not saved"
        assert os.path.isfile(os.path.join(ckpt_dir, "task_heads.pt")), \
            "task_heads.pt not saved"
        assert os.path.isfile(os.path.join(ckpt_dir, "tree_state.pt")), \
            "tree_state.pt not saved"
        assert os.path.isfile(os.path.join(ckpt_dir, "accuracy_matrix.json")), \
            "accuracy_matrix.json not saved"
        assert os.path.isfile(os.path.join(ckpt_dir, "training_log.json")), \
            "training_log.json not saved"

        # Verify accuracy_matrix.json content
        with open(os.path.join(ckpt_dir, "accuracy_matrix.json")) as f:
            data = json.load(f)
        assert data["acc_matrix"] == [[87.5]], f"Wrong acc_matrix: {data}"
        assert data["task_id"] == 0

        # Save final results
        learner.save_final_results(87.5, -5.0, 120.0)
        assert os.path.isfile(os.path.join(tmpdir, "final_results.json"))
        assert os.path.isfile(os.path.join(tmpdir, "summary.txt"))

        # Verify lora_weights.pt is loadable
        lora_state = torch.load(os.path.join(ckpt_dir, "lora_weights.pt"),
                                weights_only=True)
        assert len(lora_state) > 0, "lora_weights.pt is empty"
        print(f"  [OK] {len(lora_state)} LoRA tensors saved, JSON files verified")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  [FAIL] {e}")
    errors.append(f"Fix 4: {e}")

# ──────────────────────────────────────────────────────────────
# Fix 5: --device and --output_dir CLI args exist in train.py
# ──────────────────────────────────────────────────────────────
print("\n[5] train.py CLI args (--device, --output_dir) ...")
try:
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True, text=True, timeout=30
    )
    help_text = result.stdout + result.stderr
    assert "--device" in help_text,     "--device arg missing from train.py"
    assert "--output_dir" in help_text, "--output_dir arg missing from train.py"
    print("  [OK] --device and --output_dir found in train.py --help")
except Exception as e:
    print(f"  [FAIL] {e}")
    errors.append(f"Fix 5: {e}")

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"  RESULT: {len(errors)} FAILURE(S)")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  RESULT: ALL 5 FIXES VERIFIED [OK]")
    print("=" * 60)
    print()
    print("  Ready to push to server and run:")
    print()
    print("  CUDA_VISIBLE_DEVICES=0 \\")
    print("  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\")
    print("  python train.py \\")
    print("    --dataset cifar100 \\")
    print("    --data_root ./data \\")
    print("    --n_tasks 10 \\")
    print("    --epochs 5 \\")
    print("    --batch_size 64 \\")
    print("    --output_dir ./runs/cifar100_fixed")
    print()
    sys.exit(0)
