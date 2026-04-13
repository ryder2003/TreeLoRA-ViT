#!/bin/bash
# Paper Reproduction Training Commands
# TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs
# arXiv:2506.10355v1

echo "=================================================="
echo "TreeLoRA Paper Reproduction - Training Commands"
echo "=================================================="
echo ""

# ============================================================================
# 1. CIFAR-100 (10 tasks, 10 classes each)
# ============================================================================
echo "1. Split CIFAR-100 (Paper's Main Benchmark)"
echo "   Expected: Acc=65-70%, BWT=-8% to -12%"
echo ""
echo "   Full training (10 epochs per task):"
echo "   python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --batch_size 64 --lr 0.003 --reg 1.5"
echo ""
echo "   Quick test (2 epochs per task):"
echo "   python train.py --dataset cifar100 --n_tasks 10 --epochs 2 --batch_size 64 --lr 0.003 --reg 1.5"
echo ""
echo "   Smoke test (2 tasks, 1 epoch):"
echo "   python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --batch_size 16 --no_pretrained"
echo ""

# ============================================================================
# 2. ImageNet-R (20 tasks, 10 classes each)
# ============================================================================
echo "2. Split ImageNet-R (Challenging Benchmark)"
echo "   Expected: Acc=55-60%, BWT=-12% to -18%"
echo "   Requires: python download_datasets.py (first time)"
echo ""
echo "   Full training (8 epochs per task):"
echo "   python train.py --dataset imagenet_r --n_tasks 20 --epochs 8 --batch_size 32 --lr 0.002 --reg 2.0"
echo ""
echo "   Quick test (10 tasks, 2 epochs):"
echo "   python train.py --dataset imagenet_r --n_tasks 10 --epochs 2 --batch_size 32 --lr 0.002 --reg 2.0"
echo ""

# ============================================================================
# 3. CUB-200 (10 tasks, 20 classes each)
# ============================================================================
echo "3. Split CUB-200 (Fine-grained Classification)"
echo "   Expected: Acc=60-65%, BWT=-10% to -15%"
echo "   Requires: python download_datasets.py (first time)"
echo ""
echo "   Full training (10 epochs per task):"
echo "   python train.py --dataset cub200 --n_tasks 10 --epochs 10 --batch_size 32 --lr 0.003 --reg 1.5"
echo ""
echo "   Quick test (5 tasks, 2 epochs):"
echo "   python train.py --dataset cub200 --n_tasks 5 --epochs 2 --batch_size 32 --lr 0.003 --reg 1.5"
echo ""

# ============================================================================
# Ablation Studies
# ============================================================================
echo "=================================================="
echo "Ablation Studies"
echo "=================================================="
echo ""
echo "A. Regularization Strength (λ):"
echo "   python train.py --dataset cifar100 --reg 0.5   # Weak regularization"
echo "   python train.py --dataset cifar100 --reg 1.0   # Medium regularization"
echo "   python train.py --dataset cifar100 --reg 1.5   # Strong regularization (paper default)"
echo "   python train.py --dataset cifar100 --reg 2.0   # Very strong regularization"
echo ""
echo "B. Tree Depth:"
echo "   python train.py --dataset cifar100 --lora_depth 3   # Shallow tree"
echo "   python train.py --dataset cifar100 --lora_depth 5   # Paper default"
echo "   python train.py --dataset cifar100 --lora_depth 7   # Deep tree"
echo ""
echo "C. LoRA Rank:"
echo "   python train.py --dataset cifar100 --lora_rank 2 --lora_alpha 4.0   # Low rank"
echo "   python train.py --dataset cifar100 --lora_rank 4 --lora_alpha 8.0   # Paper default"
echo "   python train.py --dataset cifar100 --lora_rank 8 --lora_alpha 16.0  # High rank"
echo ""
echo "D. Baseline (No TreeLoRA Regularization):"
echo "   python train.py --dataset cifar100 --reg 0.0   # SeqLoRA baseline"
echo ""

# ============================================================================
# Verification
# ============================================================================
echo "=================================================="
echo "Verification"
echo "=================================================="
echo ""
echo "Before training, verify implementation:"
echo "   python verify_paper_implementation.py"
echo ""
echo "Expected output: All 5 tests should PASS"
echo ""

# ============================================================================
# GPU Configuration
# ============================================================================
echo "=================================================="
echo "GPU Configuration"
echo "=================================================="
echo ""
echo "Single GPU (default):"
echo "   python train.py --dataset cifar100"
echo ""
echo "Specific GPU:"
echo "   CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100"
echo ""
echo "CPU only (slow, for testing):"
echo "   python train.py --dataset cifar100 --device cpu"
echo ""
echo "Memory optimization (if OOM):"
echo "   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --dataset cifar100 --batch_size 32"
echo ""

# ============================================================================
# Output Management
# ============================================================================
echo "=================================================="
echo "Output Management"
echo "=================================================="
echo ""
echo "Custom output directory:"
echo "   python train.py --dataset cifar100 --output_dir ./runs/my_experiment"
echo ""
echo "Default output: ./runs/<dataset>_<timestamp>/"
echo "   - config.json           # Experiment configuration"
echo "   - task_X/               # Checkpoints after each task"
echo "   - final_results.json    # Complete results"
echo "   - summary.txt           # Human-readable summary"
echo ""

# ============================================================================
# Expected Training Times (on RTX 2080 Ti)
# ============================================================================
echo "=================================================="
echo "Expected Training Times (RTX 2080 Ti)"
echo "=================================================="
echo ""
echo "CIFAR-100 (10 tasks, 10 epochs):  ~5-10 minutes"
echo "ImageNet-R (20 tasks, 8 epochs):  ~50-60 minutes"
echo "CUB-200 (10 tasks, 10 epochs):    ~15-20 minutes"
echo ""

# ============================================================================
# Troubleshooting
# ============================================================================
echo "=================================================="
echo "Troubleshooting"
echo "=================================================="
echo ""
echo "If accuracy is too high (>90% on CIFAR-100):"
echo "  1. Verify LoRA reset is happening every task"
echo "  2. Increase regularization: --reg 2.0"
echo "  3. Reduce epochs: --epochs 5"
echo ""
echo "If accuracy is too low (<50% on CIFAR-100):"
echo "  1. Decrease regularization: --reg 0.5"
echo "  2. Increase learning rate: --lr 0.005"
echo "  3. Increase epochs: --epochs 15"
echo ""
echo "If out of memory:"
echo "  1. Reduce batch size: --batch_size 32"
echo "  2. Use memory optimization: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  3. Reduce number of workers: --num_workers 2"
echo ""

echo "=================================================="
echo "For detailed documentation, see:"
echo "  - PAPER_REPRODUCTION.md"
echo "  - CHANGES.md"
echo "  - README.md"
echo "=================================================="
