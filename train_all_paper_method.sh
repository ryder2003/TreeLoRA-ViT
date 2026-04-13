#!/bin/bash
################################################################################
# train_all_paper_method.sh
# 
# Complete training for all 3 datasets using paper's method
# Expected results: CIFAR-100=88.54%, ImageNet-R=71.94%, CUB-200=73.66%
#
# Usage:
#   chmod +x train_all_paper_method.sh
#   screen -S treelora
#   ./train_all_paper_method.sh
#   # Press Ctrl+A, D to detach
################################################################################

set -e

echo "=============================================================================="
echo "  TreeLoRA Training (Paper's Method)"
echo "  Expected: CIFAR-100=88.54%, ImageNet-R=71.94%, CUB-200=73.66%"
echo "  Started at: $(date)"
echo "=============================================================================="

GPU_ID=0
DATA_ROOT="./data"

################################################################################
# 1. CIFAR-100 (10 tasks × 10 classes, batch_size=64)
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  DATASET 1/3: CIFAR-100"
echo "  Expected: 88.54% Acc"
echo "████████████████████████████████████████████████████████████████████████████"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --dataset cifar100 \
    --data_root "$DATA_ROOT" \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5 \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo "✓ CIFAR-100 completed successfully"
    latest_dir=$(ls -td runs/cifar100_* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        grep "Average Accuracy" "$latest_dir/summary.txt" 2>/dev/null || true
    fi
else
    echo "✗ CIFAR-100 FAILED"
    exit 1
fi

################################################################################
# 2. ImageNet-R (10 tasks × 20 classes, batch_size=32)
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  DATASET 2/3: ImageNet-R"
echo "  Expected: 71.94% Acc"
echo "████████████████████████████████████████████████████████████████████████████"

# Clear GPU memory
echo "Clearing GPU memory..."
pkill -9 python 2>/dev/null || true
sleep 5

# Use memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --dataset imagenet_r \
    --data_root "$DATA_ROOT" \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5 \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo "✓ ImageNet-R completed successfully"
    latest_dir=$(ls -td runs/imagenet_r_* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        grep "Average Accuracy" "$latest_dir/summary.txt" 2>/dev/null || true
    fi
else
    echo "✗ ImageNet-R FAILED"
    exit 1
fi

################################################################################
# 3. CUB-200 (10 tasks × 20 classes, batch_size=32)
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  DATASET 3/3: CUB-200"
echo "  Expected: 73.66% Acc"
echo "████████████████████████████████████████████████████████████████████████████"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --dataset cub200 \
    --data_root "$DATA_ROOT" \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5 \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --num_workers 4

if [ $? -eq 0 ]; then
    echo "✓ CUB-200 completed successfully"
    latest_dir=$(ls -td runs/cub200_* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        grep "Average Accuracy" "$latest_dir/summary.txt" 2>/dev/null || true
    fi
else
    echo "✗ CUB-200 FAILED"
    exit 1
fi

################################################################################
# Summary
################################################################################
echo ""
echo "=============================================================================="
echo "  ALL TRAINING COMPLETED"
echo "  Finished at: $(date)"
echo "=============================================================================="
echo ""
echo "Results Summary:"
echo "----------------"

for dataset in cifar100 imagenet_r cub200; do
    latest_dir=$(ls -td runs/${dataset}_* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        echo ""
        echo "[$dataset]"
        grep -E "(Average Accuracy|Backward Transfer)" "$latest_dir/summary.txt" 2>/dev/null || echo "  Results not found"
    fi
done

echo ""
echo "=============================================================================="
echo "  Expected vs Actual:"
echo "=============================================================================="
echo "  CIFAR-100:   Expected=88.54%"
echo "  ImageNet-R:  Expected=71.94%"
echo "  CUB-200:     Expected=73.66%"
echo ""
echo "  View detailed results:"
echo "    cat runs/*/summary.txt"
echo ""
echo "  Commit to git:"
echo "    git add runs/"
echo "    git add -f runs/**/*.json"
echo "    git add -f runs/**/*.txt"
echo "    git commit -m 'Add training results (paper method)'"
echo "    git push origin main"
echo "=============================================================================="
