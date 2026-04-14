#!/bin/bash
################################################################################
# train_all_datasets_class_incremental.sh
# 
# Complete training script for all 3 datasets using class-incremental learning
# (matches paper's evaluation protocol)
#
# Usage on SSH server:
#   chmod +x train_all_datasets_class_incremental.sh
#   screen -S treelora
#   ./train_all_datasets_class_incremental.sh
#   # Press Ctrl+A, D to detach
#   # screen -r treelora to reattach
################################################################################

set -e  # Exit on error

echo "=============================================================================="
echo "  TreeLoRA Class-Incremental Training (Paper's Method)"
echo "  Starting at: $(date)"
echo "=============================================================================="

# Configuration
GPU_ID=0
DATA_ROOT="./data"
OUTPUT_BASE="./runs_class_incremental"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to train a dataset
train_dataset() {
    local dataset=$1
    local n_tasks=$2
    local epochs=$3
    local batch_size=$4
    local lr=$5
    local reg=$6
    
    echo ""
    echo "=============================================================================="
    echo "  Training: $dataset"
    echo "  Tasks: $n_tasks | Epochs: $epochs | Batch: $batch_size | LR: $lr | Reg: $reg"
    echo "  Started at: $(date)"
    echo "=============================================================================="
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_class_incremental.py \
        --dataset "$dataset" \
        --data_root "$DATA_ROOT" \
        --n_tasks "$n_tasks" \
        --epochs "$epochs" \
        --batch_size "$batch_size" \
        --lr "$lr" \
        --reg "$reg" \
        --num_workers 4
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ $dataset training completed successfully at $(date)"
        echo ""
        
        # Show results
        latest_dir=$(ls -td "$OUTPUT_BASE"/${dataset}_* 2>/dev/null | head -1)
        if [ -n "$latest_dir" ]; then
            echo "Results for $dataset:"
            echo "----------------------------------------"
            cat "$latest_dir/summary.txt" 2>/dev/null || echo "Summary not found"
            echo "----------------------------------------"
        fi
    else
        echo ""
        echo "✗ $dataset training FAILED with exit code $exit_code at $(date)"
        echo ""
        return $exit_code
    fi
}

################################################################################
# 1. CIFAR-100 (10 tasks × 10 classes)
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  DATASET 1/3: CIFAR-100"
echo "████████████████████████████████████████████████████████████████████████████"

train_dataset "cifar100" 10 10 64 0.003 2.0

################################################################################
# 2. ImageNet-R (10 tasks × 20 classes)
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  DATASET 2/3: ImageNet-R"
echo "████████████████████████████████████████████████████████████████████████████"

# Clear GPU memory before ImageNet-R
echo "Clearing GPU memory..."
pkill -9 python 2>/dev/null || true
sleep 5

# Use memory optimization for ImageNet-R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_dataset "imagenet_r" 10 10 16 0.003 2.0

################################################################################
# 3. CUB-200 (10 tasks × 20 classes)
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████"
echo "  DATASET 3/3: CUB-200"
echo "████████████████████████████████████████████████████████████████████████████"

train_dataset "cub200" 10 10 32 0.003 2.0

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
    latest_dir=$(ls -td "$OUTPUT_BASE"/${dataset}_* 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        echo ""
        echo "[$dataset]"
        grep -E "(Average Accuracy|Backward Transfer)" "$latest_dir/summary.txt" 2>/dev/null || echo "  Results not found"
    fi
done

echo ""
echo "=============================================================================="
echo "  Next Steps:"
echo "=============================================================================="
echo "1. Review results:"
echo "   cat runs_class_incremental/*/summary.txt"
echo ""
echo "2. Commit results to git (excludes large .pt files):"
echo "   git add runs_class_incremental/"
echo "   git add -f runs_class_incremental/**/*.json"
echo "   git add -f runs_class_incremental/**/*.txt"
echo "   git commit -m 'Add class-incremental training results'"
echo "   git push origin main"
echo ""
echo "3. Pull on local machine:"
echo "   git pull origin main"
echo ""
echo "4. Analyze locally:"
echo "   python analyze_results.py --output_dir runs_class_incremental/cifar100_*"
echo "=============================================================================="
