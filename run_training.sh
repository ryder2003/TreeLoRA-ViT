#!/bin/bash
# run_training.sh
# Recommended training configurations for TreeLoRA

# CIFAR-100 (10 tasks) - Optimized for better retention
echo "Training CIFAR-100 with optimized hyperparameters..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar100 \
    --data_root ./data \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5 \
    --lora_rank 4 \
    --lora_alpha 8.0

# ImageNet-R (20 tasks) - Lower batch size for memory
# echo "Training ImageNet-R with optimized hyperparameters..."
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --dataset imagenet_r \
#     --data_root ./data \
#     --n_tasks 20 \
#     --epochs 8 \
#     --batch_size 32 \
#     --lr 0.002 \
#     --reg 2.0 \
#     --lora_rank 4 \
#     --lora_alpha 8.0

# CUB-200 (10 tasks)
# echo "Training CUB-200 with optimized hyperparameters..."
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --dataset cub200 \
#     --data_root ./data \
#     --n_tasks 10 \
#     --epochs 10 \
#     --batch_size 32 \
#     --lr 0.003 \
#     --reg 1.5 \
#     --lora_rank 4 \
#     --lora_alpha 8.0
