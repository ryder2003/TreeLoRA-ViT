# Complete SSH Server Training Workflow

## 📋 Overview

This guide covers:
1. ✅ Training all 3 datasets on SSH server
2. ✅ Saving checkpoints and results
3. ✅ Pushing results to GitHub (excluding large files)
4. ✅ Pulling and analyzing locally
5. ✅ Testing (if needed)

---

## 🚀 Quick Start (Copy-Paste Commands)

### On SSH Server

```bash
# 1. Connect to server
ssh user@your-server.com

# 2. Navigate to repo
cd ~/TreeLoRa  # or wherever your repo is

# 3. Pull latest code
git pull origin main

# 4. Make training script executable
chmod +x train_all_datasets_class_incremental.sh

# 5. Start training in screen session
screen -S treelora
./train_all_datasets_class_incremental.sh

# 6. Detach from screen (training continues in background)
# Press: Ctrl+A, then D

# 7. Logout (training continues)
exit
```

### Monitor Progress (Reconnect Anytime)

```bash
# Reconnect to server
ssh user@your-server.com

# Reattach to screen
screen -r treelora

# Or check latest results without attaching
tail -f ~/TreeLoRa/runs_class_incremental/*/summary.txt
```

### After Training Completes

```bash
# View all results
cat runs_class_incremental/*/summary.txt

# Commit and push (excludes large .pt files automatically)
git add runs_class_incremental/
git add -f runs_class_incremental/**/*.json
git add -f runs_class_incremental/**/*.txt
git commit -m "Add class-incremental training results for all datasets"
git push origin main
```

### On Local Machine

```bash
# Pull results
git pull origin main

# Analyze CIFAR-100
python analyze_results.py --output_dir runs_class_incremental/cifar100_*

# Analyze ImageNet-R
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_*

# Analyze CUB-200
python analyze_results.py --output_dir runs_class_incremental/cub200_*

# View plots
open runs_class_incremental/cifar100_*/*.png
```

---

## 📝 Detailed Step-by-Step Guide

### Step 1: Prepare SSH Server

```bash
# Connect
ssh user@your-server.com

# Navigate to repo
cd ~/TreeLoRa

# Pull latest code (includes all new files)
git pull origin main

# Verify new files exist
ls -la train_class_incremental.py
ls -la continual_learner_class_incremental.py
ls -la datasets_class_incremental.py
ls -la train_all_datasets_class_incremental.sh

# Make script executable
chmod +x train_all_datasets_class_incremental.sh

# Verify GPU is available
nvidia-smi
```

### Step 2: Start Training

#### Option A: Train All Datasets (Recommended)

```bash
# Start screen session (allows training to continue after disconnect)
screen -S treelora

# Run complete training script
./train_all_datasets_class_incremental.sh

# Detach from screen (Ctrl+A, then D)
# Training continues in background

# Logout
exit
```

#### Option B: Train Individual Datasets

```bash
# CIFAR-100 only
screen -S cifar100
python train_class_incremental.py \
    --dataset cifar100 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5
# Ctrl+A, D to detach

# ImageNet-R only (with OOM fix)
screen -S imagenet_r
pkill -9 python  # Clear GPU first
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_class_incremental.py \
    --dataset imagenet_r \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.003 \
    --reg 1.5
# Ctrl+A, D to detach

# CUB-200 only
screen -S cub200
python train_class_incremental.py \
    --dataset cub200 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5
# Ctrl+A, D to detach
```

### Step 3: Monitor Progress

```bash
# Reconnect to server
ssh user@your-server.com

# Check if training is still running
ps aux | grep python

# Reattach to screen
screen -r treelora

# Or check specific dataset
screen -r cifar100
screen -r imagenet_r
screen -r cub200

# View latest results without attaching
tail -f ~/TreeLoRa/runs_class_incremental/cifar100_*/summary.txt

# Check GPU usage
watch -n 1 nvidia-smi
```

### Step 4: After Training Completes

```bash
# View all results
cat runs_class_incremental/*/summary.txt

# Or view individually
cat runs_class_incremental/cifar100_*/summary.txt
cat runs_class_incremental/imagenet_r_*/summary.txt
cat runs_class_incremental/cub200_*/summary.txt

# Check directory structure
tree runs_class_incremental/ -L 2
```

### Step 5: Commit Results to Git

```bash
# Add results directory
git add runs_class_incremental/

# Force add JSON and text files (small)
git add -f runs_class_incremental/**/*.json
git add -f runs_class_incremental/**/*.txt

# Verify what will be committed (should NOT include .pt files)
git status

# Commit
git commit -m "Add class-incremental training results for CIFAR-100, ImageNet-R, and CUB-200"

# Push to GitHub
git push origin main

# Logout
exit
```

### Step 6: Pull and Analyze Locally

```bash
# On your local machine
cd ~/TreeLoRa  # or wherever your local repo is

# Pull results from server
git pull origin main

# Verify results were pulled
ls -la runs_class_incremental/

# Analyze CIFAR-100
python analyze_results.py --output_dir runs_class_incremental/cifar100_20260413_*

# Analyze ImageNet-R
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_20260413_*

# Analyze CUB-200
python analyze_results.py --output_dir runs_class_incremental/cub200_20260413_*

# View plots (macOS)
open runs_class_incremental/cifar100_*/*.png
open runs_class_incremental/imagenet_r_*/*.png
open runs_class_incremental/cub200_*/*.png

# View plots (Linux)
xdg-open runs_class_incremental/cifar100_*/*.png

# View plots (Windows)
start runs_class_incremental/cifar100_*/*.png
```

---

## 🧪 Testing (Optional)

### Quick Test Before Full Training

```bash
# On SSH server
screen -S test
python test_class_incremental.py
# Should complete in ~5 minutes

# If test passes, proceed with full training
./train_all_datasets_class_incremental.sh
```

### Test Individual Dataset

```bash
# Test CIFAR-100 with 2 tasks, 2 epochs (~5 min)
python train_class_incremental.py \
    --dataset cifar100 \
    --n_tasks 2 \
    --epochs 2 \
    --batch_size 32

# Expected: 40-60% accuracy (lower is normal for class-incremental)
```

---

## 📊 Expected Training Times

| Dataset | Tasks | Epochs | Batch Size | GPU | Time |
|---------|-------|--------|------------|-----|------|
| CIFAR-100 | 10 | 10 | 64 | RTX 2080 Ti | ~60 min |
| ImageNet-R | 10 | 10 | 16 | RTX 2080 Ti | ~40 min |
| CUB-200 | 10 | 10 | 32 | RTX 2080 Ti | ~10 min |
| **Total** | | | | | **~110 min** |

---

## 📁 What Gets Saved

### Saved to Git (Small Files)
```
runs_class_incremental/
├── cifar100_20260413_123456/
│   ├── config.json              ✅ Committed
│   ├── final_results.json       ✅ Committed
│   ├── summary.txt              ✅ Committed
│   ├── task_0/
│   │   ├── accuracy_matrix.json ✅ Committed
│   │   └── training_log.json    ✅ Committed
│   └── ...
```

### NOT Saved to Git (Large Files)
```
runs_class_incremental/
├── cifar100_20260413_123456/
│   ├── task_0/
│   │   ├── model_state.pt       ❌ Excluded (large)
│   │   └── tree_state.pt        ❌ Excluded (large)
│   └── ...
```

---

## 🔧 Troubleshooting

### Training Killed / OOM

```bash
# Check if process was killed
dmesg | grep -i kill

# If OOM, reduce batch size
python train_class_incremental.py \
    --dataset imagenet_r \
    --batch_size 8  # Reduced from 16

# Or clear GPU memory first
pkill -9 python
nvidia-smi
```

### Screen Session Lost

```bash
# List all screen sessions
screen -ls

# Reattach to specific session
screen -r treelora

# If session is "Attached" elsewhere
screen -d -r treelora  # Detach and reattach
```

### Git Push Fails (Files Too Large)

```bash
# Check file sizes
du -sh runs_class_incremental/**/*.pt

# If .pt files are being committed, update .gitignore
echo "runs_class_incremental/**/*.pt" >> .gitignore

# Remove from staging
git reset HEAD runs_class_incremental/**/*.pt

# Commit again
git commit -m "Add results (excluding large checkpoints)"
git push origin main
```

### Results Not Showing Locally

```bash
# Verify files were pushed
git log --stat | head -50

# Force pull
git fetch origin
git reset --hard origin/main

# Check if files exist
ls -la runs_class_incremental/
```

---

## 📋 Complete Command Checklist

### On SSH Server

```bash
# [ ] Connect
ssh user@server

# [ ] Navigate
cd ~/TreeLoRa

# [ ] Pull latest
git pull origin main

# [ ] Make executable
chmod +x train_all_datasets_class_incremental.sh

# [ ] Start training
screen -S treelora
./train_all_datasets_class_incremental.sh

# [ ] Detach (Ctrl+A, D)

# [ ] Logout
exit

# [ ] Monitor (optional, reconnect anytime)
ssh user@server
screen -r treelora

# [ ] After completion, view results
cat runs_class_incremental/*/summary.txt

# [ ] Commit and push
git add runs_class_incremental/
git add -f runs_class_incremental/**/*.json
git add -f runs_class_incremental/**/*.txt
git commit -m "Add class-incremental results"
git push origin main

# [ ] Logout
exit
```

### On Local Machine

```bash
# [ ] Pull results
git pull origin main

# [ ] Analyze
python analyze_results.py --output_dir runs_class_incremental/cifar100_*
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_*
python analyze_results.py --output_dir runs_class_incremental/cub200_*

# [ ] View plots
open runs_class_incremental/*/*.png
```

---

## 🎯 Expected Results

| Dataset | Accuracy | BWT | Status |
|---------|----------|-----|--------|
| CIFAR-100 | 65-70% | -8% to -12% | ✅ Matches paper |
| ImageNet-R | 55-60% | -12% to -18% | ✅ Matches paper |
| CUB-200 | 60-65% | -10% to -15% | ✅ Matches paper |

**Remember**: Lower accuracy (65-70%) is CORRECT for class-incremental learning!

---

## 📞 Quick Reference

```bash
# Start training
screen -S treelora && ./train_all_datasets_class_incremental.sh

# Monitor
screen -r treelora

# View results
cat runs_class_incremental/*/summary.txt

# Push to git
git add runs_class_incremental/ && \
git add -f runs_class_incremental/**/*.{json,txt} && \
git commit -m "Add results" && \
git push origin main

# Pull locally
git pull origin main

# Analyze
python analyze_results.py --output_dir runs_class_incremental/cifar100_*
```

---

**Your checkpoints WILL be saved, but only small result files will be committed to git. Large .pt files stay on the server.** ✅
