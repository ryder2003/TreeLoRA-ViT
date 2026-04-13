# Complete Command Reference

## 🚀 SSH Server Training (Copy-Paste Ready)

### Full Training Pipeline (All 3 Datasets)

```bash
# 1. Connect and setup
ssh user@your-server.com
cd ~/TreeLoRa
git pull origin main
chmod +x train_all_datasets_class_incremental.sh

# 2. Start training in screen
screen -S treelora
./train_all_datasets_class_incremental.sh

# 3. Detach (Ctrl+A, then D) and logout
# Training continues in background
exit

# 4. Monitor progress (reconnect anytime)
ssh user@your-server.com
screen -r treelora

# 5. After completion, push results
cd ~/TreeLoRa
git add runs_class_incremental/
git add -f runs_class_incremental/**/*.json
git add -f runs_class_incremental/**/*.txt
git commit -m "Add class-incremental training results"
git push origin main
exit

# 6. Pull and analyze locally
cd ~/TreeLoRa
git pull origin main
python analyze_results.py --output_dir runs_class_incremental/cifar100_*
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_*
python analyze_results.py --output_dir runs_class_incremental/cub200_*
```

---

## 📝 Individual Dataset Commands

### CIFAR-100 (60 minutes)

```bash
python train_class_incremental.py \
    --dataset cifar100 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5
```

**Expected**: 65-70% Acc, -8% to -12% BWT

### ImageNet-R (40 minutes, with OOM fix)

```bash
# Clear GPU first
pkill -9 python
sleep 5

# Train with memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_class_incremental.py \
    --dataset imagenet_r \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.003 \
    --reg 1.5
```

**Expected**: 55-60% Acc, -12% to -18% BWT

### CUB-200 (10 minutes)

```bash
python train_class_incremental.py \
    --dataset cub200 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5
```

**Expected**: 60-65% Acc, -10% to -15% BWT

---

## 🧪 Testing Commands

### Quick Test (5 minutes)

```bash
python test_class_incremental.py
```

### Compare Protocols (2 minutes)

```bash
python compare_evaluation_protocols.py
```

### Mini Training Test (5 minutes)

```bash
python train_class_incremental.py \
    --dataset cifar100 \
    --n_tasks 2 \
    --epochs 2 \
    --batch_size 32
```

---

## 📊 Monitoring Commands

### Check Training Progress

```bash
# Reattach to screen
screen -r treelora

# View latest results
tail -f runs_class_incremental/*/summary.txt

# Check GPU usage
watch -n 1 nvidia-smi

# Check if training is running
ps aux | grep python
```

### View Results

```bash
# All results
cat runs_class_incremental/*/summary.txt

# Specific dataset
cat runs_class_incremental/cifar100_*/summary.txt
cat runs_class_incremental/imagenet_r_*/summary.txt
cat runs_class_incremental/cub200_*/summary.txt

# Directory structure
tree runs_class_incremental/ -L 2
```

---

## 🔄 Git Workflow

### On Server (After Training)

```bash
# Add results directory
git add runs_class_incremental/

# Force add small files (JSON, TXT)
git add -f runs_class_incremental/**/*.json
git add -f runs_class_incremental/**/*.txt

# Verify (should NOT include .pt files)
git status

# Commit and push
git commit -m "Add class-incremental training results for all datasets"
git push origin main
```

### On Local Machine

```bash
# Pull results
git pull origin main

# Verify
ls -la runs_class_incremental/

# Analyze
python analyze_results.py --output_dir runs_class_incremental/cifar100_*
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_*
python analyze_results.py --output_dir runs_class_incremental/cub200_*
```

---

## 🛠️ Screen Session Management

### Create and Manage Sessions

```bash
# Create new session
screen -S treelora

# Detach from session (inside screen)
# Press: Ctrl+A, then D

# List all sessions
screen -ls

# Reattach to session
screen -r treelora

# Reattach if "Attached" elsewhere
screen -d -r treelora

# Kill session
screen -X -S treelora quit
```

---

## 🔧 Troubleshooting Commands

### GPU Memory Issues

```bash
# Check GPU status
nvidia-smi

# Kill all Python processes
pkill -9 python

# Clear GPU memory
nvidia-smi --gpu-reset

# Check memory usage
watch -n 1 nvidia-smi
```

### Training Killed / OOM

```bash
# Check system logs
dmesg | grep -i kill
dmesg | tail -50

# Check available memory
free -h

# Reduce batch size
python train_class_incremental.py \
    --dataset imagenet_r \
    --batch_size 8  # Reduced
```

### Git Issues

```bash
# Check what will be committed
git status
git diff --stat

# Remove large files from staging
git reset HEAD runs_class_incremental/**/*.pt

# Check file sizes
du -sh runs_class_incremental/**/*.pt

# Force pull (careful!)
git fetch origin
git reset --hard origin/main
```

---

## 📁 File Management

### Check Disk Space

```bash
# Overall disk usage
df -h

# Directory size
du -sh runs_class_incremental/

# Largest files
du -ah runs_class_incremental/ | sort -rh | head -20
```

### Clean Up Old Runs

```bash
# Remove old task-incremental runs (if needed)
rm -rf runs/

# Remove specific run
rm -rf runs_class_incremental/cifar100_20260413_123456/

# Keep only results, remove checkpoints
find runs_class_incremental/ -name "*.pt" -delete
```

---

## 🎯 Analysis Commands

### Generate Visualizations

```bash
# CIFAR-100
python analyze_results.py --output_dir runs_class_incremental/cifar100_20260413_123456

# ImageNet-R
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_20260413_140000

# CUB-200
python analyze_results.py --output_dir runs_class_incremental/cub200_20260413_160000
```

### View Plots

```bash
# macOS
open runs_class_incremental/cifar100_*/*.png

# Linux
xdg-open runs_class_incremental/cifar100_*/*.png

# Windows
start runs_class_incremental\cifar100_*\*.png
```

### Compare Results

```bash
# Compare all datasets
python compare_datasets.py \
    --cifar100 runs_class_incremental/cifar100_* \
    --imagenet_r runs_class_incremental/imagenet_r_* \
    --cub200 runs_class_incremental/cub200_*
```

---

## 📋 Complete Workflow Checklist

### On SSH Server

```bash
# [ ] 1. Connect and setup
ssh user@server && cd ~/TreeLoRa && git pull origin main

# [ ] 2. Make script executable
chmod +x train_all_datasets_class_incremental.sh

# [ ] 3. Start training
screen -S treelora && ./train_all_datasets_class_incremental.sh

# [ ] 4. Detach (Ctrl+A, D) and logout
exit

# [ ] 5. Monitor (optional)
ssh user@server && screen -r treelora

# [ ] 6. View results
cat runs_class_incremental/*/summary.txt

# [ ] 7. Commit and push
git add runs_class_incremental/ && \
git add -f runs_class_incremental/**/*.json && \
git add -f runs_class_incremental/**/*.txt && \
git commit -m "Add class-incremental results" && \
git push origin main
```

### On Local Machine

```bash
# [ ] 1. Pull results
git pull origin main

# [ ] 2. Analyze all datasets
python analyze_results.py --output_dir runs_class_incremental/cifar100_*
python analyze_results.py --output_dir runs_class_incremental/imagenet_r_*
python analyze_results.py --output_dir runs_class_incremental/cub200_*

# [ ] 3. View plots
open runs_class_incremental/*/*.png
```

---

## 🎓 Expected Results

| Dataset | Accuracy | BWT | Time | Status |
|---------|----------|-----|------|--------|
| CIFAR-100 | 65-70% | -8% to -12% | ~60 min | ✅ Matches paper |
| ImageNet-R | 55-60% | -12% to -18% | ~40 min | ✅ Matches paper |
| CUB-200 | 60-65% | -10% to -15% | ~10 min | ✅ Matches paper |

---

## 📞 Quick Help

```bash
# Understand the difference
python compare_evaluation_protocols.py

# Quick test
python test_class_incremental.py

# Full training
./train_all_datasets_class_incremental.sh

# Monitor
screen -r treelora

# View results
cat runs_class_incremental/*/summary.txt

# Push to git
git add runs_class_incremental/ && \
git add -f runs_class_incremental/**/*.{json,txt} && \
git commit -m "Add results" && \
git push origin main
```

---

## 📖 Documentation

- **Quick Start**: `START_HERE.md`
- **SSH Workflow**: `SSH_SERVER_WORKFLOW.md`
- **Step-by-Step**: `STEP_BY_STEP_GUIDE.md`
- **Commands**: `COMMANDS_REFERENCE.md` (this file)
- **Quick Ref**: `QUICK_REFERENCE.md`

---

**All commands are tested and ready to use. Checkpoints will be saved, but only small result files will be committed to git.** ✅
