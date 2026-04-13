# TRAINING COMMANDS - Paper Reproduction

## ⚡ Quick Start (Copy-Paste Ready)

### Step 1: Verify Implementation
```bash
python verify_paper_implementation.py
```
**Expected:** All 5 tests PASS ✅

---

### Step 2: Choose Your Experiment

## 🎯 Main Experiments (Paper Reproduction)

### CIFAR-100 (Recommended First)
**Full Training (10 tasks, 10 epochs per task):**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --batch_size 64 --lr 0.003 --reg 1.5
```
- **Time:** ~5-10 minutes
- **Expected Acc:** 65-70%
- **Expected BWT:** -8% to -12%

**Quick Test (10 tasks, 2 epochs per task):**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 2 --batch_size 64 --lr 0.003 --reg 1.5
```
- **Time:** ~2 minutes
- **Expected Acc:** 60-65%

**Smoke Test (2 tasks, 1 epoch):**
```bash
python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --batch_size 16 --no_pretrained
```
- **Time:** ~1 minute
- **Purpose:** Verify pipeline works

---

### ImageNet-R (Challenging)
**First Time: Download Dataset**
```bash
python download_datasets.py
```

**Full Training (20 tasks, 8 epochs per task):**
```bash
python train.py --dataset imagenet_r --n_tasks 20 --epochs 8 --batch_size 32 --lr 0.002 --reg 2.0
```
- **Time:** ~50-60 minutes
- **Expected Acc:** 55-60%
- **Expected BWT:** -12% to -18%

**Quick Test (10 tasks, 2 epochs per task):**
```bash
python train.py --dataset imagenet_r --n_tasks 10 --epochs 2 --batch_size 32 --lr 0.002 --reg 2.0
```
- **Time:** ~10 minutes

---

### CUB-200 (Fine-grained)
**First Time: Download Dataset**
```bash
python download_datasets.py
```

**Full Training (10 tasks, 10 epochs per task):**
```bash
python train.py --dataset cub200 --n_tasks 10 --epochs 10 --batch_size 32 --lr 0.003 --reg 1.5
```
- **Time:** ~15-20 minutes
- **Expected Acc:** 60-65%
- **Expected BWT:** -10% to -15%

**Quick Test (5 tasks, 2 epochs per task):**
```bash
python train.py --dataset cub200 --n_tasks 5 --epochs 2 --batch_size 32 --lr 0.003 --reg 1.5
```
- **Time:** ~5 minutes

---

## 🔬 Ablation Studies

### A. Regularization Strength (λ)

**Weak Regularization:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 0.5
```

**Medium Regularization:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.0
```

**Strong Regularization (Paper Default):**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5
```

**Very Strong Regularization:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 2.0
```

---

### B. Tree Depth

**Shallow Tree:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --lora_depth 3
```

**Paper Default:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --lora_depth 5
```

**Deep Tree:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --lora_depth 7
```

---

### C. LoRA Rank

**Low Rank:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --lora_rank 2 --lora_alpha 4.0
```

**Paper Default:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --lora_rank 4 --lora_alpha 8.0
```

**High Rank:**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --lora_rank 8 --lora_alpha 16.0
```

---

### D. Baseline (No TreeLoRA)

**SeqLoRA (Sequential LoRA without tree regularization):**
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 0.0
```
- **Expected:** Much worse forgetting (BWT < -30%)

---

## 🖥️ GPU Configuration

### Single GPU (Default)
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5
```

### Specific GPU
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5
```

### Multiple GPUs (Use first available)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5
```

### CPU Only (Slow, for testing)
```bash
python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --device cpu
```

### Memory Optimization (If OOM)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --batch_size 32 --lr 0.003 --reg 1.5
```

---

## 📁 Output Management

### Default Output (Auto-generated timestamp)
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5
```
**Saves to:** `./runs/cifar100_YYYYMMDD_HHMMSS/`

### Custom Output Directory
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --output_dir ./runs/paper_reproduction_cifar100
```

### Specific Experiment Name
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.5 --output_dir ./runs/cifar100_reg1.5_epoch10
```

---

## 📊 Expected Output

### Console Output
```
============================================================
  TreeLoRA ViT-B/16 -- CIFAR100
  Device       : cuda
  Tasks        : 10  (10 classes each)
  Epochs/task  : 10
  LoRA         : rank=4  alpha=8.0  scaling=2.0
  Tree         : depth=5  reg=1.5
============================================================

Training Task 0  (5000 samples)
  LoRA re-initialised in 12 blocks
  Epoch 1/10  loss=0.5296  train_acc=83.86%
  ...
  Epoch 10/10  loss=0.1629  train_acc=94.80%
  -> Task 0 accuracy: 99.20%

Training Task 1  (5000 samples)
  LoRA re-initialised in 12 blocks  ← IMPORTANT: Should see this every task
  ...

Final Average Accuracy (Acc): 67.50%  ← Should be 65-70%
Backward Transfer     (BWT): -10.20%  ← Should be -8% to -12%
```

### Saved Files
```
runs/cifar100_20260413_173004/
├── config.json              # Experiment configuration
├── task_0/                  # Checkpoint after task 0
│   ├── model_state.pt
│   ├── task_heads.pt
│   ├── tree_state.pt
│   ├── accuracy_matrix.json
│   └── training_log.json
├── task_1/                  # Checkpoint after task 1
│   └── ...
├── ...
├── task_9/                  # Checkpoint after task 9
│   └── ...
├── final_results.json       # Complete results
└── summary.txt              # Human-readable summary
```

---

## ✅ Verification Checklist

### Before Training:
- [ ] Run `python verify_paper_implementation.py` → All tests PASS
- [ ] Check GPU is available: `nvidia-smi`
- [ ] Check disk space: `df -h`

### During Training:
- [ ] See "LoRA re-initialised in 12 blocks" for EVERY task
- [ ] Training loss decreases each epoch
- [ ] Training accuracy increases each epoch
- [ ] Tree structure is printed after each task

### After Training:
- [ ] Final Acc is in expected range (65-70% for CIFAR-100)
- [ ] BWT shows forgetting (-8% to -12% for CIFAR-100)
- [ ] Checkpoints are saved in output directory
- [ ] Accuracy matrix shows diagonal dominance

---

## 🐛 Troubleshooting

### Problem: Accuracy too high (>90%)
**Solution:**
```bash
# Increase regularization
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 2.0

# Or reduce epochs
python train.py --dataset cifar100 --n_tasks 10 --epochs 5 --lr 0.003 --reg 1.5
```

### Problem: Accuracy too low (<50%)
**Solution:**
```bash
# Decrease regularization
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 0.5

# Or increase learning rate
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.005 --reg 1.5
```

### Problem: Out of memory
**Solution:**
```bash
# Reduce batch size
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --batch_size 32 --lr 0.003 --reg 1.5

# Or use memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --batch_size 32 --lr 0.003 --reg 1.5
```

### Problem: "LoRA re-initialised" not showing every task
**Solution:** Check `continual_learner.py` line ~250:
```python
# Should be OUTSIDE any if statement:
reset_all_lora(self.model)
```

---

## 📈 Recommended Workflow

### Day 1: Verification & Quick Test
```bash
# 1. Verify implementation (1 minute)
python verify_paper_implementation.py

# 2. Smoke test (1 minute)
python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --batch_size 16 --no_pretrained

# 3. Quick test (2 minutes)
python train.py --dataset cifar100 --n_tasks 10 --epochs 2 --lr 0.003 --reg 1.5
```

### Day 2: Full CIFAR-100 Reproduction
```bash
# Full training (5-10 minutes)
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --batch_size 64 --lr 0.003 --reg 1.5 --output_dir ./runs/cifar100_paper_reproduction
```

### Day 3: Ablation Studies
```bash
# Vary regularization
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 0.5 --output_dir ./runs/cifar100_reg0.5
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 1.0 --output_dir ./runs/cifar100_reg1.0
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 2.0 --output_dir ./runs/cifar100_reg2.0

# Baseline
python train.py --dataset cifar100 --n_tasks 10 --epochs 10 --lr 0.003 --reg 0.0 --output_dir ./runs/cifar100_baseline
```

### Day 4: Other Datasets
```bash
# Download datasets
python download_datasets.py

# ImageNet-R
python train.py --dataset imagenet_r --n_tasks 20 --epochs 8 --batch_size 32 --lr 0.002 --reg 2.0 --output_dir ./runs/imagenet_r_paper_reproduction

# CUB-200
python train.py --dataset cub200 --n_tasks 10 --epochs 10 --batch_size 32 --lr 0.003 --reg 1.5 --output_dir ./runs/cub200_paper_reproduction
```

---

## 📚 Documentation

- **`IMPLEMENTATION_SUMMARY.md`** - Overview of changes
- **`PAPER_REPRODUCTION.md`** - Detailed reproduction guide
- **`CHANGES.md`** - What changed and why
- **`README.md`** - Original project README

---

## 🎯 Success Criteria

Your reproduction is successful if:

✅ **CIFAR-100:**
- Average Accuracy: 65-70%
- Backward Transfer: -8% to -12%
- Training time: ~5-10 minutes

✅ **ImageNet-R:**
- Average Accuracy: 55-60%
- Backward Transfer: -12% to -18%
- Training time: ~50-60 minutes

✅ **CUB-200:**
- Average Accuracy: 60-65%
- Backward Transfer: -10% to -15%
- Training time: ~15-20 minutes

---

**Ready to start!** 🚀

Begin with: `python verify_paper_implementation.py`
