# Step-by-Step Guide: Fixing TreeLoRA to Match the Paper

## Current Situation

✅ **Your implementation works correctly** - you got 97.18% accuracy  
❌ **But it uses a different evaluation protocol than the paper**

## The Issue

Your code uses **task-incremental** evaluation (97% accuracy), but the paper uses **class-incremental** evaluation (65-70% accuracy).

## Solution Overview

I've created a complete class-incremental implementation that matches the paper. Here's what to do:

---

## Step 1: Understand the Difference

Run this to see exactly what's different:

```bash
python compare_evaluation_protocols.py
```

**Key takeaway**: 
- Task-incremental = 10 separate heads, labels remapped → 97% accuracy
- Class-incremental = 1 unified head, no remapping → 65-70% accuracy

---

## Step 2: Test the New Implementation

Quick 5-minute test to verify everything works:

```bash
python test_class_incremental.py
```

**Expected output**:
```
✓ Labels are NOT remapped (correct!)
✓ Single unified head with 100 classes (correct!)
Task 0 accuracy: 40-60%
Task 1 accuracy: 40-60%
Final Acc: 40-50%
✓ All tests passed!
```

**Note**: Lower accuracy is EXPECTED and CORRECT for class-incremental!

---

## Step 3: Train CIFAR-100 (Class-Incremental)

Now run the full training with the paper's evaluation protocol:

```bash
python train_class_incremental.py \
    --dataset cifar100 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5
```

**Expected results**:
- Final Accuracy: 65-70%
- BWT: -8% to -12%
- Training time: ~60 minutes

---

## Step 4: Fix ImageNet-R OOM Issue

Before training ImageNet-R, clear GPU memory:

```bash
# Kill any lingering processes
pkill -9 python

# Verify GPU is clear
nvidia-smi

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

**Expected results**:
- Final Accuracy: 55-60%
- BWT: -12% to -18%

---

## Step 5: Train CUB-200

```bash
python train_class_incremental.py \
    --dataset cub200 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5
```

**Expected results**:
- Final Accuracy: 60-65%
- BWT: -10% to -15%

---

## Step 6: Compare Results

After training, compare the two protocols:

```bash
# Your original results (task-incremental)
cat runs/cifar100_*/summary.txt

# New results (class-incremental)
cat runs_class_incremental/cifar100_*/summary.txt
```

---

## What Changed?

### New Files Created

1. **`continual_learner_class_incremental.py`**
   - Single unified head (100 classes for CIFAR-100)
   - No label remapping
   - Same head used for all tasks

2. **`datasets_class_incremental.py`**
   - Loads data without remapping labels
   - Labels stay in original [0-99] range

3. **`train_class_incremental.py`**
   - Training script for class-incremental learning
   - Matches paper's evaluation protocol

4. **`compare_evaluation_protocols.py`**
   - Shows the difference between protocols
   - Demonstrates label distributions

5. **`test_class_incremental.py`**
   - Quick test to verify implementation
   - Runs 2 tasks in ~5 minutes

6. **`EVALUATION_PROTOCOL_FIX.md`**
   - Detailed explanation of the fix
   - Technical comparison

---

## Key Differences

| Aspect | Task-Incremental (Old) | Class-Incremental (New) |
|--------|------------------------|-------------------------|
| **Script** | `train.py` | `train_class_incremental.py` |
| **Heads** | 10 separate (10 classes each) | 1 unified (100 classes) |
| **Labels** | Remapped to [0-9] per task | Original [0-99] |
| **Evaluation** | Load task-specific head | Same head for all |
| **CIFAR-100 Acc** | 97.18% | 65-70% |
| **Matches Paper** | ❌ No | ✅ Yes |

---

## Expected Performance (Class-Incremental)

| Dataset | Tasks | Classes | Acc (Paper) | Acc (Expected) | BWT |
|---------|-------|---------|-------------|----------------|-----|
| CIFAR-100 | 10 | 100 | ~68% | 65-70% | -8% to -12% |
| ImageNet-R | 10 | 200 | ~58% | 55-60% | -12% to -18% |
| CUB-200 | 10 | 200 | ~63% | 60-65% | -10% to -15% |

---

## Troubleshooting

### Q: Why is accuracy lower now?

**A**: This is CORRECT! Class-incremental is much harder:
- Model must distinguish ALL 100 classes simultaneously
- No oracle task boundaries
- Inter-task confusion is possible

### Q: Should I use task-incremental or class-incremental?

**A**: Use class-incremental to match the paper. Task-incremental is easier but less realistic.

### Q: Can I still use the old implementation?

**A**: Yes! Both are valid:
- `train.py` = task-incremental (97% accuracy)
- `train_class_incremental.py` = class-incremental (65-70% accuracy)

### Q: ImageNet-R still OOM?

**A**: Try:
```bash
# Reduce batch size further
--batch_size 8

# Or use gradient accumulation (modify code)
# Or use CPU (very slow)
--device cpu
```

---

## Summary

✅ **Your original implementation was correct** for task-incremental learning  
✅ **New implementation matches the paper** (class-incremental)  
✅ **Lower accuracy (65-70%) is expected and correct**  
✅ **All three datasets should now match paper's results**

---

## Quick Commands

```bash
# 1. Understand the difference
python compare_evaluation_protocols.py

# 2. Quick test (5 min)
python test_class_incremental.py

# 3. Train CIFAR-100 (60 min)
python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 10

# 4. Train ImageNet-R (with OOM fix)
pkill -9 python
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_class_incremental.py --dataset imagenet_r --n_tasks 10 --epochs 10 --batch_size 16

# 5. Train CUB-200
python train_class_incremental.py --dataset cub200 --n_tasks 10 --epochs 10 --batch_size 32
```

---

## Next Steps

1. ✅ Run `compare_evaluation_protocols.py` to understand
2. ✅ Run `test_class_incremental.py` to verify (5 min)
3. ✅ Train CIFAR-100 with class-incremental (60 min)
4. ✅ Train ImageNet-R (fix OOM first)
5. ✅ Train CUB-200
6. ✅ Compare results with paper

Good luck! Your implementation is solid - this is just switching to the paper's evaluation protocol.
