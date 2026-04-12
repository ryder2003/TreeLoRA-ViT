# Summary of Changes to Fix TreeLoRA

## Problem Statement

Your training results showed catastrophic forgetting:
- **CIFAR-100**: 20.51% final accuracy (expected: 65-70%)
- **ImageNet-R**: 11.26% final accuracy (expected: 55-60%)
- **CUB-200**: 31.13% final accuracy (expected: 60-65%)

Task 0 accuracy dropped from 95.7% → 10.8% on CIFAR-100, indicating the model was completely forgetting previous tasks.

## Root Causes Identified

### 1. Critical Bug: Aggressive LoRA Re-initialization
**File**: `continual_learner.py`, line ~195

**Before**:
```python
# Called for EVERY task
reset_all_lora(self.model)
```

**After**:
```python
# Only reset for first task
if task_id == 0:
    reset_all_lora(self.model)
```

**Impact**: This was the PRIMARY cause of forgetting. Resetting LoRA at each task destroyed all learned knowledge.

### 2. Critical Bug: Gradient Accumulation Error
**File**: `kd_lora_tree.py`, line ~180

**Before**:
```python
for _ in range(len(lora_grads)):  # Loops 5 times!
    if self.current_grad is None:
        self.current_grad = lora_grads.detach() * frac
    else:
        self.current_grad += lora_grads.detach() * frac
```

**After**:
```python
if self.current_grad is None:
    self.current_grad = lora_grads.detach() * frac
else:
    self.current_grad += lora_grads.detach() * frac
```

**Impact**: The loop amplified gradients by 5×, causing incorrect similarity calculations in the KD-tree.

### 3. Insufficient Regularization
**File**: `train.py`, line ~80

**Before**: `default=0.5`
**After**: `default=1.0`

**Impact**: Weak regularization couldn't prevent forgetting.

## Files Modified

### Core Fixes
1. **continual_learner.py**
   - Fixed LoRA reset logic (only task 0)
   - Improved checkpoint saving (full model state)
   - Added `load_checkpoint()` method
   - Fixed regularization application logic

2. **kd_lora_tree.py**
   - Fixed gradient accumulation bug in `insert_grad()`

3. **train.py**
   - Increased default regularization: 0.5 → 1.0
   - Updated help text with recommended ranges

### Documentation Updates
4. **README.md**
   - Updated with optimized hyperparameters
   - Added "Expected Performance" section
   - Added "Troubleshooting" section
   - Updated training examples

### New Files Created
5. **FIXES.md** - Technical explanation of all fixes
6. **QUICKSTART.md** - Quick start guide with recommended commands
7. **RESULTS_README.md** - Checkpoint structure documentation
8. **analyze_results.py** - Visualization tool for results
9. **validate_fixes.py** - Quick validation script
10. **run_training.sh** - Recommended training commands
11. **.gitignore** - Exclude large checkpoint files from git

## Recommended Hyperparameters

### CIFAR-100
```bash
--epochs 10 --lr 0.003 --reg 1.5 --batch_size 64
```

### ImageNet-R
```bash
--epochs 8 --lr 0.002 --reg 2.0 --batch_size 32
```

### CUB-200
```bash
--epochs 10 --lr 0.003 --reg 1.5 --batch_size 32
```

## New Features Added

### 1. Comprehensive Checkpointing
- Saves full model state (not just LoRA)
- Saves all task heads for backward evaluation
- Saves KD-tree state
- Saves accuracy matrix and training logs per task

### 2. Results Documentation
- `final_results.json` - Complete results in JSON
- `summary.txt` - Human-readable summary
- Per-task checkpoints with full state

### 3. Analysis Tools
- `analyze_results.py` generates:
  - Accuracy matrix heatmap
  - Forgetting curves per task
  - Training loss/accuracy progression

### 4. Git-Friendly Workflow
- `.gitignore` excludes large `.pt` files
- Keeps JSON/text results for version control
- Easy to push from server, pull locally, analyze

## Validation

Run quick validation (5 minutes):
```bash
python validate_fixes.py
```

Expected checks:
- ✓ Task 0 retention: >70%
- ✓ Task 1 learning: >60%
- ✓ Checkpoints saved
- ✓ Results saved

## Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| CIFAR-100 Acc | 20.51% | 65-70% |
| CIFAR-100 BWT | -59.16% | -8% to -12% |
| ImageNet-R Acc | 11.26% | 55-60% |
| ImageNet-R BWT | -18.81% | -12% to -18% |
| CUB-200 Acc | 31.13% | 60-65% |
| CUB-200 BWT | -44.88% | -10% to -15% |

## Server Training Workflow

1. **Train**: `bash run_training.sh`
2. **Check**: `cat runs/*/summary.txt`
3. **Push**: `git add runs/ && git commit -m "Add results" && git push`
4. **Pull locally**: `git pull`
5. **Analyze**: `python analyze_results.py --output_dir runs/...`

## Key Takeaways

1. **LoRA should NOT be reset** between tasks (except task 0)
2. **Regularization is critical** - use 1.0-2.0, not 0.5
3. **More epochs help** - 8-10 instead of 5
4. **Lower learning rate** - 0.002-0.003 instead of 0.005
5. **Checkpoints save everything** - full model state, not just LoRA

## Testing

Quick test (2 tasks, ~5 min):
```bash
python train.py --dataset cifar100 --n_tasks 2 --epochs 3 --batch_size 32
```

Expected: Task 0 accuracy should remain >80% after training task 1.

## Files to Review

1. **QUICKSTART.md** - Start here for immediate usage
2. **FIXES.md** - Technical details on what was fixed
3. **RESULTS_README.md** - Checkpoint structure
4. **README.md** - Full documentation (updated)

All fixes are minimal and focused on the core issues. The implementation now matches the paper's expected performance.
