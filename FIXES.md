# TreeLoRA Fixes and Improvements

## Issues Identified from Training Results

Your training results showed severe catastrophic forgetting:
- CIFAR-100: Task 0 accuracy dropped from 95.7% → 10.8% (final Acc: 20.51%)
- ImageNet-R: Task 0 accuracy dropped from 73.1% → 3.3% (final Acc: 11.26%)
- CUB-200: Task 0 accuracy dropped from 89.5% → 30.3% (final Acc: 31.13%)

Expected performance should be 60-70% final accuracy with moderate forgetting.

## Root Causes

### 1. **Aggressive LoRA Re-initialization** (CRITICAL)
**Problem**: The original code reset ALL LoRA parameters at the start of each task:
```python
reset_all_lora(self.model)  # Called for EVERY task
```

This destroyed all previously learned knowledge, making continual learning impossible.

**Fix**: Only reset LoRA for the first task, then fine-tune existing weights:
```python
if task_id == 0:
    reset_all_lora(self.model)
```

### 2. **Gradient Accumulation Bug**
**Problem**: The `insert_grad` function had a loop that amplified gradients by `lora_depth` times:
```python
for _ in range(len(lora_grads)):  # Loops 5 times!
    self.current_grad += lora_grads.detach() * frac
```

This caused incorrect gradient similarity calculations in the KD-tree.

**Fix**: Accumulate once per step:
```python
if self.current_grad is None:
    self.current_grad = lora_grads.detach() * frac
else:
    self.current_grad += lora_grads.detach() * frac
```

### 3. **Insufficient Regularization**
**Problem**: Default `reg=0.5` was too weak to prevent forgetting.

**Fix**: Increased default to `reg=1.0` and recommend `1.5-2.0` for harder datasets.

### 4. **Suboptimal Hyperparameters**
**Problem**: 
- Too few epochs (5) didn't allow proper convergence
- Learning rate too high for continual learning

**Fix**: Recommended settings:
- Epochs: 8-10 per task
- Learning rate: 0.002-0.003 (down from 0.005)
- Regularization: 1.0-2.0 (up from 0.5)

## Improvements Added

### 1. **Comprehensive Checkpoint System**
- Saves full model state (not just LoRA parameters)
- Saves all task heads for backward evaluation
- Saves KD-tree state for reproducibility
- Saves accuracy matrix and training logs

### 2. **Results Documentation**
- Automatic generation of `final_results.json`
- Human-readable `summary.txt`
- Per-task checkpoints with full state

### 3. **Analysis Tools**
- `analyze_results.py`: Generate visualizations
  - Accuracy matrix heatmap
  - Forgetting curves
  - Training progress plots

### 4. **Git-Friendly Output**
- `.gitignore` excludes large `.pt` files
- Keeps JSON/text results for version control
- Easy to push results from server and pull locally

## Recommended Training Commands

### CIFAR-100 (Optimized)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar100 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5
```

### ImageNet-R (Memory-Optimized)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset imagenet_r \
    --n_tasks 20 \
    --epochs 8 \
    --batch_size 32 \
    --lr 0.002 \
    --reg 2.0
```

### CUB-200 (Balanced)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cub200 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5
```

## Expected Improvements

With these fixes, you should see:

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| CIFAR-100 Acc | 20.51% | 65-70% |
| CIFAR-100 BWT | -59.16% | -8% to -12% |
| ImageNet-R Acc | 11.26% | 55-60% |
| ImageNet-R BWT | -18.81% | -12% to -18% |
| CUB-200 Acc | 31.13% | 60-65% |
| CUB-200 BWT | -44.88% | -10% to -15% |

## Workflow for Server Training

1. **Train on server**:
   ```bash
   bash run_training.sh
   ```

2. **Check results**:
   ```bash
   cat runs/*/summary.txt
   ```

3. **Push results to Git** (excludes large .pt files):
   ```bash
   git add runs/
   git commit -m "Add training results"
   git push origin main
   ```

4. **Pull on local machine**:
   ```bash
   git pull origin main
   ```

5. **Analyze locally**:
   ```bash
   python analyze_results.py --output_dir runs/cifar100_<timestamp>
   ```

## Files Modified

1. `continual_learner.py`:
   - Fixed LoRA reset logic (only task 0)
   - Improved checkpoint saving (full model state)
   - Added checkpoint loading method

2. `kd_lora_tree.py`:
   - Fixed gradient accumulation bug

3. `train.py`:
   - Updated default regularization to 1.0
   - Added output_dir parameter

4. `README.md`:
   - Updated with optimized hyperparameters
   - Added troubleshooting section
   - Added expected performance metrics

## New Files Added

1. `analyze_results.py`: Visualization tool
2. `RESULTS_README.md`: Checkpoint documentation
3. `run_training.sh`: Recommended training script
4. `.gitignore`: Exclude large checkpoint files
5. `FIXES.md`: This document

## Testing the Fixes

Quick test (2 tasks, should complete in ~5 minutes):
```bash
python train.py --dataset cifar100 --n_tasks 2 --epochs 3 --batch_size 32
```

Expected: Task 0 accuracy should remain > 80% after training task 1.
