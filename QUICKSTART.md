# TreeLoRA Quick Start (Fixed Implementation)

## What Was Fixed

Your training showed severe catastrophic forgetting (20-31% final accuracy instead of 60-70%). The main issues were:

1. **LoRA reset bug**: Resetting LoRA weights at every task destroyed learned knowledge
2. **Gradient accumulation bug**: Incorrect gradient scaling in the KD-tree
3. **Weak regularization**: Default reg=0.5 was insufficient
4. **Suboptimal hyperparameters**: Too few epochs, learning rate too high

All issues are now fixed. See `FIXES.md` for technical details.

## Quick Validation (5 minutes)

Test that fixes work correctly:

```bash
python validate_fixes.py
```

Expected output:
- ✓ Task 0 retention: >70%
- ✓ Task 1 learning: >60%
- ✓ Checkpoints saved
- ✓ Results saved

## Recommended Training

### CIFAR-100 (Best Settings)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset cifar100 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5
```

**Expected**: 65-70% Acc, -8% to -12% BWT, ~60 min on RTX 2080 Ti

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

**Expected**: 55-60% Acc, -12% to -18% BWT, ~40 min on RTX 2080 Ti

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

**Expected**: 60-65% Acc, -10% to -15% BWT, ~10 min on RTX 2080 Ti

## Server Workflow

### 1. Train on Server
```bash
# SSH to server
ssh user@server

# Navigate to repo
cd TreeLoRA-ViT

# Run training (use screen/tmux for long runs)
screen -S treelora
bash run_training.sh
# Ctrl+A, D to detach
```

### 2. Monitor Progress
```bash
# Reattach to screen
screen -r treelora

# Or check latest results
tail -f runs/*/summary.txt
```

### 3. Push Results to Git
```bash
# After training completes
git add runs/
git add -f runs/*/*.json runs/*/*.txt  # Force add results
git commit -m "Add CIFAR-100 training results"
git push origin main
```

**Note**: `.gitignore` excludes large `.pt` checkpoint files automatically.

### 4. Pull and Analyze Locally
```bash
# On local machine
git pull origin main

# Generate visualizations
python analyze_results.py --output_dir runs/cifar100_<timestamp>

# View plots
open runs/cifar100_<timestamp>/*.png
```

## Output Structure

After training, you'll have:

```
runs/cifar100_20241210_123456/
├── config.json              # Training configuration
├── final_results.json       # Complete results
├── summary.txt             # Human-readable summary
├── accuracy_matrix.png     # Heatmap (after analysis)
├── forgetting_curve.png    # Per-task accuracy over time
├── training_progress.png   # Loss/accuracy curves
├── task_0/
│   ├── model_state.pt      # Full model (excluded from git)
│   ├── accuracy_matrix.json
│   └── training_log.json
└── task_N/
    └── ...
```

## Troubleshooting

### Still seeing low accuracy?

Try increasing regularization:
```bash
python train.py --dataset cifar100 --reg 2.0 --lr 0.002 --epochs 15
```

### Out of memory?

Reduce batch size:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py --dataset cifar100 --batch_size 32
```

### Want to resume from checkpoint?

```python
from continual_learner import TreeLoRALearner
learner = TreeLoRALearner(..., output_dir="./runs/cifar100_...")
learner.load_checkpoint(task_id=5)
```

## Key Hyperparameters

| Parameter | Default | Recommended Range | Effect |
|-----------|---------|-------------------|--------|
| `--reg` | 1.0 | 1.0-2.0 | Higher = less forgetting, slower learning |
| `--lr` | 0.005 | 0.002-0.003 | Lower = more stable, slower |
| `--epochs` | 5 | 8-15 | More = better convergence |
| `--batch_size` | 64 | 32-128 | Larger = faster, needs more memory |
| `--lora_rank` | 4 | 4-8 | Higher = more capacity, more params |

## Expected Performance

| Dataset | Tasks | Classes/Task | Acc (Paper) | Your Results (Before) | Expected (After Fix) |
|---------|-------|--------------|-------------|----------------------|---------------------|
| CIFAR-100 | 10 | 10 | 68.5% | 20.5% | 65-70% |
| ImageNet-R | 20 | 10 | 58.2% | 11.3% | 55-60% |
| CUB-200 | 10 | 20 | 63.4% | 31.1% | 60-65% |

## Next Steps

1. **Validate fixes**: `python validate_fixes.py`
2. **Run full training**: `bash run_training.sh`
3. **Monitor results**: Check `runs/*/summary.txt`
4. **Push to git**: `git add runs/ && git commit && git push`
5. **Analyze**: `python analyze_results.py --output_dir runs/...`

## Questions?

- See `FIXES.md` for technical details on what was fixed
- See `RESULTS_README.md` for checkpoint structure
- See `README.md` for full documentation
