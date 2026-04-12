# TreeLoRA Results and Checkpoints

## Directory Structure

After training, the output directory contains:

```
runs/<dataset>_<timestamp>/
├── config.json              # Training configuration
├── final_results.json       # Complete results with accuracy matrix
├── summary.txt             # Human-readable summary
├── task_0/                 # Checkpoint after task 0
│   ├── model_state.pt      # Full model state (LoRA + head)
│   ├── task_heads.pt       # All task heads
│   ├── tree_state.pt       # KD-tree gradients
│   ├── accuracy_matrix.json
│   └── training_log.json
├── task_1/
│   └── ...
└── task_N/
    └── ...
```

## Loading Checkpoints

To resume or analyze from a checkpoint:

```python
from continual_learner import TreeLoRALearner
import torch

# Initialize learner with same config
learner = TreeLoRALearner(
    num_tasks=10,
    classes_per_task=10,
    lora_rank=4,
    lora_alpha=8.0,
    device=torch.device("cuda"),
    output_dir="./runs/cifar100_20241210_123456"
)

# Load checkpoint from specific task
learner.load_checkpoint(task_id=5)
```

## Analyzing Results

Use the analysis script to generate visualizations:

```bash
python analyze_results.py --output_dir ./runs/cifar100_20241210_123456
```

This generates:
- `accuracy_matrix.png` - Heatmap of task accuracies
- `forgetting_curve.png` - Per-task accuracy over time
- `training_progress.png` - Loss and accuracy curves

## Pushing Results to Git

After training on server:

```bash
# Add only results (not large model files)
git add runs/*/config.json
git add runs/*/final_results.json
git add runs/*/summary.txt
git add runs/*/*.json
git add runs/*/*.png

git commit -m "Add training results for <dataset>"
git push origin main
```

## Expected Performance

Based on the paper, expected final accuracies:

- **CIFAR-100 (10 tasks)**: ~65-70% Acc, ~-10% BWT
- **ImageNet-R (20 tasks)**: ~55-60% Acc, ~-15% BWT  
- **CUB-200 (10 tasks)**: ~60-65% Acc, ~-12% BWT

## Troubleshooting Low Accuracy

If you see severe forgetting (Acc < 30%):

1. **Increase regularization**: `--reg 1.5` or `--reg 2.0`
2. **Lower learning rate**: `--lr 0.003` or `--lr 0.002`
3. **More epochs per task**: `--epochs 10`
4. **Check GPU memory**: Reduce `--batch_size` if OOM errors occur
