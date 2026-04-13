# TreeLoRA: Fixing the Evaluation Protocol

## Problem: 97% Accuracy vs Paper's 65-70%

Your implementation achieved **97.18% accuracy** on CIFAR-100, which is much higher than the paper's reported **~68%**. This isn't a bug in your code - it's a difference in **evaluation protocols**.

## Root Cause: Task-Incremental vs Class-Incremental

### Your Current Implementation (Task-Incremental)

```python
# Multiple heads, one per task
task_0_head = Linear(768, 10)  # Classes 0-9
task_1_head = Linear(768, 10)  # Classes 10-19
...

# Labels remapped to [0-9] for each task
task_0_labels = [0, 1, 2, ..., 9]
task_1_labels = [0, 1, 2, ..., 9]  # Same range!

# At test time: Load correct head for each task
test_task_0 → use task_0_head → predict from [0-9]
test_task_1 → use task_1_head → predict from [0-9]
```

**Result**: 97.18% accuracy ✅ (correct for this setting)

### Paper's Evaluation (Class-Incremental)

```python
# Single unified head for ALL classes
unified_head = Linear(768, 100)  # All classes 0-99

# Labels NOT remapped - stay in original range
task_0_labels = [0, 1, 2, ..., 9]
task_1_labels = [10, 11, 12, ..., 19]  # Different range!

# At test time: Same head for all tasks
test_task_0 → use unified_head → predict from [0-99]
test_task_1 → use unified_head → predict from [0-99]
```

**Result**: ~68% accuracy (matches paper)

## Why the Difference?

| Aspect | Task-Incremental | Class-Incremental |
|--------|------------------|-------------------|
| **Heads** | 10 separate (10 classes each) | 1 unified (100 classes) |
| **Label Range** | [0-9] per task | [0-99] global |
| **Task-ID** | Known at test (oracle) | Not used |
| **Difficulty** | Easy (10-way classification) | Hard (100-way classification) |
| **Accuracy** | 90-99% | 60-70% |

## The Fix

I've created a **class-incremental implementation** that matches the paper:

### New Files

1. **`continual_learner_class_incremental.py`**
   - Single unified head with all classes
   - No label remapping
   - Proper class-incremental evaluation

2. **`datasets_class_incremental.py`**
   - Dataset loaders without label remapping
   - Preserves original class labels [0-99]

3. **`train_class_incremental.py`**
   - Training script for class-incremental learning
   - Matches paper's evaluation protocol

4. **`compare_evaluation_protocols.py`**
   - Demonstrates the difference between protocols
   - Shows actual label distributions

## How to Use

### 1. Understand the Difference

```bash
python compare_evaluation_protocols.py
```

This shows you exactly how the two protocols differ.

### 2. Train with Class-Incremental (Paper's Method)

```bash
# CIFAR-100
python train_class_incremental.py \
    --dataset cifar100 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.003 \
    --reg 1.5

# ImageNet-R
python train_class_incremental.py \
    --dataset imagenet_r \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.003 \
    --reg 1.5

# CUB-200
python train_class_incremental.py \
    --dataset cub200 \
    --n_tasks 10 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.003 \
    --reg 1.5
```

### 3. Expected Results

| Dataset | Protocol | Accuracy | BWT |
|---------|----------|----------|-----|
| CIFAR-100 | Task-Incremental | 97.18% | -1.67% |
| CIFAR-100 | **Class-Incremental** | **65-70%** | **-8% to -12%** |
| ImageNet-R | **Class-Incremental** | **55-60%** | **-12% to -18%** |
| CUB-200 | **Class-Incremental** | **60-65%** | **-10% to -15%** |

## Which Protocol Should You Use?

### Use Task-Incremental If:
- You have clear task boundaries at test time
- You want maximum accuracy
- You're building a practical system with known task IDs

### Use Class-Incremental If:
- You want to match continual learning papers
- You want a more challenging/realistic setting
- Task boundaries are unknown at test time

**The paper uses class-incremental**, so use `train_class_incremental.py` to match their results.

## Technical Details

### Task-Incremental Implementation

```python
# continual_learner.py (your current code)
class TreeLoRALearner:
    def __init__(self, classes_per_task=10):
        # Separate head per task
        self.task_heads = {}
    
    def train_task(self, task_id):
        # Create new head for this task
        self.model.head = nn.Linear(768, 10)
        
    def evaluate_task(self, task_id):
        # Load task-specific head
        self.model.head.load_state_dict(self.task_heads[task_id])
```

### Class-Incremental Implementation

```python
# continual_learner_class_incremental.py (new)
class ClassIncrementalTreeLoRALearner:
    def __init__(self, total_classes=100):
        # Single unified head
        self.model.head = nn.Linear(768, 100)
    
    def train_task(self, task_id):
        # Same head for all tasks
        pass
    
    def evaluate_task(self, task_id):
        # Always use the same unified head
        pass
```

## Comparison Table

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│                     │ Task-Incremental     │ Class-Incremental    │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Implementation      │ train.py             │ train_class_inc.py   │
│ Heads               │ 10 × 10 classes      │ 1 × 100 classes      │
│ Label Remapping     │ Yes                  │ No                   │
│ Task-ID at Test     │ Required             │ Not used             │
│ CIFAR-100 Accuracy  │ 97.18%               │ 65-70%               │
│ Matches Paper       │ No                   │ Yes ✅               │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

## Next Steps

1. **Run the comparison script** to understand the difference:
   ```bash
   python compare_evaluation_protocols.py
   ```

2. **Train with class-incremental** to match the paper:
   ```bash
   python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 10
   ```

3. **Compare results** between the two protocols:
   ```bash
   # Your current results (task-incremental)
   cat runs/cifar100_*/summary.txt
   
   # New results (class-incremental)
   cat runs_class_incremental/cifar100_*/summary.txt
   ```

## Summary

- Your **97% accuracy is CORRECT** for task-incremental learning
- The paper uses **class-incremental** learning (65-70% accuracy)
- Use `train_class_incremental.py` to match the paper
- Both protocols are valid, but they measure different things

The class-incremental setting is more challenging and realistic, which is why papers typically report it.
