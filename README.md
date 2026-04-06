# TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs

This repository contains an implementation of **TreeLoRA**, a continual learning method designed for Vision Transformers (ViT) guided by a Hierarchical Gradient-Similarity Tree.

TreeLoRA injects separate Low-Rank Adaptation (LoRA) modules into the Query and Value projections of a pre-trained frozen ViT backbone to accommodate new learning tasks, while actively organizing them in a KD-tree structure to prevent catastrophic forgetting and facilitate forward transfer.

## Project Structure

- `train.py`: Top-level training script and entry point for continual learning experiments.
- `continual_learner.py`: Core logic for managing the continual learning process and task transitions.
- `datasets.py`: Data loaders and split configurations for benchmarks like Split CIFAR-100, ImageNet-R, and CUB-200.
- `kd_lora_tree.py`: The hierarchical KD-tree implementation managing LoRA adapters and routing.
- `lora.py`: Dynamic LoRA adapter modules injected into the attention layers.
- `vit_backbone.py`: The frozen ViT-B/16 architecture from the `timm` library.
- `test_quick.py` & `test_train.py`: Quick verification scripts to ensure the pipeline runs correctly without errors (smoke tests).

## Requirements

Ensure you have Python 3.8+ installed. Install the necessary dependencies via:

```bash
pip install -r requirements.txt
```

The core dependencies are:
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `timm >= 0.9.0`
- `numpy >= 1.24.0`
- `tqdm >= 4.65.0`
- `Pillow >= 9.5.0`

## Dataset Preparation

This implementation supports **Split CIFAR-100**, **Split ImageNet-R**, and **Split CUB-200**.
CIFAR-100 will be downloaded automatically when running for the first time.
ImageNet-R and CUB-200 require manual downloading and placement into the `./data` directory.

## Training

Use `train.py` to initiate the continual learning process. The script will automatically train task by task and evaluate the performance sequentially.

### General Usage
```bash
python train.py --dataset [dataset_name] --n_tasks [num_tasks] --epochs [num_epochs]
```

### Full Split CIFAR-100 Run
Trains on CIFAR-100 across 10 continual learning tasks, 5 epochs per task:
```bash
python train.py --dataset cifar100 --n_tasks 10 --epochs 5 --lora_rank 4 --lora_alpha 8.0
```

### Split ImageNet-R
(*Note: Ensure ImageNet-R dataset is available in `./data`*)
```bash
python train.py --dataset imagenet_r --n_tasks 20 --epochs 5
```

### Split CUB-200
(*Note: Ensure CUB-200 dataset is available in `./data`*)
```bash
python train.py --dataset cub200 --n_tasks 10 --epochs 5
```

### Advanced Training Options
You can configure a variety of hyperparameters:
- `--lora_rank`: Defines the rank of the LoRA matrices (default: 4).
- `--lora_alpha`: Scaling parameter for LoRA (default: 8.0).
- `--lora_depth`: The maximum allowed depth of the KD-tree (default: 5).
- `--reg`: TreeLoRA regularization strength (default: 0.5).
- `--lr`: Learning rate (default: 5e-3).
- `--batch_size`: Batch size per task (default: 64).

## Testing & Smoke Tests

In the continual learning framework, evaluation (testing) happens dynamically at the end of every task using the validation splits of observed tasks. Upon finishing all tasks, an "Accuracy Matrix" is generated showing the final average accuracy and **Backward Transfer (BWT)** metric.

To quickly test the environment and ensure the code structure is fully working, you can use the built-in smoke tests:

### Quick Smoke Test (test_quick.py)
This runs a rapid, shortened training sequence involving only 2 tasks for 1 epoch on CIFAR-100 using a CPU-friendly non-pretrained configuration.

```bash
python test_quick.py
```

### Minimal Training Test (train.py)
Alternatively, you can test directly through `train.py` using small parameters and bypassing the heavy pretrained ViT weights for rapid debugging:

```bash
python train.py --dataset cifar100 --n_tasks 2 --epochs 1 --batch_size 16 --no_pretrained
```
