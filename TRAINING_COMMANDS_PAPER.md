# TreeLoRA Paper Reproduction Commands (Strict Class-Incremental)

Use this flow to reproduce the paper-target CIFAR-100 result under strict class-incremental protocol.

## 1) Single Run (one seed)

```bash
python train_class_incremental.py \
  --dataset cifar100 \
  --n_tasks 10 \
  --epochs 20 \
  --batch_size 192 \
  --lr 0.005 \
  --reg 0.1 \
  --lora_rank 4 \
  --lora_alpha 8.0 \
  --lora_depth 5 \
  --seed 42 \
  --deterministic
```

## 2) Paper-Style 3 Seeds

Run three seeds and compare the average against the paper target.

```bash
python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 20 --batch_size 192 --lr 0.005 --reg 0.1 --lora_rank 4 --lora_alpha 8.0 --lora_depth 5 --seed 42 --deterministic
python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 20 --batch_size 192 --lr 0.005 --reg 0.1 --lora_rank 4 --lora_alpha 8.0 --lora_depth 5 --seed 43 --deterministic
python train_class_incremental.py --dataset cifar100 --n_tasks 10 --epochs 20 --batch_size 192 --lr 0.005 --reg 0.1 --lora_rank 4 --lora_alpha 8.0 --lora_depth 5 --seed 44 --deterministic
```

## 3) Protocol Notes

- Entrypoint must be `train_class_incremental.py` for strict paper protocol.
- Unified global head is used (100 classes for CIFAR-100).
- No task-ID is used at test time.
- No label remapping is used in class-incremental datasets.

## 4) Expected Target

- Paper reference target (CIFAR-100): around 88.54% average accuracy.
- Acceptance target for this repository: within ±0.5 points after 3-seed mean.
