"""
datasets.py
-----------
Dataset utilities for the continual-learning benchmarks used in the paper:
  - Split CIFAR-100  (10 tasks × 10 classes each)    → auto-download
  - Split ImageNet-R (20 tasks × 10 classes each)    → run download_datasets.py first
  - Split CUB-200    (10 tasks × 20 classes each)    → run download_datasets.py first

All functions accept a single `data_root` (e.g. "./data") and construct
the correct sub-paths internally.

Path conventions (after running download_datasets.py):
  data_root/
      cifar-100-python/          ← torchvision auto-manages
      imagenet-r/
          train/<200 class dirs>/
          val/<200 class dirs>/
      CUB_200_2011/
          train/<200 class dirs>/
          test/<200 class dirs>/

All datasets use 224 × 224 resized images with ImageNet normalisation
to match the ViT-B/16 pretrained input distribution.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ──────────────────────────────────────────────────────────────────────────────
# Standard transforms (same as used in the paper for ViT-B/16)
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Task-aware dataset wrapper: slice by index + remap labels without mutation
# ──────────────────────────────────────────────────────────────────────────────

class _TaskSubset(torch.utils.data.Dataset):
    """
    Slices a base dataset to a subset of indices and remaps class labels
    so that they are 0-indexed within the task (label - label_offset).
    """

    def __init__(self, base_dataset, indices, label_offset: int):
        self.base = base_dataset
        self.indices = indices
        self.label_offset = label_offset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        return img, label - self.label_offset


def _make_loaders(
    train_subset: _TaskSubset,
    eval_subset: _TaskSubset,
    batch_size: int,
    num_workers: int,
):
    tr_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    ev_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return tr_loader, ev_loader


# ──────────────────────────────────────────────────────────────────────────────
# Split CIFAR-100
# ──────────────────────────────────────────────────────────────────────────────

def get_split_cifar100(
    data_root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split CIFAR-100 into `n_tasks` tasks.
    Each task gets 100 // n_tasks consecutive class indices.
    Train split = CIFAR-100 train set; eval split = CIFAR-100 test set.

    Returns
    -------
    task_dataloaders : list[(train_loader, test_loader)]  — one per task
    class_splits     : list[list[int]]                    — class indices per task
    """
    classes_per_task = 100 // n_tasks
    class_splits = [
        list(range(t * classes_per_task, (t + 1) * classes_per_task))
        for t in range(n_tasks)
    ]

    full_train = datasets.CIFAR100(
        data_root, train=True,  download=True, transform=get_train_transform()
    )
    full_test = datasets.CIFAR100(
        data_root, train=False, download=True, transform=get_val_transform()
    )

    train_targets = list(full_train.targets)
    test_targets  = list(full_test.targets)

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)
        offset = t * classes_per_task

        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets)  if c in class_set]

        tr_loader, te_loader = _make_loaders(
            _TaskSubset(full_train, tr_indices, offset),
            _TaskSubset(full_test,  te_indices, offset),
            batch_size, num_workers,
        )
        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ──────────────────────────────────────────────────────────────────────────────
# Split ImageNet-R  (requires download_datasets.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_split_imagenet_r(
    data_root: str = "./data",
    n_tasks: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split ImageNet-R (200 classes) into `n_tasks` tasks.

    Expects the following structure under data_root (created by download_datasets.py):
        data_root/imagenet-r/train/<200 class dirs>/
        data_root/imagenet-r/val/<200 class dirs>/

    Returns
    -------
    task_dataloaders : list[(train_loader, val_loader)]
    class_splits     : list[list[int]]
    """
    ir_root   = os.path.join(data_root, "imagenet-r")
    train_dir = os.path.join(ir_root, "train")
    val_dir   = os.path.join(ir_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet-R train split not found at:\n  {train_dir}\n\n"
            "Please run:  python download_datasets.py\n"
            "(or:  python download_datasets.py --skip_cifar100 --skip_cub200)"
        )
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet-R val split not found at:\n  {val_dir}\n\n"
            "Please run:  python download_datasets.py"
        )

    classes_per_task = 200 // n_tasks

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transform())
    full_val   = datasets.ImageFolder(val_dir,   transform=get_val_transform())

    # ImageFolder sorts class names alphabetically → deterministic class ids
    sorted_class_ids = sorted(full_train.class_to_idx.values())
    class_splits = [
        sorted_class_ids[t * classes_per_task: (t + 1) * classes_per_task]
        for t in range(n_tasks)
    ]

    train_targets = [s[1] for s in full_train.samples]
    val_targets   = [s[1] for s in full_val.samples]

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)
        offset    = t * classes_per_task

        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        va_indices = [i for i, c in enumerate(val_targets)   if c in class_set]

        tr_loader, va_loader = _make_loaders(
            _TaskSubset(full_train, tr_indices, offset),
            _TaskSubset(full_val,   va_indices, offset),
            batch_size, num_workers,
        )
        task_dataloaders.append((tr_loader, va_loader))

    return task_dataloaders, class_splits


# ──────────────────────────────────────────────────────────────────────────────
# Split CUB-200-2011  (requires download_datasets.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_split_cub200(
    data_root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split CUB-200-2011 (200 classes) into `n_tasks` tasks.

    Expects the following structure under data_root (created by download_datasets.py):
        data_root/CUB_200_2011/train/<200 class dirs>/
        data_root/CUB_200_2011/test/<200 class dirs>/

    Returns
    -------
    task_dataloaders : list[(train_loader, test_loader)]
    class_splits     : list[list[int]]
    """
    cub_root  = os.path.join(data_root, "CUB_200_2011")
    train_dir = os.path.join(cub_root, "train")
    test_dir  = os.path.join(cub_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"CUB-200 train split not found at:\n  {train_dir}\n\n"
            "Please run:  python download_datasets.py\n"
            "(or:  python download_datasets.py --skip_cifar100 --skip_imagenet_r)"
        )
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"CUB-200 test split not found at:\n  {test_dir}\n\n"
            "Please run:  python download_datasets.py"
        )

    classes_per_task = 200 // n_tasks

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transform())
    full_test  = datasets.ImageFolder(test_dir,  transform=get_val_transform())

    sorted_class_ids = sorted(full_train.class_to_idx.values())
    class_splits = [
        sorted_class_ids[t * classes_per_task: (t + 1) * classes_per_task]
        for t in range(n_tasks)
    ]

    train_targets = [s[1] for s in full_train.samples]
    test_targets  = [s[1] for s in full_test.samples]

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)
        offset    = t * classes_per_task

        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets)  if c in class_set]

        tr_loader, te_loader = _make_loaders(
            _TaskSubset(full_train, tr_indices, offset),
            _TaskSubset(full_test,  te_indices, offset),
            batch_size, num_workers,
        )
        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing CIFAR-100 split ...")
    loaders, splits = get_split_cifar100(data_root="./data", n_tasks=10)
    print(f"  Tasks: {len(loaders)}")
    for t, (tr, te) in enumerate(loaders):
        print(f"    Task {t}: classes {splits[t]} | "
              f"train={len(tr.dataset)}  test={len(te.dataset)}")
    x, y = next(iter(loaders[0][0]))
    print(f"  Batch: {x.shape}, labels sample: {y[:5].tolist()}")
    print("CIFAR-100 OK ✔")
