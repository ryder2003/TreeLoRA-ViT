"""
datasets.py
-----------
Dataset utilities for the continual-learning benchmarks used in the paper:
  - Split CIFAR-100  (10 tasks × 10 classes each)
  - Split ImageNet-R (20 tasks × 10 classes each)   [optional – large download]
  - Split CUB-200    (10 tasks × 20 classes each)   [optional]

All datasets use 224 × 224 resized images with ImageNet normalisation
to match the ViT-B/16 pretrained input distribution.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Standard transforms (same as used in the paper for ViT-B/16)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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


# ---------------------------------------------------------------------------
# Helper: split a dataset by class labels
# ---------------------------------------------------------------------------

def _split_by_classes(dataset, class_list):
    """Return a Subset containing only samples whose label is in class_list."""
    targets = torch.tensor(dataset.targets)
    indices = [i for i, t in enumerate(targets) if t.item() in class_list]
    return Subset(dataset, indices)


def _remap_targets(subset: Subset, class_list):
    """
    Remap targets in a Subset so that they are 0-indexed within the task.
    E.g. for task with classes [50, 51, ..., 59], class 50 → 0.
    """
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    remapped = []
    for idx in subset.indices:
        orig_target = subset.dataset.targets[idx]
        subset.dataset.targets[idx] = class_to_idx[orig_target]
    return subset


# ---------------------------------------------------------------------------
# Split CIFAR-100
# ---------------------------------------------------------------------------

def get_split_cifar100(
    data_root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split CIFAR-100 into `n_tasks` tasks.
    Each task contains 100 // n_tasks consecutive class indices.

    Returns:
        task_dataloaders : list of (train_loader, test_loader) per task
        class_splits     : list of class-index lists per task
    """
    classes_per_task = 100 // n_tasks
    class_splits = [
        list(range(t * classes_per_task, (t + 1) * classes_per_task))
        for t in range(n_tasks)
    ]

    train_transform = get_train_transform()
    val_transform   = get_val_transform()

    full_train = datasets.CIFAR100(data_root, train=True,  download=True, transform=train_transform)
    full_test  = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)

    # We need to NOT mutate the single dataset targets multiple times,
    # so we clone targets once and use a task-local remap at DataLoader time.
    train_targets = list(full_train.targets)
    test_targets  = list(full_test.targets)

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)
        offset = t * classes_per_task      # local label offset

        # Indices for this task
        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets)  if c in class_set]

        tr_subset = _TaskSubset(full_train, tr_indices, offset)
        te_subset = _TaskSubset(full_test,  te_indices, offset)

        tr_loader = DataLoader(
            tr_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        te_loader = DataLoader(
            te_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ---------------------------------------------------------------------------
# Helper dataset wrapper: remaps labels without mutation
# ---------------------------------------------------------------------------

class _TaskSubset(torch.utils.data.Dataset):
    """Dataset slice for a single CL task with remapped class labels."""

    def __init__(self, base_dataset, indices, label_offset: int):
        self.base = base_dataset
        self.indices = indices
        self.label_offset = label_offset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        return img, label - self.label_offset

    @property
    def classes_per_task(self):
        return None


# ---------------------------------------------------------------------------
# Split ImageNet-R  (downloaded separately)
# ---------------------------------------------------------------------------

def get_split_imagenet_r(
    data_root: str = "./data/imagenet-r",
    n_tasks: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split ImageNet-R (200 classes) into `n_tasks` tasks.
    Expects ImageNet-R to be organised as ImageFolder at `data_root`.

    Download from: https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
    """
    classes_per_task = 200 // n_tasks

    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet-R train split not found at {train_dir}.\n"
            "Download from https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar "
            "and extract to " + data_root
        )

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transform())
    full_test  = datasets.ImageFolder(test_dir,  transform=get_val_transform())

    sorted_classes = sorted(full_train.class_to_idx.values())
    class_splits = [
        sorted_classes[t * classes_per_task: (t + 1) * classes_per_task]
        for t in range(n_tasks)
    ]

    train_targets = [s[1] for s in full_train.samples]
    test_targets  = [s[1] for s in full_test.samples]

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)
        offset = t * classes_per_task
        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets)  if c in class_set]

        tr_loader = DataLoader(
            _TaskSubset(full_train, tr_indices, offset),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        te_loader = DataLoader(
            _TaskSubset(full_test, te_indices, offset),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ---------------------------------------------------------------------------
# Split CUB-200  (10 tasks × 20 classes)
# ---------------------------------------------------------------------------

def get_split_cub200(
    data_root: str = "./data/CUB_200_2011",
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split CUB-200-2011 (200 classes) into `n_tasks` tasks.
    Expects CUB-200 to be organised as ImageFolder at `data_root`.

    Download:
        1. wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
        2. tar -xzf CUB_200_2011.tgz
        3. Organise into train/val splits using the provided train_test_split.txt
           OR use the following structure:
             data/CUB_200_2011/train/<class_folders>/
             data/CUB_200_2011/test/<class_folders>/
    """
    classes_per_task = 200 // n_tasks

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"CUB-200 train split not found at {train_dir}.\n"
            "Download from https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz\n"
            "Extract and organise into train/test ImageFolder format at " + data_root
        )

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transform())
    full_test = datasets.ImageFolder(test_dir, transform=get_val_transform())

    sorted_classes = sorted(full_train.class_to_idx.values())
    class_splits = [
        sorted_classes[t * classes_per_task: (t + 1) * classes_per_task]
        for t in range(n_tasks)
    ]

    train_targets = [s[1] for s in full_train.samples]
    test_targets = [s[1] for s in full_test.samples]

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)
        offset = t * classes_per_task
        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets) if c in class_set]

        tr_loader = DataLoader(
            _TaskSubset(full_train, tr_indices, offset),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        te_loader = DataLoader(
            _TaskSubset(full_test, te_indices, offset),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loaders, splits = get_split_cifar100(data_root="./data", n_tasks=10)
    print(f"Tasks: {len(loaders)}")
    for t, (tr, te) in enumerate(loaders):
        print(f"  Task {t}: classes {splits[t]}  | train={len(tr.dataset)} test={len(te.dataset)}")
    x, y = next(iter(loaders[0][0]))
    print(f"Batch shape: {x.shape}, labels: {y[:5]}")
