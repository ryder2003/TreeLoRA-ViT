"""
datasets_class_incremental.py
------------------------------
Dataset loaders for CLASS-INCREMENTAL continual learning.

Key difference: NO label remapping - original class labels [0-99] are preserved.
This matches the standard continual learning evaluation protocol.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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


# ---------------------------------------------------------------------------
# Split CIFAR-100 (Class-Incremental)
# ---------------------------------------------------------------------------

def get_split_cifar100_class_incremental(
    data_root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split CIFAR-100 for class-incremental learning.
    
    NO label remapping - labels stay in [0-99] range.
    Each task sees a subset of classes, but all use the same label space.
    
    Returns:
        task_dataloaders: list of (train_loader, test_loader) per task
        class_splits: list of class indices per task
    """
    classes_per_task = 100 // n_tasks
    class_splits = [
        list(range(t * classes_per_task, (t + 1) * classes_per_task))
        for t in range(n_tasks)
    ]

    full_train = datasets.CIFAR100(
        data_root, train=True, download=True, transform=get_train_transform()
    )
    full_test = datasets.CIFAR100(
        data_root, train=False, download=True, transform=get_val_transform()
    )

    train_targets = full_train.targets
    test_targets = full_test.targets

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)

        # Get indices for this task's classes
        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets) if c in class_set]

        # Create subsets WITHOUT label remapping
        train_subset = Subset(full_train, tr_indices)
        test_subset = Subset(full_test, te_indices)

        tr_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        te_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ---------------------------------------------------------------------------
# Split ImageNet-R (Class-Incremental)
# ---------------------------------------------------------------------------

def get_split_imagenet_r_class_incremental(
    data_root: str = "./data",
    n_tasks: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split ImageNet-R for class-incremental learning.
    NO label remapping.
    """
    ir_root = os.path.join(data_root, "imagenet-r")
    train_dir = os.path.join(ir_root, "train")
    val_dir = os.path.join(ir_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet-R train split not found at: {train_dir}\n"
            "Please run: python download_datasets.py"
        )
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet-R val split not found at: {val_dir}\n"
            "Please run: python download_datasets.py"
        )

    classes_per_task = 200 // n_tasks

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transform())
    full_val = datasets.ImageFolder(val_dir, transform=get_val_transform())

    sorted_class_ids = sorted(full_train.class_to_idx.values())
    class_splits = [
        sorted_class_ids[t * classes_per_task: (t + 1) * classes_per_task]
        for t in range(n_tasks)
    ]

    train_targets = [s[1] for s in full_train.samples]
    val_targets = [s[1] for s in full_val.samples]

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)

        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        va_indices = [i for i, c in enumerate(val_targets) if c in class_set]

        train_subset = Subset(full_train, tr_indices)
        val_subset = Subset(full_val, va_indices)

        tr_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        va_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        task_dataloaders.append((tr_loader, va_loader))

    return task_dataloaders, class_splits


# ---------------------------------------------------------------------------
# Split CUB-200 (Class-Incremental)
# ---------------------------------------------------------------------------

def get_split_cub200_class_incremental(
    data_root: str = "./data",
    n_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Split CUB-200 for class-incremental learning.
    NO label remapping.
    """
    cub_root = os.path.join(data_root, "CUB_200_2011")
    train_dir = os.path.join(cub_root, "train")
    test_dir = os.path.join(cub_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"CUB-200 train split not found at: {train_dir}\n"
            "Please run: python download_datasets.py"
        )
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"CUB-200 test split not found at: {test_dir}\n"
            "Please run: python download_datasets.py"
        )

    classes_per_task = 200 // n_tasks

    full_train = datasets.ImageFolder(train_dir, transform=get_train_transform())
    full_test = datasets.ImageFolder(test_dir, transform=get_val_transform())

    sorted_class_ids = sorted(full_train.class_to_idx.values())
    class_splits = [
        sorted_class_ids[t * classes_per_task: (t + 1) * classes_per_task]
        for t in range(n_tasks)
    ]

    train_targets = [s[1] for s in full_train.samples]
    test_targets = [s[1] for s in full_test.samples]

    task_dataloaders = []
    for t, classes in enumerate(class_splits):
        class_set = set(classes)

        tr_indices = [i for i, c in enumerate(train_targets) if c in class_set]
        te_indices = [i for i, c in enumerate(test_targets) if c in class_set]

        train_subset = Subset(full_train, tr_indices)
        test_subset = Subset(full_test, te_indices)

        tr_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        te_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        task_dataloaders.append((tr_loader, te_loader))

    return task_dataloaders, class_splits


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing CIFAR-100 class-incremental split...")
    loaders, splits = get_split_cifar100_class_incremental(
        data_root="./data", n_tasks=10
    )
    print(f"  Tasks: {len(loaders)}")
    
    for t, (tr, te) in enumerate(loaders):
        print(f"    Task {t}: classes {splits[t]} | "
              f"train={len(tr.dataset)}  test={len(te.dataset)}")
        
        # Check labels are NOT remapped
        x, y = next(iter(tr))
        print(f"      Sample labels (should be in {splits[t]}): {y[:5].tolist()}")
    
    print("CIFAR-100 class-incremental OK ✔")
