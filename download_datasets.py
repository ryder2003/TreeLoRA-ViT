"""
download_datasets.py
--------------------
Downloads and organises all datasets required for TreeLoRA continual learning:

  1. Split CIFAR-100   → auto-downloaded by torchvision (already works)
  2. ImageNet-R        → downloaded + split into train (80%) / val (20%)
  3. CUB-200-2011      → downloaded + organised using official train_test_split.txt

Expected final structure
------------------------
data/
  cifar-100-python/          ← torchvision auto-downloads
  imagenet-r/
      train/<200 class dirs>/
      val/<200 class dirs>/
  CUB_200_2011/
      train/<200 class dirs>/
      test/<200 class dirs>/

Usage
-----
    python download_datasets.py              # downloads both
    python download_datasets.py --skip_imagenet_r
    python download_datasets.py --skip_cub200
    python download_datasets.py --data_root D:/datasets
"""

import argparse
import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Progress bar helper (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

def _make_progress_hook(filename: str):
    """Return a reporthook for urllib.request.urlretrieve that shows progress."""
    prev = {"blocks": 0}

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * downloaded / total_size)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(
                f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB  {filename}",
                end="",
                flush=True,
            )
        else:
            mb_done = downloaded / 1e6
            print(f"\r  Downloaded {mb_done:.1f} MB  {filename}", end="", flush=True)

    return hook


def _download(url: str, dest: Path):
    """Download url → dest with a progress bar. Skip if dest already exists."""
    if dest.exists():
        print(f"  ✔  Already downloaded: {dest.name}")
        return
    print(f"  ↓  Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_make_progress_hook(dest.name))
    print()  # newline after progress


def _extract_tar(archive: Path, extract_to: Path, *, delete_after: bool = False):
    """Extract a .tar or .tgz file to extract_to."""
    print(f"  📦  Extracting {archive.name} → {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(extract_to)
    print(f"  ✔  Extracted.")
    if delete_after:
        archive.unlink()
        print(f"  🗑   Deleted archive {archive.name}.")


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet-R
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"


def setup_imagenet_r(data_root: Path, train_split: float = 0.8, seed: int = 42):
    """
    Download ImageNet-R and create an 80/20 train/val split.

    The raw archive extracts to:
        <extract_dir>/imagenet-r/<200 class dirs>/<images>

    We reorganise to:
        data_root/imagenet-r/train/<class>/<images>
        data_root/imagenet-r/val/<class>/<images>
    """
    imagenet_r_dir = data_root / "imagenet-r"
    train_dir = imagenet_r_dir / "train"
    val_dir   = imagenet_r_dir / "val"

    if train_dir.exists() and val_dir.exists():
        n_train_cls = len(list(train_dir.iterdir()))
        n_val_cls   = len(list(val_dir.iterdir()))
        if n_train_cls == 200 and n_val_cls == 200:
            print("  ✔  ImageNet-R already organised (train/val). Skipping.")
            return

    # ── Download ──────────────────────────────────────────────────────────────
    archive = data_root / "imagenet-r.tar"
    _download(IMAGENET_R_URL, archive)

    # ── Extract ───────────────────────────────────────────────────────────────
    raw_dir = data_root / "_imagenet_r_raw"
    if not (raw_dir / "imagenet-r").exists():
        _extract_tar(archive, raw_dir)
    else:
        print("  ✔  Archive already extracted (raw).")

    # The tar extracts to raw_dir/imagenet-r/<class>/<image>
    src_root = raw_dir / "imagenet-r"
    if not src_root.is_dir():
        # Fallback: maybe it extracted directly without subfolder
        src_root = raw_dir

    class_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    print(f"  ℹ   Found {len(class_dirs)} classes in ImageNet-R raw data.")

    rng = random.Random(seed)

    print(f"  🔀  Splitting into train ({train_split*100:.0f}%) / val ({(1-train_split)*100:.0f}%) ...")

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images = sorted(
            [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in
             {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}]
        )
        rng.shuffle(images)
        split_idx = max(1, int(len(images) * train_split))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        train_cls_dir = train_dir / cls_name
        val_cls_dir   = val_dir   / cls_name
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True,   exist_ok=True)

        for img in train_imgs:
            dst = train_cls_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
        for img in val_imgs:
            dst = val_cls_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    # Count final images
    n_train = sum(len(list(d.iterdir())) for d in train_dir.iterdir())
    n_val   = sum(len(list(d.iterdir())) for d in val_dir.iterdir())
    print(f"  ✔  ImageNet-R ready: {n_train} train images | {n_val} val images")

    # Clean up raw extraction folder (optional — saves ~1.5 GB)
    print("  🗑   Cleaning up raw extraction folder ...")
    shutil.rmtree(raw_dir, ignore_errors=True)
    if archive.exists():
        archive.unlink()
    print("  ✔  Cleanup done.")


# ──────────────────────────────────────────────────────────────────────────────
# CUB-200-2011
# ──────────────────────────────────────────────────────────────────────────────

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


def setup_cub200(data_root: Path):
    """
    Download CUB-200-2011 and organise using the official train_test_split.txt.

    The raw archive extracts to:
        <extract_dir>/CUB_200_2011/
            images/<class dirs>/<images>
            images.txt           (image_id → relative_path)
            train_test_split.txt (image_id → is_training [1=train, 0=test])
            classes.txt          (class_id → class_name)

    We reorganise to:
        data_root/CUB_200_2011/train/<class>/<image>
        data_root/CUB_200_2011/test/<class>/<image>
    """
    cub_dir   = data_root / "CUB_200_2011"
    train_dir = cub_dir / "train"
    test_dir  = cub_dir / "test"

    if train_dir.exists() and test_dir.exists():
        n_tr = len(list(train_dir.iterdir()))
        n_te = len(list(test_dir.iterdir()))
        if n_tr == 200 and n_te == 200:
            print("  ✔  CUB-200-2011 already organised (train/test). Skipping.")
            return

    # ── Download ──────────────────────────────────────────────────────────────
    archive = data_root / "CUB_200_2011.tgz"
    _download(CUB_URL, archive)

    # ── Extract ───────────────────────────────────────────────────────────────
    raw_dir = data_root / "_cub_raw"
    if not (raw_dir / "CUB_200_2011").exists():
        _extract_tar(archive, raw_dir)
    else:
        print("  ✔  Archive already extracted (raw).")

    src_root = raw_dir / "CUB_200_2011"

    # ── Parse metadata files ──────────────────────────────────────────────────
    images_file = src_root / "images.txt"
    split_file  = src_root / "train_test_split.txt"

    if not images_file.exists():
        raise FileNotFoundError(
            f"Expected images.txt inside {src_root}. "
            "The archive structure may be different — check the contents manually."
        )

    # image_id → relative path (e.g. "001.Black_footed_Albatross/Black_footed_Albatross_0001.jpg")
    id_to_path = {}
    with open(images_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                id_to_path[parts[0]] = parts[1]

    # image_id → 1 (train) or 0 (test)
    id_to_split = {}
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_split[parts[0]] = int(parts[1])

    images_src = src_root / "images"

    print(f"  🔀  Organising {len(id_to_path)} CUB-200 images into train/test ...")

    for img_id, rel_path in id_to_path.items():
        is_train = id_to_split.get(img_id, 1)
        src = images_src / rel_path
        # class folder name is the first component of rel_path
        cls_name = rel_path.split("/")[0]
        dst_base  = train_dir if is_train else test_dir
        dst_dir   = dst_base / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)

    n_train = sum(len(list(d.iterdir())) for d in train_dir.iterdir())
    n_test  = sum(len(list(d.iterdir())) for d in test_dir.iterdir())
    print(f"  ✔  CUB-200 ready: {n_train} train images | {n_test} test images")

    # Clean up
    print("  🗑   Cleaning up raw extraction folder ...")
    shutil.rmtree(raw_dir, ignore_errors=True)
    if archive.exists():
        archive.unlink()
    print("  ✔  Cleanup done.")


# ──────────────────────────────────────────────────────────────────────────────
# CIFAR-100 (just verify torchvision can download it)
# ──────────────────────────────────────────────────────────────────────────────

def verify_cifar100(data_root: Path):
    """Trigger torchvision CIFAR-100 auto-download to confirm it works."""
    try:
        from torchvision import datasets as tvd
        from torchvision import transforms as T

        print("  ↓  Verifying CIFAR-100 (torchvision auto-download) ...")
        tvd.CIFAR100(str(data_root), train=True, download=True,
                     transform=T.ToTensor())
        tvd.CIFAR100(str(data_root), train=False, download=True,
                     transform=T.ToTensor())
        print("  ✔  CIFAR-100 ready.")
    except Exception as e:
        print(f"  ⚠  CIFAR-100 verification failed: {e}")
        print("     It will still be auto-downloaded when you run train.py.")


# ──────────────────────────────────────────────────────────────────────────────
# Verify final structure
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(data_root: Path):
    print("\n" + "─" * 60)
    print("  Dataset Summary")
    print("─" * 60)

    def _count(path: Path):
        if not path.exists():
            return "MISSING"
        classes = [d for d in path.iterdir() if d.is_dir()]
        imgs = sum(
            len([f for f in d.iterdir() if f.is_file()])
            for d in classes
        )
        return f"{len(classes)} classes, {imgs} images"

    # CIFAR-100
    cifar_dir = data_root / "cifar-100-python"
    print(f"  CIFAR-100        : {'✔ present' if cifar_dir.exists() else 'will auto-download'}")

    # ImageNet-R
    ir_train = data_root / "imagenet-r" / "train"
    ir_val   = data_root / "imagenet-r" / "val"
    print(f"  ImageNet-R train : {_count(ir_train)}")
    print(f"  ImageNet-R val   : {_count(ir_val)}")

    # CUB-200
    cub_train = data_root / "CUB_200_2011" / "train"
    cub_test  = data_root / "CUB_200_2011" / "test"
    print(f"  CUB-200 train    : {_count(cub_train)}")
    print(f"  CUB-200 test     : {_count(cub_test)}")

    print("─" * 60)
    print("  Run training with:")
    print("    python train.py --dataset cifar100   --n_tasks 10 --epochs 5")
    print("    python train.py --dataset imagenet_r --n_tasks 20 --epochs 5")
    print("    python train.py --dataset cub200     --n_tasks 10 --epochs 5")
    print("─" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download & organise datasets for TreeLoRA continual learning"
    )
    p.add_argument("--data_root", type=str, default="./data",
                   help="Root directory where datasets will be stored (default: ./data)")
    p.add_argument("--skip_cifar100",    action="store_true",
                   help="Skip CIFAR-100 verification")
    p.add_argument("--skip_imagenet_r",  action="store_true",
                   help="Skip ImageNet-R download/organise")
    p.add_argument("--skip_cub200",      action="store_true",
                   help="Skip CUB-200 download/organise")
    p.add_argument("--train_split",      type=float, default=0.8,
                   help="Train fraction for ImageNet-R (default: 0.8)")
    p.add_argument("--seed",             type=int,   default=42,
                   help="Random seed for ImageNet-R train/val split")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TreeLoRA Dataset Downloader")
    print(f"  Data root : {data_root}")
    print(f"{'='*60}\n")

    if not args.skip_cifar100:
        print("▶  CIFAR-100")
        verify_cifar100(data_root)
        print()

    if not args.skip_imagenet_r:
        print("▶  ImageNet-R  (~1.5 GB download, ~3 GB extracted)")
        setup_imagenet_r(data_root, train_split=args.train_split, seed=args.seed)
        print()

    if not args.skip_cub200:
        print("▶  CUB-200-2011  (~1.1 GB download)")
        setup_cub200(data_root)
        print()

    print_summary(data_root)


if __name__ == "__main__":
    main()
