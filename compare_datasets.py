"""
compare_datasets.py
-------------------
Compare TreeLoRA performance across CIFAR-100, ImageNet-R, and CUB-200.

Usage:
    python compare_datasets.py \
      --cifar100 ./runs/cifar100_20260413_023153 \
      --imagenet_r ./runs/imagenet_r_20260413_100257 \
      --cub200 ./runs/cub200_20260413_120000
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results(output_dir):
    results_path = os.path.join(output_dir, "final_results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path, "r") as f:
        return json.load(f)

def compare_datasets(cifar_dir, imagenet_dir, cub_dir):
    datasets = {
        "CIFAR-100": cifar_dir,
        "ImageNet-R": imagenet_dir,
        "CUB-200": cub_dir
    }
    
    results = {}
    for name, path in datasets.items():
        if path and os.path.exists(path):
            results[name] = load_results(path)
    
    if not results:
        print("No valid results found!")
        return
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("  TreeLoRA Performance Comparison")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Tasks':<8} {'Avg Acc':<12} {'BWT':<12} {'Time (min)':<12}")
    print("-" * 80)
    
    for name, res in results.items():
        if res:
            print(f"{name:<15} {res['n_tasks']:<8} "
                  f"{res['final_accuracy']:>10.2f}% "
                  f"{res['backward_transfer']:>10.2f}% "
                  f"{res['training_time_seconds']/60:>10.1f}")
    
    print("=" * 80)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    accs = [results[n]['final_accuracy'] for n in names]
    bwts = [results[n]['backward_transfer'] for n in names]
    
    # Accuracy comparison
    bars1 = ax1.bar(names, accs, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax1.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax1.set_title('Final Average Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars1, accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # BWT comparison
    bars2 = ax2.bar(names, bwts, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_ylabel('Backward Transfer (%)', fontsize=12)
    ax2.set_title('Backward Transfer (Forgetting)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, bwt in zip(bars2, bwts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{bwt:.1f}%', ha='center', va='bottom' if bwt > 0 else 'top', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=300)
    print(f"\n[saved] Comparison plot -> dataset_comparison.png\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar100", type=str, help="CIFAR-100 results directory")
    parser.add_argument("--imagenet_r", type=str, help="ImageNet-R results directory")
    parser.add_argument("--cub200", type=str, help="CUB-200 results directory")
    args = parser.parse_args()
    
    compare_datasets(args.cifar100, args.imagenet_r, args.cub200)

if __name__ == "__main__":
    main()
