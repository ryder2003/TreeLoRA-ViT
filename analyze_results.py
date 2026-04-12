"""
analyze_results.py
------------------
Analyze and visualize TreeLoRA training results from saved checkpoints.

Usage:
    python analyze_results.py --output_dir ./runs/cifar100_20241210_123456
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(output_dir):
    """Load final results from output directory."""
    results_path = os.path.join(output_dir, "final_results.json")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        return None
    
    with open(results_path, "r") as f:
        return json.load(f)


def plot_accuracy_matrix(acc_matrix, output_dir):
    """Plot accuracy matrix heatmap."""
    n_tasks = len(acc_matrix)
    
    # Create full matrix (fill upper triangle with NaN)
    full_matrix = np.full((n_tasks, n_tasks), np.nan)
    for i, row in enumerate(acc_matrix):
        for j, val in enumerate(row):
            full_matrix[i, j] = val
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(full_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)])
    ax.set_yticklabels([f"T{i}" for i in range(n_tasks)])
    
    ax.set_xlabel("Task Evaluated", fontsize=12)
    ax.set_ylabel("After Training Task", fontsize=12)
    ax.set_title("Accuracy Matrix (%)", fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(n_tasks):
        for j in range(len(acc_matrix[i])):
            text = ax.text(j, i, f"{acc_matrix[i][j]:.1f}",
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_matrix.png"), dpi=300)
    print(f"  [saved] Accuracy matrix plot -> {output_dir}/accuracy_matrix.png")


def plot_forgetting_curve(acc_matrix, output_dir):
    """Plot forgetting curve for each task."""
    n_tasks = len(acc_matrix)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for task_id in range(n_tasks):
        # Get accuracy on this task after each subsequent training phase
        accuracies = [acc_matrix[i][task_id] for i in range(task_id, n_tasks)]
        training_phases = list(range(task_id, n_tasks))
        
        ax.plot(training_phases, accuracies, marker='o', label=f"Task {task_id}")
    
    ax.set_xlabel("After Training Task", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Forgetting Curve (Per-Task Accuracy Over Time)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forgetting_curve.png"), dpi=300)
    print(f"  [saved] Forgetting curve -> {output_dir}/forgetting_curve.png")


def plot_training_progress(results, output_dir):
    """Plot training loss and accuracy progression."""
    training_log = results.get("training_log", [])
    
    if not training_log:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss progression
    for task_log in training_log:
        task_id = task_log["task_id"]
        epochs = [e["epoch"] for e in task_log["epochs"]]
        losses = [e["avg_loss"] for e in task_log["epochs"]]
        
        x_offset = task_id * len(epochs)
        x_vals = [x_offset + e for e in range(len(epochs))]
        
        ax1.plot(x_vals, losses, marker='o', label=f"Task {task_id}")
    
    ax1.set_xlabel("Training Step (Epoch)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss Progression", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy progression
    for task_log in training_log:
        task_id = task_log["task_id"]
        epochs = [e["epoch"] for e in task_log["epochs"]]
        accs = [e["train_acc"] for e in task_log["epochs"]]
        
        x_offset = task_id * len(epochs)
        x_vals = [x_offset + e for e in range(len(epochs))]
        
        ax2.plot(x_vals, accs, marker='o', label=f"Task {task_id}")
    
    ax2.set_xlabel("Training Step (Epoch)", fontsize=12)
    ax2.set_ylabel("Train Accuracy (%)", fontsize=12)
    ax2.set_title("Training Accuracy Progression", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=300)
    print(f"  [saved] Training progress -> {output_dir}/training_progress.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze TreeLoRA results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to output directory with results")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} does not exist")
        return
    
    print(f"\nAnalyzing results from: {args.output_dir}\n")
    
    results = load_results(args.output_dir)
    if results is None:
        return
    
    print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
    print(f"Backward Transfer: {results['backward_transfer']:.2f}%")
    print(f"Training Time: {results['training_time_seconds']/60:.1f} min\n")
    
    acc_matrix = results["acc_matrix"]
    
    # Generate plots
    plot_accuracy_matrix(acc_matrix, args.output_dir)
    plot_forgetting_curve(acc_matrix, args.output_dir)
    plot_training_progress(results, args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
