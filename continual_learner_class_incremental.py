"""
continual_learner_class_incremental.py
---------------------------------------
CLASS-INCREMENTAL TreeLoRA implementation matching the paper.

Key differences from task-incremental:
- Single unified head with ALL classes (e.g., 100 for CIFAR-100)
- No label remapping - original labels [0-99] preserved
- Model must distinguish all classes simultaneously
- Much harder setting, expected accuracy: 60-70% (matches paper)

This is the standard evaluation protocol for continual learning papers.
"""

import json
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from vit_backbone import ViTBackbone
from lora import inject_lora_to_vit, get_lora_params, reset_all_lora
from kd_lora_tree import KD_LoRA_Tree


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_average_accuracy(acc_matrix):
    """Average Accuracy: mean of per-task accuracy after training all tasks."""
    n = len(acc_matrix)
    return sum(acc_matrix[n - 1][j] for j in range(n)) / n


def compute_backward_transfer(acc_matrix):
    """Backward Transfer: measures forgetting."""
    n = len(acc_matrix)
    if n <= 1:
        return 0.0
    bwt = sum(acc_matrix[n - 1][j] - acc_matrix[j][j] for j in range(n - 1))
    return bwt / (n - 1)


# ---------------------------------------------------------------------------
# Class-Incremental Learner
# ---------------------------------------------------------------------------

class ClassIncrementalTreeLoRALearner:
    """
    TreeLoRA for CLASS-INCREMENTAL continual learning.
    
    Uses a single unified head with all classes, no label remapping.
    This matches the standard continual learning evaluation protocol.
    """

    def __init__(
        self,
        num_tasks: int,
        total_classes: int,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_depth: int = 5,
        reg: float = 1.0,
        lr: float = 5e-3,
        device: torch.device = None,
        pretrained: bool = True,
        output_dir: str = None,
    ):
        self.num_tasks = num_tasks
        self.total_classes = total_classes
        self.classes_per_task = total_classes // num_tasks
        self.lr = lr
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.reg = reg
        self.output_dir = output_dir

        # ViT-B/16 with UNIFIED head for ALL classes
        self.model = ViTBackbone(
            num_classes=total_classes,  # Single head with all classes
            pretrained=pretrained
        )

        # Inject LoRA
        inject_lora_to_vit(self.model, rank=lora_rank, alpha=lora_alpha)
        self.model = self.model.to(self.device)

        # KD-tree manager
        self.tree = KD_LoRA_Tree(
            num_tasks=num_tasks,
            lora_depth=lora_depth,
            reg=reg,
        )

        # Tracking
        self.acc_matrix = []
        self.training_log = []
        self.seen_tasks = []

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def _make_optimizer(self):
        """Optimizer over LoRA params + unified head."""
        params = list(get_lora_params(self.model))
        param_tensors = [p for _, p in params] + list(self.model.head.parameters())
        return torch.optim.Adam(param_tensors, lr=self.lr)

    def _collect_lora_A_live(self):
        """Collect LoRA-A parameters as live tensors."""
        params = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                params.append(param)
        return params if params else None

    def save_checkpoint(self, task_id: int):
        """Save checkpoint after each task."""
        if not self.output_dir:
            return

        ckpt_dir = os.path.join(self.output_dir, f"task_{task_id}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save full model state
        model_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
        }
        torch.save(model_state, os.path.join(ckpt_dir, "model_state.pt"))

        # Save tree state
        tree_state = {
            "all_accumulate_grads": [
                g.cpu().clone() if g is not None else None
                for g in self.tree.all_accumulate_grads
            ],
        }
        torch.save(tree_state, os.path.join(ckpt_dir, "tree_state.pt"))

        # Save accuracy matrix
        with open(os.path.join(ckpt_dir, "accuracy_matrix.json"), "w") as f:
            json.dump({
                "acc_matrix": self.acc_matrix,
                "task_id": task_id,
            }, f, indent=2)

        # Save training log
        with open(os.path.join(ckpt_dir, "training_log.json"), "w") as f:
            json.dump(self.training_log, f, indent=2)

        print(f"  [saved] Checkpoint -> {ckpt_dir}")

    def save_final_results(self, final_acc: float, bwt: float, elapsed: float):
        """Save comprehensive final results."""
        if not self.output_dir:
            return

        results = {
            "final_accuracy": final_acc,
            "backward_transfer": bwt,
            "training_time_seconds": elapsed,
            "n_tasks": self.num_tasks,
            "total_classes": self.total_classes,
            "classes_per_task": self.classes_per_task,
            "acc_matrix": self.acc_matrix,
            "training_log": self.training_log,
            "evaluation_protocol": "class_incremental",
        }

        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Human-readable summary
        summary_path = os.path.join(self.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TreeLoRA ViT-B/16 - CLASS-INCREMENTAL Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Protocol: CLASS-INCREMENTAL\n")
            f.write(f"  - Single unified head ({self.total_classes} classes)\n")
            f.write(f"  - No label remapping\n")
            f.write(f"  - Model distinguishes all classes simultaneously\n\n")
            f.write(f"Average Accuracy (Acc) : {final_acc:.2f}%\n")
            f.write(f"Backward Transfer (BWT): {bwt:.2f}%\n")
            f.write(f"Training time          : {elapsed/60:.1f} min ({elapsed:.0f}s)\n")
            f.write(f"Tasks                  : {self.num_tasks}\n")
            f.write(f"Total classes          : {self.total_classes}\n")
            f.write(f"Classes per task       : {self.classes_per_task}\n\n")

            f.write("Accuracy Matrix:\n")
            header = "         " + "  ".join(f"T{j:02d}" for j in range(self.num_tasks))
            f.write(header + "\n")
            for i, row in enumerate(self.acc_matrix):
                cells = "  ".join(f"{v:5.1f}" for v in row)
                padding = "  ".join("  ---" for _ in range(self.num_tasks - len(row)))
                f.write(f"  T{i:02d}:  {cells}  {padding}\n")

        print(f"  [saved] Final results -> {self.output_dir}")

    def train_task(
        self,
        task_id: int,
        train_loader,
        epochs: int = 5,
    ):
        """
        Train on a single task using class-incremental learning.
        
        Key: Uses the SAME unified head for all tasks, no head switching.
        """
        print(f"\n{'='*60}")
        print(f"  Training Task {task_id}  ({len(train_loader.dataset)} samples)")
        print(f"  Classes: {task_id * self.classes_per_task} to "
              f"{(task_id + 1) * self.classes_per_task - 1}")
        print(f"{'='*60}")

        # CRITICAL: Reset LoRA for EVERY task (paper's approach)
        # Tree regularization provides knowledge transfer
        reset_all_lora(self.model)
        print("  LoRA re-initialized for this task")

        optimizer = self._make_optimizer()
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs * len(train_loader),
            eta_min=self.lr * 0.01,
        )

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        task_log = {
            "task_id": task_id,
            "epochs": [],
        }

        for epoch in range(epochs):
            self.tree.new_epoch_init(len(train_loader))
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(
                train_loader,
                desc=f"Task {task_id} Epoch {epoch+1}/{epochs}",
                leave=False,
            )

            for step, (images, labels) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.reg > 0:
                    self.tree.step()

                optimizer.zero_grad()

                # Forward pass
                logits = self.model(images)
                loss = criterion(logits, labels)

                # TreeLoRA regularization
                if self.reg > 0 and task_id > 0:
                    lora_A_params = self._collect_lora_A_live()
                    if lora_A_params is not None:
                        _grad_current = torch.stack(
                            [p.reshape(-1) for p in lora_A_params], dim=0
                        )
                        self.tree.insert_grad(_grad_current.detach())
                        prev_id_matrix = self.tree.tree_search(task_id, self.device)
                        reg_loss = self.tree.get_loss(
                            _grad_current, loss, task_id, prev_id_matrix
                        )
                        # We ADD the reg_loss, since tree_lora_loss returns -dot_product
                        # Minimizing (loss + reg_loss) will push the dot_product to be POSITIVE
                        loss = loss + reg_loss
                elif self.reg > 0 and task_id == 0:
                    lora_A_params = self._collect_lora_A_live()
                    if lora_A_params is not None:
                        _grad_current = torch.stack(
                            [p.reshape(-1) for p in lora_A_params], dim=0
                        )
                        self.tree.insert_grad(_grad_current.detach())

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in get_lora_params(self.model)]
                    + list(self.model.head.parameters()),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100*correct/total:.1f}%",
                )

            avg_loss = running_loss / max(len(train_loader), 1)
            train_acc = 100 * correct / max(total, 1)
            print(
                f"  Epoch {epoch+1}/{epochs}  "
                f"loss={avg_loss:.4f}  train_acc={train_acc:.2f}%"
            )

            task_log["epochs"].append({
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 4),
                "train_acc": round(train_acc, 2),
            })

        self.training_log.append(task_log)

        # End of task: store gradients and rebuild tree
        if self.reg > 0:
            self.tree.end_task(task_id)

    @torch.no_grad()
    def evaluate_task(self, task_id: int, test_loader) -> float:
        """
        Evaluate on a specific task using the unified head.
        
        No head switching - always uses the same head for all tasks.
        """
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        self.model.train()
        return 100 * correct / max(total, 1)

    def run(
        self,
        task_dataloaders,
        epochs: int = 5,
    ):
        """
        Run full class-incremental continual learning benchmark.
        """
        t0 = time.time()

        for task_id, (train_loader, test_loader) in enumerate(task_dataloaders):
            # Train
            self.train_task(task_id, train_loader, epochs=epochs)
            self.seen_tasks.append(task_id)

            # Evaluate on all seen tasks (using same unified head)
            row = []
            for prev_task_id in range(task_id + 1):
                _, prev_test_loader = task_dataloaders[prev_task_id]
                acc = self.evaluate_task(prev_task_id, prev_test_loader)
                row.append(acc)
                print(f"  -> Task {prev_task_id} accuracy: {acc:.2f}%")

            self.acc_matrix.append(row)

            # Save checkpoint
            self.save_checkpoint(task_id)

        # Compute metrics
        final_acc = compute_average_accuracy(self.acc_matrix)
        bwt = compute_backward_transfer(self.acc_matrix)
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"  Final Average Accuracy (Acc): {final_acc:.2f}%")
        print(f"  Backward Transfer     (BWT): {bwt:.2f}%")
        print(f"{'='*60}\n")

        self.save_final_results(final_acc, bwt, elapsed)

        return self.acc_matrix, final_acc, bwt
