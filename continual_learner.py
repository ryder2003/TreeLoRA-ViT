"""
continual_learner.py
--------------------
TreeLoRA continual learning trainer for Vision Transformers.

Aligned with the official TreeLoRA repo (ZinYY/TreeLoRA) training pattern:
  - Collect LoRA-A *parameters* (live, differentiable) as _grad_current
  - Compute reg_loss via tree_lora_loss (gradient alignment)
  - loss = task_loss - reg_loss  (single backward call)

Training recipe (per task):
  1. **Re-initialise LoRA adapters** (Kaiming/zero) so each task learns
     fresh deltas from the frozen backbone
  2. Replace classification head to match task's number of classes
  3. Run epochs:  forward → loss → collect LoRA-A params → tree_search →
                  compute reg loss → loss - reg_loss → backward → update
  4. After each task: end_task() stores grads and rebuilds tree

Evaluation: per-task accuracy + Average Accuracy (Acc) + Backward Transfer (BWT)
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
    """
    Average Accuracy (Acc): mean of per-task accuracy after training all tasks.

    acc_matrix[i][j] = accuracy on task j after training on task i (j <= i)
    """
    n = len(acc_matrix)
    return sum(acc_matrix[n - 1][j] for j in range(n)) / n


def compute_backward_transfer(acc_matrix):
    """
    Backward Transfer (BWT): measures forgetting.
    BWT = (1/T-1) * Σ_{j=1}^{T-1} (acc[T-1][j] - acc[j][j])
    Negative BWT = forgetting.
    """
    n = len(acc_matrix)
    if n <= 1:
        return 0.0
    bwt = sum(acc_matrix[n - 1][j] - acc_matrix[j][j] for j in range(n - 1))
    return bwt / (n - 1)


# ---------------------------------------------------------------------------
# Main Learner
# ---------------------------------------------------------------------------

class TreeLoRALearner:
    """
    Full TreeLoRA continual learning system for ViT.

    Args:
        num_tasks       : number of CL tasks
        classes_per_task: number of classes per task
        lora_rank       : LoRA rank r (default 4)
        lora_alpha      : LoRA alpha α (default 8)
        lora_depth      : KD-tree depth (default 5 – paper value for ViT)
        reg             : regularisation weight λ (default 0.5)
        lr              : learning rate for LoRA params + head
        device          : torch device
        pretrained      : load ImageNet-21k weights (set False for smoke tests)
        output_dir      : directory for saving checkpoints and logs (None = no saving)
    """

    def __init__(
        self,
        num_tasks: int,
        classes_per_task: int,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_depth: int = 5,
        reg: float = 0.5,
        lr: float = 5e-3,
        device: torch.device = None,
        pretrained: bool = True,
        output_dir: str = None,
    ):
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.lr = lr
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.reg = reg
        self.output_dir = output_dir

        # ViT-B/16 backbone (frozen) + initial head
        self.model = ViTBackbone(
            num_classes=classes_per_task, pretrained=pretrained
        )

        # Inject LoRA into Q and V projections of all 12 attention blocks
        inject_lora_to_vit(self.model, rank=lora_rank, alpha=lora_alpha)

        # Move the entire model (including new LoRA parameters) to device
        self.model = self.model.to(self.device)

        # KD-tree manager
        self.tree = KD_LoRA_Tree(
            num_tasks=num_tasks,
            lora_depth=lora_depth,
            reg=reg,
        )

        # Accuracy tracking: acc_matrix[task_trained][task_eval]
        self.acc_matrix = []

        # Per-task classification heads (saved for backward evaluation)
        self.task_heads = {}

        # Training log
        self.training_log = []

        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Task head management
    # ------------------------------------------------------------------

    def _set_task_head(self, task_id: int):
        """Replace the classification head for the current task."""
        self.model.head = nn.Linear(
            self.model.embed_dim, self.classes_per_task
        ).to(self.device)

    def _make_optimizer(self):
        """Adam optimiser over LoRA params + current task head."""
        params = list(get_lora_params(self.model))
        param_tensors = [p for _, p in params] + list(self.model.head.parameters())
        return torch.optim.Adam(param_tensors, lr=self.lr)

    def _collect_lora_A_live(self):
        """
        Collect LoRA-A parameters as LIVE tensors (differentiable).

        This matches the official repo's pattern:
            _grad_current = []
            for name_, param_ in model.named_parameters():
                if "loranew_A" in name_:
                    _grad_current.append(param_)

        Returns:
            list of parameter tensors, or None if no LoRA-A found
        """
        params = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                params.append(param)
        return params if params else None

    # ------------------------------------------------------------------
    # Checkpoint saving / loading
    # ------------------------------------------------------------------

    def save_checkpoint(self, task_id: int):
        """Save LoRA weights, all task heads, tree state, and accuracy matrix."""
        if not self.output_dir:
            return

        ckpt_dir = os.path.join(self.output_dir, f"task_{task_id}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Save ALL model parameters including LoRA
        model_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
        }
        torch.save(model_state, os.path.join(ckpt_dir, "model_state.pt"))

        # 2. Save all task heads
        heads_state = {
            tid: {k: v.cpu().clone() for k, v in head.items()}
            for tid, head in self.task_heads.items()
        }
        torch.save(heads_state, os.path.join(ckpt_dir, "task_heads.pt"))

        # 3. Save tree state (accumulated gradients)
        tree_state = {
            "all_accumulate_grads": [
                g.cpu().clone() if g is not None else None
                for g in self.tree.all_accumulate_grads
            ],
        }
        torch.save(tree_state, os.path.join(ckpt_dir, "tree_state.pt"))

        # 4. Save accuracy matrix
        with open(os.path.join(ckpt_dir, "accuracy_matrix.json"), "w") as f:
            json.dump({
                "acc_matrix": self.acc_matrix,
                "task_id": task_id,
            }, f, indent=2)

        # 5. Save training log
        with open(os.path.join(ckpt_dir, "training_log.json"), "w") as f:
            json.dump(self.training_log, f, indent=2)

        print(f"  [saved] Checkpoint -> {ckpt_dir}")

    def load_checkpoint(self, task_id: int):
        """Load model state from a checkpoint."""
        if not self.output_dir:
            return False

        ckpt_dir = os.path.join(self.output_dir, f"task_{task_id}")
        model_path = os.path.join(ckpt_dir, "model_state.pt")
        
        if not os.path.exists(model_path):
            return False

        model_state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state)
        print(f"  [loaded] Checkpoint from {ckpt_dir}")
        return True

    def save_final_results(self, final_acc: float, bwt: float, elapsed: float):
        """Save a comprehensive final results summary."""
        if not self.output_dir:
            return

        results = {
            "final_accuracy": final_acc,
            "backward_transfer": bwt,
            "training_time_seconds": elapsed,
            "num_tasks": self.num_tasks,
            "classes_per_task": self.classes_per_task,
            "acc_matrix": self.acc_matrix,
            "training_log": self.training_log,
        }

        results_path = os.path.join(self.output_dir, "final_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Also write a human-readable summary
        summary_path = os.path.join(self.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("TreeLoRA ViT-B/16 - Final Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Average Accuracy (Acc) : {final_acc:.2f}%\n")
            f.write(f"Backward Transfer (BWT): {bwt:.2f}%\n")
            f.write(f"Training time          : {elapsed/60:.1f} min ({elapsed:.0f}s)\n")
            f.write(f"Tasks                  : {self.num_tasks}\n")
            f.write(f"Classes per task       : {self.classes_per_task}\n\n")

            f.write("Accuracy Matrix (rows = after training task T, cols = task evaluated):\n")
            header = "         " + "  ".join(f"T{j:02d}" for j in range(self.num_tasks))
            f.write(header + "\n")
            for i, row in enumerate(self.acc_matrix):
                cells   = "  ".join(f"{v:5.1f}" for v in row)
                padding = "  ".join("  ---" for _ in range(self.num_tasks - len(row)))
                f.write(f"  T{i:02d}:  {cells}  {padding}\n")

        print(f"  [saved] Final results -> {self.output_dir}")

    # ------------------------------------------------------------------
    # Training one task
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader,
        epochs: int = 5,
    ):
        """
        Train the model on a single task using TreeLoRA.

        Follows the official repo's training pattern exactly:
        0. Re-initialise LoRA adapters (fresh deltas for each task)
        1. Forward pass → task loss
        2. Collect live LoRA-A params → stack → insert_grad (detached copy)
        3. If task_id > 0: tree_search → get_loss → loss = loss - reg_loss
        4. Single backward + step

        Args:
            task_id      : 0-indexed task identifier
            train_loader : DataLoader for this task's training split
            epochs       : number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"  Training Task {task_id}  ({len(train_loader.dataset)} samples)")
        print(f"{'='*60}")

        # Only reset LoRA for the FIRST task
        # Subsequent tasks fine-tune existing LoRA weights with tree regularization
        if task_id == 0:
            reset_all_lora(self.model)

        self._set_task_head(task_id)
        optimizer = self._make_optimizer()

        # Cosine annealing over all epochs for this task
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

                # Book-keep tree step counter
                if self.reg > 0:
                    self.tree.step()

                optimizer.zero_grad()

                # Forward pass
                logits = self.model(images)
                loss = criterion(logits, labels)

                # -------------------------------------------------------
                # TreeLoRA regularisation (matches official Tree_LoRA.py)
                # -------------------------------------------------------
                if self.reg > 0 and task_id > 0:
                    # Collect live LoRA-A parameters (differentiable)
                    lora_A_params = self._collect_lora_A_live()

                    if lora_A_params is not None:
                        # Stack into (lora_depth, dim*rank) — live tensor
                        _grad_current = torch.stack(
                            [p.reshape(-1) for p in lora_A_params], dim=0
                        )

                        # Insert detached copy for tree's gradient tracking
                        self.tree.insert_grad(_grad_current.detach())

                        # Compute regularisation for previous tasks
                        prev_id_matrix = self.tree.tree_search(
                            task_id, self.device
                        )
                        reg_loss = self.tree.get_loss(
                            _grad_current, loss, task_id, prev_id_matrix
                        )
                        # Official pattern: loss = loss - reg_loss
                        loss = loss - reg_loss
                elif self.reg > 0 and task_id == 0:
                    # For first task, still collect gradients but no regularization
                    lora_A_params = self._collect_lora_A_live()
                    if lora_A_params is not None:
                        _grad_current = torch.stack(
                            [p.reshape(-1) for p in lora_A_params], dim=0
                        )
                        self.tree.insert_grad(_grad_current.detach())

                # Single backward pass
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in get_lora_params(self.model)]
                    + list(self.model.head.parameters()),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()

                # Metrics (use .item() to avoid graph retention)
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

        # End of task: store gradients and rebuild KD-tree
        if self.reg > 0:
            self.tree.end_task(task_id)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_task(self, task_id: int, test_loader) -> float:
        """
        Evaluate the model on a specific task's test set.

        CRITICAL: Must use the correct task head for the task being evaluated.
        The head should already be loaded before calling this function.
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

    # ------------------------------------------------------------------
    # Full continual learning loop
    # ------------------------------------------------------------------

    def run(
        self,
        task_dataloaders,   # list of (train_loader, test_loader)
        epochs: int = 5,
    ):
        """
        Run the full continual learning benchmark.

        After each task, evaluates on all seen tasks and records the
        accuracy matrix used to compute Acc and BWT.

        Args:
            task_dataloaders : list of (train_loader, test_loader) per task
            epochs           : epochs per task

        Returns:
            acc_matrix : T × T accuracy matrix
            final_acc  : Average Accuracy (Acc)
            bwt        : Backward Transfer (BWT)
        """
        t0 = time.time()

        for task_id, (train_loader, test_loader) in enumerate(task_dataloaders):
            # Train
            self.train_task(task_id, train_loader, epochs=epochs)

            # Save current head
            self.task_heads[task_id] = {
                k: v.clone() for k, v in self.model.head.state_dict().items()
            }

            # Evaluate on all tasks seen so far
            row = []
            for prev_task_id in range(task_id + 1):
                # CRITICAL: Load the head for the task being evaluated
                self.model.head.load_state_dict(self.task_heads[prev_task_id])
                _, prev_test_loader = task_dataloaders[prev_task_id]
                acc = self.evaluate_task(prev_task_id, prev_test_loader)
                row.append(acc)
                print(f"  -> Task {prev_task_id} accuracy: {acc:.2f}%")

            # Restore current task head for next training iteration
            self.model.head.load_state_dict(self.task_heads[task_id])
            self.acc_matrix.append(row)

            # Save checkpoint after each task
            self.save_checkpoint(task_id)

        # Compute metrics
        final_acc = compute_average_accuracy(self.acc_matrix)
        bwt = compute_backward_transfer(self.acc_matrix)
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"  Final Average Accuracy (Acc): {final_acc:.2f}%")
        print(f"  Backward Transfer     (BWT): {bwt:.2f}%")
        print(f"{'='*60}\n")

        # Save final results
        self.save_final_results(final_acc, bwt, elapsed)

        return self.acc_matrix, final_acc, bwt
