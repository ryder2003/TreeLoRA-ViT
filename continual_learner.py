"""
continual_learner.py
--------------------
TreeLoRA continual learning trainer for Vision Transformers.

Aligned with the official TreeLoRA repo (ZinYY/TreeLoRA) training pattern:
  - Collect LoRA-A *parameters* (live, differentiable) as _grad_current
  - Compute reg_loss via tree_lora_loss (gradient alignment)
  - loss = task_loss - reg_loss  (single backward call)

Training recipe (per task):
  1. Replace classification head to match task's number of classes
  2. Run epochs:  forward → loss → collect LoRA-A params → tree_search →
                  compute reg loss → loss - reg_loss → backward → update
  3. After each task: end_task() stores grads and rebuilds tree

Evaluation: per-task accuracy + Average Accuracy (Acc) + Backward Transfer (BWT)
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from vit_backbone import ViTBackbone
from lora import inject_lora_to_vit, get_lora_params
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
    ):
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.lr = lr
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.reg = reg

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

        self._set_task_head(task_id)
        optimizer = self._make_optimizer()
        criterion = nn.CrossEntropyLoss()

        self.model.train()

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
                if self.reg > 0:
                    # Collect live LoRA-A parameters (differentiable)
                    lora_A_params = self._collect_lora_A_live()

                    if lora_A_params is not None:
                        # Stack into (lora_depth, dim*rank) — live tensor
                        _grad_current = torch.stack(
                            [p.reshape(-1) for p in lora_A_params], dim=0
                        )

                        # Insert detached copy for tree's gradient tracking
                        self.tree.insert_grad(_grad_current.detach())

                        # Compute regularisation if we have previous tasks
                        if task_id > 0:
                            prev_id_matrix = self.tree.tree_search(
                                task_id, self.device
                            )
                            reg_loss = self.tree.get_loss(
                                _grad_current, loss, task_id, prev_id_matrix
                            )
                            # Official pattern: loss = loss - reg_loss
                            loss = loss - reg_loss

                # Single backward pass
                loss.backward()
                optimizer.step()

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

        Note: we re-attach the task head for task_id, so the head must be saved.
        In the current setup we evaluate each task right after training; for
        backward transfer we store per-task heads.
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
        # We store per-task classification heads for backward evaluation
        task_heads = {}

        for task_id, (train_loader, test_loader) in enumerate(task_dataloaders):
            # Train
            self.train_task(task_id, train_loader, epochs=epochs)

            # Save current head
            task_heads[task_id] = {
                k: v.clone() for k, v in self.model.head.state_dict().items()
            }

            # Evaluate on all tasks seen so far
            row = []
            for prev_task_id in range(task_id + 1):
                # Restore that task's head temporarily
                self.model.head.load_state_dict(task_heads[prev_task_id])
                _, prev_test_loader = task_dataloaders[prev_task_id]
                acc = self.evaluate_task(prev_task_id, prev_test_loader)
                row.append(acc)
                print(f"  → Task {prev_task_id} accuracy: {acc:.2f}%")

            # Restore current task head
            self.model.head.load_state_dict(task_heads[task_id])
            self.acc_matrix.append(row)

        # Compute metrics
        final_acc = compute_average_accuracy(self.acc_matrix)
        bwt = compute_backward_transfer(self.acc_matrix)

        print(f"\n{'='*60}")
        print(f"  Final Average Accuracy (Acc): {final_acc:.2f}%")
        print(f"  Backward Transfer     (BWT): {bwt:.2f}%")
        print(f"{'='*60}\n")

        return self.acc_matrix, final_acc, bwt
