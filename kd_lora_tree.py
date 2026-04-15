"""
kd_lora_tree.py
---------------
KD-Tree of LoRA adapters guided by hierarchical gradient similarity.

Directly adapted from the official TreeLoRA repository
(ZinYY/TreeLoRA) for Vision Transformers.

Key classes:
    KDTreeNode      – one node in the gradient-similarity KD-tree
    KD_LoRA_Tree    – manages the full tree + bandit-based search logic
"""

import copy
import math
import torch


# ---------------------------------------------------------------------------
# KD-Tree Node  (exact port from official repo)
# ---------------------------------------------------------------------------

class KDTreeNode:
    """
    One node in the hierarchical gradient-similarity KD-tree.

    At each depth level the node uses the gradient vectors at that depth
    to split tasks into two children based on L1-distance to the mean
    vector (paper-consistent similarity criterion).

    Args:
        task_indices  : global task indices contained in this node
        depth         : current depth (root = 0)
        grads_tensor  : (num_tasks, lora_depth, feature_dim) – all accumulated grads
        lora_depth    : maximum tree depth (paper default = 5 for ViT)
    """

    def __init__(self, task_indices, depth, grads_tensor, lora_depth):
        self.task_indices = task_indices
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.lora_depth = lora_depth
        self.mean_vector = None
        self.median_similarity = None

        self._build_node(grads_tensor)

    def _build_node(self, grads_tensor):
        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            return

        # Grads at this depth for the tasks in this node: (N, D)
        current_grads = grads_tensor[self.task_indices, self.depth, :]

        # Mean vector for this split dimension
        self.mean_vector = current_grads.mean(dim=0)                # (D,)

        # L1 distance to the mean vector (smaller is more similar)
        similarities = torch.norm(
            current_grads - self.mean_vector.unsqueeze(0), p=1, dim=1
        )
        self.median_similarity = torch.median(similarities).item()

        left_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() <= self.median_similarity
        ]
        right_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() > self.median_similarity
        ]

        # Avoid degenerate splits
        if len(left_indices) == 0 or len(right_indices) == 0:
            mid = len(self.task_indices) // 2
            left_indices = self.task_indices[:mid]
            right_indices = self.task_indices[mid:]

        self.left = KDTreeNode(left_indices, self.depth + 1, grads_tensor, self.lora_depth)
        self.right = KDTreeNode(right_indices, self.depth + 1, grads_tensor, self.lora_depth)

    def __str__(self, level=0):
        indent = "  " * level
        if self.is_leaf:
            return f"{indent}Leaf(depth={self.depth}, tasks={self.task_indices})\n"
        mean_str = ", ".join(f"{x:.4f}" for x in self.mean_vector[:2].tolist())
        result = (
            f"{indent}Node(depth={self.depth}, tasks={self.task_indices}, "
            f"mean_vector=[{mean_str}, ...], "
            f"median_sim={self.median_similarity:.4f})\n"
        )
        if self.left:
            result += self.left.__str__(level + 1)
        if self.right:
            result += self.right.__str__(level + 1)
        return result


# ---------------------------------------------------------------------------
# Gradient-similarity regularisation loss
# ---------------------------------------------------------------------------

def tree_lora_loss(current_grad, all_grad, task_id, prev_id_matrix):
    """
    Maximise gradient alignment between current task and selected previous tasks.

    For each depth d, push the current gradient towards the gradient of the
    selected previous task at that depth:
        loss = -Σ_d  <current_grad[d], all_grad[prev_id_matrix[d]][d]>

    Args:
        current_grad   : (lora_depth, feature_dim) – current task grads
        all_grad       : (num_tasks, lora_depth, feature_dim) – all previous grads
        task_id        : current task ID (unused here, kept for API compatibility)
        prev_id_matrix : (lora_depth,) – selected previous task per depth

    Returns:
        scalar loss tensor
    """
    reg_loss = None
    for depth_id, prev_task_id in enumerate(prev_id_matrix):
        dot = -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
        reg_loss = dot if reg_loss is None else reg_loss + dot
    return reg_loss


# ---------------------------------------------------------------------------
# Main KD-LoRA tree manager
# ---------------------------------------------------------------------------

class KD_LoRA_Tree:
    """
    Manages the hierarchical gradient-similarity KD-tree and the
    Lower-Confidence-Bound (LCB) bandit-based adapter path search.

    Workflow per task:
        1. new_epoch_init(dataloader_len)    – reset epoch-level state
        2. step()                            – update round counter
        3. insert_grad(lora_grads)           – accumulate gradient estimates
        4. tree_search(task_id, device)      – find best previous task per depth
        5. get_loss(grads, loss, ...)        – compute regularisation loss
        6. end_task(task_id)                 – store grads, rebuild tree

    Args:
        num_tasks  : total number of continual-learning tasks
        lora_depth : max tree depth = number of LoRA layers (default 5)
        reg        : regularisation strength λ (default 0.5)
    """

    def __init__(self, num_tasks: int, lora_depth: int = 5, reg: float = 0.5):
        self.num_tasks = num_tasks
        self.lora_depth = lora_depth
        self.reg = reg

        self.all_accumulate_grads = [None] * num_tasks  # stored per-task gradients
        self.kd_tree_root = None
        self.current_grad = None     # running gradient accumulator for current task

        # Epoch-level state (reset by new_epoch_init)
        self.all_grad = None
        self.all_grad_device = None
        self.sim = None
        self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = 1
        self.tmp_reg = reg

    # ------------------------------------------------------------------
    # Epoch initialisation
    # ------------------------------------------------------------------

    def new_epoch_init(self, train_dataloader_len: int):
        self.current_grad = None
        self.all_grad = None
        self.all_grad_device = None
        self.sim = None
        self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = max(train_dataloader_len, 1)

    def step(self):
        """Increment round counter and update the annealed regularisation coefficient."""
        self.tmp_rounds += 1
        self.tmp_reg = self.reg * self.tmp_rounds / self.total_rounds

    # ------------------------------------------------------------------
    # Gradient accumulation
    # ------------------------------------------------------------------

    def insert_grad(self, lora_grads: torch.Tensor):
        """
        Accumulate LoRA parameter estimates averaged over the epoch.

        Tracks the mean LoRA gradient estimate over one epoch.

        Args:
            lora_grads : (lora_depth, feature_dim)  current-step LoRA-A values
        """
        frac = 1.0 / self.total_rounds
        if self.current_grad is None:
            self.current_grad = lora_grads.detach() * frac
        else:
            self.current_grad += lora_grads.detach() * frac

    # ------------------------------------------------------------------
    # Tree search (LCB bandit)
    # ------------------------------------------------------------------

    def tree_search(self, task_id: int, device: torch.device) -> torch.Tensor:
        """
        For each LoRA depth, select the previous task whose gradient is most
        similar to the current task using an LCB (Lower Confidence Bound) bandit
        applied hierarchically over the KD-Tree (Formula from paper Eq. 2).
        
        Returns:
            prev_id_matrix : (lora_depth,) int tensor – selected task per depth
        """
        if self.all_grad is None:
            valid_grads = torch.stack(self.all_accumulate_grads[:task_id], dim=0)
            self.all_grad = valid_grads.to(device, non_blocking=True)
            self.all_grad_device = self.all_grad

        lora_d = min(self.lora_depth, self.all_grad.shape[1])
        if self.sim is None:
            self.sim = torch.zeros(task_id, lora_d, device=device)
            self.num_of_selected = torch.zeros(self.num_tasks, lora_d, device=device)

        # Average distance computed from tracking
        mu = self.sim.clone() 
        valid_mask = self.num_of_selected[:task_id, :] > 0
        mu[valid_mask] = mu[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]
        
        # Determine t for the bandit calculation
        # total_rounds = train_dataloader_len
        # tmp_rounds = current batch index
        # We model t globally (t >= 1)
        t_global = max(1, self.total_rounds * (self.tmp_rounds + 1))
        
        prev_id_matrix = torch.zeros(lora_d, dtype=torch.long, device=device)
        
        # Traverse the tree for EACH depth
        for d in range(lora_d):
            if self.kd_tree_root is None:
                # Fallback to standard argmin if tree is not built yet
                raw_lcb = mu[:, d] - 2 * torch.sqrt(math.log(t_global) / (self.num_of_selected[:task_id, d] + 1e-5))
                prev_id_matrix[d] = torch.argmin(raw_lcb)
                continue
                
            def compute_lcb(node) -> float:
                if node.is_leaf:
                    # For a leaf k in L: LCB_k = \mu_k - 2*sqrt(log(t)/n_k)
                    best_lcb = float('inf')
                    for k in node.task_indices:
                        if k >= task_id: 
                            continue
                        n_k = self.num_of_selected[k, d].item()
                        mu_k = mu[k, d].item()
                        lcb = mu_k - 2.0 * math.sqrt(math.log(t_global) / max(n_k, 1e-5))
                        if lcb < best_lcb:
                            best_lcb = lcb
                    return best_lcb if best_lcb != float('inf') else 1e9

                # For an internal node:
                # Equation (2): max { min_{j in C} [ LCB_j - delta ] }
                # Note: node.median_similarity is the threshold (delta)
                left_lcb = compute_lcb(node.left) if node.left else float('inf')
                right_lcb = compute_lcb(node.right) if node.right else float('inf')
                
                # Children LCB values adjusted by similarity threshold (delta)
                delta = node.median_similarity if node.median_similarity is not None else 0.0
                min_adjusted_child = min(left_lcb - delta, right_lcb - delta)
                
                # We return the max constraint exactly as in the LCB conditional formula
                # However, for minimization descent it acts as a bound
                return max(min_adjusted_child, -1e9)
            
            # Now actually search down the tree to select the leaf with minimum LCB
            current = self.kd_tree_root
            while not current.is_leaf:
                left_val = compute_lcb(current.left) if current.left else float('inf')
                right_val = compute_lcb(current.right) if current.right else float('inf')
                
                if left_val < right_val:
                    current = current.left
                else:
                    current = current.right
            
            # Select the exact task index from this leaf
            best_task, best_lcb = -1, float('inf')
            for k in current.task_indices:
                if k >= task_id: continue
                n_k = self.num_of_selected[k, d].item()
                mu_k = mu[k, d].item()
                lcb_val = mu_k - 2.0 * math.sqrt(math.log(t_global) / max(n_k, 1e-5))
                if lcb_val < best_lcb:
                    best_lcb = lcb_val
                    best_task = k
            
            if best_task == -1: 
                best_task = 0 # failsafe
            prev_id_matrix[d] = best_task

        # Update selection counts
        self.num_of_selected[prev_id_matrix, torch.arange(lora_d, device=device)] += 1
        self._update_similarity(prev_id_matrix, device)

        return prev_id_matrix

    def _update_similarity(self, prev_id_matrix: torch.Tensor, device: torch.device):
        """Update L1-distance similarity tracker for selected tasks."""
        if self.sim is None:
            return
        for depth_idx, prev_id in enumerate(prev_id_matrix):
            # Accumulate distance (not negative distance): lower is better.
            self.sim[prev_id, depth_idx] += torch.sum(
                torch.abs(
                    self.current_grad[depth_idx]
                    - self.all_grad[prev_id, depth_idx]
                )
            ).item()

    # ------------------------------------------------------------------
    # Regularisation loss
    # ------------------------------------------------------------------

    def get_loss(
        self,
        lora_grads: torch.Tensor,
        task_loss: torch.Tensor,
        task_id: int,
        prev_id_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the TreeLoRA gradient-alignment regularisation loss and
        scale it to match the magnitude of the task loss.
        """
        reg_loss = tree_lora_loss(
            lora_grads, self.all_grad_device, task_id, prev_id_matrix
        )
        # Adaptive scaling matching official repo (ZinYY/TreeLoRA):
        # reg_loss / (reg_loss.detach().clone() + 1e-5)  ≈ 1.0 for large |reg_loss|
        # This normalizes to the scale of the task loss
        reg_loss = (
            reg_loss / (reg_loss.detach().clone() + 1e-5)
            * task_loss.detach().clone()
            * self.tmp_reg
        )
        return reg_loss

    # ------------------------------------------------------------------
    # End-of-task tree update
    # ------------------------------------------------------------------

    def end_task(self, task_id: int):
        """
        Store the accumulated gradients and rebuild the KD-tree to include
        the newly completed task.

        Matches the official repo's end_task pattern:
            lora_depth = self.current_grad.shape[0]
        """
        if self.current_grad is None:
            return

        self.all_accumulate_grads[task_id] = self.current_grad

        # Collect all valid gradient tensors up to and including task_id
        valid_grads = [
            g for g in self.all_accumulate_grads[: task_id + 1]
            if g is not None
        ]
        if not valid_grads:
            return

        grads_tensor = copy.deepcopy(torch.stack(valid_grads))  # (T, depth, D)

        # Compute gradient differences between consecutive tasks (official pattern)
        for i in range(grads_tensor.shape[0] - 1, 0, -1):
            grads_tensor[i] = grads_tensor[i] - grads_tensor[i - 1]

        task_ids = [
            i
            for i, g in enumerate(self.all_accumulate_grads[: task_id + 1])
            if g is not None
        ]

        # Enforce configured max depth while respecting available tensor depth.
        lora_depth = min(self.lora_depth, grads_tensor.shape[1])
        grads_tensor = grads_tensor[:, :lora_depth, :]

        self.kd_tree_root = KDTreeNode(
            task_indices=task_ids,
            depth=0,
            grads_tensor=grads_tensor,
            lora_depth=lora_depth,
        )

        print(f"KD Tree updated after task {task_id}:")
        print(str(self.kd_tree_root))


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    tree = KD_LoRA_Tree(num_tasks=10, lora_depth=5, reg=0.5)

    # Simulate 3 tasks being added
    for t in range(3):
        tree.new_epoch_init(train_dataloader_len=50)
        for step in range(50):
            tree.step()
            fake_grads = torch.randn(5, 256)   # 5 LoRA layers, 256-dim features
            tree.insert_grad(fake_grads)
        tree.end_task(t)

    print("Tree construction [OK]")
    device = torch.device("cpu")
    tree.new_epoch_init(10)
    tree.step()
    fake_grads = torch.randn(5, 256)
    tree.insert_grad(fake_grads)
    prev_ids = tree.tree_search(task_id=3, device=device)
    print(f"Tree search returned prev_id_matrix: {prev_ids}")
