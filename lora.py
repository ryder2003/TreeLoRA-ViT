"""
lora.py
-------
Low-Rank Adaptation (LoRA) module for Vision Transformers.

Based on the paper:
  "TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs
   Guided by a Hierarchical Gradient-Similarity Tree"

LoRA update rule (Eq. in paper):
    h = W0 * x + ΔW * x = W0 * x + B * A * x
    where  W0 ∈ R^{d×k},  B ∈ R^{d×r},  A ∈ R^{r×k},  rank r << min(d,k)

Hyperparameters (paper default):
    rank  r = 4
    alpha α = 8   →  scaling = α / r = 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a LoRA bypass:
        out = W0 * x  +  scaling * B * A * x

    The original weight W0 is frozen; only A and B are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze the original weight
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Trainable LoRA matrices  (named "loranew_A" / "loranew_B" to match
        # the official repo's gradient-collection convention)
        self.loranew_A = nn.Parameter(torch.empty(rank, in_features))
        self.loranew_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialise A with Kaiming uniform (paper / original LoRA convention)
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
        # B is zero-initialised so that ΔW = 0 at the start of training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear output (frozen)
        base_out = self.base_layer(x)
        # LoRA delta
        lora_out = F.linear(F.linear(x, self.loranew_A), self.loranew_B)
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.base_layer.in_features}, "
            f"out={self.base_layer.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.2f}"
        )


# ---------------------------------------------------------------------------
# Inject LoRA into ViT attention blocks
# ---------------------------------------------------------------------------

def inject_lora_to_vit(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    verbose: bool = True,
) -> nn.Module:
    """
    Replace the Query (W_q) and Value (W_v) projection matrices in every
    transformer block of a timm ViT with LoRA-wrapped versions.

    The timm ViT-B/16 attention block has a single fused `qkv` projection
    (nn.Linear with out_features = 3 * embed_dim).  We split it and re-wrap
    Q and V individually as LoRA layers, leaving K frozen.

    Args:
        model   : ViTBackbone (or the inner .vit timm model)
        rank    : LoRA rank r (paper default = 4)
        alpha   : LoRA alpha α (paper default = 8)
        verbose : Print injection summary

    Returns:
        model with LoRA injected in-place
    """
    # Accept both ViTBackbone and raw timm ViT
    vit = model.vit if hasattr(model, "vit") else model

    injected = 0
    for block_idx, block in enumerate(vit.blocks):
        attn = block.attn

        # timm ViT fuses Q, K, V into a single qkv linear layer
        # Shape: (3 * embed_dim, embed_dim)
        if not hasattr(attn, "qkv"):
            if verbose:
                print(f"  Block {block_idx}: no 'qkv' attribute found, skipping.")
            continue

        qkv: nn.Linear = attn.qkv
        embed_dim = qkv.in_features        # 768
        head_dim = embed_dim // attn.num_heads

        # Wrap the entire qkv with LoRA (targets Q and V positions)
        # We replace attn.qkv with a LoRAQKV that applies LoRA only to Q and V slices
        attn.qkv = LoRAQKV(qkv, rank=rank, alpha=alpha)
        injected += 1

    if verbose:
        print(f"LoRA injected into {injected} transformer blocks "
              f"(rank={rank}, alpha={alpha}, scaling={alpha/rank:.2f})")
        # Count trainable vs total parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")

    return model


class LoRAQKV(nn.Module):
    """
    Replaces the fused QKV projection in timm ViT attention.

    The original QKV linear maps  x  →  [Q | K | V]  (concatenated).
    We apply LoRA adapters to the Q and V slices only:
        Q_out = Q_weight * x  +  scaling * B_q * A_q * x
        K_out = K_weight * x                              (frozen)
        V_out = V_weight * x  +  scaling * B_v * A_v * x

    This matches the paper's setup: LoRA on W_q and W_v.
    """

    def __init__(self, base_qkv: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base_qkv = base_qkv
        self.rank = rank
        self.scaling = alpha / rank

        in_features = base_qkv.in_features     # 768
        out_features = base_qkv.out_features   # 2304  (3 * 768)
        self.slice = out_features // 3         # 768

        # Freeze original QKV
        for p in self.base_qkv.parameters():
            p.requires_grad = False

        # LoRA for Q
        self.loranew_A = nn.Parameter(torch.empty(rank, in_features))
        self.loranew_B = nn.Parameter(torch.zeros(self.slice, rank))
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))

        # LoRA for V
        self.loranew_A_v = nn.Parameter(torch.empty(rank, in_features))
        self.loranew_B_v = nn.Parameter(torch.zeros(self.slice, rank))
        nn.init.kaiming_uniform_(self.loranew_A_v, a=math.sqrt(5))

    def reset_parameters(self):
        """Re-initialize LoRA parameters (called at the start of each new task)."""
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B)
        nn.init.kaiming_uniform_(self.loranew_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original QKV  (B, N, 3*embed_dim)
        qkv = self.base_qkv(x)
        q, k, v = qkv.split(self.slice, dim=-1)

        # Apply LoRA delta to Q and V
        dq = F.linear(F.linear(x, self.loranew_A), self.loranew_B) * self.scaling
        dv = F.linear(F.linear(x, self.loranew_A_v), self.loranew_B_v) * self.scaling

        return torch.cat([q + dq, k, v + dv], dim=-1)


# ---------------------------------------------------------------------------
# LoRA parameter utilities
# ---------------------------------------------------------------------------

def get_lora_params(model: nn.Module):
    """Yield (name, param) for every LoRA parameter in the model."""
    for name, param in model.named_parameters():
        if "loranew_" in name:
            yield name, param


def reset_all_lora(model: nn.Module):
    """
    Re-initialize all LoRA adapter parameters in the model.

    Called at the start of each new task so that each task learns
    fresh LoRA deltas from the frozen pretrained backbone.  The KD-tree
    regularisation handles knowledge transfer between tasks.
    """
    vit = model.vit if hasattr(model, "vit") else model
    reset_count = 0
    for block in vit.blocks:
        attn = block.attn
        if hasattr(attn, "qkv") and isinstance(attn.qkv, LoRAQKV):
            attn.qkv.reset_parameters()
            reset_count += 1
    if reset_count > 0:
        print(f"  LoRA re-initialised in {reset_count} blocks")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from vit_backbone import ViTBackbone

    model = ViTBackbone(num_classes=100, pretrained=False)
    inject_lora_to_vit(model, rank=4, alpha=8.0)

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 100)

    # Verify only LoRA + head params are trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable param groups: {len(trainable_names)}")
    for n in trainable_names[:6]:
        print(f"  {n}")
