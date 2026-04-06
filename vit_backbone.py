"""
vit_backbone.py
---------------
Loads a ViT-B/16 pretrained on ImageNet-21k via timm.
All backbone parameters are frozen; only injected LoRA weights are trained.

Architecture (from the paper):
  - 12 Transformer encoder blocks
  - 12 attention heads
  - Embedding dimension: 768
  - MLP dimension: 3072
  - Patch size: 16 x 16  (input: 224 x 224  →  14x14 = 196 patches + 1 CLS token)
"""

import torch
import torch.nn as nn
import timm


class ViTBackbone(nn.Module):
    """
    Frozen ViT-B/16 backbone with a task-specific classification head.

    The backbone weights are loaded from ImageNet-21k pretraining.
    Only the LoRA adapter parameters and the classification head are trainable.
    """

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        # Load ViT-B/16 pretrained on ImageNet-21k
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,           # Remove the default head
        )
        self.embed_dim = self.vit.embed_dim  # 768
        self.num_heads = self.vit.blocks[0].attn.num_heads  # 12
        self.num_layers = len(self.vit.blocks)  # 12

        # Task-specific classification head (trainable)
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Freeze entire backbone by default
        self._freeze_backbone()

    # ------------------------------------------------------------------
    # Freeze / Unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self):
        """Freeze all ViT backbone parameters."""
        for name, param in self.vit.named_parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze all ViT backbone parameters (for full fine-tuning baseline)."""
        for name, param in self.vit.named_parameters():
            param.requires_grad = True

    def get_backbone_blocks(self):
        """Return the list of 12 transformer encoder blocks."""
        return self.vit.blocks

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (B, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
        """
        features = self.vit(x)   # (B, 768) – CLS token representation
        logits = self.head(features)
        return logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw CLS-token features without the classification head."""
        return self.vit(x)

    def get_trainable_params(self):
        """Return names and parameter counts of all trainable parameters."""
        trainable = [(n, p.numel()) for n, p in self.named_parameters() if p.requires_grad]
        total_trainable = sum(n for _, n in trainable)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total_trainable, total

    def print_trainable_summary(self):
        _, total_trainable, total = self.get_trainable_params()
        pct = 100 * total_trainable / total if total > 0 else 0
        print(f"Trainable parameters: {total_trainable:,} / {total:,}  ({pct:.3f}%)")


# -----------------------------------------------------------------------
# Quick sanity check
# -----------------------------------------------------------------------
if __name__ == "__main__":
    model = ViTBackbone(num_classes=100, pretrained=False)
    model.print_trainable_summary()

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # Expected: (2, 100)
