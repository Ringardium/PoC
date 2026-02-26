"""ReID Metric Learning Model.

Backbone (DINOv2 / MobileNetV3) + Projection Head for pet re-identification.
Trained with metric learning (TripletLoss + ArcFace) to produce embeddings
that are close for same-identity pets and far for different ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ProjectionHead(nn.Module):
    """MLP projection head: backbone_dim -> embed_dim.

    Structure: Linear -> BN -> ReLU -> Linear -> BN
    """

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        hidden = max(in_dim, embed_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReIDModel(nn.Module):
    """Pet ReID model: frozen/trainable backbone + projection head.

    Args:
        backbone_name: 'dinov2_vits14' or 'mobilenet_v3_small'
        embed_dim: Output embedding dimension.
        freeze_backbone: If True, backbone weights are frozen.
        pretrained: Load pretrained weights for backbone.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        embed_dim: int = 256,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim

        self.backbone, self.backbone_dim = self._build_backbone(
            backbone_name, pretrained
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = ProjectionHead(self.backbone_dim, embed_dim)

    @staticmethod
    def _build_backbone(name: str, pretrained: bool):
        """Build backbone and return (module, output_dim)."""
        if name == "dinov2_vits14":
            model = torch.hub.load(
                "facebookresearch/dinov2", name, pretrained=pretrained, verbose=False
            )
            out_dim = model.embed_dim  # 384 for vits14
            return model, out_dim

        elif name == "mobilenet_v3_small":
            from torchvision import models as tv_models

            weights = (
                tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            model = tv_models.mobilenet_v3_small(weights=weights)
            out_dim = model.classifier[0].in_features  # 576
            model.classifier = nn.Identity()
            return model, out_dim

        else:
            raise ValueError(f"Unknown backbone: {name}")

    def forward(
        self, x: torch.Tensor, return_backbone: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, 3, H, W].
            return_backbone: If True, return raw backbone features (no head).

        Returns:
            L2-normalized embeddings [B, embed_dim].
        """
        feat = self.backbone(x)

        if return_backbone:
            return F.normalize(feat, p=2, dim=1)

        emb = self.head(feat)
        return F.normalize(emb, p=2, dim=1)

    def get_embedding_dim(self) -> int:
        return self.embed_dim

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only embedding extraction."""
        self.eval()
        return self.forward(x)

    def save_inference(self, path: str):
        """Save only backbone + head weights for inference."""
        torch.save(
            {
                "backbone_name": self.backbone_name,
                "embed_dim": self.embed_dim,
                "backbone_dim": self.backbone_dim,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_inference(cls, path: str, device: str = "cpu") -> "ReIDModel":
        """Load inference-ready model."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            backbone_name=ckpt["backbone_name"],
            embed_dim=ckpt["embed_dim"],
            freeze_backbone=True,
            pretrained=False,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
