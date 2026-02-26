"""ReID Loss Functions.

Online hard-mining TripletLoss + ArcFace for metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OnlineTripletLoss(nn.Module):
    """Online hard-mining triplet loss.

    For each anchor, selects the hardest positive (farthest same-id)
    and hardest negative (closest different-id) within the batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings.
            labels: [B] integer identity labels.

        Returns:
            Scalar triplet loss.
        """
        # Pairwise distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        # Masks
        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        diff = ~same

        # Hardest positive: max distance among same identity
        pos_dist = dist.clone()
        pos_dist[~same] = 0.0
        hardest_pos, _ = pos_dist.max(dim=1)  # [B]

        # Hardest negative: min distance among different identity
        neg_dist = dist.clone()
        neg_dist[~diff] = float("inf")
        hardest_neg, _ = neg_dist.min(dim=1)  # [B]

        # Triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)

        # Only count anchors that have at least one positive and one negative
        valid = (same.sum(dim=1) > 1) & (diff.sum(dim=1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss[valid].mean()


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace).

    Projects embeddings onto a hypersphere and adds angular margin
    to the target logit, pushing the model to produce tighter clusters.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of identities.
        scale: Feature scale (s).
        margin: Angular margin (m) in radians.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # threshold to avoid numerical instability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings.
            labels: [B] integer identity labels (0-indexed, < num_classes).

        Returns:
            Scalar cross-entropy loss with angular margin.
        """
        # Cosine similarity
        cosine = F.linear(embeddings, F.normalize(self.weight, dim=1))  # [B, C]
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Monotonically decreasing condition
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot target
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        logits = torch.where(one_hot.bool(), phi, cosine)
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)

    def update_num_classes(self, num_classes: int):
        """Expand weight matrix for new identities (e.g., incremental learning)."""
        if num_classes <= self.num_classes:
            return
        old_w = self.weight.data
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, old_w.shape[1]))
        nn.init.xavier_uniform_(self.weight)
        self.weight.data[: old_w.shape[0]] = old_w
        self.num_classes = num_classes


class CombinedReIDLoss(nn.Module):
    """Combined TripletLoss + ArcFace loss.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of identities.
        triplet_margin: Triplet loss margin.
        arcface_scale: ArcFace scale.
        arcface_margin: ArcFace angular margin.
        triplet_weight: Weight for triplet loss.
        arcface_weight: Weight for arcface loss.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        triplet_margin: float = 0.3,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.5,
        triplet_weight: float = 1.0,
        arcface_weight: float = 0.5,
    ):
        super().__init__()
        self.triplet = OnlineTripletLoss(margin=triplet_margin)
        self.arcface = ArcFaceLoss(
            embed_dim=embed_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )
        self.triplet_weight = triplet_weight
        self.arcface_weight = arcface_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Returns:
            total_loss, dict with individual losses for logging.
        """
        loss_triplet = self.triplet(embeddings, labels)
        loss_arcface = self.arcface(embeddings, labels)

        total = self.triplet_weight * loss_triplet + self.arcface_weight * loss_arcface

        return total, {
            "triplet": loss_triplet.item(),
            "arcface": loss_arcface.item(),
            "total": total.item(),
        }
