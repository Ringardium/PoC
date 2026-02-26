"""Custom loss functions for knowledge distillation and class expansion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss


class ResponseDistillationLoss(nn.Module):
    """KL-divergence loss on detection head outputs with temperature scaling.

    Computes soft-target distillation on both classification logits and
    box regression (DFL) distributions from the Detect head.
    """

    def __init__(self, temperature: float = 4.0, nc: int = 80, reg_max: int = 16):
        super().__init__()
        self.T = temperature
        self.nc = nc
        self.reg_max = reg_max

    def forward(
        self, student_preds: list[torch.Tensor], teacher_preds: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute response distillation loss.

        Args:
            student_preds: List of [B, no, H, W] from student Detect head.
            teacher_preds: List of [B, no, H, W] from teacher Detect head.

        Returns:
            Scalar distillation loss.
        """
        loss = torch.tensor(0.0, device=student_preds[0].device)
        n_levels = min(len(student_preds), len(teacher_preds))

        for i in range(n_levels):
            s_pred = student_preds[i]
            t_pred = teacher_preds[i]

            # Handle spatial size mismatch
            if s_pred.shape[2:] != t_pred.shape[2:]:
                t_pred = F.interpolate(t_pred, size=s_pred.shape[2:], mode="bilinear", align_corners=False)

            # Classification logits: last nc channels
            s_cls = s_pred[:, self.reg_max * 4 :].permute(0, 2, 3, 1).reshape(-1, self.nc)
            t_cls = t_pred[:, self.reg_max * 4 :].permute(0, 2, 3, 1).reshape(-1, self.nc)

            s_log_soft = F.log_softmax(s_cls / self.T, dim=-1)
            t_soft = F.softmax(t_cls / self.T, dim=-1)
            loss += F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (self.T ** 2)

            # Box regression DFL distribution matching
            s_box = s_pred[:, : self.reg_max * 4].permute(0, 2, 3, 1).reshape(-1, 4, self.reg_max)
            t_box = t_pred[:, : self.reg_max * 4].permute(0, 2, 3, 1).reshape(-1, 4, self.reg_max)
            s_box_soft = F.log_softmax(s_box / self.T, dim=-1)
            t_box_soft = F.softmax(t_box / self.T, dim=-1)
            loss += F.kl_div(s_box_soft, t_box_soft, reduction="batchmean") * (self.T ** 2) * 0.25

        return loss / max(n_levels, 1)


class FeatureDistillationLoss(nn.Module):
    """L2 loss between student and teacher intermediate feature maps.

    Handles channel dimension mismatch via learnable 1x1 conv adapters and
    spatial mismatch via adaptive average pooling.
    """

    def __init__(self, student_channels: list[int], teacher_channels: list[int], device: str = "cpu"):
        super().__init__()
        self.adapters = nn.ModuleList()
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            if s_ch != t_ch:
                self.adapters.append(nn.Conv2d(s_ch, t_ch, 1, bias=False).to(device))
            else:
                self.adapters.append(nn.Identity())

    def forward(
        self, student_features: dict[int, torch.Tensor], teacher_features: dict[int, torch.Tensor], layer_indices: list[int]
    ) -> torch.Tensor:
        """Compute feature distillation loss.

        Args:
            student_features: Dict of {layer_idx: feature_tensor} from student.
            teacher_features: Dict of {layer_idx: feature_tensor} from teacher.
            layer_indices: List of layer indices to distill.

        Returns:
            Scalar feature distillation loss.
        """
        loss = torch.tensor(0.0, device=next(iter(student_features.values())).device)
        count = 0

        for i, idx in enumerate(layer_indices):
            if idx not in student_features or idx not in teacher_features:
                continue

            s_feat = student_features[idx]
            t_feat = teacher_features[idx]

            # Adapt channels
            if i < len(self.adapters):
                s_feat = self.adapters[i](s_feat)

            # Handle spatial mismatch
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])

            # Normalize and compute L2
            s_norm = F.normalize(s_feat.flatten(2), dim=-1)
            t_norm = F.normalize(t_feat.flatten(2), dim=-1)
            loss += F.mse_loss(s_norm, t_norm.detach())
            count += 1

        return loss / max(count, 1)


class DistillationLoss:
    """Combined detection + distillation loss.

    Wraps v8DetectionLoss and adds response/feature distillation terms.
    Designed to replace model.criterion for teacher-student training.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_loss: FeatureDistillationLoss | None = None,
        feature_layers: list[int] | None = None,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ):
        self.task_loss = v8DetectionLoss(student_model)
        self.teacher = teacher_model
        self.alpha = alpha
        self.feature_loss_fn = feature_loss
        self.feature_layers = feature_layers or []
        self.student_features = student_features or {}
        self.teacher_features = teacher_features or {}

        # Proxy attributes needed by the trainer
        self.stride = self.task_loss.stride
        self.nc = self.task_loss.nc
        self.no = self.task_loss.no
        self.reg_max = self.task_loss.reg_max
        self.device = self.task_loss.device

        self.response_loss = ResponseDistillationLoss(
            temperature=temperature,
            nc=self.nc,
            reg_max=self.reg_max,
        )

    def __call__(
        self, student_preds, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute combined task + distillation loss.

        Args:
            student_preds: Student model predictions (from Detect head).
            batch: Training batch dict with 'img', 'cls', 'bboxes', etc.

        Returns:
            (loss_sum, loss_items): Scaled loss tensor and detached items for logging.
        """
        # 1. Standard task loss (student vs ground truth)
        task_loss, task_items = self.task_loss(student_preds, batch)

        # 2. Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])

        # 3. Extract raw feature maps for response distillation
        # Student (training mode): preds is already [feat0, feat1, feat2]
        s_feats = student_preds if isinstance(student_preds, list) else [student_preds]
        # Teacher (eval mode): preds is (decoded_tensor, [feat0, feat1, feat2])
        if isinstance(teacher_preds, tuple) and len(teacher_preds) == 2:
            t_feats = teacher_preds[1]
            if not isinstance(t_feats, list):
                t_feats = [t_feats]
        else:
            t_feats = teacher_preds if isinstance(teacher_preds, list) else [teacher_preds]

        # 4. Response distillation loss
        distill_loss = self.response_loss(s_feats, t_feats)

        # 5. Feature distillation loss (optional)
        feat_loss = torch.tensor(0.0, device=self.device)
        if self.feature_loss_fn is not None and self.feature_layers:
            feat_loss = self.feature_loss_fn(
                self.student_features, self.teacher_features, self.feature_layers
            )

        # 6. Combine: (1-alpha)*task + alpha*(response + feature)
        total_distill = distill_loss + feat_loss
        combined_loss = (1 - self.alpha) * task_loss + self.alpha * total_distill.expand_as(task_loss)

        # 7. Extend loss items for logging: [box, cls, dfl, distill]
        distill_item = total_distill.detach()
        if distill_item.dim() == 0:
            distill_item = distill_item.unsqueeze(0)
        extended_items = torch.cat([task_items, distill_item])

        return combined_loss, extended_items
