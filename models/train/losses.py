"""Custom loss functions for knowledge distillation and class expansion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if not student_preds or not teacher_preds:
            device = student_preds[0].device if student_preds else "cpu"
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=student_preds[0].device)
        n_levels = min(len(student_preds), len(teacher_preds))

        for i in range(n_levels):
            s_pred = student_preds[i]
            t_pred = teacher_preds[i]

            # Validate tensor shape: must be [B, C, H, W] with enough channels
            min_channels = self.reg_max * 4 + self.nc
            if s_pred.dim() != 4 or s_pred.shape[1] < min_channels:
                continue
            if t_pred.dim() != 4 or t_pred.shape[1] < min_channels:
                continue

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
        """Compute feature distillation loss using channel-wise cosine similarity."""
        if not student_features or not teacher_features:
            device = self.adapters[0].weight.device if len(self.adapters) > 0 and hasattr(self.adapters[0], "weight") else "cpu"
            return torch.tensor(0.0, device=device)

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

            # Channel-wise cosine similarity loss (dimension-invariant, range [0, 2])
            s_flat = s_feat.flatten(2)  # [B, C, H*W]
            t_flat = t_feat.flatten(2).detach()
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)  # [B, C]
            loss += (1 - cos_sim).mean()
            count += 1

        return loss / max(count, 1)


class DistillationLoss:
    """Combined detection + distillation loss.

    Wraps the original task loss and adds response/feature distillation terms.
    Designed to replace model.criterion for teacher-student training.
    """

    def __init__(
        self,
        task_loss,
        student_model,
        teacher_model,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_loss: FeatureDistillationLoss | None = None,
        feature_layers: list[int] | None = None,
        student_features: dict | None = None,
        teacher_features: dict | None = None,
    ):
        self.task_loss = task_loss
        self.teacher = teacher_model
        self.alpha = alpha
        self.feature_loss_fn = feature_loss
        self.feature_layers = feature_layers or []
        self.student_features = student_features if student_features is not None else {}
        self.teacher_features = teacher_features if teacher_features is not None else {}

        # Get attributes from Detect head (works with both v8DetectionLoss and E2ELoss)
        detect_head = student_model.model[-1]
        self.stride = detect_head.stride
        self.nc = detect_head.nc
        self.no = detect_head.no
        self.reg_max = detect_head.reg_max
        self.device = next(student_model.parameters()).device

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

        # 2. Teacher forward pass — keep the output for response distillation
        # AND let registered hooks populate self.teacher_features for feature distillation.
        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])

        # 3a. Response distillation — KL on detection-head feature maps.
        # _extract_feats normalizes the various ultralytics output shapes
        # (training list, v10/v11 dict, eval-mode (decoded, raw) tuple).
        student_feats = self._extract_feats(student_preds)
        teacher_feats = self._extract_feats(teacher_preds)
        resp_loss = self.response_loss(student_feats, teacher_feats)

        # 3b. Feature distillation — L2/cosine on backbone feature hooks (optional).
        feat_loss = torch.tensor(0.0, device=self.device)
        if self.feature_loss_fn is not None and self.feature_layers:
            feat_loss = self.feature_loss_fn(
                self.student_features, self.teacher_features, self.feature_layers
            )

        # 4. Combine. Both terms are 0-d scalars and weighted equally; alpha
        # controls the task ↔ distill split as before.
        total_distill = resp_loss + feat_loss
        combined_loss = (1 - self.alpha) * task_loss + self.alpha * total_distill.expand_as(task_loss)

        # 5. Extend loss items for logging: [task_items..., distill]
        distill_item = total_distill.detach()
        if distill_item.dim() == 0:
            distill_item = distill_item.unsqueeze(0)
        extended_items = torch.cat([task_items, distill_item])

        return combined_loss, extended_items

    @staticmethod
    def _extract_feats(preds):
        """Extract raw feature map list from model predictions.

        Handles multiple ultralytics output formats:
        - Eval-mode tuple ``(decoded_tensor, raw_features_list)``
        - Training list/tuple of feature tensors per FPN level
        - v10/v11 training dict with key ``"one2many"`` or ``"feats"``
        - Single tensor (rare)
        """
        # Eval mode: (decoded_tensor, raw_features_list_or_tuple)
        if (
            isinstance(preds, tuple)
            and len(preds) == 2
            and isinstance(preds[0], torch.Tensor)
            and isinstance(preds[1], (list, tuple))
            and len(preds[1]) > 0
            and isinstance(preds[1][0], torch.Tensor)
        ):
            return list(preds[1])
        # Training mode: list/tuple of feature tensors
        if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], torch.Tensor):
            return list(preds)
        if isinstance(preds, dict):
            for key in ("one2many", "feats"):
                v = preds.get(key)
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return list(v)
            for v in preds.values():
                if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    return list(v)
        if isinstance(preds, torch.Tensor):
            return [preds]
        return []
