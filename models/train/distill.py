"""Mode 3: Knowledge distillation (teacher-student training).

Transfer knowledge from a large teacher model to a smaller student model
using response-based and feature-based distillation.
"""

import copy

import click
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    # Removed in newer ultralytics versions
    def de_parallel(model):
        return model.module if hasattr(model, "module") else model

from .config import DistillConfig
from .losses import DistillationLoss, FeatureDistillationLoss
from .utils import build_train_args, print_banner

try:
    from ultralytics.cfg import get_cfg, DEFAULT_CFG
except ImportError:
    DEFAULT_CFG = None
    get_cfg = None


class DistillationTrainer(DetectionTrainer):
    """Custom trainer that adds knowledge distillation loss.

    During each training step:
    1. Forward pass through student model (normal, with gradients)
    2. Forward pass through teacher model (no gradients)
    3. Compute: total = (1-alpha)*task_loss + alpha*distill_loss
    """

    def __init__(
        self,
        teacher_path: str,
        temperature: float = 4.0,
        alpha: float = 0.5,
        distill_features: bool = True,
        feature_layers: list[int] | None = None,
        cfg=None,
        overrides=None,
        _callbacks=None,
    ):
        self.teacher_path = teacher_path
        self.temperature = temperature
        self.alpha = alpha
        self.distill_features = distill_features
        self.feature_layers = feature_layers or []
        self._teacher = None
        self._student_features = {}
        self._teacher_features = {}
        # Newer ultralytics requires cfg to be non-None
        if cfg is None and DEFAULT_CFG is not None:
            cfg = DEFAULT_CFG
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def _setup_train(self, world_size=1):
        """Extended setup: load teacher and inject distillation loss."""
        try:
            super()._setup_train(world_size)
        except TypeError:
            # Newer ultralytics: _setup_train() takes no arguments
            super()._setup_train()

        # Load teacher model
        self._teacher = self._load_teacher()

        # Set up feature hooks if enabled
        feature_loss_fn = None
        if self.distill_features:
            feature_loss_fn = self._setup_feature_distillation()

        # Replace student model's criterion with distillation loss
        student = de_parallel(self.model)
        student.criterion = DistillationLoss(
            student_model=student,
            teacher_model=self._teacher,
            temperature=self.temperature,
            alpha=self.alpha,
            feature_loss=feature_loss_fn,
            feature_layers=self.feature_layers,
            student_features=self._student_features,
            teacher_features=self._teacher_features,
        )

        click.echo(f"[INFO] Distillation loss injected (T={self.temperature}, alpha={self.alpha})")

    def _load_teacher(self) -> nn.Module:
        """Load and freeze teacher model."""
        click.echo(f"[INFO] Loading teacher model: {self.teacher_path}")

        ckpt = torch.load(self.teacher_path, map_location="cpu", weights_only=False)
        teacher = (ckpt.get("ema") or ckpt["model"]).float()
        teacher = teacher.to(self.device).eval()

        for param in teacher.parameters():
            param.requires_grad = False

        click.echo(f"[INFO] Teacher loaded: {teacher.nc} classes, frozen")
        return teacher

    def _setup_feature_distillation(self) -> FeatureDistillationLoss | None:
        """Register forward hooks for intermediate feature distillation."""
        student = de_parallel(self.model)

        # Auto-detect layers feeding into Detect head
        if not self.feature_layers:
            detect = student.model[-1]
            if hasattr(detect, "f"):
                f = detect.f
                self.feature_layers = f if isinstance(f, list) else [f]
            else:
                click.echo("[WARN] Cannot auto-detect feature layers, skipping feature distillation")
                return None

        click.echo(f"[INFO] Feature distillation layers: {self.feature_layers}")

        # Probe channel dimensions
        student_channels = []
        teacher_channels = []

        for idx in self.feature_layers:
            # Get output channels from student layer
            s_layer = student.model[idx]
            s_ch = self._get_out_channels(s_layer)
            student_channels.append(s_ch)

            # Get output channels from teacher layer
            t_layer = self._teacher.model[idx]
            t_ch = self._get_out_channels(t_layer)
            teacher_channels.append(t_ch)

        click.echo(f"  Student channels: {student_channels}")
        click.echo(f"  Teacher channels: {teacher_channels}")

        # Register hooks
        for idx in self.feature_layers:
            student.model[idx].register_forward_hook(self._make_hook(self._student_features, idx))
            self._teacher.model[idx].register_forward_hook(self._make_hook(self._teacher_features, idx))

        # Create feature loss with adapters
        feature_loss = FeatureDistillationLoss(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            device=str(self.device),
        )

        # Add adapter parameters to optimizer
        if any(p.requires_grad for p in feature_loss.parameters()):
            adapter_params = [p for p in feature_loss.parameters() if p.requires_grad]
            self.optimizer.add_param_group({"params": adapter_params, "weight_decay": 0.0})
            click.echo(f"  Added {len(adapter_params)} adapter parameters to optimizer")

        return feature_loss

    @staticmethod
    def _get_out_channels(layer) -> int:
        """Extract output channel count from a model layer."""
        if hasattr(layer, "cv2"):
            # Bottleneck or C2f module
            return layer.cv2.conv.out_channels
        if hasattr(layer, "conv"):
            return layer.conv.out_channels
        if hasattr(layer, "out_channels"):
            return layer.out_channels
        # Fallback: inspect last conv in module
        last_conv = None
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv.out_channels
        return 256  # Default fallback

    @staticmethod
    def _make_hook(storage: dict, idx: int):
        """Create a forward hook that stores output feature maps."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[idx] = output
        return hook

    def get_validator(self):
        """Return validator with extended loss names for distillation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "distill_loss"
        validator = super().get_validator()
        # Reset parent's loss_names back so validator works correctly
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label loss items including distillation loss."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        """Return a formatted training progress string with distillation loss."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )


def run_distill(config: DistillConfig) -> dict:
    """Execute knowledge distillation training.

    Args:
        config: Distillation configuration.

    Returns:
        Dict with save_dir, best_model, last_model paths.
    """
    print_banner("distill", config)

    # Build training arguments
    train_args = build_train_args(config)
    train_args["data"] = config.data
    train_args["model"] = config.student_model

    click.echo("\nStarting knowledge distillation...")

    trainer = DistillationTrainer(
        teacher_path=config.teacher_model,
        temperature=config.temperature,
        alpha=config.alpha,
        distill_features=config.distill_features,
        feature_layers=config.feature_layers,
        overrides=train_args,
    )
    trainer.train()

    save_dir = trainer.save_dir
    summary = {
        "save_dir": str(save_dir),
        "best_model": str(save_dir / "weights" / "best.pt"),
        "last_model": str(save_dir / "weights" / "last.pt"),
    }

    click.echo("\n" + "=" * 60)
    click.echo("Knowledge distillation completed!")
    click.echo(f"  Best student model: {summary['best_model']}")
    click.echo("=" * 60)

    return summary
