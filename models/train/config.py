"""Configuration dataclasses for Pet Tracker training module."""

import warnings
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import List, Optional

import torch
import yaml


def _detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LossConfig:
    """Loss weight configuration for YOLO training."""

    box: float = 7.5  # Box regression loss gain
    cls: float = 0.5  # Classification loss gain
    dfl: float = 1.5  # Distribution focal loss gain


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""

    mosaic: float = 1.0
    copy_paste: float = 0.3
    mixup: float = 0.15
    erasing: float = 0.4
    scale: float = 0.9
    degrees: float = 10.0
    translate: float = 0.2
    perspective: float = 0.0005
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    close_mosaic: int = 10


@dataclass
class TrainConfig:
    """Base training configuration shared across all modes."""

    model: str = ""
    data: str = ""
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = field(default_factory=_detect_device)
    workers: int = 8
    project: str = "runs/train"
    name: str = "pet_tracker"
    patience: int = 50
    resume: bool = False
    optimizer: str = "auto"
    iou: float = 0.7  # NMS IoU threshold for validation
    conf: float = 0.001  # Confidence threshold for validation
    lr0: float = 0.01
    lrf: float = 0.01
    warmup_epochs: float = 3.0
    loss: LossConfig = field(default_factory=LossConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class FinetuneConfig(TrainConfig):
    """Fine-tuning configuration (same-class refinement)."""

    freeze_layers: int = 0  # Number of layers to freeze from start (0 = none)

    def __post_init__(self):
        if not self.name or self.name == "pet_tracker":
            self.name = "finetune"


@dataclass
class ExpandConfig(TrainConfig):
    """Class expansion configuration."""

    old_model: str = ""  # Path to existing model with old classes
    combined_data: str = ""  # data.yaml with ALL classes (old + new)
    freeze_layers: int = 10  # Freeze first N layers to preserve old knowledge
    lr0: float = 0.001  # Lower LR to prevent catastrophic forgetting

    def __post_init__(self):
        if not self.name or self.name == "pet_tracker":
            self.name = "expand"


@dataclass
class DistillConfig(TrainConfig):
    """Knowledge distillation configuration."""

    teacher_model: str = ""  # Path to large teacher model
    student_model: str = ""  # Path to small student model or YOLO config
    temperature: float = 4.0  # Distillation temperature
    alpha: float = 0.5  # Balance: 0 = task only, 1 = distill only
    distill_features: bool = True  # Enable intermediate feature distillation
    feature_layers: List[int] = field(default_factory=list)  # Layers to distill (empty = auto)

    def __post_init__(self):
        if not self.name or self.name == "pet_tracker":
            self.name = "distill"


@dataclass
class MergeConfig:
    """Dataset merge configuration."""

    datasets: List[str] = field(default_factory=list)  # data.yaml paths
    output_dir: str = "merged"
    old_model: str = ""  # Existing model to preserve class order
    preserve_splits: bool = True  # Keep original train/val/test splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    keep_unmapped: bool = False  # Keep classes not in mapping
    symlink: bool = False  # Use symlinks instead of copying
    include_classes: List[str] = field(default_factory=list)  # Only include these class names (empty = all)


@dataclass
class ReIDConfig:
    """ReID metric learning configuration."""

    # 데이터
    data_root: str = ""  # {identity}/{img.jpg, ...} 구조 폴더
    query_ratio: float = 0.2  # 평가 시 query/gallery 분할 비율

    # 모델
    backbone: str = "dinov2_vits14"  # dinov2_vits14, mobilenet_v3_small
    embed_dim: int = 256  # 최종 임베딩 차원
    freeze_backbone: bool = False  # backbone 가중치 고정
    pretrained: bool = True

    # 학습
    epochs: int = 60
    p: int = 8  # PK sampler: P identities per batch
    k: int = 4  # PK sampler: K images per identity
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # cosine, step
    step_size: int = 20
    step_gamma: float = 0.5
    device: str = field(default_factory=_detect_device)
    workers: int = 4
    imgsz: int = 224  # 입력 이미지 크기

    # 손실 함수
    loss: str = "combined"  # triplet, arcface, combined
    triplet_margin: float = 0.3
    arcface_scale: float = 30.0
    arcface_margin: float = 0.5
    triplet_weight: float = 1.0
    arcface_weight: float = 0.5

    # Augmentation
    random_crop: bool = True
    color_jitter: float = 0.3
    random_erasing: float = 0.5
    horizontal_flip: float = 0.5

    # 저장
    project: str = "runs/reid"
    name: str = "pet_reid"
    save_interval: int = 5

    def __post_init__(self):
        if self.backbone not in ("dinov2_vits14", "mobilenet_v3_small"):
            raise ValueError(f"Unsupported backbone: {self.backbone}")


def load_config(yaml_path: str, config_class: type) -> TrainConfig:
    """Load configuration from a YAML file.

    Unknown YAML keys are dropped with a warning instead of raising —
    config schemas evolve over time and old yaml files shouldn't crash a run.
    Typos are still surfaced (just not fatally).

    Args:
        yaml_path: Path to YAML config file.
        config_class: Target dataclass type (FinetuneConfig, ExpandConfig, DistillConfig).

    Returns:
        Populated config dataclass instance.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Handle device: auto
    if raw.get("device") == "auto":
        raw["device"] = _detect_device()

    # Handle nested configs (only for TrainConfig subclasses that use LossConfig/AugmentConfig)
    loss_val = raw.pop("loss", None)
    augment_dict = raw.pop("augment", None)

    # Drop unknown keys before constructing the dataclass — emit one warning
    # listing them so the user can spot typos without the run dying.
    known_fields = {f.name for f in fields(config_class)}
    unknown = [k for k in raw if k not in known_fields]
    if unknown:
        valid_hint = ", ".join(sorted(known_fields))
        warnings.warn(
            f"[load_config] {yaml_path}: ignoring unknown keys {unknown} for "
            f"{config_class.__name__}. Valid keys: {valid_hint}",
            stacklevel=2,
        )
        for k in unknown:
            raw.pop(k, None)

    config = config_class(**raw)

    if loss_val is not None and isinstance(loss_val, dict) and hasattr(config, "loss"):
        config.loss = LossConfig(**loss_val)
    elif loss_val is not None and isinstance(loss_val, str) and hasattr(config, "loss"):
        config.loss = loss_val
    if augment_dict is not None and isinstance(augment_dict, dict) and hasattr(config, "augment"):
        config.augment = AugmentConfig(**augment_dict)

    return config


def save_config(config: TrainConfig, yaml_path: str) -> None:
    """Save configuration to a YAML file."""
    data = asdict(config)
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
