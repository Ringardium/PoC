"""
Pet Tracker Training Module

Supports three training modes:
  1. finetune - Improve detection with existing classes (same-class refinement)
  2. expand   - Add new detection classes while preserving old knowledge
  3. distill  - Transfer knowledge from large teacher to small student model

Usage:
  python -m train finetune --data data.yaml --model weights/modelv11x.pt
  python -m train expand --old-model weights/modelv11x.pt --combined-data expanded.yaml
  python -m train distill --teacher weights/modelv11x.pt --student yolo11n.pt --data data.yaml
"""

from .config import (
    AugmentConfig,
    DistillConfig,
    ExpandConfig,
    FinetuneConfig,
    LossConfig,
    MergeConfig,
    ReIDConfig,
    TrainConfig,
    load_config,
    save_config,
)


def run_finetune(*args, **kwargs):
    """Fine-tune model on additional data with existing classes."""
    from .finetune import run_finetune as _run
    return _run(*args, **kwargs)


def run_expand(*args, **kwargs):
    """Add new detection classes while preserving old knowledge."""
    from .expand import run_expand as _run
    return _run(*args, **kwargs)


def run_distill(*args, **kwargs):
    """Transfer knowledge from large teacher to small student model."""
    from .distill import run_distill as _run
    return _run(*args, **kwargs)


def run_merge(*args, **kwargs):
    """Merge multiple datasets with automatic class index remapping."""
    from .merge import run_merge as _run
    return _run(*args, **kwargs)


def run_reid(*args, **kwargs):
    """Train pet ReID model with metric learning."""
    from .reid_train import run_reid as _run
    return _run(*args, **kwargs)


__all__ = [
    "run_finetune",
    "run_expand",
    "run_distill",
    "run_merge",
    "run_reid",
    "TrainConfig",
    "FinetuneConfig",
    "ExpandConfig",
    "DistillConfig",
    "MergeConfig",
    "ReIDConfig",
    "LossConfig",
    "AugmentConfig",
    "load_config",
    "save_config",
]
