"""Shared utilities for Pet Tracker training module."""

import copy

import click
import torch
import torch.nn as nn


def print_banner(mode: str, config) -> None:
    """Print formatted training configuration banner."""
    click.echo("=" * 60)
    click.echo(f"Pet Tracker Training - {mode}")
    click.echo("=" * 60)

    fields = {
        "finetune": [
            ("Model", "model"),
            ("Data", "data"),
            ("Epochs", "epochs"),
            ("Image Size", "imgsz"),
            ("Batch Size", "batch"),
            ("Device", "device"),
            ("Learning Rate", "lr0"),
            ("Freeze Layers", "freeze_layers"),
        ],
        "expand": [
            ("Old Model", "old_model"),
            ("Combined Data", "combined_data"),
            ("Epochs", "epochs"),
            ("Image Size", "imgsz"),
            ("Batch Size", "batch"),
            ("Device", "device"),
            ("Learning Rate", "lr0"),
            ("Freeze Layers", "freeze_layers"),
        ],
        "distill": [
            ("Teacher Model", "teacher_model"),
            ("Student Model", "student_model"),
            ("Data", "data"),
            ("Epochs", "epochs"),
            ("Image Size", "imgsz"),
            ("Batch Size", "batch"),
            ("Device", "device"),
            ("Temperature", "temperature"),
            ("Alpha", "alpha"),
            ("Feature Distillation", "distill_features"),
        ],
    }

    for label, attr in fields.get(mode, []):
        if hasattr(config, attr):
            click.echo(f"  {label}: {getattr(config, attr)}")

    click.echo("")
    click.echo("Loss Weights:")
    click.echo(f"  box: {config.loss.box}  |  cls: {config.loss.cls}  |  dfl: {config.loss.dfl}")
    click.echo("=" * 60)


def build_train_args(config) -> dict:
    """Build ultralytics train() keyword arguments from config."""
    args = {
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "project": config.project,
        "name": config.name,
        "patience": config.patience,
        "resume": config.resume,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "warmup_epochs": config.warmup_epochs,
        # Loss weights
        "cls": config.loss.cls,
        "box": config.loss.box,
        "dfl": config.loss.dfl,
        # Augmentations
        "mosaic": config.augment.mosaic,
        "copy_paste": config.augment.copy_paste,
        "mixup": config.augment.mixup,
        "erasing": config.augment.erasing,
        "scale": config.augment.scale,
        "degrees": config.augment.degrees,
        "translate": config.augment.translate,
        "perspective": config.augment.perspective,
        "hsv_h": config.augment.hsv_h,
        "hsv_s": config.augment.hsv_s,
        "hsv_v": config.augment.hsv_v,
        "fliplr": config.augment.fliplr,
        "flipud": config.augment.flipud,
        "close_mosaic": config.augment.close_mosaic,
        # General
        "save": True,
        "plots": True,
        "verbose": True,
    }
    return args


def expand_detection_head(old_model_path: str, nc_new: int, new_names: dict):
    """Expand YOLO detection head to support additional classes.

    Performs model surgery on the Detect head's cv3 (classification) layers,
    expanding from nc_old to nc_new output channels while preserving all
    existing weights.

    Args:
        old_model_path: Path to existing .pt model file.
        nc_new: New total number of classes.
        new_names: Dict mapping class index to class name.

    Returns:
        Modified YOLO model with expanded detection head.
    """
    from ultralytics.nn.tasks import DetectionModel

    ckpt = torch.load(old_model_path, map_location="cpu", weights_only=False)
    old_model = (ckpt.get("ema") or ckpt["model"]).float()
    nc_old = old_model.model[-1].nc

    if nc_new <= nc_old:
        raise ValueError(
            f"New class count ({nc_new}) must be greater than old ({nc_old}). "
            f"For same-class training, use finetune mode."
        )

    click.echo(f"[INFO] Expanding detection head: {nc_old} -> {nc_new} classes")

    # Build new model from same architecture with updated nc
    yaml_cfg = copy.deepcopy(old_model.yaml)
    yaml_cfg["nc"] = nc_new

    new_model = DetectionModel(yaml_cfg, nc=nc_new, verbose=False)

    # Transfer weights
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()

    transferred = 0
    expanded = 0

    for key in new_sd:
        if key not in old_sd:
            continue

        old_param = old_sd[key]
        new_param = new_sd[key]

        if old_param.shape == new_param.shape:
            # Exact match: copy directly
            new_sd[key] = old_param
            transferred += 1
        elif old_param.dim() >= 1 and old_param.shape[0] == nc_old and new_param.shape[0] == nc_new:
            # Classification output layer (out_channels dimension)
            new_sd[key][:nc_old] = old_param
            expanded += 1
        elif old_param.dim() >= 1 and old_param.shape[0] != new_param.shape[0]:
            # Other mismatched layers: skip (keep random init)
            click.echo(f"  [SKIP] {key}: {old_param.shape} -> {new_param.shape}")
        else:
            new_sd[key] = old_param
            transferred += 1

    new_model.load_state_dict(new_sd)
    new_model.names = new_names
    new_model.nc = nc_new

    click.echo(f"  Transferred: {transferred} layers | Expanded: {expanded} layers")
    return new_model


def get_feature_channels(model, input_size: int = 640, device: str = "cpu") -> list:
    """Probe model to determine channel dimensions at each layer.

    Args:
        model: YOLO detection model.
        input_size: Input image size.
        device: Device for probing.

    Returns:
        List of output channel counts per layer.
    """
    model = model.to(device).eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    channels = []

    x = dummy
    for m in model.model:
        if hasattr(m, "f") and m.f != -1:
            # Concat or other multi-input module
            break
        x = m(x)
        if isinstance(x, torch.Tensor):
            channels.append(x.shape[1])

    return channels


def format_results(results) -> dict:
    """Format ultralytics training results into a summary dict."""
    return {
        "save_dir": str(results.save_dir),
        "best_model": str(results.save_dir / "weights" / "best.pt"),
        "last_model": str(results.save_dir / "weights" / "last.pt"),
    }
