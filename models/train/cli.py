"""CLI entry point for Pet Tracker training module.

Usage:
  python -m train finetune --data data.yaml --model weights/modelv11x.pt
  python -m train expand --old-model weights/modelv11x.pt --combined-data expanded.yaml
  python -m train distill --teacher weights/modelv11x.pt --student yolo11n.pt --data data.yaml
"""

import click

from .config import (
    AugmentConfig,
    DistillConfig,
    ExpandConfig,
    FinetuneConfig,
    LossConfig,
    MergeConfig,
    ReIDConfig,
    load_config,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _get_explicit(ctx: click.Context) -> set[str]:
    """Return set of parameter names explicitly passed on the command line."""
    explicit = set()
    for param in ctx.command.params:
        source = ctx.get_parameter_source(param.name)
        if source == click.core.ParameterSource.COMMANDLINE:
            explicit.add(param.name)
    return explicit


def _apply_overrides(config, ctx: click.Context, params: dict):
    """Apply only explicitly-passed CLI values onto a config object.

    If --config was used, only CLI args the user actually typed override
    the YAML values. If no --config, all params (including defaults) apply.
    """
    explicit = _get_explicit(ctx)
    use_config_file = "config_path" in explicit or params.get("config_path") is not None
    has_config = use_config_file

    # Map of CLI param name -> (config attr, transform_fn or None)
    # Build loss and augment from individual params
    loss_keys = {"cls_loss", "box_loss", "dfl_loss"}
    augment_keys = {
        "mosaic", "copy_paste", "mixup", "erasing", "scale", "degrees",
        "translate", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "close_mosaic",
    }
    skip = loss_keys | augment_keys | {"config_path"}

    # Simple 1:1 attribute overrides
    for param_name, value in params.items():
        if param_name in skip:
            continue
        if has_config and param_name not in explicit:
            continue
        if hasattr(config, param_name) and value is not None:
            setattr(config, param_name, value)

    # Loss config
    if not has_config or (explicit & loss_keys):
        loss_dict = {}
        for k, attr in [("cls_loss", "cls"), ("box_loss", "box"), ("dfl_loss", "dfl")]:
            if not has_config or k in explicit:
                loss_dict[attr] = params[k]
            else:
                loss_dict[attr] = getattr(config.loss, attr)
        config.loss = LossConfig(**loss_dict)

    # Augment config
    if not has_config or (explicit & augment_keys):
        aug_dict = {}
        for k in augment_keys:
            if not has_config or k in explicit:
                aug_dict[k] = params[k]
            else:
                aug_dict[k] = getattr(config.augment, k)
        config.augment = AugmentConfig(**aug_dict)


# ── Shared option decorators ──────────────────────────────────────────


def common_options(f):
    """Decorator for options shared across all training modes."""
    for opt in reversed([
        click.option("--epochs", default=100, help="Number of training epochs"),
        click.option("--imgsz", default=640, help="Input image size"),
        click.option("--batch", default=16, help="Batch size"),
        click.option("--device", default=None, help="Device: auto-detect, '0' (CUDA), 'mps' (Apple), 'cpu'"),
        click.option("--workers", default=8, help="Data loading workers"),
        click.option("--project", default="runs/train", help="Project save directory"),
        click.option("--name", default=None, help="Experiment name"),
        click.option("--patience", default=50, help="Early stopping patience"),
        click.option("--optimizer", default="auto", help="Optimizer: SGD, Adam, AdamW, auto"),
        click.option("--config", "config_path", default=None, help="Path to YAML config file"),
        click.option("--resume", is_flag=True, help="Resume training from last checkpoint"),
        # Loss weights
        click.option("--cls-loss", default=0.5, help="Classification loss weight"),
        click.option("--box-loss", default=7.5, help="Box regression loss weight"),
        click.option("--dfl-loss", default=1.5, help="Distribution focal loss weight"),
        # Augmentations
        click.option("--mosaic", default=1.0, help="Mosaic augmentation (0.0-1.0)"),
        click.option("--copy-paste", default=0.3, help="Copy-paste augmentation (0.0-1.0)"),
        click.option("--mixup", default=0.15, help="MixUp augmentation (0.0-1.0)"),
        click.option("--erasing", default=0.4, help="Random erasing (0.0-1.0)"),
        click.option("--scale", default=0.9, help="Scale augmentation (+/- gain)"),
        click.option("--degrees", default=10.0, help="Rotation (+/- degrees)"),
        click.option("--translate", default=0.2, help="Translation (+/- fraction)"),
        click.option("--hsv-h", default=0.015, help="HSV-Hue augmentation"),
        click.option("--hsv-s", default=0.7, help="HSV-Saturation augmentation"),
        click.option("--hsv-v", default=0.4, help="HSV-Value augmentation"),
        click.option("--fliplr", default=0.5, help="Horizontal flip probability"),
        click.option("--flipud", default=0.0, help="Vertical flip probability"),
        click.option("--close-mosaic", default=10, help="Disable mosaic for last N epochs"),
    ]):
        f = opt(f)
    return f


# ── CLI Group ─────────────────────────────────────────────────────────


@click.group()
def cli():
    """Pet Tracker Training Tool

    \b
    Three training modes:
      finetune  Improve model with more data (same classes)
      expand    Add new detection classes to existing model
      distill   Compress large model into small model (knowledge distillation)
    """
    pass


# ── Mode 1: Fine-tuning ──────────────────────────────────────────────


@cli.command()
@click.option("--model", default="weights/modelv11x.pt", help="Base model weights path")
@click.option("--data", required=True, help="Path to data.yaml")
@click.option("--freeze-layers", default=0, help="Freeze first N model layers (0 = none)")
@click.option("--lr0", default=0.01, help="Initial learning rate")
@click.option("--lrf", default=0.01, help="Final LR as fraction of lr0")
@click.option("--warmup-epochs", default=3.0, help="Warmup epochs")
@common_options
@click.pass_context
def finetune(ctx, **params):
    """Fine-tune model on additional data with existing classes.

    \b
    Examples:
      python -m train finetune --data data.yaml --model weights/modelv11x.pt
      python -m train finetune --config expand_train_config.yaml --data data.yaml --cls-loss 0.8
    """
    from .finetune import run_finetune

    config_path = params.get("config_path")
    config = load_config(config_path, FinetuneConfig) if config_path else FinetuneConfig()

    _apply_overrides(config, ctx, params)
    run_finetune(config)


# ── Mode 2: Class Expansion ──────────────────────────────────────────


@cli.command()
@click.option("--old-model", required=True, help="Existing model with old classes")
@click.option("--combined-data", required=True, help="data.yaml with ALL classes (old + new)")
@click.option("--freeze-layers", default=10, help="Freeze first N layers (default 10)")
@click.option("--lr0", default=0.001, help="Initial learning rate (lower to preserve old knowledge)")
@common_options
@click.pass_context
def expand(ctx, **params):
    """Add new detection classes while preserving old class knowledge.

    \b
    Examples:
      python -m train expand --old-model weights/modelv11x.pt --combined-data data.yaml
      python -m train expand --config expand_train_config.yaml \\
          --old-model weights/modelv11x.pt --combined-data data.yaml
    """
    from .expand import run_expand

    config_path = params.get("config_path")
    config = load_config(config_path, ExpandConfig) if config_path else ExpandConfig()

    _apply_overrides(config, ctx, params)
    # Expand-specific required fields
    explicit = _get_explicit(ctx)
    if "old_model" in explicit:
        config.old_model = params["old_model"]
    if "combined_data" in explicit:
        config.combined_data = params["combined_data"]

    run_expand(config)


# ── Mode 3: Knowledge Distillation ───────────────────────────────────


@cli.command()
@click.option("--teacher", required=True, help="Large teacher model path")
@click.option("--student", required=True, help="Small student model path or YOLO config YAML")
@click.option("--data", required=True, help="Path to data.yaml")
@click.option("--temperature", default=4.0, help="Distillation temperature (higher = softer)")
@click.option("--alpha", default=0.5, help="Distill weight: 0.0 = task only, 1.0 = distill only")
@click.option("--distill-features/--no-distill-features", default=True, help="Enable feature distillation")
@click.option("--lr0", default=0.01, help="Initial learning rate")
@common_options
@click.pass_context
def distill(ctx, **params):
    """Transfer knowledge from large teacher to small student model.

    \b
    Examples:
      python -m train distill --teacher weights/modelv11x.pt --student yolo11n.pt --data data.yaml
      python -m train distill --config distill_config.yaml \\
          --teacher weights/modelv11x.pt --student yolo11n.pt --data data.yaml
    """
    from .distill import run_distill

    config_path = params.get("config_path")
    config = load_config(config_path, DistillConfig) if config_path else DistillConfig()

    _apply_overrides(config, ctx, params)
    # Distill-specific required fields
    explicit = _get_explicit(ctx)
    if "teacher" in explicit:
        config.teacher_model = params["teacher"]
    if "student" in explicit:
        config.student_model = params["student"]

    run_distill(config)


# ── Mode 4: Dataset Merge ────────────────────────────────────


@cli.command()
@click.option("--datasets", required=True, multiple=True, help="Paths to data.yaml files (repeat for multiple)")
@click.option("--output", default="merged", help="Output directory for merged dataset")
@click.option("--old-model", default="", help="Existing model to preserve class order")
@click.option("--preserve-splits/--resplit", default=True, help="Keep original train/val/test splits")
@click.option("--train-ratio", default=0.7, help="Train split ratio (when --resplit)")
@click.option("--val-ratio", default=0.15, help="Val split ratio (when --resplit)")
@click.option("--test-ratio", default=0.15, help="Test split ratio (when --resplit)")
@click.option("--keep-unmapped", is_flag=True, help="Keep classes not in mapping")
@click.option("--symlink", is_flag=True, help="Use symlinks instead of copying images")
@click.option("--include-classes", default="", help="Comma-separated class names to include (empty = all)")
def merge(**params):
    """Merge multiple YOLO datasets with automatic class index remapping.

    \b
    Examples:
      python -m train merge --datasets dataset/A/data.yaml --datasets dataset/B/data.yaml --output merged
      python -m train merge --datasets ds1/data.yaml --datasets ds2/data.yaml --old-model weights/modelv11x.pt
      python -m train merge --datasets ds1/data.yaml --datasets ds2/data.yaml --resplit --train-ratio 0.8
      python -m train merge --datasets ds1/data.yaml --datasets ds2/data.yaml --include-classes dog,cat
    """
    from .merge import run_merge

    include_raw = params["include_classes"].strip()
    include_list = [c.strip() for c in include_raw.split(",") if c.strip()] if include_raw else []

    config = MergeConfig(
        datasets=list(params["datasets"]),
        output_dir=params["output"],
        old_model=params["old_model"],
        preserve_splits=params["preserve_splits"],
        train_ratio=params["train_ratio"],
        val_ratio=params["val_ratio"],
        test_ratio=params["test_ratio"],
        keep_unmapped=params["keep_unmapped"],
        symlink=params["symlink"],
        include_classes=include_list,
    )
    run_merge(config)


# ── Mode 5: Dataset Prepare ─────────────────────────────────


@cli.command()
@click.option("--input", "input_dir", required=True, help="Folder with mixed image+label files")
@click.option("--output", "output_dir", required=True, help="Output directory for YOLO structure")
@click.option("--val-ratio", default=0.15, help="Validation split ratio (default 0.15)")
@click.option("--test-ratio", default=0.0, help="Test split ratio (default 0.0 = no test split)")
@click.option("--class-names", default="", help="Comma-separated class names in order (e.g. 'background,pet,person,bowl')")
@click.option("--seed", default=42, help="Random seed for reproducible splits")
@click.option("--symlink", is_flag=True, help="Use symlinks instead of copying files")
def prepare(input_dir, output_dir, val_ratio, test_ratio, class_names, seed, symlink):
    """Reorganize flat image+label folder into YOLO directory structure.

    \b
    Converts:
      flat_folder/001.jpg + 001.txt  →  output/train/images/001.jpg
                                         output/train/labels/001.txt
                                         output/val/images/...
                                         output/data.yaml

    \b
    Examples:
      python -m models.train prepare --input data/raw --output data/prepared
      python -m models.train prepare --input data/raw --output data/prepared --val-ratio 0.2
      python -m models.train prepare --input data/raw --output data/prepared --class-names "background,pet,person,bowl"
    """
    from .prepare import run_prepare

    names_dict = None
    if class_names.strip():
        names_dict = {i: name.strip() for i, name in enumerate(class_names.split(","))}

    run_prepare(
        input_dir=input_dir,
        output_dir=output_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        class_names=names_dict,
        seed=seed,
        symlink=symlink,
    )


# ── Mode 6: Dataset Gather ─────────────────────────────────


@cli.command()
@click.option("--target", required=True, help="Target data.yaml (files moved here, classes used as base)")
@click.option("--source", "sources", required=True, multiple=True, help="Source data.yaml files (repeat for multiple)")
@click.option("--keep-unmapped", is_flag=True, help="Keep label lines for unmapped classes")
@click.option("--include-classes", default="", help="Comma-separated class names to include (empty = all)")
@click.option("--dry-run", is_flag=True, help="Preview without moving files")
def gather(target, sources, keep_unmapped, include_classes, dry_run):
    """Move source dataset files into target with automatic label remapping.

    \b
    Reads data.yaml from each dataset to resolve classes. Source labels are
    remapped to match the target's class IDs, then images are MOVED (not
    copied) into the target splits. Target data.yaml is updated with any
    new classes.

    \b
    Examples:
      python -m models.train gather --target A/data.yaml --source B/data.yaml --source C/data.yaml
      python -m models.train gather --target A/data.yaml --source B/data.yaml --dry-run
      python -m models.train gather --target A/data.yaml --source B/data.yaml --include-classes dog,cat
    """
    from .prepare import run_gather

    include = [c.strip() for c in include_classes.split(",") if c.strip()] if include_classes.strip() else None

    run_gather(
        target_yaml=target,
        source_yamls=list(sources),
        keep_unmapped=keep_unmapped,
        include_classes=include,
        dry_run=dry_run,
    )


# ── Mode 7: Dataset Analyze ────────────────────────────────


@cli.command()
@click.option("--data", required=True, help="Path to data.yaml")
@click.option("--split", default="all", help="Split to analyze: train, val, test, all (default: all)")
def analyze(data, split):
    """Show class distribution (instance counts, image counts, imbalance ratio).

    \b
    Examples:
      python -m models.train analyze --data dataset/data.yaml
      python -m models.train analyze --data dataset/data.yaml --split train
    """
    from .prepare import run_analyze

    run_analyze(data_yaml=data, split=split)


# ── Mode 8: Dataset Sample ────────────────────────────────


@cli.command()
@click.option("--data", required=True, help="Path to source data.yaml")
@click.option("--output", required=True, help="Output directory for sampled dataset")
@click.option("--strategy", default="undersample", type=click.Choice(["undersample", "oversample", "cap"]),
              help="Sampling strategy (default: undersample)")
@click.option("--max-per-class", default=None, type=int, help="Max images per class (undersample/cap)")
@click.option("--min-per-class", default=None, type=int, help="Min images per class (oversample)")
@click.option("--seed", default=42, help="Random seed")
@click.option("--no-copy-val", is_flag=True, help="Skip copying val/test splits")
def sample(data, output, strategy, max_per_class, min_per_class, seed, no_copy_val):
    """Create a class-balanced dataset by sampling.

    \b
    Strategies:
      undersample  Reduce majority to match minority (or --max-per-class)
      oversample   Duplicate minority to match majority (or --min-per-class)
      cap          Limit each class to --max-per-class without duplication

    \b
    Examples:
      python -m models.train sample --data dataset/data.yaml --output balanced --strategy undersample
      python -m models.train sample --data dataset/data.yaml --output balanced --strategy cap --max-per-class 500
      python -m models.train sample --data dataset/data.yaml --output balanced --strategy oversample --min-per-class 1000
    """
    from .prepare import run_sample

    run_sample(
        data_yaml=data,
        output_dir=output,
        strategy=strategy,
        max_per_class=max_per_class,
        min_per_class=min_per_class,
        seed=seed,
        copy_val=not no_copy_val,
    )


# ── Mode 9: ReID Metric Learning ──────────────────────────────────


@cli.command()
@click.option("--data-root", required=True, help="Root folder with identity subfolders ({id}/{img.jpg})")
@click.option("--backbone", default="dinov2_vits14", type=click.Choice(["dinov2_vits14", "mobilenet_v3_small"]),
              help="Backbone model (default: dinov2_vits14)")
@click.option("--embed-dim", default=256, help="Output embedding dimension")
@click.option("--freeze-backbone", is_flag=True, help="Freeze backbone weights (train head only)")
@click.option("--epochs", default=60, help="Training epochs")
@click.option("--p", default=8, help="PK sampler: identities per batch")
@click.option("--k", default=4, help="PK sampler: images per identity")
@click.option("--lr", default=1e-4, type=float, help="Learning rate")
@click.option("--loss", "loss_type", default="combined", type=click.Choice(["triplet", "arcface", "combined"]),
              help="Loss function (default: combined)")
@click.option("--triplet-margin", default=0.3, help="Triplet loss margin")
@click.option("--arcface-scale", default=30.0, help="ArcFace scale")
@click.option("--arcface-margin", default=0.5, help="ArcFace angular margin")
@click.option("--imgsz", default=224, help="Input image size")
@click.option("--device", default=None, help="Device (auto-detect if not set)")
@click.option("--workers", default=4, help="Data loading workers")
@click.option("--project", default="runs/reid", help="Project save directory")
@click.option("--name", default="pet_reid", help="Experiment name")
@click.option("--config", "config_path", default=None, help="Path to YAML config file")
@click.option("--query-ratio", default=0.2, help="Query split ratio for evaluation")
@click.option("--save-interval", default=5, help="Evaluate & save every N epochs")
@click.option("--scheduler", default="cosine", type=click.Choice(["cosine", "step"]),
              help="LR scheduler")
@click.option("--warmup-epochs", default=5, help="Warmup epochs")
def reid(config_path, **params):
    """Train pet ReID model with metric learning (TripletLoss + ArcFace).

    \b
    Dataset structure:
      data_root/
        identity_A/
          img_001.jpg, img_002.jpg, ...
        identity_B/
          img_001.jpg, ...

    Mix of reference photos and CCTV crops recommended (ratio ~2:8).

    \b
    Examples:
      python -m models.train reid --data-root data/reid --backbone dinov2_vits14
      python -m models.train reid --data-root data/reid --backbone mobilenet_v3_small --freeze-backbone
      python -m models.train reid --data-root data/reid --loss triplet --epochs 100
      python -m models.train reid --config configs/reid_default.yaml --data-root data/reid
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from .reid_train import run_reid

    if config_path:
        config = load_config(config_path, ReIDConfig)
    else:
        config = ReIDConfig()

    # Apply CLI overrides
    direct_map = {
        "data_root": "data_root",
        "backbone": "backbone",
        "embed_dim": "embed_dim",
        "freeze_backbone": "freeze_backbone",
        "epochs": "epochs",
        "p": "p",
        "k": "k",
        "lr": "lr",
        "loss_type": "loss",
        "triplet_margin": "triplet_margin",
        "arcface_scale": "arcface_scale",
        "arcface_margin": "arcface_margin",
        "imgsz": "imgsz",
        "device": "device",
        "workers": "workers",
        "project": "project",
        "name": "name",
        "query_ratio": "query_ratio",
        "save_interval": "save_interval",
        "scheduler": "scheduler",
        "warmup_epochs": "warmup_epochs",
    }

    for cli_key, cfg_attr in direct_map.items():
        val = params.get(cli_key)
        if val is not None:
            setattr(config, cfg_attr, val)

    run_reid(config)


if __name__ == "__main__":
    cli()
