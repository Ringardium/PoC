"""Mode 1: Fine-tuning (same-class refinement).

Improve detection quality by training the existing model on additional data
while keeping the same class structure.
"""

import click
from ultralytics import YOLO

from .config import FinetuneConfig
from .utils import build_train_args, format_results, print_banner


def run_finetune(config: FinetuneConfig) -> dict:
    """Execute fine-tuning training.

    Loads the pre-trained model and continues training on the provided dataset.
    Supports backbone freezing, custom loss weights, and full augmentation control.

    Args:
        config: Fine-tuning configuration.

    Returns:
        Dict with save_dir, best_model, last_model paths.
    """
    print_banner("finetune", config)

    model = YOLO(config.model)
    click.echo(f"[INFO] Loaded model: {config.model}")
    click.echo(f"[INFO] Classes: {model.names}")

    train_args = build_train_args(config)
    train_args["data"] = config.data

    if config.freeze_layers > 0:
        train_args["freeze"] = config.freeze_layers
        click.echo(f"[INFO] Freezing first {config.freeze_layers} layers")

    click.echo("\nStarting fine-tuning...")
    results = model.train(**train_args)

    summary = format_results(results)
    click.echo("\n" + "=" * 60)
    click.echo("Fine-tuning completed!")
    click.echo(f"  Best model: {summary['best_model']}")
    click.echo("=" * 60)

    return summary
