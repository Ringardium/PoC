"""Mode 2: Class expansion (add new detection classes).

Expand the detection head to support additional classes while preserving
knowledge of existing classes through weight transfer and replay training.
"""

import copy
import tempfile
from pathlib import Path

import click
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK

from .config import ExpandConfig
from .utils import build_train_args, expand_detection_head, format_results, print_banner


class ExpandTrainer(DetectionTrainer):
    """Custom trainer that uses a pre-built expanded detection model."""

    def __init__(self, expanded_model, overrides=None, _callbacks=None):
        self._expanded_model = expanded_model
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return the pre-built expanded model instead of building a new one."""
        return self._expanded_model

    def setup_model(self):
        """Override to use the already-constructed expanded model."""
        self.model = self._expanded_model
        return None


def run_expand(config: ExpandConfig) -> dict:
    """Execute class expansion training.

    Steps:
    1. Load the old model and inspect its class structure.
    2. Parse combined_data.yaml to determine new class count and names.
    3. Perform model surgery: expand Detect head cv3 layers.
    4. Train on the combined dataset (old + new classes) with backbone freezing.

    Args:
        config: Class expansion configuration.

    Returns:
        Dict with save_dir, best_model, last_model paths.
    """
    print_banner("expand", config)

    # 1. Parse combined data to get new nc and names
    data_yaml = Path(config.combined_data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Combined data YAML not found: {config.combined_data}")

    import yaml

    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)

    nc_new = data_dict["nc"]
    new_names = data_dict["names"]
    if isinstance(new_names, list):
        new_names = {i: name for i, name in enumerate(new_names)}

    click.echo(f"[INFO] Target classes ({nc_new}): {new_names}")

    # 2. Perform model surgery
    expanded_model = expand_detection_head(config.old_model, nc_new, new_names)

    # 3. Build training arguments
    train_args = build_train_args(config)
    train_args["data"] = config.combined_data
    train_args["model"] = config.old_model
    train_args["task"] = "detect"

    if config.freeze_layers > 0:
        train_args["freeze"] = config.freeze_layers
        click.echo(f"[INFO] Freezing first {config.freeze_layers} layers")

    # 4. Train with the expanded model
    click.echo("\nStarting class expansion training...")

    trainer = ExpandTrainer(
        expanded_model=expanded_model,
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
    click.echo("Class expansion training completed!")
    click.echo(f"  Best model: {summary['best_model']}")
    click.echo(f"  Classes: {new_names}")
    click.echo("=" * 60)

    return summary
