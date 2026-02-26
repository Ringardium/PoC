"""Dataset merge tool for combining multiple YOLO datasets with class index remapping.

Automatically parses data.yaml files, builds unified class mapping,
remaps label file class indices, and generates a combined dataset.
"""

import os
import random
import shutil
from pathlib import Path

import click
import torch
import yaml


def _resolve_split_dir(img_path_str: str, base_dir: Path, data: dict) -> Path | None:
    """Try multiple strategies to resolve a split image directory path.

    Handles various YOLO data.yaml path conventions:
    - Absolute paths
    - Relative to data.yaml parent dir
    - Relative to 'path' field
    - Relative to 'path' field resolved from data.yaml parent
    """
    candidates = []
    img_path = Path(img_path_str)

    if img_path.is_absolute():
        candidates.append(img_path)
    else:
        # Strategy 1: Relative to data.yaml directory
        candidates.append(base_dir / img_path)

        # Strategy 2: Relative to 'path' field (if present)
        if "path" in data:
            path_field = Path(data["path"])
            if path_field.is_absolute():
                candidates.append(path_field / img_path)
            else:
                candidates.append(base_dir / path_field / img_path)

        # Strategy 3: Strip leading "../" and try from base_dir
        # Handles Roboflow pattern: data.yaml has "../train/images" but actual
        # structure is data.yaml + train/ in same directory
        stripped = img_path
        while str(stripped).startswith(".."):
            stripped = Path(*stripped.parts[1:]) if len(stripped.parts) > 1 else stripped
            candidates.append(base_dir / stripped)

        # Strategy 4: Just the last component (e.g. "images" from "../train/images")
        candidates.append(base_dir / img_path.name)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    return None


def _parse_data_yaml(yaml_path: str) -> dict:
    """Parse a YOLO data.yaml and return normalized structure.

    Returns:
        Dict with keys: path, names (dict {id: name}), nc, splits (dict of split -> images/labels dirs).
    """
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base_dir = yaml_path.parent

    # Normalize names to {int: str} dict
    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    # Handle 'valid' vs 'val' key (Roboflow uses 'valid')
    split_keys = {
        "train": data.get("train"),
        "val": data.get("val") or data.get("valid"),
        "test": data.get("test"),
    }

    # Resolve split paths
    splits = {}
    for split_name, img_path_str in split_keys.items():
        if not img_path_str:
            continue

        img_dir = _resolve_split_dir(img_path_str, base_dir, data)

        if img_dir is None:
            click.echo(f"  [WARN] {split_name} directory not found: {img_path_str}")
            click.echo(f"         Searched from: {base_dir}")
            if "path" in data:
                click.echo(f"         data.yaml 'path' field: {data['path']}")
            continue

        # Derive labels dir (sibling directory)
        label_dir = img_dir.parent / "labels"
        if not label_dir.exists():
            click.echo(f"  [WARN] labels directory not found: {label_dir}")
            continue

        splits[split_name] = {"images": img_dir, "labels": label_dir}

    return {
        "path": str(base_dir),
        "names": names,
        "nc": data.get("nc", len(names)),
        "splits": splits,
        "source_yaml": str(yaml_path),
    }


def _load_model_classes(model_path: str) -> dict:
    """Extract class names from a YOLO model checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt["model"]
    names = getattr(model, "names", None)
    if names is None:
        names = ckpt.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def _build_unified_mapping(
    datasets: list[dict], base_classes: dict, include_classes: list[str] | None = None
) -> tuple[dict, dict]:
    """Build unified class mapping across all datasets.

    Args:
        datasets: List of parsed data.yaml dicts.
        base_classes: Existing classes to preserve (from model or first dataset).
        include_classes: If provided, only these class names are included.
                         Classes not in this list are skipped (not mapped).

    Returns:
        (unified_classes, per_dataset_mappings)
        unified_classes: {unified_id: name}
        per_dataset_mappings: {dataset_index: {original_id: unified_id}}
    """
    # Filter base_classes if include_classes is set
    if include_classes:
        include_set = set(include_classes)
        unified = {k: v for k, v in base_classes.items() if v in include_set}
    else:
        unified = dict(base_classes)

    name_to_id = {name: idx for idx, name in unified.items()}
    next_id = max(unified.keys()) + 1 if unified else 0

    per_dataset = {}

    for i, ds in enumerate(datasets):
        mapping = {}
        for org_id, cls_name in ds["names"].items():
            if include_classes and cls_name not in include_set:
                continue
            if cls_name in name_to_id:
                mapping[org_id] = name_to_id[cls_name]
            else:
                mapping[org_id] = next_id
                unified[next_id] = cls_name
                name_to_id[cls_name] = next_id
                next_id += 1
        per_dataset[i] = mapping

    return unified, per_dataset


def _remap_label_file(label_path: Path, class_mapping: dict, keep_unmapped: bool) -> list[str]:
    """Read a YOLO label file and remap class indices.

    Returns:
        List of remapped label lines.
    """
    lines = []
    seen = set()

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            org_cls = int(parts[0])

            if org_cls in class_mapping:
                parts[0] = str(class_mapping[org_cls])
                new_line = " ".join(parts)
                if new_line not in seen:
                    seen.add(new_line)
                    lines.append(new_line)
            elif keep_unmapped:
                new_line = " ".join(parts)
                if new_line not in seen:
                    seen.add(new_line)
                    lines.append(new_line)

    return lines


def _collect_image_files(image_dir: Path) -> list[Path]:
    """Collect all image files from a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [f for f in image_dir.iterdir() if f.suffix.lower() in exts]
    return sorted(files)


def run_merge(config) -> dict:
    """Execute dataset merge.

    Args:
        config: MergeConfig instance.

    Returns:
        Dict with output_dir, data_yaml, unified_classes, stats.
    """
    output_dir = Path(config.output_dir)

    click.echo("=" * 60)
    click.echo("Dataset Merge Tool")
    click.echo("=" * 60)

    # 1. Parse all data.yaml files
    datasets = []
    for ds_path in config.datasets:
        ds = _parse_data_yaml(ds_path)
        datasets.append(ds)
        click.echo(f"\n[Dataset] {Path(ds_path).parent.name}")
        click.echo(f"  Classes ({ds['nc']}): {ds['names']}")
        click.echo(f"  Splits: {list(ds['splits'].keys())}")

    # 2. Load base classes from model or empty
    base_classes = {}
    if config.old_model:
        base_classes = _load_model_classes(config.old_model)
        click.echo(f"\n[Model] {config.old_model}")
        click.echo(f"  Classes: {base_classes}")

    # 3. Build unified class mapping
    include = config.include_classes if config.include_classes else None
    unified_classes, per_dataset_mappings = _build_unified_mapping(datasets, base_classes, include)

    if include:
        click.echo(f"\n[Filter] Including only: {config.include_classes}")

    click.echo("\n" + "=" * 60)
    click.echo("Unified Class Mapping")
    click.echo("=" * 60)
    for ds_idx, ds in enumerate(datasets):
        ds_name = Path(ds["source_yaml"]).parent.name
        mapping = per_dataset_mappings[ds_idx]
        click.echo(f"\n  [{ds_name}]")
        for org_id, new_id in sorted(mapping.items()):
            org_name = ds["names"][org_id]
            click.echo(f"    {org_id} ({org_name}) -> {new_id}")

    click.echo(f"\n  Result ({len(unified_classes)} classes):")
    for idx in sorted(unified_classes.keys()):
        status = "existing" if idx in base_classes else "new"
        click.echo(f"    {idx}: {unified_classes[idx]} ({status})")
    click.echo("=" * 60)

    # 4. Create output directories
    splits_to_create = ["train", "val", "test"]
    for split in splits_to_create:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 5. Process each dataset
    total_count = 0
    class_stats = {
        split: {cls_id: 0 for cls_id in unified_classes}
        for split in splits_to_create
    }
    split_stats = {split: 0 for split in splits_to_create}

    for ds_idx, ds in enumerate(datasets):
        ds_name = Path(ds["source_yaml"]).parent.name
        mapping = per_dataset_mappings[ds_idx]

        click.echo(f"\nProcessing: {ds_name}")

        if config.preserve_splits:
            # Keep original train/val/test splits
            for split_name, split_dirs in ds["splits"].items():
                if split_name not in splits_to_create:
                    continue
                img_dir = split_dirs["images"]
                lbl_dir = split_dirs["labels"]

                images = _collect_image_files(img_dir)
                processed = 0

                for img_path in images:
                    label_path = lbl_dir / f"{img_path.stem}.txt"
                    if not label_path.exists():
                        continue

                    # Remap labels
                    remapped = _remap_label_file(label_path, mapping, config.keep_unmapped)
                    if not remapped:
                        continue

                    # Unique filename
                    new_name = f"{ds_name}_{total_count:06d}{img_path.suffix}"
                    total_count += 1
                    processed += 1
                    split_stats[split_name] += 1

                    # Copy or symlink image
                    dst_img = output_dir / split_name / "images" / new_name
                    if config.symlink:
                        dst_img.symlink_to(img_path.resolve())
                    else:
                        shutil.copy2(img_path, dst_img)

                    # Write remapped label
                    dst_label = output_dir / split_name / "labels" / f"{Path(new_name).stem}.txt"
                    with open(dst_label, "w") as f:
                        f.write("\n".join(remapped) + "\n")

                    # Update stats
                    for line in remapped:
                        cls_id = int(line.split()[0])
                        class_stats[split_name][cls_id] += 1

                click.echo(f"  {split_name}: {processed} images")
        else:
            # Collect all images then re-split
            all_images = []
            for split_dirs in ds["splits"].values():
                img_dir = split_dirs["images"]
                lbl_dir = split_dirs["labels"]
                for img_path in _collect_image_files(img_dir):
                    label_path = lbl_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        all_images.append((img_path, label_path))

            random.shuffle(all_images)
            n = len(all_images)
            train_end = int(n * config.train_ratio)
            val_end = train_end + int(n * config.val_ratio)

            split_assignments = (
                [("train", img_lbl) for img_lbl in all_images[:train_end]]
                + [("val", img_lbl) for img_lbl in all_images[train_end:val_end]]
                + [("test", img_lbl) for img_lbl in all_images[val_end:]]
            )

            for split_name, (img_path, label_path) in split_assignments:
                remapped = _remap_label_file(label_path, mapping, config.keep_unmapped)
                if not remapped:
                    continue

                new_name = f"{ds_name}_{total_count:06d}{img_path.suffix}"
                total_count += 1
                split_stats[split_name] += 1

                dst_img = output_dir / split_name / "images" / new_name
                if config.symlink:
                    dst_img.symlink_to(img_path.resolve())
                else:
                    shutil.copy2(img_path, dst_img)

                dst_label = output_dir / split_name / "labels" / f"{Path(new_name).stem}.txt"
                with open(dst_label, "w") as f:
                    f.write("\n".join(remapped) + "\n")

                for line in remapped:
                    cls_id = int(line.split()[0])
                    class_stats[split_name][cls_id] += 1

            click.echo(f"  Re-split: train {split_stats['train']} | val {split_stats['val']} | test {split_stats['test']}")

    # 6. Generate data.yaml
    names_list = [""] * (max(unified_classes.keys()) + 1)
    for cls_id, cls_name in unified_classes.items():
        names_list[cls_id] = cls_name

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(unified_classes),
        "names": names_list,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # 7. Print statistics
    click.echo("\n" + "=" * 60)
    click.echo(f"Merge completed! Total: {total_count} images")
    click.echo("=" * 60)

    click.echo(f"\n{'Split':<10} {'Count':>8}")
    click.echo("-" * 20)
    for split in splits_to_create:
        click.echo(f"{split:<10} {split_stats[split]:>8}")

    click.echo(f"\n{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    click.echo("-" * 60)
    for cls_id in sorted(unified_classes.keys()):
        cls_name = unified_classes[cls_id]
        t = class_stats["train"].get(cls_id, 0)
        v = class_stats["val"].get(cls_id, 0)
        te = class_stats["test"].get(cls_id, 0)
        click.echo(f"{cls_id}:{cls_name:<18} {t:>8} {v:>8} {te:>8} {t + v + te:>8}")
    click.echo("=" * 60)

    click.echo(f"\ndata.yaml: {yaml_path.resolve()}")

    return {
        "output_dir": str(output_dir),
        "data_yaml": str(yaml_path),
        "unified_classes": unified_classes,
        "total_images": total_count,
        "split_stats": split_stats,
    }
