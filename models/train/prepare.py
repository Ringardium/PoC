"""Dataset preparation tool: reorganize flat image+label folders into YOLO structure.

Handles the common case where images (.jpg/.png) and labels (.txt) are mixed
in the same directory, splitting them into the standard YOLO layout:

    output/
    ├── train/images/  train/labels/
    ├── val/images/    val/labels/
    └── data.yaml

Usage:
    python -m models.train prepare --input /path/to/flat_folder --output /path/to/dataset
    python -m models.train prepare --input /path/to/flat_folder --output /path/to/dataset --val-ratio 0.2
"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple

import click
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_paired_files(input_dir: Path) -> List[Tuple[Path, Path]]:
    """Find image files that have a matching .txt label file in the same directory."""
    pairs = []
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def _extract_classes(pairs: List[Tuple[Path, Path]]) -> dict:
    """Scan all label files and collect unique class IDs."""
    class_ids = set()
    for _, label_path in pairs:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    return sorted(class_ids)


def run_prepare(
    input_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.0,
    class_names: dict | None = None,
    seed: int = 42,
    symlink: bool = False,
) -> dict:
    """Reorganize a flat folder into YOLO directory structure.

    Args:
        input_dir: Folder containing mixed .jpg/.png and .txt files.
        output_dir: Output root directory.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for test (0 = no test split).
        class_names: Optional {class_id: name} mapping. Auto-detected if None.
        seed: Random seed for reproducible splits.
        symlink: Use symlinks instead of copying files.

    Returns:
        Dict with output_dir, data_yaml path, and stats.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    click.echo("=" * 60)
    click.echo("Dataset Prepare Tool")
    click.echo("=" * 60)
    click.echo(f"  Input:  {input_path}")
    click.echo(f"  Output: {output_path}")

    # 1. Find image-label pairs
    pairs = _find_paired_files(input_path)
    if not pairs:
        click.echo("[ERROR] No image+label pairs found. Ensure .jpg/.png files have matching .txt files.")
        return {"error": "no_pairs"}

    click.echo(f"  Found {len(pairs)} image-label pairs")

    # Also check for images without labels
    all_images = [f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    orphan_count = len(all_images) - len(pairs)
    if orphan_count > 0:
        click.echo(f"  [WARN] {orphan_count} images have no matching .txt label (skipped)")

    # 2. Detect classes
    class_ids = _extract_classes(pairs)
    click.echo(f"  Classes found: {class_ids}")

    if class_names is None:
        class_names = {cid: f"class_{cid}" for cid in class_ids}

    # 3. Shuffle and split
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    n = len(shuffled)
    val_end = int(n * (1 - val_ratio - test_ratio))
    test_start = int(n * (1 - test_ratio)) if test_ratio > 0 else n

    splits = {
        "train": shuffled[:val_end],
        "val": shuffled[val_end:test_start],
    }
    if test_ratio > 0:
        splits["test"] = shuffled[test_start:]

    # 4. Create directories and copy/link files
    for split_name, split_pairs in splits.items():
        img_dir = output_path / split_name / "images"
        lbl_dir = output_path / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            dst_img = img_dir / img_path.name
            dst_lbl = lbl_dir / label_path.name

            if symlink:
                if not dst_img.exists():
                    dst_img.symlink_to(img_path)
                if not dst_lbl.exists():
                    dst_lbl.symlink_to(label_path)
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(label_path, dst_lbl)

    # 5. Generate data.yaml
    names_list = [""] * (max(class_names.keys()) + 1) if class_names else []
    for cid, name in class_names.items():
        if cid < len(names_list):
            names_list[cid] = name

    data_yaml = {
        "path": str(output_path),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": names_list,
    }
    if test_ratio > 0:
        data_yaml["test"] = "test/images"

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # 6. Print summary
    click.echo("\n" + "=" * 60)
    click.echo("Preparation completed!")
    click.echo("=" * 60)

    click.echo(f"\n{'Split':<10} {'Count':>8}")
    click.echo("-" * 20)
    for split_name, split_pairs in splits.items():
        click.echo(f"{split_name:<10} {len(split_pairs):>8}")

    click.echo(f"\n  data.yaml: {yaml_path}")
    click.echo(f"  Classes: {class_names}")
    click.echo("=" * 60)

    return {
        "output_dir": str(output_path),
        "data_yaml": str(yaml_path),
        "total": len(pairs),
        "splits": {k: len(v) for k, v in splits.items()},
        "class_names": class_names,
    }


# ── Gather: move files + remap labels into one target dataset ────────


SPLITS = ("train", "val", "test")


def run_gather(
    target_yaml: str,
    source_yamls: list[str],
    keep_unmapped: bool = False,
    include_classes: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Move files from source datasets into the target dataset with label remapping.

    Reads each data.yaml to resolve paths and class names, builds a unified
    class mapping based on the target's classes, remaps source labels to match,
    then **moves** images and writes remapped labels into the target splits.

    Args:
        target_yaml: data.yaml of the target dataset (folder 1).
        source_yamls: data.yaml paths of source datasets (folders 2, 3, ...).
        keep_unmapped: Keep label lines for classes not in any mapping.
        include_classes: If set, only gather these class names.
        dry_run: Preview without moving/writing.

    Returns:
        Dict with stats.
    """
    from .merge import (
        _parse_data_yaml,
        _build_unified_mapping,
        _remap_label_file,
        _collect_image_files,
    )

    click.echo("=" * 60)
    click.echo("Dataset Gather Tool (move + remap)")
    click.echo("=" * 60)

    # 1. Parse target and source data.yamls
    target_ds = _parse_data_yaml(target_yaml)
    target_path = Path(target_ds["path"]).resolve()
    click.echo(f"\n[Target] {target_path.name}")
    click.echo(f"  Classes ({target_ds['nc']}): {target_ds['names']}")
    click.echo(f"  Splits: {list(target_ds['splits'].keys())}")

    source_datasets = []
    for sy in source_yamls:
        ds = _parse_data_yaml(sy)
        source_datasets.append(ds)
        click.echo(f"\n[Source] {Path(ds['source_yaml']).parent.name}")
        click.echo(f"  Classes ({ds['nc']}): {ds['names']}")
        click.echo(f"  Splits: {list(ds['splits'].keys())}")

    if dry_run:
        click.echo("\n  ** DRY RUN — no files will be moved **")

    # 2. Build unified class mapping (target classes as base)
    base_classes = dict(target_ds["names"])
    include = include_classes if include_classes else None
    unified_classes, per_dataset_mappings = _build_unified_mapping(
        source_datasets, base_classes, include,
    )

    # Show mapping
    click.echo("\n" + "-" * 60)
    click.echo("Class Mapping")
    click.echo("-" * 60)
    for ds_idx, ds in enumerate(source_datasets):
        ds_name = Path(ds["source_yaml"]).parent.name
        mapping = per_dataset_mappings[ds_idx]
        click.echo(f"\n  [{ds_name}]")
        for org_id, new_id in sorted(mapping.items()):
            org_name = ds["names"][org_id]
            new_name = unified_classes[new_id]
            arrow = "" if org_id == new_id and org_name == new_name else f" (was {org_id})"
            click.echo(f"    {org_name} → {new_id}:{new_name}{arrow}")

    if include:
        click.echo(f"\n  [Filter] Only: {include_classes}")

    click.echo(f"\n  Unified ({len(unified_classes)} classes): {unified_classes}")
    click.echo("-" * 60)

    # 3. Process each source dataset — move images, write remapped labels
    total_moved = 0
    split_stats = {sp: 0 for sp in SPLITS}
    class_stats = {sp: {} for sp in SPLITS}

    for ds_idx, ds in enumerate(source_datasets):
        ds_name = Path(ds["source_yaml"]).parent.name
        mapping = per_dataset_mappings[ds_idx]
        click.echo(f"\nMoving: {ds_name}")

        for split in SPLITS:
            if split not in ds["splits"]:
                continue
            if split not in target_ds["splits"]:
                # Create split dirs in target if they don't exist yet
                tgt_img = target_path / split / "images"
                tgt_lbl = target_path / split / "labels"
            else:
                tgt_img = target_ds["splits"][split]["images"]
                tgt_lbl = target_ds["splits"][split]["labels"]

            src_img_dir = ds["splits"][split]["images"]
            src_lbl_dir = ds["splits"][split]["labels"]

            if not dry_run:
                tgt_img.mkdir(parents=True, exist_ok=True)
                tgt_lbl.mkdir(parents=True, exist_ok=True)

            images = _collect_image_files(src_img_dir)
            processed = 0

            for img_path in images:
                label_path = src_lbl_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                # Remap labels
                remapped = _remap_label_file(label_path, mapping, keep_unmapped)
                if not remapped:
                    continue

                # Resolve filename conflict
                dst_img = tgt_img / img_path.name
                dst_lbl = tgt_lbl / f"{img_path.stem}.txt"
                if dst_img.exists():
                    new_stem = f"{ds_name}_{img_path.stem}"
                    dst_img = tgt_img / f"{new_stem}{img_path.suffix}"
                    dst_lbl = tgt_lbl / f"{new_stem}.txt"

                if dry_run:
                    if processed < 3:
                        click.echo(f"    {img_path.name} → {dst_img.name}")
                else:
                    # Move image, write remapped label (original label replaced)
                    shutil.move(str(img_path), str(dst_img))
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(remapped) + "\n")
                    # Remove original label
                    if label_path.exists():
                        label_path.unlink()

                processed += 1

                # Track class stats
                for line in remapped:
                    cid = int(line.split()[0])
                    class_stats[split][cid] = class_stats[split].get(cid, 0) + 1

            split_stats[split] += processed
            total_moved += processed

            if processed > 0:
                click.echo(f"  {split}: {processed} images")

    # 4. Update target data.yaml with unified classes
    if not dry_run:
        names_list = [""] * (max(unified_classes.keys()) + 1)
        for cid, name in unified_classes.items():
            names_list[cid] = name

        target_yaml_path = Path(target_yaml).resolve()
        with open(target_yaml_path) as f:
            existing = yaml.safe_load(f)

        existing["nc"] = len(unified_classes)
        existing["names"] = names_list

        with open(target_yaml_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

        click.echo(f"\n  Updated: {target_yaml_path}")

    # 5. Summary
    click.echo("\n" + "=" * 60)
    click.echo(f"Gather completed! Total: {total_moved} images moved")
    click.echo("=" * 60)

    click.echo(f"\n{'Split':<10} {'Moved':>8}")
    click.echo("-" * 20)
    for sp in SPLITS:
        if split_stats[sp]:
            click.echo(f"{sp:<10} {split_stats[sp]:>8}")

    if any(class_stats[sp] for sp in SPLITS):
        click.echo(f"\n{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8}")
        click.echo("-" * 55)
        all_cids = set()
        for sp in SPLITS:
            all_cids.update(class_stats[sp].keys())
        for cid in sorted(all_cids):
            name = unified_classes.get(cid, f"class_{cid}")
            t = class_stats["train"].get(cid, 0)
            v = class_stats["val"].get(cid, 0)
            te = class_stats["test"].get(cid, 0)
            click.echo(f"{cid}:{name:<23} {t:>8} {v:>8} {te:>8}")

    click.echo("=" * 60)

    return {
        "target": str(target_path),
        "unified_classes": unified_classes,
        "total_moved": total_moved,
        "split_stats": split_stats,
    }


# ── Analyze: class distribution per split ────────────────────────────


def _parse_data_yaml_simple(yaml_path: str) -> dict:
    """Parse data.yaml and resolve split label directories."""
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base_dir = yaml_path.parent
    root = Path(data["path"]) if Path(data.get("path", "")).is_absolute() else base_dir / data.get("path", ".")

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    splits = {}
    for key in ("train", "val", "test"):
        val = data.get(key) or data.get("valid" if key == "val" else "")
        if not val:
            continue
        img_dir = root / val
        lbl_dir = img_dir.parent / "labels"
        if lbl_dir.exists():
            splits[key] = lbl_dir

    return {"names": names, "splits": splits, "root": root}


def _scan_labels(label_dir: Path) -> dict:
    """Scan label dir and return {class_id: instance_count} and per-image info.

    Returns:
        (class_counts, image_classes)
        class_counts: {cls_id: int}
        image_classes: list of (label_path, {cls_id: count})
    """
    class_counts = {}
    image_classes = []

    for lbl in sorted(label_dir.iterdir()):
        if lbl.suffix != ".txt":
            continue
        per_image = {}
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                per_image[cid] = per_image.get(cid, 0) + 1
                class_counts[cid] = class_counts.get(cid, 0) + 1
        image_classes.append((lbl, per_image))

    return class_counts, image_classes


def run_analyze(data_yaml: str, split: str = "all") -> dict:
    """Analyze class distribution in a dataset.

    Args:
        data_yaml: Path to data.yaml.
        split: Which split to analyze ('train', 'val', 'test', 'all').

    Returns:
        Dict with per-split class counts and image counts.
    """
    info = _parse_data_yaml_simple(data_yaml)
    names = info["names"]

    click.echo("=" * 70)
    click.echo("Dataset Class Analysis")
    click.echo("=" * 70)
    click.echo(f"  data.yaml: {data_yaml}")
    click.echo(f"  Classes: {len(names)}")

    target_splits = list(info["splits"].keys()) if split == "all" else [split]
    all_stats = {}

    for sp in target_splits:
        if sp not in info["splits"]:
            click.echo(f"\n  [WARN] Split '{sp}' not found")
            continue

        class_counts, image_classes = _scan_labels(info["splits"][sp])
        total_images = len(image_classes)
        total_instances = sum(class_counts.values())

        all_stats[sp] = {
            "class_counts": class_counts,
            "total_images": total_images,
            "total_instances": total_instances,
        }

        click.echo(f"\n── {sp} ({total_images} images, {total_instances} instances) ──")
        click.echo(f"  {'ID':<4} {'Class':<20} {'Instances':>10} {'Images':>8} {'Ratio':>8}")
        click.echo("  " + "-" * 54)

        # Count images containing each class
        images_per_class = {}
        for _, per_img in image_classes:
            for cid in per_img:
                images_per_class[cid] = images_per_class.get(cid, 0) + 1

        max_count = max(class_counts.values()) if class_counts else 1
        for cid in sorted(class_counts.keys()):
            cnt = class_counts[cid]
            img_cnt = images_per_class.get(cid, 0)
            ratio = cnt / max_count
            name = names.get(cid, f"class_{cid}")
            bar = "█" * int(ratio * 20)
            click.echo(f"  {cid:<4} {name:<20} {cnt:>10} {img_cnt:>8} {ratio:>7.1%} {bar}")

        # Imbalance warning
        if class_counts:
            min_c = min(class_counts.values())
            max_c = max(class_counts.values())
            if max_c > min_c * 5:
                click.echo(f"\n  ⚠ Imbalance ratio: {max_c / min_c:.1f}x (max/min)")
                click.echo(f"    Consider: python -m models.train sample --data {data_yaml} --strategy undersample")

    click.echo("\n" + "=" * 70)
    return all_stats


# ── Sample: balanced dataset creation ────────────────────────────────


def run_sample(
    data_yaml: str,
    output_dir: str,
    strategy: str = "undersample",
    max_per_class: int | None = None,
    min_per_class: int | None = None,
    seed: int = 42,
    copy_val: bool = True,
) -> dict:
    """Create a class-balanced dataset by sampling from an existing one.

    Strategies:
        undersample: Reduce majority classes to match target count.
                     Target = min class count (or --max-per-class if set).
        oversample:  Duplicate minority class images to match target count.
                     Target = max class count (or --min-per-class if set).
        cap:         Keep at most --max-per-class images per class.
                     Does not duplicate, just limits.

    Sampling is image-level: each image is assigned to its rarest class
    for balanced selection.

    Args:
        data_yaml: Path to source data.yaml.
        output_dir: Output directory for sampled dataset.
        strategy: 'undersample', 'oversample', or 'cap'.
        max_per_class: Max images per class (for undersample/cap).
        min_per_class: Min images per class (for oversample).
        seed: Random seed.
        copy_val: If True, copy val/test splits unchanged.

    Returns:
        Dict with sampling stats.
    """
    random.seed(seed)
    info = _parse_data_yaml_simple(data_yaml)
    names = info["names"]
    output_path = Path(output_dir).resolve()

    click.echo("=" * 70)
    click.echo(f"Dataset Sampling — strategy: {strategy}")
    click.echo("=" * 70)

    if "train" not in info["splits"]:
        click.echo("[ERROR] No train split found in data.yaml")
        return {"error": "no_train_split"}

    # 1. Scan train labels
    train_lbl_dir = info["splits"]["train"]
    train_img_dir = train_lbl_dir.parent / "images"
    class_counts, image_classes = _scan_labels(train_lbl_dir)

    click.echo(f"  Source images: {len(image_classes)}")
    click.echo(f"  Classes: {len(class_counts)}")
    for cid in sorted(class_counts.keys()):
        click.echo(f"    {cid} ({names.get(cid, '?')}): {class_counts[cid]} instances")

    # 2. Group images by their "rarest class" (class with fewest total instances)
    # This ensures rare class images are prioritized
    class_to_images: dict[int, list[Path]] = {cid: [] for cid in class_counts}
    for lbl_path, per_img in image_classes:
        if not per_img:
            continue
        # Assign image to the rarest class it contains
        rarest_cid = min(per_img.keys(), key=lambda c: class_counts.get(c, float("inf")))
        class_to_images[rarest_cid].append(lbl_path)

    # Also track all images per class (for oversample)
    class_all_images: dict[int, list[Path]] = {cid: [] for cid in class_counts}
    for lbl_path, per_img in image_classes:
        for cid in per_img:
            class_all_images[cid].append(lbl_path)

    # 3. Determine target count per class
    counts_per_group = {cid: len(imgs) for cid, imgs in class_to_images.items() if imgs}
    min_count = min(counts_per_group.values()) if counts_per_group else 0
    max_count = max(counts_per_group.values()) if counts_per_group else 0

    if strategy == "undersample":
        target = max_per_class if max_per_class else min_count
        click.echo(f"\n  Undersample target: {target} images per class")
    elif strategy == "oversample":
        target = min_per_class if min_per_class else max_count
        click.echo(f"\n  Oversample target: {target} images per class")
    elif strategy == "cap":
        target = max_per_class if max_per_class else min_count
        click.echo(f"\n  Cap target: {target} images per class")
    else:
        click.echo(f"[ERROR] Unknown strategy: {strategy}")
        return {"error": "unknown_strategy"}

    # 4. Select images
    selected: set[Path] = set()

    for cid in sorted(class_to_images.keys()):
        pool = class_to_images[cid]
        random.shuffle(pool)

        if strategy == "undersample" or strategy == "cap":
            chosen = pool[:target]
        elif strategy == "oversample":
            if len(pool) >= target:
                chosen = pool[:target]
            else:
                # Duplicate to reach target
                chosen = list(pool)
                while len(chosen) < target:
                    chosen.extend(pool[:target - len(chosen)])
        else:
            chosen = pool

        selected.update(chosen)

    click.echo(f"  Selected: {len(selected)} unique images")

    # 5. Copy selected images + labels to output
    out_train_img = output_path / "train" / "images"
    out_train_lbl = output_path / "train" / "labels"
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)

    dup_counter = {}  # Track duplicates for oversample
    copied = 0
    for lbl_path in sorted(selected):
        stem = lbl_path.stem
        # Find matching image
        img_path = None
        for ext in IMAGE_EXTS:
            candidate = train_img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        # Handle oversampled duplicates
        count = dup_counter.get(stem, 0)
        dup_counter[stem] = count + 1
        if count > 0:
            out_name = f"{stem}_dup{count}"
        else:
            out_name = stem

        shutil.copy2(img_path, out_train_img / f"{out_name}{img_path.suffix}")
        shutil.copy2(lbl_path, out_train_lbl / f"{out_name}.txt")
        copied += 1

    # 6. Copy val/test unchanged
    val_test_stats = {}
    if copy_val:
        for sp in ("val", "test"):
            if sp not in info["splits"]:
                continue
            src_lbl = info["splits"][sp]
            src_img = src_lbl.parent / "images"
            dst_img = output_path / sp / "images"
            dst_lbl = output_path / sp / "labels"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            count = 0
            for lbl in sorted(src_lbl.iterdir()):
                if lbl.suffix != ".txt":
                    continue
                stem = lbl.stem
                img = None
                for ext in IMAGE_EXTS:
                    c = src_img / f"{stem}{ext}"
                    if c.exists():
                        img = c
                        break
                if img:
                    shutil.copy2(img, dst_img / img.name)
                    shutil.copy2(lbl, dst_lbl / lbl.name)
                    count += 1
            val_test_stats[sp] = count
            click.echo(f"  {sp}: {count} images (copied as-is)")

    # 7. Generate data.yaml
    names_list = [""] * (max(names.keys()) + 1) if names else []
    for cid, name in names.items():
        if cid < len(names_list):
            names_list[cid] = name

    out_yaml = {
        "path": str(output_path),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names_list,
    }
    if "test" in val_test_stats:
        out_yaml["test"] = "test/images"

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(out_yaml, f, default_flow_style=False, allow_unicode=True)

    # 8. Verify result distribution
    click.echo(f"\n── Result distribution ──")
    result_counts, result_images = _scan_labels(out_train_lbl)
    for cid in sorted(result_counts.keys()):
        orig = class_counts.get(cid, 0)
        new = result_counts[cid]
        name = names.get(cid, f"class_{cid}")
        click.echo(f"  {cid} ({name}): {orig} → {new}")

    click.echo(f"\n  data.yaml: {yaml_path}")
    click.echo("=" * 70)

    return {
        "output_dir": str(output_path),
        "data_yaml": str(yaml_path),
        "strategy": strategy,
        "train_images": copied,
        "original_counts": class_counts,
        "sampled_counts": result_counts,
    }
