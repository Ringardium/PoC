"""ReID Dataset and PK Sampler.

Expected folder structure:
    data_root/
        identity_001/
            img_001.jpg
            img_002.jpg
        identity_002/
            img_001.jpg
            ...

Identities can be pet names, global IDs, or any folder name.
Each subfolder = one identity, each image = one observation.
Mix of reference photos and CCTV crops is recommended (ratio ~2:8).
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# torchvision transforms
import torchvision.transforms as T


# ── Augmentation builders ──────────────────────────────────────────────


def build_train_transforms(
    imgsz: int = 224,
    random_crop: bool = True,
    color_jitter: float = 0.3,
    random_erasing: float = 0.5,
    horizontal_flip: float = 0.5,
) -> T.Compose:
    """Build training augmentation pipeline."""
    transforms = []

    if random_crop:
        transforms.append(T.Resize((imgsz + 32, imgsz + 32)))
        transforms.append(T.RandomCrop((imgsz, imgsz)))
    else:
        transforms.append(T.Resize((imgsz, imgsz)))

    if horizontal_flip > 0:
        transforms.append(T.RandomHorizontalFlip(p=horizontal_flip))

    if color_jitter > 0:
        transforms.append(
            T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter * 0.3,
            )
        )

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if random_erasing > 0:
        transforms.append(T.RandomErasing(p=random_erasing, scale=(0.02, 0.3)))

    return T.Compose(transforms)


def build_eval_transforms(imgsz: int = 224) -> T.Compose:
    """Build evaluation transform (deterministic)."""
    return T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ────────────────────────────────────────────────────────────


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ReIDDataset(Dataset):
    """ReID dataset: one folder per identity, images inside.

    Args:
        data_root: Root directory with identity subfolders.
        transform: torchvision transform to apply.
        min_images: Skip identities with fewer images than this.
    """

    def __init__(
        self,
        data_root: str,
        transform: T.Compose = None,
        min_images: int = 2,
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        self.identity_to_label: Dict[str, int] = {}
        self.label_to_identity: Dict[int, str] = {}
        self.label_to_indices: Dict[int, List[int]] = {}

        self._scan(min_images)

    def _scan(self, min_images: int):
        """Scan data_root for identity folders."""
        label = 0
        folders = sorted(
            d for d in self.data_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        for folder in folders:
            images = [
                str(f)
                for f in sorted(folder.iterdir())
                if f.suffix.lower() in IMAGE_EXTS
            ]
            if len(images) < min_images:
                continue

            self.identity_to_label[folder.name] = label
            self.label_to_identity[label] = folder.name
            indices = []

            for img_path in images:
                idx = len(self.samples)
                self.samples.append((img_path, label))
                indices.append(idx)

            self.label_to_indices[label] = indices
            label += 1

    @property
    def num_identities(self) -> int:
        return len(self.identity_to_label)

    @property
    def num_images(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # OpenCV read -> PIL-like RGB tensor
        img = cv2.imread(img_path)
        if img is None:
            # Fallback: random noise
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_identity_stats(self) -> Dict[str, int]:
        """Return {identity_name: image_count} for inspection."""
        return {
            self.label_to_identity[label]: len(indices)
            for label, indices in self.label_to_indices.items()
        }


# ── PK Sampler ─────────────────────────────────────────────────────────


class PKSampler(Sampler):
    """P-identities x K-images-per-identity batch sampler.

    Each batch contains exactly P*K samples: P randomly chosen identities
    with K randomly sampled images each. This ensures every batch has
    valid positive pairs for triplet mining.

    Args:
        dataset: ReIDDataset instance.
        p: Number of identities per batch.
        k: Number of images per identity.
    """

    def __init__(self, dataset: ReIDDataset, p: int = 8, k: int = 4):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.labels = list(dataset.label_to_indices.keys())

        if len(self.labels) < p:
            raise ValueError(
                f"Need at least P={p} identities, got {len(self.labels)}. "
                f"Add more identity folders or reduce --p."
            )

    def __iter__(self):
        # Shuffle identities, then yield P*K batches
        labels = self.labels.copy()
        random.shuffle(labels)

        # Repeat labels to fill enough batches
        extended = labels * ((self._num_batches() // len(labels)) + 2)

        for batch_start in range(0, len(extended) - self.p + 1, self.p):
            batch_labels = extended[batch_start : batch_start + self.p]
            indices = []

            for lbl in batch_labels:
                pool = self.dataset.label_to_indices[lbl]
                if len(pool) >= self.k:
                    chosen = random.sample(pool, self.k)
                else:
                    # Over-sample if fewer than K
                    chosen = random.choices(pool, k=self.k)
                indices.extend(chosen)

            yield from indices

    def _num_batches(self) -> int:
        total_images = len(self.dataset)
        return max(total_images // self.batch_size, 1)

    def __len__(self) -> int:
        return self._num_batches() * self.batch_size


# ── Train/query/gallery split ─────────────────────────────────────────


def split_query_gallery(
    dataset: ReIDDataset, query_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split dataset into query and gallery indices for evaluation.

    For each identity, ``query_ratio`` fraction of images go to query,
    the rest to gallery. Ensures at least 1 image in each set.

    Returns:
        (query_indices, gallery_indices)
    """
    rng = random.Random(seed)
    query_indices = []
    gallery_indices = []

    for label, indices in dataset.label_to_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        n_query = max(1, int(len(shuffled) * query_ratio))
        n_query = min(n_query, len(shuffled) - 1)  # at least 1 gallery

        query_indices.extend(shuffled[:n_query])
        gallery_indices.extend(shuffled[n_query:])

    return query_indices, gallery_indices
