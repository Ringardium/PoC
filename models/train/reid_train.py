"""ReID Metric Learning Training Loop.

Trains a pet ReID model with PK-sampled batches, TripletLoss + ArcFace,
and evaluates with Rank-1/5/10 and mAP metrics.

Usage:
    python -m models.train reid --data-root data/reid --backbone dinov2_vits14
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .config import ReIDConfig
from .reid_model import ReIDModel
from .reid_losses import OnlineTripletLoss, ArcFaceLoss, CombinedReIDLoss
from .reid_dataset import (
    ReIDDataset,
    PKSampler,
    build_train_transforms,
    build_eval_transforms,
    split_query_gallery,
)

logger = logging.getLogger(__name__)


# ── Evaluation metrics ─────────────────────────────────────────────────


@torch.no_grad()
def extract_features(
    model: ReIDModel, dataloader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings for all images in dataloader.

    Returns:
        features [N, D], labels [N]
    """
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def compute_metrics(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> Dict[str, float]:
    """Compute Rank-1, Rank-5, Rank-10, mAP.

    Args:
        query_features: [Q, D]
        query_labels: [Q]
        gallery_features: [G, D]
        gallery_labels: [G]

    Returns:
        Dict with rank1, rank5, rank10, mAP.
    """
    # Cosine distance (features already L2-normalized)
    dist = torch.cdist(query_features, gallery_features, p=2)  # [Q, G]

    # Sort gallery by distance for each query
    sorted_indices = dist.argsort(dim=1)  # [Q, G]

    q_labels = query_labels.numpy()
    g_labels = gallery_labels.numpy()

    rank_hits = {1: 0, 5: 0, 10: 0}
    average_precisions = []

    for i in range(len(q_labels)):
        target = q_labels[i]
        ordered = g_labels[sorted_indices[i].numpy()]

        # Rank-K
        for k in rank_hits:
            if target in ordered[:k]:
                rank_hits[k] += 1

        # AP
        matches = (ordered == target).astype(np.float32)
        if matches.sum() == 0:
            continue
        cumsum = np.cumsum(matches)
        precision_at_k = cumsum / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        average_precisions.append(ap)

    n = len(q_labels)
    return {
        "rank1": rank_hits[1] / n * 100,
        "rank5": rank_hits[5] / n * 100,
        "rank10": rank_hits[10] / n * 100,
        "mAP": np.mean(average_precisions) * 100 if average_precisions else 0.0,
    }


def evaluate(
    model: ReIDModel,
    dataset: ReIDDataset,
    query_indices: List[int],
    gallery_indices: List[int],
    device: torch.device,
    imgsz: int = 224,
    batch_size: int = 64,
    workers: int = 4,
) -> Dict[str, float]:
    """Full evaluation: extract features then compute metrics."""
    eval_tf = build_eval_transforms(imgsz)

    # Build eval datasets with eval transform
    query_ds = Subset(dataset, query_indices)
    gallery_ds = Subset(dataset, gallery_indices)

    # Temporarily swap transform
    orig_tf = dataset.transform
    dataset.transform = eval_tf

    query_loader = DataLoader(query_ds, batch_size=batch_size, num_workers=workers)
    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, num_workers=workers)

    q_feat, q_labels = extract_features(model, query_loader, device)
    g_feat, g_labels = extract_features(model, gallery_loader, device)

    # Restore
    dataset.transform = orig_tf

    return compute_metrics(q_feat, q_labels, g_feat, g_labels)


# ── Training loop ──────────────────────────────────────────────────────


def run_reid(config: ReIDConfig) -> Dict:
    """Main ReID training function.

    Args:
        config: ReIDConfig with all hyperparameters.

    Returns:
        Dict with best metrics and saved model path.
    """
    # ── Setup ──
    save_dir = Path(config.project) / config.name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if config.device != "cpu" else "cpu")
    if config.device not in ("cpu", "mps") and config.device.isdigit():
        device = torch.device(f"cuda:{config.device}")

    logger.info(f"Device: {device}")

    # ── Dataset ──
    train_tf = build_train_transforms(
        imgsz=config.imgsz,
        random_crop=config.random_crop,
        color_jitter=config.color_jitter,
        random_erasing=config.random_erasing,
        horizontal_flip=config.horizontal_flip,
    )

    dataset = ReIDDataset(config.data_root, transform=train_tf, min_images=2)
    num_ids = dataset.num_identities
    num_imgs = dataset.num_images

    logger.info(f"Dataset: {num_ids} identities, {num_imgs} images")
    stats = dataset.get_identity_stats()
    for name, count in sorted(stats.items()):
        logger.info(f"  {name}: {count} images")

    if num_ids < config.p:
        raise ValueError(
            f"Not enough identities ({num_ids}) for P={config.p}. "
            f"Need at least {config.p} identity folders."
        )

    # Query/gallery split for evaluation
    query_idx, gallery_idx = split_query_gallery(dataset, config.query_ratio)
    logger.info(f"Eval split: {len(query_idx)} query, {len(gallery_idx)} gallery")

    # PK Sampler
    sampler = PKSampler(dataset, p=config.p, k=config.k)
    train_loader = DataLoader(
        dataset,
        batch_size=config.p * config.k,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ──
    logger.info(f"Building model: backbone={config.backbone}, embed_dim={config.embed_dim}")
    model = ReIDModel(
        backbone_name=config.backbone,
        embed_dim=config.embed_dim,
        freeze_backbone=config.freeze_backbone,
        pretrained=config.pretrained,
    )
    model = model.to(device)

    # ── Loss ──
    if config.loss == "triplet":
        criterion = OnlineTripletLoss(margin=config.triplet_margin).to(device)
    elif config.loss == "arcface":
        criterion = ArcFaceLoss(
            embed_dim=config.embed_dim,
            num_classes=num_ids,
            scale=config.arcface_scale,
            margin=config.arcface_margin,
        ).to(device)
    else:  # combined
        criterion = CombinedReIDLoss(
            embed_dim=config.embed_dim,
            num_classes=num_ids,
            triplet_margin=config.triplet_margin,
            arcface_scale=config.arcface_scale,
            arcface_margin=config.arcface_margin,
            triplet_weight=config.triplet_weight,
            arcface_weight=config.arcface_weight,
        ).to(device)

    # ── Optimizer ──
    params = [
        {"params": model.head.parameters(), "lr": config.lr},
    ]
    if not config.freeze_backbone:
        params.append(
            {"params": model.backbone.parameters(), "lr": config.lr * 0.1}
        )
    if hasattr(criterion, "parameters"):
        params.append({"params": criterion.parameters(), "lr": config.lr})

    optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)

    # ── Scheduler ──
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.step_gamma
        )

    # ── Warmup ──
    warmup_scheduler = None
    if config.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=config.warmup_epochs
        )

    # ── Training ──
    best_rank1 = 0.0
    best_epoch = 0
    best_metrics = {}

    _print_banner(config, num_ids, num_imgs, device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            if isinstance(criterion, CombinedReIDLoss):
                loss, loss_dict = criterion(embeddings, labels)
            elif isinstance(criterion, ArcFaceLoss):
                loss = criterion(embeddings, labels)
                loss_dict = {"arcface": loss.item()}
            else:
                loss = criterion(embeddings, labels)
                loss_dict = {"triplet": loss.item()}

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        # Scheduler step
        if warmup_scheduler and epoch <= config.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        lr_now = optimizer.param_groups[0]["lr"]

        # Log
        log_msg = f"Epoch {epoch}/{config.epochs} | loss={avg_loss:.4f} | lr={lr_now:.6f} | {epoch_time:.1f}s"
        if isinstance(criterion, CombinedReIDLoss):
            log_msg += f" | trip={loss_dict.get('triplet', 0):.4f} arc={loss_dict.get('arcface', 0):.4f}"

        # Evaluate periodically
        if epoch % config.save_interval == 0 or epoch == config.epochs:
            metrics = evaluate(
                model, dataset, query_idx, gallery_idx, device,
                imgsz=config.imgsz, workers=config.workers
            )
            log_msg += (
                f" | R1={metrics['rank1']:.1f}% R5={metrics['rank5']:.1f}% "
                f"mAP={metrics['mAP']:.1f}%"
            )

            if metrics["rank1"] > best_rank1:
                best_rank1 = metrics["rank1"]
                best_epoch = epoch
                best_metrics = metrics.copy()

                # Save best
                best_path = save_dir / "best.pt"
                model.save_inference(str(best_path))
                logger.info(f"  New best saved: {best_path}")

        logger.info(log_msg)

        # Save checkpoint periodically
        if epoch % config.save_interval == 0:
            ckpt_path = save_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "metrics": metrics if "metrics" in dir() else {},
                },
                str(ckpt_path),
            )

    # Save last
    last_path = save_dir / "last.pt"
    model.save_inference(str(last_path))

    # ── Summary ──
    logger.info("=" * 60)
    logger.info(f"Training complete. Best epoch: {best_epoch}")
    logger.info(f"  Rank-1:  {best_metrics.get('rank1', 0):.1f}%")
    logger.info(f"  Rank-5:  {best_metrics.get('rank5', 0):.1f}%")
    logger.info(f"  Rank-10: {best_metrics.get('rank10', 0):.1f}%")
    logger.info(f"  mAP:     {best_metrics.get('mAP', 0):.1f}%")
    logger.info(f"  Best model: {save_dir / 'best.pt'}")
    logger.info(f"  Last model: {last_path}")
    logger.info("=" * 60)

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_model": str(save_dir / "best.pt"),
        "last_model": str(last_path),
        "save_dir": str(save_dir),
    }


def _print_banner(config: ReIDConfig, num_ids: int, num_imgs: int, device):
    """Print training configuration banner."""
    lines = [
        "=" * 60,
        "  Pet ReID Metric Learning",
        "=" * 60,
        f"  Backbone:       {config.backbone}",
        f"  Embed dim:      {config.embed_dim}",
        f"  Freeze backbone: {config.freeze_backbone}",
        f"  Loss:           {config.loss}",
        f"  Epochs:         {config.epochs}",
        f"  PK batch:       P={config.p} x K={config.k} = {config.p * config.k}",
        f"  LR:             {config.lr}",
        f"  Device:         {device}",
        f"  Image size:     {config.imgsz}",
        f"  Identities:     {num_ids}",
        f"  Total images:   {num_imgs}",
        f"  Save dir:       {config.project}/{config.name}",
        "=" * 60,
    ]
    for line in lines:
        logger.info(line)
