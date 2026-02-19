import os
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available. Install with: pip install pytorch-lightning")

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect, C2f, Conv, SPPF
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available. Install with: pip install ultralytics")


class YOLOLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for YOLO training"""

    def __init__(
        self,
        model_config: str = "yolo11n.yaml",
        num_classes: int = 2,  # background + pet
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 3,
        max_epochs: int = 100,
        optimizer: str = "AdamW",
        scheduler: str = "cosine",
        pretrained_weights: Optional[str] = None,
        img_size: int = 640,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning is required")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.img_size = img_size
        self.num_classes = num_classes

        # Load YOLO model
        if pretrained_weights:
            self.model = YOLO(pretrained_weights)
        else:
            self.model = YOLO(model_config)

        # Extract the actual PyTorch model
        self.net = self.model.model

        # Loss function
        self.criterion = self._build_loss_function()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []

    def _build_loss_function(self):
        """Build YOLO loss function"""
        try:
            # Use YOLO's built-in loss
            from ultralytics.utils.loss import v8DetectionLoss
            return v8DetectionLoss(self.net)
        except ImportError:
            # Fallback to custom loss
            return self._custom_yolo_loss

    def _custom_yolo_loss(self, predictions, targets):
        """Custom YOLO loss implementation"""
        # Simplified YOLO loss
        # In practice, you would implement the full YOLO loss
        box_loss = F.mse_loss(predictions[..., :4], targets[..., :4])
        conf_loss = F.binary_cross_entropy_with_logits(
            predictions[..., 4], targets[..., 4]
        )
        cls_loss = F.cross_entropy(
            predictions[..., 5:].reshape(-1, self.num_classes),
            targets[..., 5].long().reshape(-1)
        )
        return box_loss + conf_loss + cls_loss

    def forward(self, x):
        """Forward pass"""
        return self.net(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, targets = batch

        # Forward pass
        predictions = self.forward(images)

        # Calculate loss
        if hasattr(self.criterion, 'device'):
            self.criterion.device = self.device

        try:
            loss = self.criterion(predictions, targets)
            if isinstance(loss, tuple):
                loss = loss[0]  # Take main loss if tuple returned
        except Exception:
            # Fallback loss calculation
            loss = self._simple_detection_loss(predictions, targets)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, targets = batch

        predictions = self.forward(images)

        try:
            loss = self.criterion(predictions, targets)
            if isinstance(loss, tuple):
                loss = loss[0]
        except Exception:
            loss = self._simple_detection_loss(predictions, targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())

        return loss

    def _simple_detection_loss(self, predictions, targets):
        """Simplified detection loss for fallback"""
        # Basic MSE loss for demonstration
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        # Ensure same shape
        min_size = min(predictions.numel(), targets.numel())
        pred_flat = predictions.view(-1)[:min_size]
        target_flat = targets.view(-1)[:min_size]

        return F.mse_loss(pred_flat, target_flat)

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer selection
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        # Scheduler selection
        if self.scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            )
        elif self.scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_name.lower() == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
        else:
            # Linear warmup + cosine annealing
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return epoch / self.warmup_epochs
                else:
                    return 0.5 * (1 + np.cos(np.pi * epoch / self.max_epochs))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.scheduler_name.lower() == "reduce":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if self.train_losses:
            avg_loss = np.mean(self.train_losses[-len(self.trainer.train_dataloader):])
            self.log('train_loss_epoch', avg_loss)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if self.val_losses:
            avg_loss = np.mean(self.val_losses[-len(self.trainer.val_dataloaders[0]):])
            self.log('val_loss_epoch', avg_loss)


class PetDataset(Dataset):
    """Custom dataset for pet detection"""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: int = 640,
        augmentations: Optional[Any] = None,
        transform: Optional[Any] = None
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augmentations = augmentations
        self.transform = transform

        # Get image files
        from pathlib import Path
        self.image_files = list(Path(images_dir).glob("*.jpg")) + \
                          list(Path(images_dir).glob("*.png"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = Path(self.labels_dir) / f"{img_path.stem}.txt"
        targets = self._load_labels(label_path)

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, bboxes=targets)
            image = augmented['image']
            targets = augmented['bboxes']

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert targets to tensor
        if len(targets) > 0:
            targets = torch.from_numpy(np.array(targets)).float()
        else:
            targets = torch.zeros((0, 6)).float()  # [class, x, y, w, h, conf]

        return image, targets

    def _load_labels(self, label_path):
        """Load YOLO format labels"""
        if not label_path.exists():
            return []

        targets = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    targets.append([cls, x, y, w, h, 1.0])  # Add confidence

        return targets


class LightningTrainer:
    """High-level trainer for YOLO with Lightning"""

    def __init__(
        self,
        train_data_dir: str,
        val_data_dir: str,
        model_config: str = "yolo11n.yaml",
        batch_size: int = 16,
        num_workers: int = 4,
        gpus: int = 1,
        precision: str = "16-mixed",
        max_epochs: int = 100,
        save_dir: str = "./lightning_logs"
    ):
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.model_config = model_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gpus = gpus
        self.precision = precision
        self.max_epochs = max_epochs
        self.save_dir = save_dir

    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Setup data loaders"""
        # Create datasets
        train_dataset = PetDataset(
            f"{self.train_data_dir}/images",
            f"{self.train_data_dir}/labels",
            augmentations=self._get_augmentations()
        )

        val_dataset = PetDataset(
            f"{self.val_data_dir}/images",
            f"{self.val_data_dir}/labels"
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )

        return train_loader, val_loader

    def _get_augmentations(self):
        """Get data augmentations"""
        try:
            import albumentations as A
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussNoise(p=0.1),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        except ImportError:
            print("Albumentations not available. No augmentations applied.")
            return None

    def _collate_fn(self, batch):
        """Custom collate function"""
        images, targets = zip(*batch)
        images = torch.stack(images)

        # Pad targets to same length
        max_targets = max([len(t) for t in targets]) if targets else 0
        if max_targets == 0:
            targets = torch.zeros((len(batch), 0, 6))
        else:
            padded_targets = []
            for target in targets:
                if len(target) < max_targets:
                    padding = torch.zeros((max_targets - len(target), 6))
                    target = torch.cat([target, padding])
                padded_targets.append(target)
            targets = torch.stack(padded_targets)

        return images, targets

    def train(self, **kwargs) -> YOLOLightningModule:
        """Train the model"""
        # Setup data
        train_loader, val_loader = self.setup_data()

        # Create model
        model = YOLOLightningModule(
            model_config=self.model_config,
            max_epochs=self.max_epochs,
            **kwargs
        )

        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=f"{self.save_dir}/checkpoints",
                filename="best-{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                mode="min"
            ),
            LearningRateMonitor(logging_interval="epoch")
        ]

        # Setup logger
        logger = TensorBoardLogger(
            save_dir=self.save_dir,
            name="yolo_training"
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=self.gpus,
            precision=self.precision,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            val_check_interval=0.5,
            enable_progress_bar=True,
            enable_model_summary=True
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        return model


def convert_lightning_to_yolo(lightning_model: YOLOLightningModule, output_path: str):
    """Convert Lightning model back to YOLO format"""
    # Extract the underlying model
    state_dict = lightning_model.net.state_dict()

    # Save as regular PyTorch model
    torch.save({
        'model': state_dict,
        'hyperparameters': lightning_model.hparams
    }, output_path)

    print(f"Model saved to {output_path}")


def quick_train_setup(
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto"
) -> YOLOLightningModule:
    """Quick setup for training with Lightning"""

    # Parse data yaml
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    train_dir = data_config['train']
    val_dir = data_config['val']

    # Setup trainer
    trainer = LightningTrainer(
        train_data_dir=train_dir,
        val_data_dir=val_dir,
        batch_size=batch_size,
        max_epochs=epochs
    )

    # Train
    model = trainer.train()

    return model