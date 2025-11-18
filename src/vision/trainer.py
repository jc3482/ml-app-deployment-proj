"""
Training script for custom detector.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import yaml
from tqdm import tqdm
import logging

from .detector import CustomDetector
from .dataset import YOLODataset, collate_fn
from .loss import DetectionLoss
from .utils import calculate_map, non_max_suppression

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """
    Trainer class for object detection model.
    """

    def __init__(
        self,
        model: CustomDetector,
        train_dataset: YOLODataset,
        val_dataset: YOLODataset,
        config: Dict,
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            model: CustomDetector model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        # Training parameters
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 8)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0005)
        self.save_dir = Path(config.get("save_dir", "models/checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Loss function
        self.criterion = DetectionLoss(
            num_classes=model.num_classes,
            img_size=config.get("img_size", 640),
            cls_weight=config.get("cls_weight", 1.0),
            reg_weight=config.get("reg_weight", 5.0),
            obj_weight=config.get("obj_weight", 1.0),
        ).to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.warmup_epochs = config.get("warmup_epochs", 0)
        self.base_lr = self.learning_rate

        # Main scheduler (CosineAnnealingLR)
        self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=self.learning_rate * 0.01,
        )

        # Warmup scheduler (linear warmup)
        if self.warmup_epochs > 0:
            # Start with very small learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * 0.01

        # TensorBoard writer
        log_dir = config.get("log_dir", "runs/detection")
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.best_map = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_obj_loss = 0.0
        num_batches = 0

        # Create progress bar with smoother updates
        total_batches = len(self.train_loader)
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.epochs}",
            mininterval=0.5,  # Update at most every 0.5 seconds
            smoothing=0.1,  # Smooth the loss display
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(self.device)
            targets = {
                "images": images,
                "boxes": batch["boxes"],
                "labels": batch["labels"],
            }

            # Forward pass
            cls_pred, reg_pred, obj_pred = self.model(images)

            # Compute loss
            loss_dict = self.criterion(cls_pred, reg_pred, obj_pred, targets)
            loss = loss_dict["loss"]

            # Check for NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (more aggressive for stability)
            grad_clip_norm = self.config.get("grad_clip_norm", 5.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=grad_clip_norm
            )

            # Log gradient norm if it's too large
            if grad_norm > grad_clip_norm * 0.9:
                logger.warning(f"Large gradient norm: {grad_norm:.4f}")

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_cls_loss += loss_dict["cls_loss"].item()
            total_reg_loss += loss_dict["reg_loss"].item()
            total_obj_loss += loss_dict["obj_loss"].item()
            num_batches += 1

            # Update progress bar less frequently (every 10 batches or at end)
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                avg_loss_so_far = total_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss_so_far:.4f}"})

        # Average losses
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        avg_obj_loss = total_obj_loss / num_batches

        return {
            "loss": avg_loss,
            "cls_loss": avg_cls_loss,
            "reg_loss": avg_reg_loss,
            "obj_loss": avg_obj_loss,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_ground_truth = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False, mininterval=0.5):
            images = batch["images"].to(self.device)
            targets = {
                "images": images,
                "boxes": batch["boxes"],
                "labels": batch["labels"],
            }

            # Forward pass
            cls_pred, reg_pred, obj_pred = self.model(images)

            # Compute loss
            loss_dict = self.criterion(cls_pred, reg_pred, obj_pred, targets)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

            # Collect predictions for mAP calculation
            # (Simplified - full implementation would decode predictions)
            for i in range(len(batch["boxes"])):
                all_ground_truth.append(
                    {
                        "boxes": batch["boxes"][i],
                        "labels": batch["labels"][i],
                    }
                )
                # Placeholder for predictions
                all_predictions.append(
                    {
                        "boxes": batch["boxes"][i],  # Placeholder
                        "labels": batch["labels"][i],  # Placeholder
                        "scores": torch.ones(len(batch["labels"][i])),  # Placeholder
                    }
                )

        avg_loss = total_loss / num_batches

        # Calculate mAP (simplified)
        # map_score = calculate_map(all_predictions, all_ground_truth)
        map_score = 0.0  # Placeholder

        return {
            "val_loss": avg_loss,
            "val_map": map_score,
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.main_scheduler.state_dict(),
            "best_map": self.best_map,
            "config": self.config,
        }

        # Save latest
        checkpoint_path = self.save_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with mAP: {self.best_map:.4f}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Try to load scheduler state (may not exist for old checkpoints)
        if "scheduler_state_dict" in checkpoint:
            try:
                self.main_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (KeyError, AttributeError):
                logger.warning("Could not load scheduler state, using default")
        self.current_epoch = checkpoint["epoch"]
        self.best_map = checkpoint.get("best_map", 0.0)
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, resume_from: Optional[Path] = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate (with warmup)
            if epoch < self.warmup_epochs:
                # Linear warmup
                warmup_lr = (
                    self.base_lr * 0.01
                    + (self.base_lr - self.base_lr * 0.01) * (epoch + 1) / self.warmup_epochs
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            else:
                # Cosine annealing after warmup
                self.main_scheduler.step()

            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"Train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"Val/{key}", value, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # Save checkpoint
            is_best = val_metrics["val_map"] > self.best_map
            if is_best:
                self.best_map = val_metrics["val_map"]

            if (epoch + 1) % self.config.get("save_interval", 10) == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val mAP: {val_metrics['val_map']:.4f}"
            )

        logger.info("Training completed!")
        self.writer.close()


def train_from_config(config_path: str, device: str = "cuda"):
    """
    Train model from configuration file.

    Args:
        config_path: Path to training configuration YAML file
        device: Device to train on
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create datasets
    data_config = config["data"]
    train_dataset = YOLODataset.from_yaml(
        yaml_path=data_config["yaml_path"],
        split="train",
        img_size=data_config.get("img_size", 640),
        augment=True,
    )

    val_dataset = YOLODataset.from_yaml(
        yaml_path=data_config["yaml_path"],
        split="val",
        img_size=data_config.get("img_size", 640),
        augment=False,
    )

    # Create model
    model_config = config["model"]
    model = CustomDetector(
        num_classes=model_config.get("num_classes", 30),
        backbone_pretrained=model_config.get("backbone_pretrained", True),
        fpn_out_channels=model_config.get("fpn_out_channels", 256),
        head_hidden_channels=model_config.get("head_hidden_channels", 256),
    )

    # Create trainer
    trainer = DetectionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config["training"],
        device=device,
    )

    # Train
    trainer.train()
