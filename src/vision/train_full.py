"""
Full training script for custom detector on complete dataset.
"""

import torch
import sys
from pathlib import Path
import logging
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)

# Now import modules
from src.vision.dataset import YOLODataset
from src.vision.detector import CustomDetector
from src.vision.trainer import DetectionTrainer

# Setup logging - create logs directory first
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU (training will be slow)")

    # Training configuration
    config = {
        "data": {
            "yaml_path": "data/fridge_photos/data.yaml",
            "img_size": 640,
        },
        "model": {
            "num_classes": 30,
            "backbone_pretrained": True,
            "fpn_out_channels": 256,
            "head_hidden_channels": 256,
        },
        "training": {
            "epochs": 100,
            "batch_size": 8,  # Adjust based on GPU memory
            "learning_rate": 0.00005,  # Very conservative - regression loss was exploding
            "weight_decay": 0.0005,
            "num_workers": 4,  # Adjust based on CPU cores
            "save_dir": "models/checkpoints",
            "log_dir": f"runs/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "save_interval": 10,  # Save checkpoint every 10 epochs
            "img_size": 640,
            "cls_weight": 1.0,
            "reg_weight": 2.0,  # Reduced from 5.0 - regression loss was exploding
            "obj_weight": 1.0,
            "warmup_epochs": 10,  # Extended warmup for very conservative start
            "grad_clip_norm": 1.0,  # Very aggressive gradient clipping
        },
    }

    # Adjust batch size based on device
    if device == "cpu":
        config["training"]["batch_size"] = 2
        config["training"]["num_workers"] = 0
        logger.info("Reduced batch size to 2 for CPU training")
    elif device == "mps":
        config["training"]["batch_size"] = 4
        logger.info("Reduced batch size to 4 for MPS")

    logger.info("=" * 60)
    logger.info("Starting Full Training")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Image size: {config['data']['img_size']}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)

    # Create datasets
    logger.info("Loading datasets...")
    try:
        train_dataset = YOLODataset.from_yaml(
            yaml_path=config["data"]["yaml_path"],
            split="train",
            img_size=config["data"]["img_size"],
            augment=True,
        )

        val_dataset = YOLODataset.from_yaml(
            yaml_path=config["data"]["yaml_path"],
            split="val",
            img_size=config["data"]["img_size"],
            augment=False,
        )

        logger.info(f"✓ Training samples: {len(train_dataset)}")
        logger.info(f"✓ Validation samples: {len(val_dataset)}")

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create model
    logger.info("Creating model...")
    try:
        model = CustomDetector(
            num_classes=config["model"]["num_classes"],
            backbone_pretrained=config["model"]["backbone_pretrained"],
            fpn_out_channels=config["model"]["fpn_out_channels"],
            head_hidden_channels=config["model"]["head_hidden_channels"],
        )

        model_info = model.get_model_info()
        logger.info(f"✓ Model created")
        logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"  Trainable parameters: {model_info['trainable_parameters']:,}")

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create trainer
    logger.info("Initializing trainer...")
    try:
        trainer = DetectionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config["training"],
            device=device,
        )
        logger.info("✓ Trainer initialized")

    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        import traceback

        traceback.print_exc()
        return

    # Start training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    logger.info("You can monitor training progress with TensorBoard:")
    logger.info(f"  tensorboard --logdir {config['training']['log_dir']}")
    logger.info("=" * 60)

    try:
        trainer.train()
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {config['training']['save_dir']}/best.pth")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving current checkpoint...")
        trainer.save_checkpoint()
        logger.info("Checkpoint saved")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        logger.info("Saving current checkpoint before exit...")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main()
