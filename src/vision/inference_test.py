"""
Test inference with the trained model on sample images.
"""

import torch
import sys
from pathlib import Path
import logging
import yaml
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vision.detector import CustomDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_class_names(yaml_path):
    """Load class names from data.yaml"""
    if not Path(yaml_path).exists():
        return [f"Class {i}" for i in range(30)]

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        return data.get("names", [])


def main():
    # Configuration
    checkpoint_path = project_root / "models/checkpoints/latest.pth"
    # Also try best.pth if it exists
    best_checkpoint = project_root / "models/checkpoints/best.pth"
    if best_checkpoint.exists():
        logger.info("Found best.pth, using it instead of latest.pth")
        checkpoint_path = best_checkpoint

    data_yaml_path = project_root / "data/fridge_photos/data.yaml"
    test_images_dir = project_root / "data/fridge_photos/test/images"

    # Device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 1. Load Class Names
    class_names = load_class_names(data_yaml_path)
    num_classes = len(class_names)
    logger.info(f"Loaded {num_classes} classes: {class_names[:5]}...")

    # 2. Load Model
    logger.info(f"Loading model from {checkpoint_path}...")

    # Initialize model structure
    model = CustomDetector(num_classes=num_classes, class_names=class_names, device=device).to(
        device
    )

    # Load weights
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Model weights loaded successfully.")
    else:
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    model.eval()

    # 3. Select Test Images
    if not test_images_dir.exists():
        logger.error(f"Test images directory not found: {test_images_dir}")
        return

    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_files:
        logger.error("No images found in test directory.")
        return

    # Pick 3 random images
    sample_images = random.sample(image_files, min(3, len(image_files)))

    # 4. Run Inference
    logger.info("\n=== Starting Inference Test ===\n")

    for img_path in sample_images:
        logger.info(f"Testing image: {img_path.name}")

        try:
            # Predict
            # Lower threshold to see what the model is thinking, even if low confidence
            results = model.detect_ingredients(str(img_path), visualize=True, conf_threshold=0.1)

            ingredients = results["ingredients"]
            confidences = results["confidences"]

            print(f"  Detected {len(ingredients)} ingredients:")
            for ing, conf in zip(ingredients, confidences):
                print(f"    - {ing}: {conf:.1%}")

            if len(ingredients) == 0:
                print("    (No ingredients detected above threshold)")

            print("-" * 40)

        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")


if __name__ == "__main__":
    main()
