import torch
import yaml
import logging
from pathlib import Path
from src.vision.detector import CustomDetector


def export_model():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. Configuration
    data_yaml_path = Path("data/fridge_photos/data.yaml")
    checkpoint_path = Path("models/checkpoints/latest.pth")
    output_path = Path("models/ingredient_detector.pkl")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load Class Names
    logger.info("Loading class names...")
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
        class_names = data_config["names"]
        num_classes = len(class_names)

    # 3. Initialize Model
    logger.info(f"Initializing model with {num_classes} classes...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomDetector(num_classes=num_classes, class_names=class_names, device=device)

    # 4. Load Weights
    logger.info(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 5. Save Full Model
    logger.info(f"Saving full model to {output_path}...")
    torch.save(model, output_path)
    logger.info("Done! The model is packaged and ready to use.")
    logger.info(f"Teammate can load it using: model = torch.load('{output_path}')")


if __name__ == "__main__":
    export_model()
