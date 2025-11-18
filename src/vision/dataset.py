"""
YOLO format dataset for object detection.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import logging

from .transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class YOLODataset(Dataset):
    """
    Dataset for YOLO format object detection.

    Expected directory structure:
    data/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── labels/
        ├── img1.txt
        └── img2.txt

    Label format (YOLO):
    <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0-1).
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: int = 640,
        augment: bool = False,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize YOLO dataset.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files
            img_size: Target image size (square)
            augment: Whether to apply data augmentation
            class_names: List of class names (optional)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names or []

        # Get all image files
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        )

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

        # Get transforms
        if augment:
            self.transforms = get_train_transforms(img_size)
        else:
            self.transforms = get_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - image: Tensor of shape (3, H, W)
            - boxes: Tensor of shape (N, 4) with normalized bbox coordinates
            - labels: Tensor of shape (N,) with class indices
            - image_id: Image filename
        """
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # Load labels
        label_path = self.labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            # Try alternative path (in case of path mismatch)
            alt_label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if alt_label_path.exists():
                label_path = alt_label_path
        boxes, labels = self._load_labels(label_path, original_w, original_h)

        # Apply transforms (resize, normalize, augment)
        # Ensure boxes and labels are in correct format for albumentations
        if len(boxes) == 0:
            # Empty bboxes - albumentations needs empty list
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            boxes_list = []
            labels_list = []
        else:
            # Convert boxes to list of lists if needed
            boxes_for_aug = []
            labels_for_aug = []
            for box, label in zip(boxes, labels):
                if isinstance(box, (list, tuple)):
                    boxes_for_aug.append(list(box))
                else:
                    boxes_for_aug.append(box.tolist() if hasattr(box, "tolist") else list(box))
                labels_for_aug.append(int(label))

            transformed = self.transforms(
                image=image, bboxes=boxes_for_aug, class_labels=labels_for_aug
            )
            boxes_list = transformed.get("bboxes", [])
            labels_list = transformed.get("class_labels", [])

        image = transformed["image"]

        # Convert to tensors
        # Note: albumentations Normalize already normalizes to [0,1] range and applies mean/std
        # So we don't need to divide by 255
        if isinstance(image, np.ndarray):
            # If image is numpy array (from albumentations), convert directly
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        else:
            # Fallback: if not normalized, divide by 255
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Handle boxes and labels (albumentations may filter some boxes)
        if boxes_list and len(boxes_list) > 0:
            # Convert to tensor
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.long)
        else:
            # No objects in image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": img_path.stem,
            "original_size": (original_h, original_w),
        }

    def _load_labels(
        self, label_path: Path, img_width: int, img_height: int  # noqa: ARG002  # noqa: ARG002
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Load YOLO format labels.

        Args:
            label_path: Path to label file
            img_width: Original image width
            img_height: Original image height

        Returns:
            Tuple of (boxes, labels):
            - boxes: List of [x_center, y_center, width, height] in normalized coordinates
            - labels: List of class indices
        """
        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # YOLO format is already normalized, but we'll keep it as is
                boxes.append([x_center, y_center, width, height])
                labels.append(class_id)

        return boxes, labels

    @staticmethod
    def from_yaml(
        yaml_path: str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = False,
    ) -> "YOLODataset":
        """
        Create dataset from YOLO data.yaml file.

        Args:
            yaml_path: Path to data.yaml file
            split: Dataset split ('train', 'val', 'test')
            img_size: Target image size
            augment: Whether to apply augmentation

        Returns:
            YOLODataset instance
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Get split directory
        split_key = split if split != "val" else "val"
        if split_key not in data:
            raise ValueError(f"Split '{split_key}' not found in {yaml_path}")

        images_dir = data[split_key]
        # Convert to absolute path if relative
        if not os.path.isabs(images_dir):
            # Find project root by looking for common markers (pyproject.toml, setup.py, etc.)
            yaml_path_obj = Path(yaml_path).resolve()
            # Start from yaml file's directory and go up to find project root
            current_dir = yaml_path_obj.parent
            project_root = None

            # Go up until we find project root markers
            # We need to find the directory that contains both pyproject.toml and src/
            # Also ensure it's not a 'data' directory
            while current_dir.parent != current_dir:
                # Check for project root markers - must have pyproject.toml AND src/ directory
                has_pyproject = (current_dir / "pyproject.toml").exists()
                has_src = (current_dir / "src").exists()
                has_setup = (current_dir / "setup.py").exists()
                is_data_dir = current_dir.name == "data"

                # Strong indicator: has pyproject.toml and src directory, and is NOT a data directory
                if (has_pyproject and has_src) and not is_data_dir:
                    project_root = current_dir
                    break
                # Alternative: has setup.py and src directory, and is NOT a data directory
                elif (has_setup and has_src) and not is_data_dir:
                    project_root = current_dir
                    break

                current_dir = current_dir.parent

            # If we didn't find project root, try to infer from yaml path
            # If yaml is in data/fridge_photos/data.yaml, project root should be 2 levels up
            if project_root is None:
                # Fallback: assume project root is 2 levels up from yaml file
                # (data/fridge_photos/data.yaml -> project root)
                yaml_parts = yaml_path_obj.parts
                if "data" in yaml_parts and "fridge_photos" in yaml_parts:
                    # Find the index of 'data' and go to its parent
                    data_idx = yaml_parts.index("data")
                    if data_idx > 0:
                        project_root = Path(*yaml_parts[:data_idx])
                    else:
                        # Go up 2 levels: data/fridge_photos/data.yaml -> project root
                        project_root = yaml_path_obj.parent.parent
                else:
                    project_root = yaml_path_obj.parent

                # Verify this is actually the project root
                if (
                    not (project_root / "pyproject.toml").exists()
                    and not (project_root / "src").exists()
                ):
                    # If still not found, go up one more level
                    if project_root.parent != project_root:
                        project_root = project_root.parent

            # If path starts with 'data/', it's relative to project root
            # Otherwise, it's relative to yaml file location
            if images_dir.startswith("data/") or images_dir.startswith("./data/"):
                # Path is relative to project root
                # Remove leading 'data/' or './data/' and construct path
                clean_path = images_dir.lstrip("./")
                images_dir = str(project_root / clean_path)

                # Debug: log the resolved path
                logger.debug(f"Resolved images_dir: {images_dir}")
                logger.debug(f"Project root: {project_root}")
                logger.debug(f"Clean path: {clean_path}")
            else:
                # Path is relative to yaml file location
                yaml_dir = Path(yaml_path).parent
                images_dir = str(yaml_dir / images_dir)

        # Labels directory is in the same parent directory as images
        # e.g., train/images -> train/labels
        images_path = Path(images_dir)
        labels_dir = str(images_path.parent / "labels")

        # Verify paths exist
        if not Path(images_dir).exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not Path(labels_dir).exists():
            raise ValueError(f"Labels directory not found: {labels_dir}. Expected at: {labels_dir}")

        # Get class names
        class_names = data.get("names", [])

        return YOLODataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            img_size=img_size,
            augment=augment,
            class_names=class_names,
        )


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching variable-length targets.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary
    """
    images = torch.stack([item["image"] for item in batch])

    # Boxes and labels have variable lengths, keep as lists
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    original_sizes = [item["original_size"] for item in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "image_ids": image_ids,
        "original_sizes": original_sizes,
    }
