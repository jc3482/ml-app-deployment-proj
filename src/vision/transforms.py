"""
Data augmentation and preprocessing transforms for object detection.
"""

import numpy as np
import albumentations as A


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """
    Get training data augmentation pipeline.

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations compose transform
    """
    return A.Compose(
        [
            # Geometric transforms
            A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=(114, 114, 114), p=1.0
            ),
            # Augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            # Normalize to ImageNet statistics
            # This normalizes to [0,1] range first, then applies mean/std
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",  # YOLO format: normalized (x_center, y_center, width, height)
            label_fields=["class_labels"],
            min_visibility=0.1,  # Keep bboxes with at least 10% visibility
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """
    Get validation data preprocessing pipeline (expects bboxes).

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations compose transform
    """
    return A.Compose(
        [
            # Resize and pad (same as training, but no augmentation)
            A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=(114, 114, 114), p=1.0
            ),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
        ),
    )


def get_inference_transforms(img_size: int = 640) -> A.Compose:
    """
    Get inference data preprocessing pipeline (NO bboxes expected).

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations compose transform
    """
    return A.Compose(
        [
            # Resize and pad
            A.LongestMaxSize(max_size=img_size, interpolation=1, p=1.0),
            A.PadIfNeeded(
                min_height=img_size, min_width=img_size, border_mode=0, value=(114, 114, 114), p=1.0
            ),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0
            ),
        ]
    )


def denormalize_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert normalized YOLO bbox to pixel coordinates.

    Args:
        bbox: Normalized bbox [x_center, y_center, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Pixel coordinates [x_center, y_center, width, height]
    """
    x_center, y_center, width, height = bbox
    return np.array(
        [
            x_center * img_width,
            y_center * img_height,
            width * img_width,
            height * img_height,
        ]
    )


def yolo_to_xyxy(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert YOLO format bbox to xyxy format.

    Args:
        bbox: YOLO bbox [x_center, y_center, width, height] (normalized)
        img_width: Image width
        img_height: Image height

    Returns:
        xyxy bbox [x1, y1, x2, y2] (pixel coordinates)
    """
    x_center, y_center, width, height = bbox

    # Denormalize
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Convert to xyxy
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return np.array([x1, y1, x2, y2])


def xyxy_to_yolo(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert xyxy format bbox to YOLO format.

    Args:
        bbox: xyxy bbox [x1, y1, x2, y2] (pixel coordinates)
        img_width: Image width
        img_height: Image height

    Returns:
        YOLO bbox [x_center, y_center, width, height] (normalized)
    """
    x1, y1, x2, y2 = bbox

    # Convert to center and size
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    # Normalize
    return np.array(
        [
            x_center / img_width,
            y_center / img_height,
            width / img_width,
            height / img_height,
        ]
    )
