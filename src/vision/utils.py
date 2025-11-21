"""
Utility functions for visualization and metrics.
"""

import torch
import numpy as np
import cv2  # type: ignore
from typing import List, Dict, Optional


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,  # Ignored for classification
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.25,
) -> np.ndarray:
    """
    Visualize detected ingredients as a list on the image.

    Args:
        image: Input image (H, W, 3) in RGB format
        boxes: Ignored (kept for interface compatibility)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        class_names: List of class names
        score_threshold: Score threshold for display

    Returns:
        Annotated image
    """
    img = image.copy()

    # Collect valid predictions
    valid_preds = []
    for i, label in enumerate(labels):
        score = scores[i] if scores is not None else 1.0
        if score >= score_threshold:
            name = class_names[label] if class_names else f"Class {label}"
            valid_preds.append((name, score))

    # Sort by score descending
    valid_preds.sort(key=lambda x: x[1], reverse=True)

    # Draw text list on image (top-left corner with background)
    if valid_preds:
        # Settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        margin = 10
        line_height = 30

        # Calculate box size
        max_width = 0
        for name, score in valid_preds:
            text = f"{name}: {score:.1%}"
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, w)

        box_w = max_width + 2 * margin
        box_h = len(valid_preds) * line_height + margin

        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw text
        y = margin + 20
        for name, score in valid_preds:
            text = f"{name}: {score:.1%}"
            cv2.putText(img, text, (margin, y), font, font_scale, (255, 255, 255), thickness)
            y += line_height

    return img
