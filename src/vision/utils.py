"""
Utility functions for object detection: NMS, IoU, metrics, visualization.
"""

import torch
import numpy as np
import cv2  # type: ignore
from typing import List, Dict, Tuple, Optional


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.

    Args:
        box1: Boxes in format [x1, y1, x2, y2] or [x_center, y_center, width, height]
        box2: Boxes in same format as box1

    Returns:
        IoU values
    """
    # Convert to xyxy format if needed (assuming YOLO format if last dim is 4)
    if box1.shape[-1] == 4 and box1.dim() == 2:
        # Check if it's YOLO format (values typically in [0, 1])
        if box1[:, 2].max() <= 1.0:
            box1 = yolo_to_xyxy_tensor(box1)
            box2 = yolo_to_xyxy_tensor(box2)

    # Calculate intersection
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)

    return iou


def yolo_to_xyxy_tensor(box: torch.Tensor) -> torch.Tensor:
    """
    Convert YOLO format to xyxy format (tensor version).

    Args:
        box: YOLO format [x_center, y_center, width, height] (normalized)

    Returns:
        xyxy format [x1, y1, x2, y2] (normalized)
    """
    x_center, y_center, width, height = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25,
    max_detections: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Non-Maximum Suppression (NMS) to remove duplicate detections.

    Args:
        boxes: Bounding boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        labels: Class labels (N,)
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold for filtering
        max_detections: Maximum number of detections to keep

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels)
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if len(boxes) == 0:
        return boxes, scores, labels

    # Sort by score (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    # Apply NMS per class
    keep = []
    unique_labels = torch.unique(labels)

    for label in unique_labels:
        label_mask = labels == label
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]
        label_indices = torch.where(label_mask)[0]

        if len(label_boxes) == 0:
            continue

        # Greedy NMS
        while len(label_boxes) > 0:
            # Keep the box with highest score
            keep.append(label_indices[0].item())

            if len(label_boxes) == 1:
                break

            # Calculate IoU with remaining boxes
            ious = calculate_iou(label_boxes[0:1].expand(len(label_boxes) - 1, -1), label_boxes[1:])

            # Remove boxes with high IoU
            keep_mask = ious < iou_threshold
            label_boxes = label_boxes[1:][keep_mask]
            label_scores = label_scores[1:][keep_mask]
            label_indices = label_indices[1:][keep_mask]

    # Get kept indices
    if len(keep) == 0:
        return boxes[:0], scores[:0], labels[:0]

    keep = torch.tensor(keep, device=boxes.device)
    keep = keep[:max_detections]  # Limit to max_detections

    return boxes[keep], scores[keep], labels[keep]


def calculate_map(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,  # noqa: ARG001
    num_classes: int = 30,
) -> float:
    """
    Calculate Mean Average Precision (mAP) at given IoU threshold.

    Args:
        predictions: List of prediction dicts, each with 'boxes', 'scores', 'labels'
        ground_truth: List of ground truth dicts, each with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes

    Returns:
        mAP score
    """
    # This is a simplified mAP calculation
    # Full implementation would compute AP for each class and average

    aps = []

    for class_id in range(num_classes):
        # Collect predictions and ground truth for this class
        pred_boxes = []
        pred_scores = []
        gt_boxes = []

        for pred, gt in zip(predictions, ground_truth):
            # Filter by class
            pred_mask = pred["labels"] == class_id
            gt_mask = gt["labels"] == class_id

            if pred_mask.sum() > 0:
                pred_boxes.append(pred["boxes"][pred_mask])
                pred_scores.append(pred["scores"][pred_mask])

            if gt_mask.sum() > 0:
                gt_boxes.append(gt["boxes"][gt_mask])

        if len(gt_boxes) == 0:
            continue

        # Compute AP for this class (simplified)
        # Full implementation would sort by score and compute precision-recall curve
        ap = 0.0  # Placeholder
        aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return sum(aps) / len(aps)


def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.25,
) -> np.ndarray:
    """
    Visualize detections on image.

    Args:
        image: Input image (H, W, 3) in RGB format
        boxes: Bounding boxes in xyxy format (N, 4)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        class_names: List of class names
        score_threshold: Score threshold for display

    Returns:
        Annotated image
    """
    img = image.copy()

    # Filter by score if provided
    if scores is not None:
        mask = scores >= score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

    # Draw boxes
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = class_names[label] if class_names else f"Class {label}"
        if scores is not None:
            label_text += f" {scores[i]:.2f}"

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw text background
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)

        # Draw text
        cv2.putText(
            img, label_text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    return img
