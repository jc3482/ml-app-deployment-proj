"""
Custom object detection module for ingredient detection.
"""

from .detector import CustomDetector, IngredientDetector
from .preprocessor import ImagePreprocessor
from .dataset import YOLODataset
from .trainer import DetectionTrainer
from .loss import DetectionLoss
from .utils import visualize_detections

__all__ = [
    "CustomDetector",
    "IngredientDetector",  # Alias for backward compatibility
    "ImagePreprocessor",
    "YOLODataset",
    "DetectionTrainer",
    "DetectionLoss",
    "visualize_detections",
]
