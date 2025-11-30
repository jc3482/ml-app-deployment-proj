"""
Custom object detection module for ingredient detection.
"""

from .detector import CustomDetector, IngredientDetector
from .preprocessor import ImagePreprocessor
from .dataset import YOLODataset
from .loss import DetectionLoss
from .utils import visualize_detections

# Optional import for DetectionTrainer (requires tensorboard)
# This allows the API to run without tensorboard dependency
try:
    from .trainer import DetectionTrainer
    _HAS_TRAINER = True
except ImportError:
    # DetectionTrainer requires tensorboard, which may not be installed
    DetectionTrainer = None
    _HAS_TRAINER = False

__all__ = [
    "CustomDetector",
    "IngredientDetector",  # Alias for backward compatibility
    "ImagePreprocessor",
    "YOLODataset",
    "DetectionLoss",
    "visualize_detections",
]

# Add DetectionTrainer to __all__ only if available
if _HAS_TRAINER:
    __all__.append("DetectionTrainer")
