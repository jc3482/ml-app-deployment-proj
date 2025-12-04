"""
Custom object detection module for ingredient detection.
"""

# Optional imports to avoid dependency issues (for API-only deployments)
try:
    from .detector import CustomDetector, IngredientDetector
except ImportError:
    CustomDetector = None
    IngredientDetector = None

try:
    from .preprocessor import ImagePreprocessor
except ImportError:
    ImagePreprocessor = None

try:
    from .dataset import YOLODataset
except ImportError:
    YOLODataset = None

try:
    from .loss import DetectionLoss
except ImportError:
    DetectionLoss = None

try:
    from .utils import visualize_detections
except ImportError:
    visualize_detections = None

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
