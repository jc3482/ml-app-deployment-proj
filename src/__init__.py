"""
SmartPantry: Recipes from Your Fridge Cam
AI-powered ingredient detection and recipe recommendation system.
"""

__version__ = "0.1.0"
__author__ = "SmartPantry Team"

# Import commonly used classes (using lazy imports to avoid dependency issues)
# These can be imported directly from submodules if needed
# Use try-except to make imports optional (for API-only deployments)
# NOTE: We avoid importing from src.vision.detector here because it triggers
# tensorboard imports which have numpy version conflicts

# Lazy imports - only import when actually needed
IngredientDetector = None
ImagePreprocessor = None
IngredientEmbedder = None
RecipeRetriever = None
load_config = None
setup_logging = None

def _lazy_import_ingredient_detector():
    """Lazy import to avoid dependency issues."""
    global IngredientDetector
    if IngredientDetector is None:
        try:
            from src.vision.detector import IngredientDetector as _IngredientDetector
            IngredientDetector = _IngredientDetector
        except ImportError:
            pass
    return IngredientDetector

def _lazy_import_image_preprocessor():
    """Lazy import to avoid dependency issues."""
    global ImagePreprocessor
    if ImagePreprocessor is None:
        try:
            from src.vision.preprocessor import ImagePreprocessor as _ImagePreprocessor
            ImagePreprocessor = _ImagePreprocessor
        except ImportError:
            pass
    return ImagePreprocessor

__all__ = [
    "IngredientDetector",
    "ImagePreprocessor",
    "IngredientEmbedder",
    "RecipeRetriever",
    "load_config",
    "setup_logging",
]

