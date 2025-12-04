"""
SmartPantry: Recipes from Your Fridge Cam
AI-powered ingredient detection and recipe recommendation system.
"""

__version__ = "0.1.0"
__author__ = "SmartPantry Team"

# Import commonly used classes (using lazy imports to avoid dependency issues)
# These can be imported directly from submodules if needed
# Use try-except to make imports optional (for API-only deployments)
try:
    from src.vision.detector import IngredientDetector
except ImportError:
    IngredientDetector = None

try:
    from src.vision.preprocessor import ImagePreprocessor
except ImportError:
    ImagePreprocessor = None

try:
    from src.nlp.embedder import IngredientEmbedder
except ImportError:
    IngredientEmbedder = None

try:
    from src.nlp.retriever import RecipeRetriever
except ImportError:
    RecipeRetriever = None

try:
    from src.utils.helpers import load_config, setup_logging
except ImportError:
    load_config = None
    setup_logging = None

__all__ = [
    "IngredientDetector",
    "ImagePreprocessor",
    "IngredientEmbedder",
    "RecipeRetriever",
    "load_config",
    "setup_logging",
]

