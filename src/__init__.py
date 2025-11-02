"""
SmartPantry: Recipes from Your Fridge Cam
AI-powered ingredient detection and recipe recommendation system.
"""

__version__ = "0.1.0"
__author__ = "SmartPantry Team"

from src.vision.detector import IngredientDetector
from src.vision.preprocessor import ImagePreprocessor
from src.nlp.embedder import IngredientEmbedder
from src.nlp.retriever import RecipeRetriever
from src.utils.helpers import load_config, setup_logging

__all__ = [
    "IngredientDetector",
    "ImagePreprocessor",
    "IngredientEmbedder",
    "RecipeRetriever",
    "load_config",
    "setup_logging",
]

