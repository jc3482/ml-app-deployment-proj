"""
Recipe Matching Engine
A system for matching detected ingredients with recipes from a large dataset.

Architecture: Retrieve & Rank
- Stage 1: Fast candidate retrieval (RecipeRetriever)
- Stage 2: Precise ranking (RecipeRanker)
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .preprocess import IngredientNormalizer, IngredientOntology
from .retrieval_engine import RecipeRetriever, RecipeRanker, match_recipe
from .pipeline import RecipePipeline

__all__ = [
    'IngredientNormalizer',
    'IngredientOntology',
    'RecipeRetriever',
    'RecipeRanker',
    'RecipePipeline',
    'match_recipe'
]

