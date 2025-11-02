"""
Unit tests for recipe retriever.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.retriever import RecipeRetriever


class TestRecipeRetriever:
    """Test cases for RecipeRetriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever instance for testing."""
        return RecipeRetriever(
            recipe_database_path=None,
            top_k=10,
            min_ingredient_match=0.3,
        )
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initializes correctly."""
        assert retriever is not None
        assert retriever.top_k == 10
        assert retriever.min_ingredient_match == 0.3
    
    def test_calculate_ingredient_overlap(self, retriever):
        """Test ingredient overlap calculation."""
        detected = ["milk", "eggs", "flour"]
        recipe = ["milk", "eggs", "sugar", "butter"]
        
        overlap, matched, missing = retriever.calculate_ingredient_overlap(
            detected, recipe
        )
        
        assert overlap == 0.5  # 2 out of 4 matched
        assert len(matched) == 2
        assert len(missing) == 2
    
    def test_filter_recipes(self, retriever):
        """Test recipe filtering."""
        recipes = [
            {"cuisine": "Italian", "difficulty": "Easy", "cooking_time": 30, "overlap_score": 0.8},
            {"cuisine": "Japanese", "difficulty": "Hard", "cooking_time": 60, "overlap_score": 0.5},
            {"cuisine": "Italian", "difficulty": "Medium", "cooking_time": 45, "overlap_score": 0.2},
        ]
        
        filters = {"cuisine": ["Italian"], "max_cooking_time": 40}
        filtered = retriever.filter_recipes(recipes, filters)
        
        assert len(filtered) == 0  # Third recipe filtered by min_ingredient_match
    
    # TODO: Add more tests
    # - test_retrieve_recipes
    # - test_rank_recipes
    # - test_update_index

