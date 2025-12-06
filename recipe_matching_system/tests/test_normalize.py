"""
Tests for the ingredient normalization module.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recipe_matcher.preprocess import IngredientNormalizer


class TestIngredientNormalizer:
    """Test cases for IngredientNormalizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = IngredientNormalizer()
    
    def test_normalize_basic_ingredient(self):
        """Test basic ingredient normalization."""
        result = self.normalizer.normalize_ingredient("apple")
        assert result == "apple"
    
    def test_normalize_with_quantity(self):
        """Test normalization removes quantities."""
        result = self.normalizer.normalize_ingredient("2 cups of milk")
        assert "2" not in result
        assert "cups" not in result
        assert "milk" in result
    
    def test_normalize_with_measurements(self):
        """Test normalization removes measurement units."""
        result = self.normalizer.normalize_ingredient("100g shredded cheese")
        assert "100" not in result
        assert "g" not in result
        assert "cheese" in result
    
    def test_normalize_with_stopwords(self):
        """Test normalization removes stopwords."""
        result = self.normalizer.normalize_ingredient("fresh chopped onion")
        assert "fresh" not in result
        assert "chopped" not in result
        assert "onion" in result
    
    def test_normalize_list(self):
        """Test normalizing a list of ingredients."""
        ingredients = [
            "2 cups milk",
            "3 large apples",
            "100g cheese"
        ]
        result = self.normalizer.normalize_list(ingredients)
        
        assert len(result) == 3
        assert "milk" in result
        assert "apple" in result
        assert "cheese" in result
    
    def test_normalize_list_removes_duplicates(self):
        """Test that normalize_list removes duplicates."""
        ingredients = [
            "milk",
            "1 cup milk",
            "fresh milk"
        ]
        result = self.normalizer.normalize_list(ingredients)
        
        # Should have only one "milk" entry
        assert len(result) == 1
        assert result[0] == "milk"
    
    def test_extract_ingredients_from_recipe(self):
        """Test extracting ingredients from recipe string."""
        recipe_string = "2 cups milk, 3 apples, 100g cheese"
        result = self.normalizer.extract_ingredients_from_recipe(recipe_string)
        
        assert len(result) >= 3
        assert "milk" in result
        assert "apple" in result
        assert "cheese" in result
    
    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        result = self.normalizer.normalize_ingredient("")
        assert result == ""
    
    def test_normalize_none(self):
        """Test normalizing None."""
        result = self.normalizer.normalize_ingredient(None)
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

