"""
Tests for the recipe matching module.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recipe_matcher.retrieval_engine import match_recipe


class TestRecipeMatcher:
    """Test cases for RecipeMatcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = RecipeMatcher(fuzzy_threshold=0.8)
    
    def test_exact_match(self):
        """Test exact ingredient match."""
        score = self.matcher.fuzzy_match_ingredient("milk", "milk")
        assert score == 1.0
    
    def test_fuzzy_match(self):
        """Test fuzzy ingredient matching."""
        score = self.matcher.fuzzy_match_ingredient("tomato", "tomatoes")
        assert score > 0.8
    
    def test_substring_match(self):
        """Test substring matching."""
        score = self.matcher.fuzzy_match_ingredient("cheese", "cheddar cheese")
        assert score >= 0.9
    
    def test_no_match(self):
        """Test ingredients that don't match."""
        score = self.matcher.fuzzy_match_ingredient("apple", "banana")
        assert score < 0.5
    
    def test_match_ingredients_full_coverage(self):
        """Test matching with full ingredient coverage."""
        detected = ["milk", "apple", "cheese"]
        recipe = ["milk", "apple", "cheese"]
        
        result = self.matcher.match_ingredients(detected, recipe)
        
        assert result['coverage'] == 1.0
        assert result['num_matched'] == 3
        assert result['num_missing'] == 0
    
    def test_match_ingredients_partial_coverage(self):
        """Test matching with partial coverage."""
        detected = ["milk", "apple"]
        recipe = ["milk", "apple", "cheese", "flour"]
        
        result = self.matcher.match_ingredients(detected, recipe)
        
        assert result['coverage'] == 0.5
        assert result['num_matched'] == 2
        assert result['num_missing'] == 2
    
    def test_match_ingredients_fuzzy(self):
        """Test matching with fuzzy similarities."""
        detected = ["milk", "apple"]
        recipe = ["milk", "apples"]  # plural form
        
        result = self.matcher.match_ingredients(detected, recipe)
        
        assert result['num_matched'] == 2
        assert result['coverage'] == 1.0
    
    def test_match_recipe(self):
        """Test matching a complete recipe."""
        detected = ["milk", "apple", "butter"]
        recipe = {
            'title': 'Apple Tart',
            'normalized_ingredients': ['milk', 'apple', 'butter', 'flour']
        }
        
        result = self.matcher.match_recipe(detected, recipe)
        
        assert 'match_score' in result
        assert 'coverage' in result
        assert result['coverage'] == 0.75  # 3 out of 4
        assert 'missing_ingredients' in result
    
    def test_match_recipes_with_filter(self):
        """Test matching multiple recipes with minimum coverage."""
        detected = ["milk", "apple"]
        recipes = [
            {
                'title': 'Recipe 1',
                'normalized_ingredients': ['milk', 'apple']
            },
            {
                'title': 'Recipe 2',
                'normalized_ingredients': ['milk', 'apple', 'flour', 'sugar', 'butter']
            }
        ]
        
        # Only recipes with >= 60% coverage
        results = self.matcher.match_recipes(detected, recipes, min_coverage=0.6)
        
        # Recipe 1 has 100% coverage, Recipe 2 has 40% coverage
        assert len(results) == 1
        assert results[0]['title'] == 'Recipe 1'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

