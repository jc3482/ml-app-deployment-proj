"""
Tests for the recipe matching pipeline.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recipe_matcher.pipeline import RecipeMatchingPipeline


class TestRecipeMatchingPipeline:
    """Test cases for RecipeMatchingPipeline class."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample recipe dataset."""
        # Create temporary CSV file
        data = {
            'Title': ['Apple Pie', 'Cheese Pasta', 'Milk Shake'],
            'Ingredients': [
                'apple, flour, sugar, butter',
                'pasta, cheese, milk, butter',
                'milk, ice cream, banana'
            ],
            'Cuisine': ['American', 'Italian', 'American'],
            'TotalTimeInMins': [60, 30, 10],
            'Difficulty': ['Medium', 'Easy', 'Easy'],
            'Rating': [4.5, 4.0, 4.8],
            'Instructions': ['Mix and bake', 'Cook pasta', 'Blend all'],
        }
        df = pd.DataFrame(data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            return f.name
    
    def test_pipeline_initialization(self, sample_dataset):
        """Test pipeline initialization."""
        pipeline = RecipeMatchingPipeline(sample_dataset)
        
        assert pipeline.recipes_df is not None
        assert len(pipeline.recipes_list) > 0
    
    def test_pipeline_run_basic(self, sample_dataset):
        """Test basic pipeline run."""
        pipeline = RecipeMatchingPipeline(sample_dataset)
        
        detected = ["milk", "cheese", "butter"]
        results = pipeline.run(detected, top_n=5, min_coverage=0.3)
        
        assert 'summary' in results
        assert 'recipes' in results
        assert len(results['recipes']) > 0
    
    def test_pipeline_with_filters(self, sample_dataset):
        """Test pipeline with filters applied."""
        pipeline = RecipeMatchingPipeline(sample_dataset)
        
        detected = ["milk", "cheese"]
        results = pipeline.run(
            detected,
            top_n=5,
            min_coverage=0.3,
            cuisines=['Italian'],
            max_time=60
        )
        
        assert 'recipes' in results
        # Should find Cheese Pasta (Italian cuisine)
        if len(results['recipes']) > 0:
            assert any('Pasta' in r['title'] for r in results['recipes'])
    
    def test_pipeline_no_matches(self, sample_dataset):
        """Test pipeline when no recipes match."""
        pipeline = RecipeMatchingPipeline(sample_dataset)
        
        detected = ["xyz", "abc"]  # Non-existent ingredients
        results = pipeline.run(detected, top_n=5, min_coverage=0.8)
        
        assert results['summary']['total_recipes_found'] == 0
        assert len(results['recipes']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

