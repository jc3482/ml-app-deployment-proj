"""
RecipePipeline
Data processing and matching pipeline:
1. Raw CSV → preprocess.py (normalize) → normalized_recipes.json
2. normalized_recipes.json → preprocess.py (ontology) → ontology_recipes.json 
3. Load processed data → retrieval_engine.py (retrieve & rank) → ranked results
"""

import pandas as pd
import ast
from pathlib import Path

from recipe_matcher.preprocess import IngredientNormalizer, IngredientOntology
from recipe_matcher.retrieval_engine import RecipeRetriever, RecipeRanker
from recipe_matcher.utils.helpers import (
    load_raw_recipes,
    load_normalized_recipes,
    load_ontology_recipes,
    save_normalized_recipes,
    save_ontology_recipes
)


# =============================================================================
# Step 1: Normalize Raw Dataset
# =============================================================================

def normalize_recipe_dataset(input_path=None, output_json=True, output_csv=True):
    """
    Process raw recipe dataset with preprocess.IngredientNormalizer
    
    Pipeline:
        raw CSV → IngredientNormalizer → normalized_recipes.json/csv
    
    Args:
        input_path: Path to raw CSV (optional)
        output_json: Save JSON format
        output_csv: Save CSV format
    
    Returns:
        DataFrame with normalized_ingredients column
    """
    print("="*70)
    print("Step 1: Recipe Dataset Normalization")
    print("="*70)
    
    # Load raw recipes
    print("\nLoading raw recipe dataset...")
    df = load_raw_recipes(input_path)
    print(f"Loaded {len(df)} recipes")
    
    # Initialize normalizer
    print("\nInitializing IngredientNormalizer...")
    normalizer = IngredientNormalizer()
    
    # Parse ingredients column
    print("\nParsing ingredients...")
    if 'ingredients' in df.columns:
        if isinstance(df['ingredients'].iloc[0], str):
            try:
                df['ingredients_parsed'] = df['ingredients'].apply(ast.literal_eval)
            except:
                df['ingredients_parsed'] = df['ingredients'].apply(lambda x: [x])
        else:
            df['ingredients_parsed'] = df['ingredients']
    else:
        raise ValueError("'ingredients' column not found in dataset")
    
    # Normalize all ingredients
    print("\nApplying normalization...")
    df['normalized_ingredients'] = df['ingredients_parsed'].apply(
        lambda ing_list: normalizer.normalize_list(ing_list)
    )
    
    # Filter recipes with no valid ingredients
    df['ingredient_count'] = df['normalized_ingredients'].apply(len)
    original_count = len(df)
    df = df[df['ingredient_count'] > 0].copy()
    removed_count = original_count - len(df)
    
    print(f"Normalized {len(df)} recipes")
    print(f"Removed {removed_count} recipes with no valid ingredients")
    
    # Save
    print("\nSaving normalized dataset...")
    save_normalized_recipes(df, save_json=output_json, save_csv=output_csv)
    
    print("\n" + "="*70)
    print("Normalization Complete")
    print(f"Total recipes: {len(df)}")
    print(f"Avg ingredients per recipe: {df['ingredient_count'].mean():.1f}")
    print("="*70)
    
    return df


# =============================================================================
# Step 2: Apply Ontology Processing (Optional)
# =============================================================================

def apply_ontology_processing(input_path=None, output_json=True, output_csv=True, output_pkl=True):
    """
    Apply ontology processing to normalized dataset.
    
    Pipeline:
        normalized_recipes.json → preprocess.IngredientOntology → ontology_recipes.json/csv
    
    Args:
        input_path: Path to normalized recipes (optional)
        output_json: Save JSON format
        output_csv: Save CSV format
        output_pkl: Save PKL format
    
    Returns:
        DataFrame with ontology_ingredients column
    """
    print("="*70)
    print("Step 2: Ontology Processing")
    print("="*70)
    
    # Load normalized recipes
    print("\nLoading normalized recipe dataset...")
    df = load_normalized_recipes(input_path)
    print(f"Loaded {len(df)} recipes")
    
    # Initialize ontology
    print("\nInitializing IngredientOntology...")
    ontology = IngredientOntology()
    
    # Apply ontology mapping
    print("\nApplying ontology canonicalization...")
    df['ontology_ingredients'] = df['normalized_ingredients'].apply(
        lambda ing_list: ontology.process_list(ing_list)[0]
    )
    
    # Update count
    df['ontology_ingredient_count'] = df['ontology_ingredients'].apply(len)
    
    # Save
    print("\nSaving ontology-processed dataset...")
    save_ontology_recipes(df, save_json=output_json, save_csv=output_csv, save_pkl=output_pkl)
    
    print("\n" + "="*70)
    print("Ontology Processing Complete")
    print(f"Total recipes: {len(df)}")
    print(f"Avg ontology ingredients per recipe: {df['ontology_ingredient_count'].mean():.1f}")
    print("="*70)
    
    return df


# =============================================================================
# Main Matching Pipeline
# =============================================================================

class RecipePipeline:
    """
    Full recipe matching pipeline with Retrieve & Rank architecture.
    
    Pipeline stages:
        1. User input → normalize → ontology (optional)
        2. Retrieve: Fast candidate recall (retrieval_engine.RecipeRetriever)
        3. Rank: Precise scoring and sorting (retrieval_engine.RecipeRanker)
    
    Usage:
        # Option 1: Use ontology-processed data (recommended)
        pipeline = RecipePipeline(use_ontology=True)
        
        # Option 2: Use normalized data only
        pipeline = RecipePipeline(use_ontology=False)
        
        # Run matching
        user_ings = ["apple", "milk", "butter"]
        normalized_ings, top_recipes = pipeline.run(user_ings, top_k=5)
    """
    
    def __init__(self, use_ontology=True, recipe_path=None):
        """
        Initialize pipeline.
        
        Args:
            use_ontology: If True, use ontology-processed data; else use normalized data
            recipe_path: Custom path to recipe dataset (optional)
        """
        print("Initializing Recipe Matching Pipeline...")
        
        self.use_ontology = use_ontology
        
        # Load recipe dataset
        if use_ontology:
            try:
                self.recipes = load_ontology_recipes(recipe_path)
                self.ingredient_column = 'ontology_ingredients'
                print(f"Loaded {len(self.recipes)} ontology-processed recipes")
            except FileNotFoundError:
                print("Warning: Ontology dataset not found, falling back to normalized")
                self.recipes = load_normalized_recipes(recipe_path)
                self.ingredient_column = 'normalized_ingredients'
                self.use_ontology = False
                print(f"Loaded {len(self.recipes)} normalized recipes")
        else:
            self.recipes = load_normalized_recipes(recipe_path)
            self.ingredient_column = 'normalized_ingredients'
            print(f"Loaded {len(self.recipes)} normalized recipes")
        
        # Initialize normalizer and ontology
        self.normalizer = IngredientNormalizer()
        if self.use_ontology:
            self.ontology = IngredientOntology()
        
        # Initialize retriever and ranker
        self.retriever = RecipeRetriever(self.recipes)
        self.ranker = RecipeRanker()
        
        print("Pipeline ready (Retrieve & Rank architecture)")
    
    def process_user_ingredients(self, raw_ingredients):
        """
        Process user's raw ingredients through same pipeline as recipes.
        
        Args:
            raw_ingredients: List of raw ingredient strings
        
        Returns:
            List of processed ingredient strings
        """
        # Step 1: Normalize
        normalized = self.normalizer.normalize_list(raw_ingredients)
        
        # Step 2: Apply ontology if enabled
        if self.use_ontology:
            processed = self.ontology.process_list(normalized)[0]
        else:
            processed = normalized
        
        return processed
    
    def retrieve_candidates(self, user_ingredients, top_k=300):
        """
        Stage 1: Fast retrieval of candidate recipes.
        
        Uses RecipeRetriever for efficient candidate recall:
        - Ontology-based recall (exact matches on canonical ingredients)
        - Fuzzy recall (similarity > 75%)
        
        Args:
            user_ingredients: List of processed user ingredients
            top_k: Maximum number of candidates to retrieve
        
        Returns:
            DataFrame of candidate recipes
        """
        candidates = self.retriever.retrieve(user_ingredients, top_k=top_k)
        return candidates
    
    def rank_candidates(self, candidates, user_ingredients, top_k=5):
        """
        Stage 2: Precise ranking of retrieved candidates.
        
        Uses RecipeRanker for detailed matching and scoring.
        
        Args:
            candidates: DataFrame of candidate recipes
            user_ingredients: List of processed user ingredients
            top_k: Number of top recipes to return
        
        Returns:
            List of ranked recipe match results
        """
        ranked = self.ranker.rank(
            candidates, 
            user_ingredients, 
            top_k=top_k,
            ingredient_col=self.ingredient_column
        )
        return ranked
    
    def run(self, raw_user_ingredients, top_k=5, retrieve_k=300):
        """
        Execute full Retrieve & Rank pipeline.
        
        Pipeline:
            1. Process user ingredients (normalize + ontology)
            2. Retrieve: Fast candidate recall (top 300)
            3. Rank: Precise scoring and sorting (top K)
        
        Args:
            raw_user_ingredients: List of raw ingredient strings
            top_k: Number of top recipes to return
            retrieve_k: Number of candidates to retrieve (default 300)
        
        Returns:
            Tuple: (processed_user_ingredients, top_k_ranked_recipes)
        """
        print(f"\nSearching {len(self.recipes)} recipes...")
        
        # Step 1: Process user ingredients
        user_processed = self.process_user_ingredients(raw_user_ingredients)
        print(f"User ingredients (processed): {user_processed}")
        
        # Step 2: Retrieve candidates
        print(f"Retrieving candidates (max {retrieve_k})...")
        candidates = self.retrieve_candidates(user_processed, top_k=retrieve_k)
        print(f"Retrieved {len(candidates)} candidates")
        
        # Step 3: Rank candidates
        print(f"Ranking top {top_k} recipes...")
        ranked = self.rank_candidates(candidates, user_processed, top_k=top_k)
        
        print(f"Done! Found top {len(ranked)} matches")
        
        return user_processed, ranked
