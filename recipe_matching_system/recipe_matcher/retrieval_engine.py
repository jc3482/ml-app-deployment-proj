"""
Recipe Retrieval Engine
Combines matching, retrieval, and ranking for efficient recipe search.

This module implements a two-stage Retrieve & Rank architecture:
1. Retrieve: Fast candidate recall using ontology and fuzzy matching
2. Rank: Precise scoring and sorting of candidates

Components:
    - match_recipe(): Core matching algorithm with fuzzy string matching
    - RecipeRetriever: Stage 1 - Fast candidate retrieval (top 300)
    - RecipeRanker: Stage 2 - Precise ranking (top K)
"""

from rapidfuzz import fuzz
from typing import List, Dict

# =============================================================================
# Core Matching Algorithm
# =============================================================================

def match_recipe(user_ingredients: List[str], recipe_ingredients: List[str]) -> Dict:
    """
    Compute detailed match score between user ingredients and recipe ingredients.
    
    Uses fuzzy string matching to calculate:
    - Exact overlap count
    - Weighted fuzzy match score
    - Matched and missing ingredients
    
    Args:
        user_ingredients: List of user's processed ingredients
        recipe_ingredients: List of recipe's processed ingredients
    
    Returns:
        Dictionary with match results:
        {
            'overlap': int,           # Number of exact matches
            'fuzzy_score': float,     # Weighted fuzzy score (0-1)
            'matched': list,          # List of matched ingredients
            'missing': list,          # List of missing ingredients from recipe
            'fuzzy_pairs': list       # List of fuzzy match details
        }
    """
    exact_matches = []
    fuzzy_matches = []
    missing = []
    
    matched_recipe_ings = set()
    
    # For each recipe ingredient, find best match from user ingredients
    for recipe_ing in recipe_ingredients:
        best_score = 0
        best_user_ing = None
        
        for user_ing in user_ingredients:
            # Exact match
            if user_ing == recipe_ing:
                exact_matches.append(recipe_ing)
                matched_recipe_ings.add(recipe_ing)
                best_score = 100
                best_user_ing = user_ing
                break
            
            # Fuzzy match
            score = fuzz.ratio(user_ing, recipe_ing)
            if score > best_score:
                best_score = score
                best_user_ing = user_ing
        
        # If no exact match, check if fuzzy match is good enough
        if best_score < 100 and best_score >= 60:
            fuzzy_matches.append({
                'user': best_user_ing,
                'recipe': recipe_ing,
                'score': best_score / 100.0
            })
            matched_recipe_ings.add(recipe_ing)
        elif best_score < 60:
            missing.append(recipe_ing)
    
    # Calculate scores
    overlap = len(exact_matches)
    
    # Weighted fuzzy score: exact matches = 1.0, fuzzy matches = weighted by score
    total_score = overlap * 1.0
    for fm in fuzzy_matches:
        total_score += fm['score']
    
    # Normalize by total recipe ingredients
    if len(recipe_ingredients) > 0:
        fuzzy_score = total_score / len(recipe_ingredients)
    else:
        fuzzy_score = 0.0
    
    # All matched ingredients (exact + fuzzy)
    matched = exact_matches + [fm['user'] for fm in fuzzy_matches]
    
    return {
        'overlap': overlap,
        'fuzzy_score': fuzzy_score,
        'matched': matched,
        'missing': missing,
        'fuzzy_pairs': fuzzy_matches
    }


# =============================================================================
# Stage 1: Fast Candidate Retrieval
# =============================================================================

class RecipeRetriever:
    """
    Fast candidate retrieval using two-stage recall strategy.
    
    Stage 1: Ontology recall - exact matches on canonical ingredients
    Stage 2: Fuzzy recall - token-level similarity matching (threshold > 75%)
    
    This retrieval stage significantly reduces the search space for
    the subsequent ranking stage, improving overall efficiency.
    """
    
    def __init__(self, recipes):
        """
        Initialize retriever with recipe database.
        
        Args:
            recipes: DataFrame with columns:
                - 'title': Recipe name
                - 'normalized_ingredients': Normalized ingredient list
                - 'ontology_ingredients': Canonicalized ingredient list
                - Other metadata (image_path, instructions, etc.)
        """
        self.recipes = recipes

    def _token_recall(self, user_tokens, recipe_tokens):
        """
        Check if there is at least one matching token between user and recipe.
        
        Matching criteria:
        - Exact match (user == recipe)
        - Fuzzy match (partial_ratio > 75%)
        
        Args:
            user_tokens: List of user ingredient tokens
            recipe_tokens: List of recipe ingredient tokens
        
        Returns:
            True if at least one token matches, False otherwise
        """
        for u in user_tokens:
            for r in recipe_tokens:
                if u == r:
                    return True
                if fuzz.partial_ratio(u, r) > 75:
                    return True
        return False

    def retrieve(self, user_ingredients, top_k=300):
        """
        Retrieve candidate recipes using two-stage recall.
        
        Pipeline:
            1. Ontology recall: Match against canonical ingredients
               - Higher precision, prioritized
            2. Fuzzy recall: Match against normalized ingredients
               - Higher coverage, fallback
        
        Args:
            user_ingredients: List of processed user ingredients
                             (should be ontology-processed for best results)
            top_k: Maximum number of candidates to return (default 300)
        
        Returns:
            DataFrame of candidate recipes (unranked, up to top_k rows)
        """
        candidates = []

        for idx, row in self.recipes.iterrows():
            ontos = row.get("ontology_ingredients", [])
            norms = row.get("normalized_ingredients", [])

            # Priority 1: Ontology recall (highest precision)
            if ontos and self._token_recall(user_ingredients, ontos):
                candidates.append(idx)
                continue

            # Priority 2: Fuzzy recall on normalized ingredients (higher coverage)
            if norms and self._token_recall(user_ingredients, norms):
                candidates.append(idx)

        # Return unranked candidates (ranking happens in next stage)
        return self.recipes.loc[candidates].head(top_k)


# =============================================================================
# Stage 2: Precise Ranking
# =============================================================================

class RecipeRanker:
    """
    Precise ranking of candidate recipes.
    
    Computes detailed match scores for each candidate and returns
    top-k ranked results.
    """
    
    def __init__(self):
        pass

    def rank(self, df_candidates, user_ingredients, top_k=5, ingredient_col="ontology_ingredients"):
        """
        Rank candidate recipes by match quality.
        
        Args:
            df_candidates: DataFrame of candidate recipes
            user_ingredients: List of processed user ingredients
            top_k: Number of top recipes to return
            ingredient_col: Column name for recipe ingredients
                           ("ontology_ingredients" or "normalized_ingredients")
        
        Returns:
            List of ranked recipe dictionaries with match scores
        """
        ranked = []

        for _, row in df_candidates.iterrows():
            # Use fallback if specified column not available
            if ingredient_col in row:
                recipe_ings = row[ingredient_col]
            elif "normalized_ingredients" in row:
                recipe_ings = row["normalized_ingredients"]
            else:
                continue
            
            # Compute match score
            match = match_recipe(user_ingredients, recipe_ings)

            # Attach metadata
            match["title"] = row.get("title", "")
            match["image_path"] = row.get("image_path", "")
            match["image_name"] = row.get("image_name", "")
            match["instructions"] = row.get("instructions", "")
            match["recipe_ingredients"] = recipe_ings
            
            ranked.append(match)

        # Sort by fuzzy_score first, then exact overlap
        ranked = sorted(ranked,
                        key=lambda x: (x["fuzzy_score"], x["overlap"]),
                        reverse=True)

        return ranked[:top_k]
