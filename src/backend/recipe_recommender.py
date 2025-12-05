"""
Unified YOLO → Canonical → Recipe Matching Pipeline
Now returns full metadata:
Title, Ingredients, Instructions, Cleaned_Ingredients, Normalized_Ingredients
"""

import os
import pickle
import random
import pandas as pd

# Import pipeline pieces
from src.vision.yolo_detector.food_detector import FoodDetector

# Updated import: Use the new RecipePipeline
from recipe_matching_system.recipe_matcher.pipeline import RecipePipeline


class RecipeRecommender:
    def __init__(
        self,
        recipe_cache_path=None, # Use defaults in pipeline
        scoring="binary",      
    ):
        self.scoring = scoring

        self.detector = FoodDetector()
        # self.preprocessor = YoloPreprocessor() # Replaced by pipeline internal processing
        
        # Define common pantry items that are assumed to be available
        self.common_pantry_items = [
            "salt", 
            "pepper", 
            "butter", 
            "sugar", 
            "flour", 
            "water", 
            "oil", 
            "olive oil",
            "vegetable oil",
            "vinegar",
            "garlic",
            "onion",
            "rice",
            "pasta",
            "milk",
            "spice",
            "sauce"
        ]
        
        # Initialize the new pipeline
        # It handles loading recipes, normalization, retrieval, and ranking
        print("[RecipeRecommender] Initializing RecipePipeline...")
        self.pipeline = RecipePipeline(use_ontology=True)
        
        # Compatibility: expose recipe_dict for API health check
        # The pipeline stores recipes in a dataframe self.pipeline.recipes
        # We can create a simple dict view if needed, or just expose the dataframe length
        self.recipe_dict = self.pipeline.recipes

    # -------------------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------------------
    def recommend(self, image_path: str, top_k: int = 5):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print("\n=== Pipeline Start ===")
        print(f"Image: {image_path}")

        # 1. YOLO detection
        raw_labels = self.detector.detect(image_path)
        print("\n[1] Raw YOLO detections:", raw_labels)
        
        # 1.5. Add common pantry items to the list of ingredients
        # We append them to the raw detections so they get processed by the pipeline (normalized/ontology)
        augmented_labels = raw_labels + self.common_pantry_items
        print(f"\n[1.5] Added {len(self.common_pantry_items)} common pantry items.")

        # 2. Run Recipe Matching Pipeline (Normalize -> Retrieve -> Rank)
        # The pipeline handles normalization internally using the new ontology logic
        processed_ingredients, ranked_recipes = self.pipeline.run(augmented_labels, top_k=top_k)
        
        print("[2] Canonical fridge items (including pantry):", processed_ingredients)

        # 3. Format output to match expected API structure
        top = []
        for r in ranked_recipes:
            # Map pipeline result fields to API expected fields
            top.append({
                "title": r["title"],
                "score": float(r["fuzzy_score"]), # Use fuzzy score as main score
                "overlap_score": r["overlap"],
                "normalized_ingredients": r.get("recipe_ingredients", []),
                "cleaned_ingredients": r.get("recipe_ingredients", []), # Fallback
                "ingredients_raw": r.get("recipe_ingredients", []), # Fallback
                "instructions": r.get("instructions", ""),
                "image_name": r.get("image_name", ""),
                "image_path": r.get("image_path", ""),
                "matched_ingredients": r.get("matched", []),
                "missing_ingredients": r.get("missing", [])
            })

        # Print preview
        print("\n=== Top Matches ===")
        for r in top:
            print(f"- {r['title']} (score={r['score']:.2f})")
            print(f"  ingredients: {r['normalized_ingredients']}")
            print(f"  matched ingredients: {r['matched_ingredients']}")
            print(f"  missing ingredients: {r['missing_ingredients']}")

        print("=== Pipeline End ===\n")

        return {
            "image_path": image_path,
            "fridge_items": processed_ingredients,
            "recommendations": top
        }

    # ============================================================
    #  HELPER: Return a clean recipe detail block
    # ============================================================
    def get_recipe_details(self, result_dict, index=0):
        """
        Extracts detailed metadata for any recommended recipe.
        """
        recs = result_dict.get("recommendations", [])
        if len(recs) == 0:
            return None

        if index < 0 or index >= len(recs):
            raise IndexError("Recipe index out of range.")

        rec = recs[index]

        return {
            "title": rec["title"],
            "score": rec["score"],
            "cleaned_ingredients": rec["cleaned_ingredients"],
            "normalized_ingredients": rec["normalized_ingredients"],
            "ingredients_raw": rec["ingredients_raw"],
            "instructions": rec["instructions"],
            "image_name": rec["image_name"],
        }

# -------------------------------------------------------------
# STANDALONE DEMO
# -------------------------------------------------------------
if __name__ == "__main__":
    folder = "data/fridge_photos/test/images"
    if os.path.exists(folder):
        test_img = os.path.join(folder, random.choice(os.listdir(folder)))
        rr = RecipeRecommender()
        rr.recommend(test_img, top_k=5)
    else:
        print("Test folder not found.")