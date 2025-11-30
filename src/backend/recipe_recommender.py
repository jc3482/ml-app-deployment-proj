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
from src.vision.yolo_detector.yolo_preprocessor import YoloPreprocessor
from src.recipes.recipe_matcher import RecipeMatcher


class RecipeRecommender:
    def __init__(
        self,
        recipe_cache_path="data/normalized_recipes.pkl",
        scoring="binary",      # “binary” is what you want
    ):
        self.recipe_cache_path = recipe_cache_path
        self.scoring = scoring

        self.detector = FoodDetector()
        self.preprocessor = YoloPreprocessor()
        self.matcher = RecipeMatcher()

        self.recipe_dict = self._load_cached_recipes()

        print(f"[RecipeRecommender] Loaded {len(self.recipe_dict)} recipes.")

    # -------------------------------------------------------------
    # LOAD CACHED RECIPE METADATA
    # -------------------------------------------------------------
    def _load_cached_recipes(self):
        if not os.path.exists(self.recipe_cache_path):
            raise FileNotFoundError(
                f"Recipe cache missing at {self.recipe_cache_path}\n"
                "Run scripts/cache_normalized_recipes.py first."
            )

        with open(self.recipe_cache_path, "rb") as f:
            return pickle.load(f)

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

        # 2. Normalize to canonical ingredients
        canonical = self.preprocessor.normalize(raw_labels)
        print("[2] Canonical fridge items:", canonical)

        # 3. Match recipes
        scored_recipes = []
        for title, meta in self.recipe_dict.items():
            score = self.matcher.match(canonical, meta["normalized"])

            scored_recipes.append({
                "title": title,
                "score": float(score),
                "normalized_ingredients": meta["normalized"],
                "cleaned_ingredients": meta["cleaned"],
                "ingredients_raw": meta["ingredients_raw"],
                "instructions": meta["instructions"],
                "image_name": meta["image_name"]
            })

        # Sort by score
        ranked = sorted(scored_recipes, key=lambda x: x["score"], reverse=True)

        # Top-k
        top = ranked[:top_k]

        # Print preview
        print("\n=== Top Matches ===")
        for r in top:
            print(f"- {r['title']} (score={r['score']:.2f})")
            print(f"  ingredients: {r['normalized_ingredients']}")

        print("=== Pipeline End ===\n")

        return {
            "image_path": image_path,
            "fridge_items": canonical,
            "recommendations": top
        }

        # ============================================================
    #  HELPER: Return a clean recipe detail block
    # ============================================================
    def get_recipe_details(self, result_dict, index=0):
        """
        Extracts detailed metadata for any recommended recipe.

        Args:
            result_dict: output of self.recommend()
            index: which ranked recipe to view (default = #1)

        Returns:
            A dict containing:
                - title
                - score
                - cleaned_ingredients
                - normalized_ingredients
                - ingredients_raw
                - instructions
                - image_name
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
    test_img = os.path.join(folder, random.choice(os.listdir(folder)))

    rr = RecipeRecommender()
    rr.recommend(test_img, top_k=5)
