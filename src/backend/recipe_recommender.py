"""
Unified YOLO → Canonical → Recipe Matching Pipeline
Ready for integration into a FastAPI / Flask / Streamlit backend.

Usage:
    from src.backend.recipe_recommender import RecipeRecommender

    rr = RecipeRecommender()
    result = rr.recommend(image_path="some_image.jpg", top_k=5)
"""

import os
import pickle
import random
import pandas as pd

# Import components
from src.vision.yolo_detector.food_detector import FoodDetector
from src.vision.yolo_detector.yolo_preprocessor import YoloPreprocessor
from src.recipes.recipe_matcher import RecipeMatcher


class RecipeRecommender:
    def __init__(
        self,
        recipe_cache_path="data/normalized_recipes.pkl",
        weight_method="weighted",
    ):
        """
        Initializes the pipeline.

        Args:
            recipe_cache_path: Path to precomputed normalized recipes PKL.
            weight_method: 'weighted' or 'binary'
        """
        self.recipe_cache_path = recipe_cache_path
        self.detector = FoodDetector()
        self.preprocessor = YoloPreprocessor()
        self.matcher = RecipeMatcher()
        self.weight_method = weight_method

        self.recipe_dict = self._load_cached_recipes()

        print(f"[RecipeRecommender] Loaded {len(self.recipe_dict)} normalized recipes.")

    # ============================================================
    #  LOAD CACHED NORMALIZED RECIPES
    # ============================================================
    def _load_cached_recipes(self):
        if not os.path.exists(self.recipe_cache_path):
            raise FileNotFoundError(
                f"Normalized recipe cache not found at {self.recipe_cache_path}.\n"
                f"Run: scripts/cache_normalized_recipes.py"
            )

        with open(self.recipe_cache_path, "rb") as f:
            return pickle.load(f)  # Dict: {recipe_title: [canonical ingredient list]}

    # ============================================================
    #  SINGLE PASS PIPELINE
    # ============================================================
    def recommend(self, image_path: str, top_k: int = 5):
        """
        Process an image → detect ingredients → canonical normalization → match recipes.
        Returns structured results dict.

        Args:
            image_path: path to fridge image
            top_k: number of recipe results to return
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print("\n=== Pipeline Start ===")
        print(f"Image: {image_path}")

        # 1. YOLO detection
        raw_labels = self.detector.detect(image_path)
        print("\n[1] Raw YOLO detections:", raw_labels)

        # 2. Canonical normalization
        canonical_items = self.preprocessor.normalize(raw_labels)
        print("[2] Canonical fridge items:", canonical_items)

        # 3. Weighted scoring against cached recipes
        results = []
        for title, recipe_ings in self.recipe_dict.items():
            score = self.matcher.match(canonical_items, recipe_ings)
            results.append((title, recipe_ings, score))

        # 4. Sort recipes by score
        results.sort(key=lambda x: x[2], reverse=True)

        # 5. Construct clean return
        top_results = []
        for title, ingredients, score in results[:top_k]:
            top_results.append({
                "title": title,
                "score": float(score),
                "ingredients": ingredients,
            })

        print("\n=== Top Matches ===")
        for r in top_results:
            print(f"- {r['title']} (score={r['score']:.4f})")
            print(f"  ingredients: {r['ingredients']}")

        print("=== Pipeline End ===\n")
        return {
            "image_path": image_path,
            "fridge_items": canonical_items,
            "recommendations": top_results,
        }


# ============================================================
#  STANDALONE DEMO (optional)
# ============================================================
if __name__ == "__main__":
    folder = "data/fridge_photos/test/images"
    test_img = os.path.join(folder, random.choice(os.listdir(folder)))

    rr = RecipeRecommender()
    rr.recommend(test_img, top_k=5)
