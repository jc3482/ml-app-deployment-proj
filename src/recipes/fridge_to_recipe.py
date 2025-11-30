import pandas as pd
import os

from src.vision.yolo_detector.food_detector import FoodDetector
from src.vision.yolo_detector.yolo_preprocessor import YoloPreprocessor
from src.recipes.recipe_normalizer import RecipeNormalizer
from src.recipes.recipe_matcher import RecipeMatcher


def main(image_path, top_k=5):
    # 1. Detect fridge ingredients
    detector = FoodDetector()
    raw = detector.detect(image_path)

    # 2. Normalize YOLO labels → canonical vocab
    yp = YoloPreprocessor()
    fridge_items = yp.normalize(raw)

    print("\nDetected fridge items:", fridge_items)

    # 3. Load recipes
    df_path = "data/recipes/recipe_dataset_final_clean_v2.csv"
    df = pd.read_csv(df_path)

    # 4. Normalize recipe ingredients
    rn = RecipeNormalizer()
    df["normalized"] = df["Cleaned_Ingredients"].apply(rn.normalize_cleaned_list)

    # 5. Match recipes
    matcher = RecipeMatcher()
    df["match_score"] = df["normalized"].apply(lambda r: matcher.match(fridge_items, r))

    results = df.sort_values("match_score", ascending=False).head(top_k)
    print("\nTop recipe matches:")
    print(results[["Title", "match_score", "normalized"]])

    return results


if __name__ == "__main__":
    main("path/to/your/fridge_image.jpg", top_k=5)
