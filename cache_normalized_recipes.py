"""
Rebuild normalized recipe cache:
- normalized_recipes.pkl  (runtime)
- cached_normalized.csv   (audit/debug)
"""

import os
import ast
import pickle
import pandas as pd

from src.recipes.recipe_normalizer import RecipeNormalizer

CSV_PATH = "data/recipes/recipe_dataset_final_clean.csv"
PKL_OUT = "data/normalized_recipes.pkl"
CSV_OUT = "data/cached_normalized.csv"


def safe_parse_list(raw):
    if not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


if __name__ == "__main__":

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Recipe dataset not found at {CSV_PATH}")

    print("Loading recipes...")
    df = pd.read_csv(CSV_PATH)

    rn = RecipeNormalizer()

    final_dict = {}
    audit_rows = []

    print("Normalizing recipes...")

    for idx, row in df.iterrows():

        title = row["Title"]
        cleaned = safe_parse_list(row["Cleaned_Ingredients"])
        normalized = rn.normalize_cleaned_list(cleaned)

        # FULL STRUCTURE (FIXED)
        final_dict[title] = {
            "normalized": normalized,
            "cleaned": cleaned,
            "ingredients_raw": row.get("Ingredients", ""),
            "instructions": row.get("Instructions", ""),
            "image_name": row.get("Image_Name", "")
        }

        audit_rows.append({
            "Title": title,
            "Normalized": normalized,
            "Cleaned_Ingredients": cleaned,
            "Ingredients_Raw": row.get("Ingredients", ""),
            "Instructions": row.get("Instructions", ""),
            "Image_Name": row.get("Image_Name", "")
        })

        if idx % 500 == 0:
            print(f"{idx}/{len(df)} processed...")

    print("\nSaving PKL:", PKL_OUT)
    with open(PKL_OUT, "wb") as f:
        pickle.dump(final_dict, f)

    print("Saving CSV:", CSV_OUT)
    pd.DataFrame(audit_rows).to_csv(CSV_OUT, index=False)

    print("\n=== DONE ===")
