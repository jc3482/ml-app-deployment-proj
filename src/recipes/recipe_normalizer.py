import json
import os
import ast
from rapidfuzz import process, fuzz


class RecipeNormalizer:
    """
    Normalize recipe ingredient strings into canonical vocab.
    Works on Cleaned_Ingredients which is a Python list literal stored as a string.
    """

    def __init__(self, vocab_path="data/canonical_vocab.json", fuzzy_threshold=80):
        # Resolve project root
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        vocab_path = os.path.join(base, vocab_path)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Canonical vocab missing at {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [v.lower().strip() for v in json.load(f)]

        self.threshold = fuzzy_threshold

    # ----------------------------------------------------
    # Normalize a single ingredient string
    # ----------------------------------------------------
    def normalize_ingredient(self, ingredient: str):
        if not isinstance(ingredient, str):
            return None

        raw = ingredient.lower().strip()

        # 1. Exact match
        if raw in self.vocab:
            return raw

        # 2. Fuzzy match (RapidFuzz returns 3 values)
        result = process.extractOne(raw, self.vocab, scorer=fuzz.ratio)
        if result:
            match, score, _ = result
            if score >= self.threshold:
                return match

        # 3. Base token fallback (e.g., "chopped onion" → "onion")
        parts = raw.split()
        if len(parts) > 1:
            base = parts[-1]   # last token works better for food recipes
            result2 = process.extractOne(base, self.vocab, scorer=fuzz.ratio)
            if result2:
                match2, score2, _ = result2
                if score2 >= self.threshold:
                    return match2

        return None

    # ----------------------------------------------------
    # Normalize list of ingredients from Cleaned_Ingredients field
    # ----------------------------------------------------
    def normalize_cleaned_list(self, cleaned_list_str):
        if not isinstance(cleaned_list_str, str):
            return []

        try:
            items = ast.literal_eval(cleaned_list_str)
            if not isinstance(items, list):
                return []
        except Exception:
            return []

        normalized = []
        for ing in items:
            n = self.normalize_ingredient(ing)
            if n:
                normalized.append(n)

        # Deduplicate and sort
        return sorted(set(normalized))
