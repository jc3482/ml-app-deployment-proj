import json
import os
from rapidfuzz import process, fuzz
import re


class RecipeNormalizer:
    def __init__(self, vocab_path="data/canonical_vocab.json", fuzzy_threshold=80):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        vocab_path = os.path.join(base, vocab_path)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"canonical vocab not found at {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = [v.lower().strip() for v in json.load(f)]

        self.threshold = fuzzy_threshold

    # ----------------------------------------
    # Normalize ONE ingredient
    # ----------------------------------------
    def normalize_ingredient(self, ingredient: str):
        if not isinstance(ingredient, str):
            return None

        raw = ingredient.lower().strip()

        # remove units/measurements
        raw = re.sub(r"\d+[\w\"\'\.\-/]*", "", raw).strip()
        raw = raw.replace("(", "").replace(")", "")
        raw = raw.replace(",", "").replace(":", "")

        # direct match
        if raw in self.vocab:
            return raw

        # fuzzy match
        result = process.extractOne(raw, self.vocab, scorer=fuzz.ratio)
        if result:
            match, score, _ = result
            if score >= self.threshold:
                return match

        # fallback: try first word
        base = raw.split()[0] if raw else ""
        result = process.extractOne(base, self.vocab, scorer=fuzz.ratio)
        if result:
            match, score, _ = result
            if score >= self.threshold:
                return match

        return None

    # ----------------------------------------
    # Normalize list of ingredients
    # ----------------------------------------
    def normalize_cleaned_list(self, ingredients):
        normalized = []

        for ing in ingredients:
            n = self.normalize_ingredient(ing)
            if n:
                normalized.append(n)

        return sorted(list(set(normalized)))
