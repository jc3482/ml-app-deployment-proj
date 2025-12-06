"""
Ingredient Preprocessing Module
Combines normalization and ontology processing for recipe ingredients.

This module provides two stages of ingredient processing:
1. Normalization: Light cleaning (remove quantities, units, punctuation)
2. Ontology: Extract canonical ingredient entities

Classes:
    - IngredientNormalizer: Minimal ingredient normalizer
    - IngredientOntology: Ingredient canonicalization and entity extraction
"""

import re
import pickle
import unicodedata
from typing import List

# =============================================================================
# Stage 1: Ingredient Normalization
# =============================================================================
class IngredientNormalizer:
    """
    Safe, high-recall ingredient normalizer.
    Only cleans surface form (punctuation, units, digits, unicode).
    """

    def __init__(self):

        self.units = {
            "cup","cups","tbsp","tbs","tablespoon","tablespoons",
            "tsp","teaspoon","teaspoons",
            "pound","pounds","lb","lbs",
            "ounce","ounces","oz",
            "gram","grams","kg",
            "ml","l","liter","liters"
        }

        self.frac_map = {
            "½": "1/2", "¼": "1/4", "¾": "3/4",
            "⅛": "1/8", "⅓": "1/3", "⅔": "2/3"
        }

    def _ascii(self, text: str) -> str:
        for k, v in self.frac_map.items():
            text = text.replace(k, v)
        text = (
            text.replace("–", "-")
                .replace("—", "-")
                .replace("•", " ")
        )
        return text

    def _basic_clean(self, text: str) -> str:

        if not isinstance(text, str):
            return ""

        text = self._ascii(text)
        text = text.lower()

        text = re.sub(r"\([^)]*\)", " ", text)
        text = text.replace("-", " ")

        text = re.sub(r"\d+\s*/\s*\d+", " ", text)
        text = re.sub(r"\b\d+\.?\d*\b", " ", text)

        for u in self.units:
            text = re.sub(rf"\b{u}\b", " ", text)

        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_ingredient(self, raw: str) -> str:
        return self._basic_clean(raw)

    def normalize_list(self, ingredients):
        return [self.normalize_ingredient(i) for i in ingredients if self.normalize_ingredient(i)]


# =============================================================================
# Stage 2: Ingredient Ontology
# =============================================================================
class IngredientOntology:
    """
    Ingredient Ontology v3 — includes pantry exclusions & PKL caching.
    """

    def __init__(self):

        # Pantry ingredients to ignore in final ontology output
        self.pantry_exclusions = {
            "glass", "water", 
        }

        self.noise_phrases = [
            "torn into pieces","cut into pieces","cut into","torn into",
            "halved lengthwise","peeled and halved lengthwise",
            "peeled and sliced","thinly sliced","finely chopped",
            "freshly ground","lightly beaten","to taste","if needed",
            "such as","about","total"
        ]

        self.multi_map = {
            "extra virgin olive oil": "olive oil",
            "extra-virgin olive oil": "olive oil",
            "virgin olive oil": "olive oil",
            "olive oil": "olive oil",

            "evaporated milk": "milk",
            "whole milk": "milk",
            "heavy cream": "cream",
            "cream cheese": "cream cheese",

            "dark brown sugar": "brown sugar",
            "light brown sugar": "brown sugar",
            "brown sugar": "brown sugar",
            "powdered sugar": "sugar",
            "all purpose flour": "flour",

            "kosher salt": "salt",
            "sea salt": "salt",
            "black pepper": "pepper",
            "onion powder": "onion",
            "garlic powder": "garlic",

            "apple cider vinegar": "vinegar",
            "balsamic vinegar": "vinegar",

            "elbow macaroni": "pasta",
            "bread crumbs": "breadcrumbs",
            "panko bread crumbs": "breadcrumbs",

            "chicken broth": "broth",
            "beef broth": "broth",
            "chicken stock": "broth",
        }

        self.single_map = {
            "onions":"onion","carrots":"carrot","apples":"apple","eggs":"egg",
            "cloves":"clove","leaves":"leaf","tomatoes":"tomato","potatoes":"potato",
            "ribs":"rib","mushrooms":"mushroom","chilies":"chili","chiles":"chile",
        }

        self.noise_words = {
            "cup","cups","tbsp","tablespoon","tsp","teaspoon",
            "pound","lb","ounce","oz","gram","kg","ml","l",
            "bag","bottle","can","jar","slice","piece",

            "small","medium","large","fresh","raw","ripe","whole",
            "unsalted","salted","smoked","boneless","light","dark",
            "red","yellow","green","white",

            "lengthwise","diagonally","crosswise","inch","inches",
            "cubes","strips","packed","finely","freshly",

            "and","or","to","into","of","for","with","without",
            "the","a","an","as","from","in","on","by",
        }

        self.verb_patterns = re.compile(r"(chopped|diced|minced|sliced|grated|peeled|drained|rinsed)$")

        self.food_words = {
            "onion","garlic","carrot","celery","apple","lemon","lime","tomato","potato",
            "pepper","chili","mushroom","spinach","parsley","cilantro","basil",
            "chicken","turkey","beef","pork","sausage","bacon","ham",
            "egg","cheese","cream","milk","butter",
            "rice","flour","sugar","pasta","breadcrumbs","bread",
            "salt","vinegar","oil","broth","mustard","honey","syrup",
            "paprika"
        }

    def _tokens(self, text: str):
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text.split()

    def canonicalize(self, ing: str):
        if not ing:
            return None

        text = ing.lower()

        for p in self.noise_phrases:
            if p in text:
                text = text.replace(p, " ")

        text = re.sub(r"\s+", " ", text).strip()

        for phrase, mapped in sorted(self.multi_map.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                return mapped

        tokens = self._tokens(text)

        tokens = [t for t in tokens if t not in self.noise_words]
        tokens = [t for t in tokens if not self.verb_patterns.search(t)]

        if not tokens:
            return None

        food_hits = [t for t in tokens if t in self.food_words]

        head = food_hits[-1] if food_hits else tokens[-1]

        head = self.single_map.get(head, head)

        # 🧹 NEW: remove pantry essentials
        if head in self.pantry_exclusions:
            return None

        return head

    def process_list(self, ingredients: List[str], deduplicate=True):

        ontology = []
        raw = []

        for ing in ingredients:
            raw.append(ing)
            can = self.canonicalize(ing)
            if can:
                ontology.append(can)

        if deduplicate:
            ontology = list(dict.fromkeys(ontology))

        return ontology, raw

    def deduplicate_list(self, ingredients: List[str]):
        return list(dict.fromkeys(ingredients))

    # NEW — save ontology list to PKL
    def save_cached(self, ontology_list, path="cached_ontology.pkl"):
        with open(path, "wb") as f:
            pickle.dump(ontology_list, f)
