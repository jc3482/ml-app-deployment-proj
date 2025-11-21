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
import unicodedata
from typing import List

# =============================================================================
# Stage 1: Ingredient Normalization
# =============================================================================
class IngredientNormalizer:
    """
    Safe, high-recall ingredient normalizer.
    Only cleans surface form (punctuation, units, digits, unicode).
    Does NOT attempt to remove adjectives or verbs.
    Does NOT try to identify ingredient semantics.
    """

    def __init__(self):
        # minimal unit list (will not delete nouns)
        self.units = {
            "cup", "cups", "tbsp", "tablespoon", "tablespoons",
            "tsp", "teaspoon", "teaspoons", "pound", "pounds",
            "lb", "lbs", "ounce", "ounces", "oz", "gram", "grams",
            "kg", "ml", "l",
        }

    def _ascii(self, text: str) -> str:
        # Convert unicode fractions & weird chars → ASCII
        return (
            text.replace("½", "1/2")
                .replace("¼", "1/4")
                .replace("¾", "3/4")
                .replace("⅛", "1/8")
                .replace("⅓", "1/3")
                .replace("⅔", "2/3")
                .replace("–", "-")
                .replace("—", "-")
                .replace("•", "")
        )

    def _basic_clean(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = self._ascii(text)
        text = text.lower()

        # remove parentheses
        text = re.sub(r"\([^)]*\)", "", text)

        # remove fractions & numbers
        text = re.sub(r"\d+\s*/\s*\d+", "", text)  # 1/2
        text = re.sub(r"\b\d+\.?\d*\b", "", text)  # standalone

        # remove units (but NOT nouns)
        for u in self.units:
            text = re.sub(rf"\b{u}\b", "", text)

        # punctuation → space
        text = re.sub(r"[^\w\s\-]", " ", text)

        # collapse spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_ingredient(self, raw: str) -> str:
        return self._basic_clean(raw)

    def normalize_list(self, ingredients):
        result = []
        for ing in ingredients:
            cleaned = self.normalize_ingredient(ing)
            if cleaned:
                result.append(cleaned)
        return result


# =============================================================================
# Stage 2: Ingredient Ontology
# =============================================================================

class IngredientOntology:
    """
    Ingredient Ontology v2
    ------------------------------------------
    Extract true ingredient entities from normalized text:
      - Remove cooking actions, modifiers, and descriptive information
      - Identify multi-word ingredients (elbow macaroni, apple cider vinegar...)
      - Map to canonical forms (parsnips -> parsnip, kosher salt -> salt)
      - Deduplicate ingredient lists
    """

    def __init__(self):

        # ========= 1. Phrase-level noise (remove entire phrases like "torn into pieces") =========
        self.noise_phrases = [
            "torn into pieces",
            "torn into",
            "cut into pieces",
            "cut into",
            "halved lengthwise",
            "peeled and halved lengthwise",
            "peeled and sliced",
            "peeled and halved",
            "peeled and",
            "peeled",
            "thinly sliced",
            "coarsely grated",
            "finely chopped",
            "freshly ground",
            "lightly beaten",
            "to taste",
            "if needed",
            "such as",
            "about",
            "total",
        ]

        # ========= 2. Multi-word canonical rules (ingredient phrases → standard names) =========
        self.multi_map = {
            # Oils and fats
            "extra virgin olive oil": "olive oil",
            "extra-virgin olive oil": "olive oil",
            "virgin olive oil": "olive oil",
            "olive oil": "olive oil",

            # Dairy products
            "evaporated milk": "milk",
            "whole milk": "milk",
            "heavy cream": "cream",
            "whipping cream": "cream",
            "sour cream": "sour cream",
            "cream cheese": "cream cheese",
            "full fat cream cheese": "cream cheese",
            "full-fat cream cheese": "cream cheese",

            # Sugar and flour
            "dark brown sugar": "brown sugar",
            "light brown sugar": "brown sugar",
            "brown sugar": "brown sugar",
            "powdered sugar": "sugar",
            "confectioners sugar": "sugar",
            "all purpose flour": "flour",
            "all-purpose flour": "flour",

            # Salt, pepper, and spices
            "kosher salt": "salt",
            "sea salt": "salt",
            "table salt": "salt",
            "freshly ground black pepper": "pepper",
            "ground black pepper": "pepper",
            "black pepper": "pepper",
            "red pepper flakes": "pepper",
            "crushed red pepper flakes": "pepper",
            "onion powder": "onion",
            "garlic powder": "garlic",
            "smoked paprika": "paprika",

            # Vinegar, sauces, and liquids
            "apple cider vinegar": "vinegar",
            "red wine vinegar": "vinegar",
            "balsamic vinegar": "vinegar",
            "apple juice": "apple juice",
            "maple syrup": "maple syrup",
            "soy sauce": "soy sauce",
            "fish sauce": "fish sauce",
            "dijon mustard": "mustard",
            "grainy dijon mustard": "mustard",
            "smooth dijon mustard": "mustard",

            # Starches, pasta, and bread
            "elbow macaroni": "pasta",
            "macaroni pasta": "pasta",
            "pasta shells": "pasta",
            "bread crumbs": "breadcrumbs",
            "panko bread crumbs": "breadcrumbs",
            "italian bread": "bread",
            "white bread": "bread",

            # Meat, broth, and meat products
            "chicken broth": "broth",
            "chicken stock": "broth",
            "turkey giblet stock": "broth",
            "beef broth": "broth",
            "bone in ham": "ham",
            "hickory smoked ham": "ham",

            # Herbs and others
            "flat leaf parsley": "parsley",
            "italian parsley": "parsley",
            "sweet italian sausage": "sausage",
            "italian sausage": "sausage",
            "white miso": "miso",
            "red miso": "miso",
        }

        # ========= 3. Single-word canonical rules (plural → singular / aliases) =========
        self.single_map = {
            "onions": "onion",
            "parsnips": "parsnip",
            "carrots": "carrot",
            "apples": "apple",
            "eggs": "egg",
            "cloves": "clove",
            "leaves": "leaf",
            "tomatoes": "tomato",
            "potatoes": "potato",
            "loaves": "loaf",
            "mushrooms": "mushroom",
            "chilies": "chili",
            "chiles": "chile",
            "sausages": "sausage",
            "ribs": "rib",
        }

        # ========= 4. Noise words (non-ingredients, descriptive, containers, etc.) =========
        self.noise_words = {
            # Units and containers
            "cup","cups","tbsp","tsp","tablespoon","tablespoons","teaspoon","teaspoons",
            "pound","pounds","lb","lbs","ounce","ounces","oz","gram","grams","kg","ml","l",
            "stick","sticks","package","packages","bag","bags","bottle","bottles","can","cans",
            "jar","jars","slice","slices","piece","pieces","loaf","pan","dish","qt",

            # Adjectives, sizes, and colors
            "small","medium","large","extra","good","quality","sturdy","fresh","freshly",
            "raw","ripe","whole","unsalted","salted","smoked","hickory","boneless","bone",
            "dark","light","white","red","yellow","green",

            # Directions, shapes, and cutting methods
            "lengthwise","diagonally","crosswise","inch","inches","cubes","cube","strips","strip",

            # Other descriptive words
            "packed","coarsely","finely","lightly","fully","such","total","about","more","plus",
            "room","temperature","unsweetened","good quality","sturdy","quality",

            # Prepositions, conjunctions, and articles
            "of","for","to","into","in","on","at","by","with","without",
            "and","or","as","if","than","from","the","a","an",
        }

        # ========= 5. Core food vocabulary (head noun candidates) =========
        self.food_words = {
            # Basic vegetables and fruits
            "onion","garlic","parsnip","carrot","celery","apple","lemon","lime","tomato",
            "potato","pepper","chili","chile","squash","mushroom","broccoli","spinach",
            "parsley","cilantro","basil","sage","rosemary","thyme","oregano",

            # Meat, eggs, and dairy
            "ham","chicken","turkey","beef","pork","sausage","bacon",
            "egg","yogurt","cheddar","cheese","cream","milk","butter","miso",

            # Dry goods, grains, and noodles
            "flour","sugar","rice","pasta","macaroni","noodle","breadcrumbs","bread","loaf",
            "oats","cornmeal","polenta",

            # Seasonings and liquids
            "salt","pepper","vinegar","oil","syrup","juice","broth","stock","wine","mustard",
            "ketchup","mayonnaise","mayo","sauce","soy","honey","maple","paprika","allspice",
        }

    # ------------------------------------------------------------------
    def _tokens(self, text: str):
        """Simple tokenization: remove punctuation, lowercase, split by spaces."""
        text = re.sub(r"[^\w\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    # ------------------------------------------------------------------
    def canonicalize(self, ing: str):
        """
        Map a normalized ingredient string to a canonical ingredient name (or None).
        
        Args:
            ing: A normalized ingredient string
            
        Returns:
            Canonical ingredient name (str) or None if invalid
        """

        if not ing or not isinstance(ing, str):
            return None

        text = ing.lower()

        # 1) Remove phrase-level noise (torn into, halved lengthwise, freshly ground, etc.)
        for phrase in self.noise_phrases:
            if phrase in text:
                text = text.replace(phrase, " ")

        text = re.sub(r"\s+", " ", text).strip()

        # 2) Multi-word mapping (return immediately if matched)
        for phrase, mapped in sorted(self.multi_map.items(), key=lambda x: -len(x[0])):
            if phrase in text:
                return mapped

        # 3) Tokenize and remove noise words
        tokens = self._tokens(text)
        tokens = [t for t in tokens if t not in self.noise_words]

        if not tokens:
            return None

        # 4) Further filter obvious verbs/adverbs: words ending in ed/ing/ly
        filtered = []
        for t in tokens:
            if re.match(r".*(ed|ing|ly)$", t):
                continue
            filtered.append(t)

        tokens = filtered or tokens  # Fallback to original tokens if all filtered out

        # 5) Find "head noun" from food_words vocabulary
        food_hits = [t for t in tokens if t in self.food_words]

        if food_hits:
            head = food_hits[-1]  # Take rightmost (closest to head noun)
        else:
            # Not in food_words, use last token as fallback
            head = tokens[-1]

        # 6) Singularize (apply single-word canonical mapping)
        head = self.single_map.get(head, head)

        return head

    # ------------------------------------------------------------------
    def process_list(self, ingredients: List[str], deduplicate=True):
        """
        Process a list of ingredients:
          - Returns canonical ingredient list (deduplicated by default)
          - Returns raw text list as second element
          
        Args:
            ingredients: List of normalized ingredient strings
            deduplicate: If True, remove duplicate canonical ingredients (default: True)
            
        Returns:
            Tuple of (canonical_ingredients, raw_ingredients)
            - canonical_ingredients: List of canonical ingredient names (deduplicated)
            - raw_ingredients: Original ingredient strings
        """
        ontology = []
        raw = []

        for ing in ingredients:
            raw.append(ing)
            c = self.canonicalize(ing)
            if c:
                ontology.append(c)

        # Deduplicate while preserving order (using dict to maintain insertion order)
        if deduplicate:
            ontology = list(dict.fromkeys(ontology))

        return ontology, raw
    
    # ------------------------------------------------------------------
    def deduplicate_list(self, ingredients: List[str]) -> List[str]:
        """
        Deduplicate a list of canonical ingredients while preserving order.
        
        Args:
            ingredients: List of canonical ingredient names
            
        Returns:
            Deduplicated list (first occurrence preserved)
        """
        return list(dict.fromkeys(ingredients))
