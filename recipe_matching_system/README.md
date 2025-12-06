# SmartPantry Recipe Matching System

Intelligent recipe matching system based on ingredient detection.

## Project Overview

SmartPantry matches user ingredients (from YOLO detection or manual input) against a database of 13,000+ recipes using fuzzy matching algorithms.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process Raw Dataset (First Time)

```bash
# Step 1: Normalize raw recipes
python -m recipe_matcher.bin.main normalize

# Step 2: Apply ontology processing (optional but recommended)
python -m recipe_matcher.bin.main ontology
```

This generates:
- `data/normalized_recipes.json` (from preprocess.py - normalize)
- `data/ontology_recipes.json` (from preprocess.py - ontology, more robust)

### 3. Match Ingredients

```bash
# Using ingredient list
python -m recipe_matcher.bin.main match --ingredients "apple,milk,butter,eggs"

# Using JSON file
python -m recipe_matcher.bin.main match --input data/mock_ingredients/case1.json

# Get more recommendations
python -m recipe_matcher.bin.main match --ingredients "chicken,onion,garlic" --topk 10
```

## Data Processing Pipeline

```
data/recipe_dataset_raw.csv
         |
         v
    preprocess.py (IngredientNormalizer)
         |
         v
data/normalized_recipes.json
         |
         v
    preprocess.py (IngredientOntology)
         |
         v
data/ontology_recipes.json
         |
         v
    User Query + Retrieve & Rank Architecture
         |
         v
    retrieval_engine.py (Two-stage: Retrieve + Rank)
         |
         v
    Ranked Results
```

## Project Structure

```
recipe_matching_system/
├── data/
│   ├── recipe_dataset_raw.csv          # Raw dataset
│   ├── normalized_recipes.json         # After normalize.py
│   ├── ontology_recipes.json           # After ontology.py
│   ├── mock_ingredients/               # Test data
│   └── food_images/                    # Recipe images
│
├── recipe_matcher/
│   ├── bin/main.py                     # CLI interface
│   ├── utils/helpers.py                # Data I/O
│   ├── preprocess.py                   # Steps 1 & 2: Normalization + Ontology
│   ├── retrieval_engine.py             # Retrieve & Rank: matching, retrieval, ranking
│   └── pipeline.py                     # Overall orchestration
│
├── tests/                              # Unit tests
└── cleaning.ipynb                      # Data cleaning notebook
```

## Core Modules

### preprocess.py
Ingredient normalization and ontology processing (combines normalize + ontology).

```python
from recipe_matcher.preprocess import IngredientNormalizer, IngredientOntology

# Normalization
normalizer = IngredientNormalizer()
cleaned = normalizer.normalize_ingredient("2 cups of fresh milk")
# Result: "milk"

# Ontology
ontology = IngredientOntology()
canonical = ontology.canonicalize("parmesan cheese")
# Result: "parmesan"

# Process list with deduplication
ontology_tokens, raw_tokens = ontology.process_list(["fresh apple", "apple", "milk"])
# ontology_tokens: ["apple", "milk"]
# raw_tokens: ["fresh apple", "apple", "milk"]
```

### retrieval_engine.py
Complete Retrieve & Rank architecture with matching, retrieval, and ranking.

```python
from recipe_matcher.retrieval_engine import RecipeRetriever, RecipeRanker, match_recipe
from recipe_matcher.utils.helpers import load_ontology_recipes

# Stage 1: Fast candidate retrieval
recipes = load_ontology_recipes()
retriever = RecipeRetriever(recipes)
candidates = retriever.retrieve(["apple", "milk"], top_k=300)
# Returns: DataFrame of up to 300 candidate recipes

# Stage 2: Precise ranking
ranker = RecipeRanker()
top_5 = ranker.rank(candidates, ["apple", "milk"], top_k=5)
# Returns: List of top 5 ranked recipes with match scores

# Core matching function
result = match_recipe(["apple", "milk"], ["apple", "milk", "flour", "sugar"])
# Returns: {'overlap': 2, 'fuzzy_score': 0.5, 'matched': [...], 'missing': [...]}
```

### pipeline.py
Orchestrates the entire workflow using Retrieve & Rank architecture.

```python
from recipe_matcher.pipeline import RecipePipeline

# Initialize (uses ontology by default)
pipeline = RecipePipeline(use_ontology=True)

# Run Retrieve & Rank pipeline:
# 1. Process user ingredients
# 2. Retrieve up to 300 candidates (fast)
# 3. Rank candidates and return top 5 (precise)
user_ingredients = ["apple", "milk", "butter"]
processed, top_5 = pipeline.run(user_ingredients, top_k=5, retrieve_k=300)

# Results
for recipe in top_5:
    print(f"{recipe['title']}: {recipe['fuzzy_score']:.2%}")
    print(f"  Overlap: {recipe['overlap']}, Missing: {len(recipe['missing'])}")
```

## Testing

```bash
# Test with mock data
python -m recipe_matcher.bin.main match --input data/mock_ingredients/case1.json
python -m recipe_matcher.bin.main match --input data/mock_ingredients/case2.json
python -m recipe_matcher.bin.main match --input data/mock_ingredients/case3.json

# Run unit tests
pytest tests/
```

## Example Output

```
======================================================================
TOP 5 RECIPE RECOMMENDATIONS
======================================================================

----------------------------------------------------------------------
#1. Apple Cinnamon Pancakes
----------------------------------------------------------------------
   Fuzzy Match Score: 85.50%
   Exact Ingredient Overlap: 4

   Matched Ingredients (4):
      - apple
      - milk
      - butter
      - eggs

   Missing Ingredients (3):
      - flour
      - sugar
      - cinnamon

   Image: data/food_images/apple-cinnamon-pancakes.jpg
   Steps: Mix dry ingredients. In another bowl, whisk eggs and milk...
```

## Development

### Add Custom Matching Logic

Edit `recipe_matcher/retrieval_engine.py`:

```python
def match_recipe(user_ingredients, recipe_ingredients):
    # Implement your matching logic
    # Current implementation uses rapidfuzz for fuzzy matching
    ...
```

### Add Ontology Rules

Edit `recipe_matcher/preprocess.py`:

```python
class IngredientOntology:
    def __init__(self):
        # Add your mappings
        self.multi_map = {...}      # Multi-word mappings
        self.single_map = {...}     # Single-word mappings
        self.food_words = {...}     # Core food vocabulary
        self.noise_words = {...}    # Non-ingredient words
```

### Tune Retrieval Strategy

Edit `recipe_matcher/retrieval_engine.py`:

```python
class RecipeRetriever:
    def _token_recall(self, user_tokens, recipe_tokens):
        # Adjust fuzzy matching threshold
        if fuzz.partial_ratio(u, r) > 75:  # Change 75 to tune recall
            return True
```

## License

MIT License
