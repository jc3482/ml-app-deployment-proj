# Recipe Matching System Workflow

## Overview

This system is part of the SmartPantry project. It processes recipe data and matches user-detected ingredients with recipes from the database.

**Important: All commands should be run from the project root directory (`ml-app-deployment-proj/`).**

## Directory Structure

```
ml-app-deployment-proj/                    # ← Run commands from here!
├── data/
│   ├── recipes/
│   │   └── recipe_dataset_raw.csv        # Raw recipe data (input)
│   ├── normalized_recipes.json           # After Step 2 (normalize)
│   └── ontology_recipes.json             # After Step 3 (ontology)
│
├── recipe_matching_system/               # This subsystem
│   ├── recipe_matcher/
│   │   ├── bin/main.py                   # CLI entry point
│   │   ├── utils/helpers.py              # Data I/O (path config here)
│   │   ├── preprocess.py                 # Steps 1-2: normalize + ontology
│   │   ├── retrieval_engine.py           # Retrieve & Rank
│   │   └── pipeline.py                   # Full workflow orchestration
│   └── tests/
│
└── src/
    └── vision/                            # YOLO detector (produces ingredients)
```

**Data Flow:**
1. YOLO detector (in `src/vision/`) detects ingredients from fridge photos
2. Detected ingredients → recipe_matcher (this system) for recipe recommendations
3. Or manually provide ingredients via CLI for testing

## Data Processing Pipeline

```
../data/recipes/recipe_dataset_raw.csv
         |
         | helpers.load_raw_recipes()
         v
    preprocess.py (normalize) [轻量级recipe食材清洗器; 把原始ingredients文本变成"标准化、干净"的字符串，用于后续 fuzzy matching]
    (IngredientNormalizer)
         |
         | helpers.save_normalized_recipes()
         v
../data/normalized_recipes.json 
         |
         | helpers.load_normalized_recipes()
         v
    preprocess.py (ontology) [把 normalized 的字符串进一步"语义标准化"，如：parmesan cheese → parmesan]
    (IngredientOntology)
         |
         | helpers.save_ontology_recipes()
         v
../data/ontology_recipes.json
         |
         | helpers.load_ontology_recipes()
         v
    User Query (from YOLO detector or manual input)
         |
         v
    retrieval_engine.py  <--- Two-stage Retrieve & Rank Architecture
         |
         v
    RecipeRetriever      <--- Stage 1: Fast Candidate Retrieval (top 300) [根据用户输入食材，从上万 recipe 中快速筛选recipe]
         |                    Ontology recall + Fuzzy recall
         |                 [如果 recipe 的 canonical ingredient 与用户 canonical ingredient 匹配 → 优先收录]
                            [用 rapidfuzz.partial_ratio > 75 去 fuzzy match]
         v
    Candidate Pool          [输出：一个 candidate recipes DataFrame]
         |
         v
    RecipeRanker         <--- Stage 2: Precise Ranking (top K) [对候选 recipe 进行更精确的打分与排序（top-K）]
         |                    Detailed match scoring (match_recipe)
         |                 [计算 user ingredients vs recipe ingredients 的详细匹配分数]
         v
    Ranked Results
```

## Usage

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- rapidfuzz (for fuzzy matching)

### Step 2: Normalize Raw Dataset

**Important:** Run this command from the project root directory (`ml-app-deployment-proj/`), not from `recipe_matching_system/`.

```bash
# Make sure you're in the project root
cd ml-app-deployment-proj/

# Run normalization
python -m recipe_matcher.bin.main normalize
```

This runs:
1. `helpers.load_raw_recipes()` - Load raw CSV from `data/recipes/recipe_dataset_raw.csv`
2. `preprocess.py (normalize)` - Clean and normalize ingredients
3. `helpers.save_normalized_recipes()` - Save to `data/normalized_recipes.json`

### Step 3: Apply Ontology Processing (Optional but Recommended)

```bash
# From project root
python -m recipe_matcher.bin.main ontology
```

This runs:
1. `helpers.load_normalized_recipes()` - Load normalized data from `data/normalized_recipes.json`
2. `preprocess.py (ontology)` - Canonicalize ingredients, map synonyms
3. `helpers.save_ontology_recipes()` - Save to `data/ontology_recipes.json`

### Step 4: Match Ingredients

```bash
# From project root directory

# Using ingredient list (comma-separated)
python -m recipe_matcher.bin.main match --ingredients "apple,milk,butter"

# Using JSON file (if you have mock test data)
python -m recipe_matcher.bin.main match --input data/mock_ingredients/case1.json

# Skip ontology processing (use normalized data only)
python -m recipe_matcher.bin.main match --ingredients "apple,milk" --no-ontology

# Get more recommendations (default is 5)
python -m recipe_matcher.bin.main match --ingredients "chicken,onion,garlic" --topk 10
```

## Module Responsibilities

### helpers.py
- Pure data I/O (no business logic)
- **Path Configuration**: All paths are relative to project root
- Functions:
  - `load_raw_recipes()` - Read raw CSV from `data/recipes/recipe_dataset_raw.csv`
  - `load_normalized_recipes()` - Read normalized JSON/CSV from `data/normalized_recipes.json`
  - `load_ontology_recipes()` - Read ontology-processed JSON/CSV from `data/ontology_recipes.json`
  - `save_normalized_recipes()` - Save normalized data to `data/` directory
  - `save_ontology_recipes()` - Save ontology data to `data/` directory

### preprocess.py
- Step 1 & 2: Ingredient preprocessing (combines normalize + ontology)
- **IngredientNormalizer**: Basic text cleaning
  - Remove quantities, units, punctuation
  - Light normalization
- **IngredientOntology**: Semantic processing
  - Ingredient canonicalization
  - Synonym mapping (e.g., "parmesan cheese" -> "parmesan")
  - Multi-word ingredient handling
  - Food word detection (head noun extraction)
  - Cooking modifier removal
  - Deduplication

### retrieval_engine.py
- Complete Retrieve & Rank architecture (combines retrieve + rank + match)
- **match_recipe()**: Core matching algorithm
  - Fuzzy string matching using rapidfuzz
  - Score calculation
  - Returns match details (overlap, fuzzy_score, matched, missing)
- **RecipeRetriever**: Stage 1 - Fast candidate retrieval
  - Two-stage recall strategy:
    1. Ontology recall (canonical matches)
    2. Fuzzy recall (similarity > 75%)
  - Returns top 300 candidates (configurable)
- **RecipeRanker**: Stage 2 - Precise ranking
  - Detailed match scoring on candidates
  - Sort by (fuzzy_score, overlap)
  - Returns top K ranked results

### pipeline.py
- Orchestrates the entire workflow
- Functions:
  - `normalize_recipe_dataset()` - Run Step 1
  - `apply_ontology_processing()` - Run Step 2
- Class: `RecipePipeline` - Main matching interface with Retrieve & Rank

## Python API Usage

**Note:** When using the Python API, make sure your working directory is the project root (`ml-app-deployment-proj/`), or adjust the paths in `helpers.py` accordingly.

### Full Pipeline (Retrieve & Rank Architecture)

```python
# Run this from project root: ml-app-deployment-proj/
import sys
import os

# Make sure we're in the right directory
os.chdir('/path/to/ml-app-deployment-proj')

from recipe_matcher.pipeline import (
    normalize_recipe_dataset,
    apply_ontology_processing,
    RecipePipeline
)

# Step 1: Normalize (first time only)
# Reads: data/recipes/recipe_dataset_raw.csv
# Writes: data/normalized_recipes.json
normalize_recipe_dataset()

# Step 2: Apply ontology (first time only)
# Reads: data/normalized_recipes.json
# Writes: data/ontology_recipes.json
apply_ontology_processing()

# Step 3: Match ingredients using Retrieve & Rank
pipeline = RecipePipeline(use_ontology=True)
user_ings = ["apple", "milk", "butter", "eggs"]

# Run pipeline:
# - Retrieve: Fast candidate recall (top 300)
# - Rank: Precise scoring (top 5)
processed, top_5 = pipeline.run(user_ings, top_k=5, retrieve_k=300)

# View results
for recipe in top_5:
    print(f"{recipe['title']}: {recipe['fuzzy_score']:.2%}")
    print(f"  Overlap: {recipe['overlap']}")
    print(f"  Missing: {recipe['missing']}")
```

### Individual Components

```python
# Normalize
from recipe_matcher.preprocess import IngredientNormalizer
normalizer = IngredientNormalizer()
clean = normalizer.normalize_ingredient("2 cups of milk")
# Result: "milk"

# Ontology
from recipe_matcher.preprocess import IngredientOntology
ontology = IngredientOntology()
canonical = ontology.canonicalize("parmesan cheese")
# Result: "parmesan"

# Retrieval Engine
from recipe_matcher.retrieval_engine import RecipeRetriever, RecipeRanker, match_recipe
from recipe_matcher.utils.helpers import load_ontology_recipes

# Retrieve candidates
recipes = load_ontology_recipes()
retriever = RecipeRetriever(recipes)
candidates = retriever.retrieve(["apple", "milk"], top_k=300)
# Returns: DataFrame of candidate recipes

# Rank candidates
ranker = RecipeRanker()
top_5 = ranker.rank(candidates, ["apple", "milk"], top_k=5)
# Returns: List of ranked recipe dictionaries

# Match (low-level)
result = match_recipe(
    user_ingredients=["chicken", "onion"],
    recipe_ingredients=["chicken", "onion", "garlic", "tomato"]
)
# Returns: {'overlap': 2, 'fuzzy_score': 0.85, 'matched': [...], 'missing': [...]}
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

## Troubleshooting

### ModuleNotFoundError: rapidfuzz
```bash
pip install rapidfuzz
```

### File not found: recipe_dataset_raw.csv
Make sure the raw recipe data is located at:
```
ml-app-deployment-proj/data/recipes/recipe_dataset_raw.csv
```

And run commands from the project root directory:
```bash
cd ml-app-deployment-proj/
python -m recipe_matcher.bin.main normalize
```

### File not found: normalized_recipes.json
Run normalization first (from project root):
```bash
cd ml-app-deployment-proj/
python -m recipe_matcher.bin.main normalize
```

This will create `data/normalized_recipes.json`.

### File not found: ontology_recipes.json
Run ontology processing (from project root):
```bash
cd ml-app-deployment-proj/
python -m recipe_matcher.bin.main ontology
```

Or use normalized data only:
```bash
python -m recipe_matcher.bin.main match --ingredients "apple,milk" --no-ontology
```

### ModuleNotFoundError: recipe_matcher
Make sure you're running from the project root and the module is installed:
```bash
cd ml-app-deployment-proj/
pip install -e recipe_matching_system/
```

Or if using from the main project:
```bash
cd ml-app-deployment-proj/
pip install -e .
```

