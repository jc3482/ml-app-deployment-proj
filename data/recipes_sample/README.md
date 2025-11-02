# Recipe Dataset Sample

## Overview

This is a small sample of our recipe dataset for testing and development. We included only 100 recipes and 50 images to keep the repository size manageable for GitHub.

## Sample Statistics

- **Recipes**: 100 recipes with full details
- **Images**: 50 food images
- **Total Size**: ~872 KB
- **Format**: CSV with separate image directory

## Purpose

Our team uses this sample for:

- Quick testing of recipe retrieval pipeline
- Development without downloading the full dataset
- CI/CD pipeline testing
- Verifying NLP and embedding generation
- Demo purposes

## Dataset Structure

```
recipes_sample/
├── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
├── Food Images/
│   └── [50 recipe images]
└── README.md
```

## CSV Columns

- **Title**: Recipe name
- **Ingredients**: List of ingredients (raw format)
- **Instructions**: Cooking instructions
- **Image_Name**: Associated image filename (without extension)
- **Cleaned_Ingredients**: Processed ingredient list

## Full Dataset

The complete dataset contains significantly more data:

### Full Dataset Stats
- **Recipes**: 58,783 recipes
- **Images**: 13,582 food images
- **CSV Size**: 25 MB
- **Images Size**: 229 MB
- **Total**: ~254 MB

### Getting the Full Dataset

Contact team members for access to the full recipe dataset. It's stored separately due to size constraints.

#### Setup Instructions

1. Obtain the full dataset from team storage
2. Extract to `data/recipes/`:
   ```
   data/recipes/
   ├── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
   └── Food Images/
   ```

## Usage Examples

### Load Sample Data

```python
import pandas as pd

# Load recipes
df = pd.read_csv('data/recipes_sample/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

print(f"Total recipes: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# View first recipe
print(df.iloc[0]['Title'])
print(df.iloc[0]['Ingredients'])
```

### Test Recipe Embeddings

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for sample recipes
titles = df['Title'].tolist()
embeddings = model.encode(titles, show_progress_bar=True)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

### Test FAISS Indexing

```python
import faiss
import numpy as np

# Normalize embeddings
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Build index
dimension = embeddings_norm.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_norm.astype('float32'))

print(f"Index contains {index.ntotal} recipes")

# Test search
query = model.encode(["chicken pasta"])
query_norm = query / np.linalg.norm(query)
distances, indices = index.search(query_norm.astype('float32'), k=5)

print("Top 5 similar recipes:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {df.iloc[idx]['Title']}")
```

## Recipe Categories

The sample includes diverse recipes covering:
- Main courses
- Desserts
- Cocktails and drinks
- Side dishes
- Appetizers

## Data Quality Notes

- All recipes include title, ingredients, and instructions
- Not all recipes have associated images
- Ingredient lists are in raw format (may need cleaning)
- Instructions are complete cooking steps

## Integration with SmartPantry

This sample works with our pipeline:

1. **Recipe Loading**: Load CSV into database
2. **Embedding Generation**: Create embeddings with Sentence-BERT
3. **FAISS Indexing**: Build search index
4. **Retrieval**: Match detected ingredients to recipes

## License

Check with the original dataset source for licensing information.

## Notes for Development

- Use this sample for initial development and testing
- Switch to full dataset for final training and evaluation
- The sample is representative of the full dataset structure
- Images may not match all recipes in the sample (that's okay for testing)

