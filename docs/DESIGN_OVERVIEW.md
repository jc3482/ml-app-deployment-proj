# SmartPantry: Comprehensive Design Overview

**Team**: Stacy Che, Kexin Lyu, Samantha Wang, Zexi Wu (Allen), Tongrui Zhang (Neil)  
**Course**: Machine Learning Application Deployment  
**Last Updated**: November 2, 2025

---

## Table of Contents

1. [Project Evolution](#1-project-evolution)
2. [System Architecture](#2-system-architecture)
3. [Computer Vision Pipeline](#3-computer-vision-pipeline)
4. [Recipe Retrieval System](#4-recipe-retrieval-system)
5. [Data Structures and Algorithms](#5-data-structures-and-algorithms)
6. [User Experience and Edge Cases](#6-user-experience-and-edge-cases)
7. [Data Management](#7-data-management)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Project Evolution

### 1.1 Original Proposal vs. Final Design

**Initial Vision**:
- Food-101 dataset (101 meal categories, 101,000 images)
- Recipe1M+ dataset (1M+ recipes)
- Multiple model architectures (YOLOv8, Faster R-CNN)
- Multimodal learning with Im2Recipe
- Complex AWS deployment (EC2, S3, Lambda)

**Why We Changed**:

**Dataset Mismatch**: Food-101 shows complete plated meals, not raw fridge ingredients. Mapping "spaghetti_bolognese" to "pasta" and "tomato" was too complex and indirect for our use case.

**Data Scale Issues**: Recipe1M+ had over a million entries with messy data formats. Preprocessing ingredient lists, handling missing fields, and mapping to detection classes would take the entire semester.

**Scope Creep**: Multiple model architectures and multimodal learning were too ambitious. We needed to focus on core concepts from class (data structures, information retrieval) rather than cutting-edge research.

**Deployment Complexity**: AWS multi-service architecture was overkill for a student project and would require significant DevOps experience we don't have.

### 1.2 Final Focused Design

**Smart Refrigerator Dataset (Roboflow)**:
- 3,049 images of actual fridge contents
- 30 common ingredient classes
- Pre-labeled with YOLO annotations
- Directly matches our use case

**Food Ingredients Recipe Dataset**:
- 58,783 recipes with clean ingredient lists
- Includes cooking steps and serving sizes
- Structured CSV format (easy to process)
- Manageable size for semester project

**Single Model Focus**: YOLOv8 for detection, with clear comparison between TF-IDF and Sentence-BERT for retrieval.

**Simplified Deployment**: Docker + Hugging Face Spaces, with AWS as optional stretch goal.

---

## 2. System Architecture

### 2.1 High-Level Components

```
USER INPUT
    ↓
[Image Validation] → Check if valid fridge photo
    ↓
[Ingredient Detection] → YOLOv8 object detection
    ↓
[Multi-Photo Merging] → Combine detections, remove duplicates (Bloom filter)
    ↓
[Name Normalization] → Fuzzy matching to standard names
    ↓
[Recipe Retrieval] → TF-IDF + Sentence-BERT hybrid search
    ↓
[Recipe Ranking] → Ingredient overlap + semantic similarity
    ↓
[Recipe Clustering] → Group by cuisine and difficulty
    ↓
[User Interface] → Gradio web application
    ↓
USER OUTPUT
```

### 2.2 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Frontend** | Gradio | Simple Python-based UI, no HTML/CSS needed |
| **Backend** | Python 3.10+ | Team expertise, ML ecosystem |
| **Computer Vision** | YOLOv8 (Ultralytics) | State-of-art object detection, easy to train |
| **Image Processing** | OpenCV, Pillow | Standard tools for preprocessing |
| **NLP Baseline** | TF-IDF (scikit-learn) | Classic IR baseline from class |
| **NLP Advanced** | Sentence-BERT | Modern semantic embeddings |
| **Vector Search** | FAISS | Fast approximate nearest neighbors |
| **Data Structures** | Custom Python | Bloom filters, heaps, hash tables (class concepts) |
| **Data Storage** | Pandas DataFrame | Simple CSV-based recipe storage |
| **Config Management** | YAML | Human-readable configuration |
| **Package Manager** | uv | Fast dependency resolution |
| **Containerization** | Docker | Reproducible deployment |
| **Deployment** | Hugging Face Spaces | Free hosting for Gradio apps |

---

## 3. Computer Vision Pipeline

### 3.1 Ingredient Detection

**Model**: YOLOv8 Medium (yolov8m)
- **Input**: RGB images (640×640)
- **Output**: Bounding boxes, class labels, confidence scores
- **Classes**: 30 ingredient categories (see section 7.1)
- **Confidence Threshold**: 0.25 (configurable)
- **NMS IoU Threshold**: 0.45

**Training Strategy**:
1. Start with pre-trained YOLOv8 weights (COCO dataset)
2. Fine-tune on Smart Refrigerator dataset
3. Use data augmentation (Albumentations):
   - Random brightness/contrast
   - Random rotation
   - Horizontal flip
   - Gaussian noise (simulate poor camera quality)
4. Train for 100 epochs with early stopping
5. Validate on held-out test set

### 3.2 Image Preprocessing

**Validation Checks**:
```python
def validate_image(image_path):
    # 1. Check file format
    if not image_path.endswith(('.jpg', '.jpeg', '.png')):
        return False, "Invalid format. Use JPG or PNG"
    
    # 2. Check file size
    if os.path.getsize(image_path) > 10 * 1024 * 1024:  # 10MB
        return False, "File too large. Max 10MB"
    
    # 3. Check image quality
    img = cv2.imread(image_path)
    if img is None:
        return False, "Cannot read image"
    
    # 4. Check if image is too dark/bright
    mean_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if mean_brightness < 30 or mean_brightness > 225:
        return False, "Image too dark/bright. Try better lighting"
    
    # 5. Check resolution
    h, w = img.shape[:2]
    if h < 300 or w < 300:
        return False, "Resolution too low. Min 300×300"
    
    return True, "Valid"
```

**Preprocessing Steps**:
1. Resize to 640×640 (maintain aspect ratio, pad)
2. Normalize pixel values [0, 255] → [0, 1]
3. Convert RGB color space
4. Apply inference-time augmentation (optional)

### 3.3 Post-Detection Processing

**Confidence Filtering**:
```python
detections = [d for d in raw_detections if d['confidence'] > 0.3]
```

**Name Normalization** (fuzzy matching):
```python
from fuzzywuzzy import fuzz

def normalize_ingredient(detected_name, vocab, threshold=85):
    best_match = None
    best_score = 0
    
    for standard_name in vocab:
        score = fuzz.ratio(detected_name.lower(), standard_name.lower())
        if score > best_score and score > threshold:
            best_score = score
            best_match = standard_name
    
    return best_match or detected_name
```

**Duplicate Removal**: See Section 5.1 (Bloom Filter)

---

## 4. Recipe Retrieval System

### 4.1 Retrieval Methods Comparison

We implement three retrieval approaches and compare their performance:

#### Method A: TF-IDF with Cosine Similarity (Baseline)

**Why**: Classic information retrieval from class. Fast, interpretable, good baseline.

**Pipeline**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Build TF-IDF matrix (done once offline)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
recipe_ingredients_text = [" ".join(recipe['ingredients']) for recipe in recipes]
tfidf_matrix = vectorizer.fit_transform(recipe_ingredients_text)

# Query (at inference time)
query_text = " ".join(detected_ingredients)
query_vector = vectorizer.transform([query_text])
similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
top_k_indices = np.argsort(similarities)[-50:][::-1]
```

**Pros**:
- Fast (no neural network inference)
- Interpretable (can see term weights)
- Works well for exact ingredient matches

**Cons**:
- No semantic understanding ("chicken breast" vs "poultry")
- Bag-of-words (ignores word order)
- Sparse vectors (high dimensionality)

#### Method B: Sentence-BERT Embeddings (Advanced)

**Why**: Modern semantic search. Understands context and similarity.

**Pipeline**:
```python
from sentence_transformers import SentenceTransformer
import faiss

# Build embedding index (done once offline)
model = SentenceTransformer('all-MiniLM-L6-v2')
recipe_embeddings = model.encode(recipe_ingredients_text, show_progress_bar=True)

# Create FAISS index
dimension = recipe_embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100)
index.train(recipe_embeddings)
index.add(recipe_embeddings)

# Query (at inference time)
query_embedding = model.encode([query_text])
distances, indices = index.search(query_embedding, k=50)
```

**Pros**:
- Semantic understanding (synonyms, related concepts)
- Dense vectors (low dimensionality: 384)
- Handles paraphrasing well

**Cons**:
- Slower (neural network inference)
- Needs GPU for large-scale embedding
- Less interpretable

#### Method C: Hybrid Ranking (Our Approach)

**Why**: Combine strengths of both methods.

**Pipeline**:
```python
def hybrid_search(detected_ingredients, alpha=0.6, beta=0.3, gamma=0.1):
    # Step 1: Get TF-IDF scores
    tfidf_scores = tfidf_retrieval(detected_ingredients)  # 0-1 normalized
    
    # Step 2: Get embedding scores
    embedding_scores = embedding_retrieval(detected_ingredients)  # 0-1 normalized
    
    # Step 3: Compute ingredient overlap
    overlap_scores = []
    for recipe in recipes:
        overlap = len(set(detected_ingredients) & set(recipe['ingredients']))
        total = len(set(detected_ingredients) | set(recipe['ingredients']))
        overlap_scores.append(overlap / total)  # Jaccard similarity
    
    # Step 4: Hybrid scoring
    final_scores = (
        alpha * np.array(overlap_scores) +
        beta * np.array(embedding_scores) +
        gamma * np.array(tfidf_scores)
    )
    
    # Step 5: Rank and return top K
    top_k_indices = np.argsort(final_scores)[-20:][::-1]
    return [recipes[i] for i in top_k_indices]
```

**Weight Tuning**:
- `alpha=0.6`: Ingredient overlap (most important)
- `beta=0.3`: Semantic similarity (context)
- `gamma=0.1`: TF-IDF (keyword match)

These weights are configurable in `config.yaml`.

### 4.2 Partial Match Handling

**Problem**: User has only 3 out of 8 ingredients needed.

**Solution**: Minimum overlap threshold with penalty scaling.

```python
def retrieve_with_partial_match(detected_ingredients, min_match=0.3):
    results = []
    
    for recipe in recipes:
        overlap = len(set(detected_ingredients) & set(recipe['ingredients']))
        required = len(recipe['ingredients'])
        ratio = overlap / required
        
        if ratio >= min_match:
            # Penalty for missing ingredients
            missing = required - overlap
            penalty = 1.0 - (missing / required) * 0.5
            score = ratio * penalty
            results.append((recipe, score))
    
    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:20]]
```

**User Feedback**: Display missing ingredients clearly:
```
Recipe: Spaghetti Carbonara
✓ You have: eggs, parmesan, bacon
✗ Missing: spaghetti, black pepper
Match: 60%
```

### 4.3 Recipe Clustering

**Purpose**: Organize results by cuisine type and difficulty for better UX.

**Method**: K-means clustering on recipe features.

```python
from sklearn.cluster import KMeans

def cluster_recipes(recipes, n_clusters=5):
    # Feature extraction
    features = []
    for recipe in recipes:
        cuisine_onehot = encode_cuisine(recipe['cuisine'])
        difficulty_onehot = encode_difficulty(recipe['difficulty'])
        cooking_time = recipe['cooking_time'] / 120.0  # Normalize to [0,1]
        n_ingredients = len(recipe['ingredients']) / 20.0
        
        feature_vector = np.concatenate([
            cuisine_onehot,
            difficulty_onehot,
            [cooking_time, n_ingredients]
        ])
        features.append(feature_vector)
    
    # Clustering
    features = np.array(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Group recipes by cluster
    clustered = defaultdict(list)
    for recipe, label in zip(recipes, labels):
        clustered[label].append(recipe)
    
    return dict(clustered)
```

**Display Format**:
```
Cluster 1: Italian | Easy (5 recipes)
Cluster 2: Japanese | Medium (3 recipes)
Cluster 3: Mexican | Easy (4 recipes)
...
```

---

## 5. Data Structures and Algorithms

### 5.1 Bloom Filter (Duplicate Detection)

**Use Case**: When users upload multiple photos, we need to merge detections and remove duplicates efficiently.

**Why Bloom Filter**:
- Space-efficient: O(1) space per element (vs O(n) for hash set)
- Fast lookups: O(k) where k = number of hash functions
- Acceptable false positive rate for our use case

**Implementation**:
```python
import hashlib

class BloomFilter:
    def __init__(self, size=1000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
    
    def _hash(self, item, seed):
        h = hashlib.md5((str(item) + str(seed)).encode())
        return int(h.hexdigest(), 16) % self.size
    
    def add(self, item):
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = 1
    
    def contains(self, item):
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True  # Possibly in set (might be false positive)

def merge_detections(photos_detections):
    bloom = BloomFilter(size=1000, num_hashes=3)
    merged = []
    
    for photo_detections in photos_detections:
        for detection in photo_detections:
            ingredient_name = detection['class']
            
            # Check if already seen (approximately)
            if not bloom.contains(ingredient_name):
                bloom.add(ingredient_name)
                merged.append(detection)
            else:
                # Likely duplicate, aggregate confidence
                for existing in merged:
                    if existing['class'] == ingredient_name:
                        existing['confidence'] = max(
                            existing['confidence'],
                            detection['confidence']
                        )
                        break
    
    return merged
```

**Analysis**:
- **Size**: 1000 bits = 125 bytes (very small)
- **False Positive Rate**: ~0.7% (acceptable for duplicate detection)
- **Time Complexity**: O(k) per lookup, where k=3
- **Space Complexity**: O(m) where m=1000 bits (constant)

### 5.2 Min Heap (Top-K Ranking)

**Use Case**: Efficiently find top-20 recipes from 58,783 candidates.

**Why Min Heap**:
- Efficient top-K: O(n log k) vs O(n log n) for full sort
- Constant space: O(k) vs O(n) for sorting all
- Online algorithm: Can process streaming results

**Implementation**:
```python
import heapq

class TopKRecipes:
    def __init__(self, k=20):
        self.k = k
        self.heap = []
    
    def add(self, recipe, score):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, recipe))
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, (score, recipe))
    
    def get_top_k(self):
        # Return sorted in descending order
        return [recipe for score, recipe in sorted(self.heap, reverse=True)]

def rank_recipes(recipes, detected_ingredients):
    top_k = TopKRecipes(k=20)
    
    for recipe in recipes:
        score = compute_score(recipe, detected_ingredients)
        top_k.add(recipe, score)
    
    return top_k.get_top_k()
```

**Analysis**:
- **Time Complexity**: O(n log k) where n=58,783, k=20
- **Space Complexity**: O(k) = O(20) (constant)
- **Comparison**: Full sort would be O(n log n) ≈ 987,000 operations
- **Min Heap**: 58,783 × log(20) ≈ 255,000 operations (**~4x faster**)

### 5.3 Hash Table (Fast Recipe Lookup)

**Use Case**: Quick access to recipe details by ID.

**Implementation**:
```python
class RecipeIndex:
    def __init__(self, recipes):
        self.id_to_recipe = {}
        self.name_to_recipe = {}
        self.ingredient_to_recipes = defaultdict(list)
        
        for recipe in recipes:
            # Index by ID
            self.id_to_recipe[recipe['id']] = recipe
            
            # Index by name (for search)
            self.name_to_recipe[recipe['title'].lower()] = recipe
            
            # Inverted index: ingredient -> recipes
            for ingredient in recipe['ingredients']:
                self.ingredient_to_recipes[ingredient].append(recipe['id'])
    
    def get_by_id(self, recipe_id):
        return self.id_to_recipe.get(recipe_id)
    
    def get_by_name(self, name):
        return self.name_to_recipe.get(name.lower())
    
    def get_by_ingredient(self, ingredient):
        recipe_ids = self.ingredient_to_recipes.get(ingredient, [])
        return [self.id_to_recipe[rid] for rid in recipe_ids]
```

**Analysis**:
- **Lookup Time**: O(1) average case
- **Space Complexity**: O(n) where n = number of recipes
- **Use Case**: After ranking, quickly fetch full recipe details

---

## 6. User Experience and Edge Cases

### 6.1 Multi-Photo Upload Workflow

**Scenario**: User takes 3 photos of different fridge compartments.

**Process**:
1. User uploads Photo 1 (top shelf) → Detects: milk, eggs, butter
2. User uploads Photo 2 (middle shelf) → Detects: chicken, lettuce, tomato
3. User uploads Photo 3 (door) → Detects: milk, ketchup, mustard

**System Response**:
```python
def handle_multi_photo_upload(photos):
    all_detections = []
    
    # Step 1: Run detection on each photo
    for photo in photos:
        detections = yolo_detect(photo)
        all_detections.append(detections)
    
    # Step 2: Merge and deduplicate using Bloom filter
    merged = merge_detections(all_detections)
    
    # Step 3: Normalize names
    normalized = [normalize_ingredient(d['class']) for d in merged]
    
    # Step 4: Show to user
    return {
        'ingredients': list(set(normalized)),
        'confidence_avg': np.mean([d['confidence'] for d in merged]),
        'photos_processed': len(photos)
    }
```

**UI Display**:
```
✓ Processed 3 photos
Found 7 unique ingredients:
- milk (confidence: 92%)
- eggs (confidence: 88%)
- butter (confidence: 85%)
- chicken (confidence: 90%)
- lettuce (confidence: 78%)
- tomato (confidence: 82%)
- ketchup (confidence: 75%)
- mustard (confidence: 70%)

[Edit List] [Find Recipes]
```

### 6.2 Edge Case Handling

#### Case 1: Invalid Image Format

**Input**: User uploads a PDF or HEIC file.

**Response**:
```
❌ Error: Invalid file format
Please upload JPG or PNG images only.
[Try Again]
```

#### Case 2: Not a Fridge Photo

**Input**: User uploads a selfie or landscape photo.

**Detection Method**:
```python
def is_fridge_photo(image):
    # Simple heuristic: check for typical fridge features
    # - Rectangular shelves (edge detection)
    # - Common fridge items (at least 2 detections)
    # - Indoor lighting conditions
    
    detections = yolo_detect(image)
    
    if len(detections) == 0:
        return False, "No ingredients detected"
    
    # Check detection distribution (fridge items are usually multiple)
    if len(detections) < 2:
        return False, "This doesn't look like a fridge photo"
    
    return True, "Valid"
```

**Response**:
```
❌ Error: This doesn't appear to be a fridge photo
Please take a clear photo of your fridge contents.
Tips:
- Open the fridge door fully
- Use good lighting
- Show multiple items clearly
[Try Again]
```

#### Case 3: Low Image Quality

**Input**: Dark, blurry, or low-resolution image.

**Response**:
```
⚠️ Warning: Image quality is low
We detected ingredients but confidence is low.
Tips for better results:
- Use better lighting
- Hold phone steady
- Get closer to items
[Continue Anyway] [Retake Photo]
```

#### Case 4: No Ingredients Detected

**Input**: Empty fridge or all items unrecognizable.

**Response**:
```
❌ Error: No ingredients detected
This could mean:
- Fridge is empty
- Items are not in our database (30 classes)
- Photo quality is too poor
[Try Different Angle] [See What We Can Detect]
```

#### Case 5: No Recipes Found

**Input**: Detected ingredients don't match any recipes.

**Response**:
```
❌ No recipes found with all ingredients
Found ingredients: celery, radish

Would you like to:
[✓] Enable partial match mode (show recipes with some ingredients)
[✓] Add more photos (detect more ingredients)
[✓] Manually add ingredients
```

**Partial Match Results**:
```
Showing recipes with at least 30% ingredient match:

Recipe: Garden Salad
✓ You have: celery, radish
✗ Missing: lettuce, cucumber, tomato
Match: 40% | [View Recipe]
```

### 6.3 Manual Ingredient Editing

**Scenario**: Detection missed an item or detected wrong item.

**UI Interface**:
```
Detected Ingredients:
[x] milk        [x] eggs        [x] chicken
[x] lettuce     [x] tomato      [x] cheese

[+ Add Ingredient]

Search: [bacon        ] [Add]

Remove incorrect detections by clicking [x]
```

**Backend**:
```python
def update_ingredient_list(detected, added, removed):
    # Start with detections
    ingredients = set(detected)
    
    # Apply user edits
    ingredients.update(added)
    ingredients.difference_update(removed)
    
    # Re-normalize
    ingredients = [normalize_ingredient(ing) for ing in ingredients]
    
    return list(set(ingredients))
```

---

## 7. Data Management

### 7.1 Dataset Details

#### Smart Refrigerator Dataset (Fridge Photos)

**Source**: Roboflow Universe  
**Link**: https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr

**Statistics**:
- **Total Images**: 3,049
- **Train Split**: 2,895 images (95%)
- **Validation Split**: 103 images (3.4%)
- **Test Split**: 51 images (1.7%)
- **Format**: YOLOv8 (images + .txt annotations)
- **Image Size**: Variable (640×640 recommended)
- **Annotation Format**: Normalized bounding box coordinates

**Classes** (30 total):
```
1. milk             11. butter           21. yogurt
2. eggs             12. cheese           22. soda
3. chicken          13. beef             23. juice
4. lettuce          14. pork             24. water
5. tomato           15. fish             25. bread
6. carrot           16. shrimp           26. apple
7. potato           17. tofu             27. orange
8. onion            18. beans            28. banana
9. garlic           19. rice             29. strawberry
10. pepper          20. pasta            30. broccoli
```

**Git Strategy**:
- **Full Dataset** (~400 MB): `.gitignore`, stored locally
- **Sample Dataset** (3.1 MB): Committed to Git
  - 20 train images
  - 5 validation images
  - 5 test images

#### Food Ingredients Recipe Dataset

**Source**: Kaggle  
**Link**: https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

**Statistics**:
- **Total Recipes**: 58,783
- **Format**: CSV with columns:
  - `Title`: Recipe name
  - `Ingredients`: Comma-separated ingredient list
  - `Instructions`: Step-by-step cooking instructions
  - `Image_Name`: Associated food image filename
  - `Cleaned_Ingredients`: Preprocessed ingredient list
- **Associated Images**: 13,582 food photos
- **Image Size**: Variable (need resizing)

**Sample Entry**:
```csv
Title,Ingredients,Instructions,Image_Name,Cleaned_Ingredients
"Spaghetti Carbonara","eggs, parmesan cheese, bacon, spaghetti, black pepper","1. Cook spaghetti. 2. Fry bacon...","carbonara_001.jpg","eggs,parmesan,bacon,spaghetti,black pepper"
```

**Git Strategy**:
- **Full Dataset** (~254 MB): `.gitignore`, stored locally
- **Sample Dataset** (872 KB): Committed to Git
  - 100 recipes (CSV)
  - 50 associated images

### 7.2 Data Directory Structure

```
data/
├── README.md                    # Dataset overview and setup
├── raw/                         # Original downloaded data (not in Git)
│   ├── food-101/               # Optional: Food-101 dataset
│   └── recipe1m/               # Optional: Recipe1M+ dataset
├── processed/                   # Preprocessed data (not in Git)
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
├── embeddings/                  # Cached embeddings (not in Git)
│   ├── recipe_embeddings.npy   # Sentence-BERT embeddings
│   └── tfidf_matrix.pkl        # TF-IDF matrix
├── fridge_photos/              # Full fridge dataset (not in Git)
│   ├── train/
│   │   ├── images/             # 2,895 images
│   │   └── labels/             # 2,895 .txt files
│   ├── valid/
│   │   ├── images/             # 103 images
│   │   └── labels/             # 103 .txt files
│   ├── test/
│   │   ├── images/             # 51 images
│   │   └── labels/             # 51 .txt files
│   ├── data.yaml               # YOLO config
│   └── DATASET_INFO.md
├── fridge_photos_sample/       # Sample dataset (IN GIT)
│   ├── train/
│   │   ├── images/             # 20 images
│   │   └── labels/             # 20 .txt files
│   ├── valid/
│   │   ├── images/             # 5 images
│   │   └── labels/             # 5 .txt files
│   ├── test/
│   │   ├── images/             # 5 images
│   │   └── labels/             # 5 .txt files
│   ├── data.yaml
│   └── README.md
├── recipes/                     # Full recipe dataset (not in Git)
│   ├── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
│   └── Food Images/            # 13,582 images
└── recipes_sample/             # Sample recipes (IN GIT)
    ├── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
    └── Food Images/            # 50 images
```

### 7.3 Data Pipeline

```python
# 1. Load fridge photos for training
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # Start with pretrained weights
results = model.train(
    data='data/fridge_photos/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)

# 2. Load and preprocess recipes
import pandas as pd

recipes_df = pd.read_csv('data/recipes/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

# Clean and preprocess
recipes_df['Cleaned_Ingredients'] = recipes_df['Ingredients'].apply(clean_ingredients)
recipes_df['Ingredient_List'] = recipes_df['Cleaned_Ingredients'].str.split(',')

# 3. Build embeddings (done once, cached)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
recipe_texts = recipes_df['Cleaned_Ingredients'].tolist()
embeddings = model.encode(recipe_texts, show_progress_bar=True)
np.save('data/embeddings/recipe_embeddings.npy', embeddings)

# 4. Build FAISS index
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100)
index.train(embeddings)
index.add(embeddings)
faiss.write_index(index, 'data/embeddings/faiss_index.bin')

# 5. Build TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(recipe_texts)
pickle.dump((vectorizer, tfidf_matrix), open('data/embeddings/tfidf_matrix.pkl', 'wb'))
```

---

## 8. Evaluation Strategy

### 8.1 Computer Vision Metrics

**Metrics**: Precision, Recall, F1, mAP (mean Average Precision)

**Evaluation Script**:
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/yolo/best.pt')

# Evaluate on test set
results = model.val(data='data/fridge_photos/data.yaml', split='test')

print(f"Precision: {results.box.p:.3f}")
print(f"Recall: {results.box.r:.3f}")
print(f"mAP50: {results.box.map50:.3f}")
print(f"mAP50-95: {results.box.map:.3f}")
```

**Target Performance**:
- **Precision**: ≥ 0.75 (minimize false positives)
- **Recall**: ≥ 0.70 (detect most ingredients)
- **mAP50**: ≥ 0.65 (good localization)

### 8.2 Recipe Retrieval Metrics

**Metrics**: Recall@K, nDCG@K, MRR (Mean Reciprocal Rank)

**Recall@K**: What fraction of relevant recipes are in top-K?
```python
def recall_at_k(retrieved, relevant, k=10):
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)
```

**nDCG@K**: Normalized Discounted Cumulative Gain (considers ranking quality)
```python
def ndcg_at_k(retrieved, relevant, k=10):
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1)=0
    
    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0
```

**MRR**: How quickly do we find the first relevant recipe?
```python
def mean_reciprocal_rank(retrieved, relevant):
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0
```

**Test Set Creation**:
```python
# Create ground truth: ingredient list → relevant recipes
test_cases = [
    {
        'ingredients': ['chicken', 'lettuce', 'tomato', 'cheese'],
        'relevant_recipes': ['Caesar Salad', 'Chicken Wrap', 'Cobb Salad']
    },
    # ... 50-100 test cases
]

# Evaluate each method
for method in ['tfidf', 'embedding', 'hybrid']:
    recall_scores = []
    ndcg_scores = []
    mrr_scores = []
    
    for test in test_cases:
        retrieved = retrieve_recipes(test['ingredients'], method=method)
        recall_scores.append(recall_at_k(retrieved, test['relevant_recipes'], k=10))
        ndcg_scores.append(ndcg_at_k(retrieved, test['relevant_recipes'], k=10))
        mrr_scores.append(mean_reciprocal_rank(retrieved, test['relevant_recipes']))
    
    print(f"{method.upper()} Results:")
    print(f"  Recall@10: {np.mean(recall_scores):.3f}")
    print(f"  nDCG@10: {np.mean(ndcg_scores):.3f}")
    print(f"  MRR: {np.mean(mrr_scores):.3f}")
```

**Target Performance**:
- **Recall@10**: ≥ 0.60 (find 60% of relevant recipes in top-10)
- **nDCG@10**: ≥ 0.55 (good ranking quality)
- **MRR**: ≥ 0.40 (first relevant recipe in top-3 on average)

### 8.3 End-to-End System Test

**Test Scenario**:
1. Upload real fridge photo
2. System detects ingredients
3. System retrieves recipes
4. Human evaluator rates top-5 recipes: Relevant / Somewhat Relevant / Not Relevant

**Evaluation**:
```python
ratings = {
    'relevant': 3 points,
    'somewhat': 1 point,
    'not_relevant': 0 points
}

score = sum(ratings[r] for r in top_5_ratings) / 15  # Max score = 15
```

**Target**: Average score ≥ 0.65 (mostly relevant results)

---

## 9. Deployment Architecture

### 9.1 Docker Container

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/fridge_photos_sample/ ./data/fridge_photos_sample/
COPY data/recipes_sample/ ./data/recipes_sample/
COPY config.yaml ./

# Install dependencies
RUN uv pip install --system -e .

# Expose Gradio port
EXPOSE 7860

# Run application
CMD ["python", "app/main.py"]
```

**Build and Run**:
```bash
# Build image
docker build -t smartpantry:latest .

# Run container
docker run -p 7860:7860 smartpantry:latest

# Access at http://localhost:7860
```

### 9.2 Hugging Face Spaces Deployment

**Steps**:
1. Create new Space on Hugging Face
2. Select Gradio SDK
3. Push code to Space repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/username/smartpantry
   git push hf main
   ```
4. Space automatically builds and deploys

**Space Configuration** (`README.md` header):
```yaml
---
title: SmartPantry
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app/main.py
pinned: false
---
```

**Advantages**:
- Free hosting
- Automatic SSL/HTTPS
- GPU available (paid tier)
- Easy sharing (public URL)

### 9.3 AWS Deployment (Optional)

**Architecture**:
```
User → ALB (Load Balancer) → EC2 (Gradio App) → S3 (Models/Data)
```

**EC2 Instance**:
- **Type**: g4dn.xlarge (GPU)
- **OS**: Ubuntu 22.04
- **Docker**: Run container on instance

**S3 Buckets**:
- `smartpantry-models`: Store YOLOv8 weights, embeddings
- `smartpantry-data`: Store recipe database

**Deployment Script**:
```bash
# On EC2 instance
aws s3 cp s3://smartpantry-models/yolo/best.pt models/yolo/best.pt
aws s3 cp s3://smartpantry-models/embeddings/ data/embeddings/ --recursive
docker-compose up -d
```

**Cost Estimate** (per month):
- EC2 g4dn.xlarge: ~$120/month
- S3 storage (10 GB): ~$0.23/month
- Data transfer: ~$9/month (100 GB)
- **Total**: ~$130/month

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goals**:
- Setup project structure
- Download and organize datasets
- Implement data loading pipelines
- Basic exploratory data analysis

**Deliverables**:
- [x] Project repository with skeleton code
- [x] Sample datasets committed to Git
- [ ] Notebook: `01_data_exploration.ipynb`
- [ ] Data loading utilities in `src/utils/helpers.py`

**Success Criteria**:
- Can load and visualize fridge photos
- Can parse recipe CSV
- Basic statistics computed (class distribution, recipe length)

### Phase 2: Computer Vision (Weeks 3-4)

**Goals**:
- Train YOLOv8 on fridge dataset
- Implement image preprocessing
- Evaluate detection performance
- Save best model weights

**Deliverables**:
- [ ] Notebook: `02_model_training.ipynb`
- [ ] Detection module: `src/vision/detector.py`
- [ ] Preprocessing module: `src/vision/preprocessor.py`
- [ ] Trained model: `models/yolo/best.pt`
- [ ] Evaluation report with mAP, Precision, Recall

**Success Criteria**:
- mAP50 ≥ 0.65
- Precision ≥ 0.75
- Can detect ingredients in test images

### Phase 3: Recipe Retrieval (Weeks 5-6)

**Goals**:
- Implement TF-IDF baseline
- Implement Sentence-BERT embeddings
- Build FAISS index
- Compare retrieval methods
- Implement hybrid ranking

**Deliverables**:
- [ ] Notebook: `03_recipe_retrieval.ipynb`
- [ ] Retrieval module: `src/nlp/retriever.py`
- [ ] Embedding module: `src/nlp/embedder.py`
- [ ] Cached embeddings and indices in `data/embeddings/`
- [ ] Comparison report: TF-IDF vs Embeddings vs Hybrid

**Success Criteria**:
- Recall@10 ≥ 0.60
- nDCG@10 ≥ 0.55
- Hybrid method outperforms baselines

### Phase 4: Data Structures (Weeks 7-8)

**Goals**:
- Implement Bloom filter for duplicate detection
- Implement min heap for top-K ranking
- Implement hash table for recipe lookup
- Benchmark performance improvements

**Deliverables**:
- [ ] Bloom filter: `src/utils/bloom_filter.py`
- [ ] Top-K heap: `src/utils/ranking.py`
- [ ] Recipe index: `src/utils/recipe_index.py`
- [ ] Performance benchmarks

**Success Criteria**:
- Bloom filter achieves <1% false positive rate
- Min heap is 3-4x faster than full sorting
- Hash lookups are O(1)

### Phase 5: Integration & UI (Weeks 9-10)

**Goals**:
- Build Gradio interface
- Connect all components (detection → retrieval → ranking)
- Implement edge case handling
- Add multi-photo support
- User testing and feedback

**Deliverables**:
- [ ] Gradio app: `app/main.py`
- [ ] UI components: `app/templates/` and `app/static/`
- [ ] End-to-end tests: `tests/integration/test_pipeline.py`
- [ ] User testing results

**Success Criteria**:
- Can upload photo and get recipe recommendations
- Multi-photo upload works correctly
- Edge cases handled gracefully
- User satisfaction score ≥ 4/5

### Phase 6: Deployment (Weeks 11-12)

**Goals**:
- Dockerize application
- Deploy to Hugging Face Spaces
- (Optional) Deploy to AWS
- Documentation and demo video

**Deliverables**:
- [ ] Dockerfile and docker-compose.yml
- [ ] Deployed Space: https://huggingface.co/spaces/username/smartpantry
- [ ] Deployment guide: `deployment/README.md`
- [ ] Final report and presentation
- [ ] Demo video (3-5 minutes)

**Success Criteria**:
- Docker container runs successfully
- Hugging Face Space is publicly accessible
- Documentation is complete and clear
- Demo video showcases all features

### Phase 7: Evaluation & Report (Week 13)

**Goals**:
- Comprehensive evaluation on test set
- Compare all retrieval methods
- Analyze failure cases
- Write final report
- Prepare presentation

**Deliverables**:
- [ ] Evaluation notebook: `notebooks/04_final_evaluation.ipynb`
- [ ] Final report (PDF)
- [ ] Presentation slides
- [ ] Code cleanup and documentation

**Success Criteria**:
- All metrics computed and reported
- Failure cases analyzed
- Report explains design decisions and results
- Presentation ready for class

---

## 11. Key Design Decisions

### 11.1 Why YOLOv8 over Faster R-CNN?

**Decision**: Use YOLOv8 for ingredient detection.

**Reasoning**:
1. **Speed**: YOLOv8 is real-time (30+ FPS), Faster R-CNN is slower (5-10 FPS)
2. **Ease of Use**: Ultralytics library is well-documented and easy to fine-tune
3. **Performance**: YOLOv8 achieves comparable or better mAP than Faster R-CNN
4. **Community**: Large community, many pre-trained models available
5. **Project Scope**: Single model focus aligns with our semester timeline

**Trade-offs**:
- YOLOv8 may struggle with very small objects (not an issue for fridge items)
- Faster R-CNN might have slightly better precision (acceptable trade-off for speed)

### 11.2 Why Hybrid Retrieval over Single Method?

**Decision**: Combine TF-IDF, embeddings, and ingredient overlap.

**Reasoning**:
1. **Complementary Strengths**:
   - TF-IDF: Good for exact keyword matching
   - Embeddings: Good for semantic similarity
   - Overlap: Direct ingredient count matters most
2. **Better Results**: Hybrid consistently outperforms single methods in IR literature
3. **Configurable**: Can tune weights for different user preferences
4. **Learning Objective**: Demonstrates understanding of multiple IR techniques

**Trade-offs**:
- More complex to implement and tune
- Requires computing multiple scores (acceptable overhead)

### 11.3 Why Gradio over Custom Web App?

**Decision**: Use Gradio for frontend.

**Reasoning**:
1. **Speed**: Build UI in Python, no HTML/CSS/JavaScript needed
2. **ML-Friendly**: Designed for ML demos, handles file uploads well
3. **Deployment**: Works seamlessly with Hugging Face Spaces
4. **Team Skills**: Team is comfortable with Python, not web dev
5. **Iteration**: Quick to prototype and iterate

**Trade-offs**:
- Less customizable than Flask/React
- UI styling options are limited
- Good enough for project scope

### 11.4 Why Sample Datasets in Git?

**Decision**: Commit 30 sample images and 100 sample recipes to Git.

**Reasoning**:
1. **Team Collaboration**: All members can test code immediately without large downloads
2. **CI/CD**: Automated tests can run on sample data
3. **Documentation**: Examples for users trying to understand the system
4. **Git Performance**: 3.1 MB is acceptable for Git (under 5 MB guideline)

**Trade-offs**:
- Slightly larger repository size
- Need to keep sample and full datasets in sync (acceptable maintenance)

---

## 12. Risk Management

### 12.1 Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| YOLOv8 accuracy too low | Medium | High | Start training early, try data augmentation, use pretrained weights |
| Recipe retrieval poor quality | Medium | High | Implement multiple methods, tune hybrid weights, user feedback loop |
| Large dataset handling | High | Medium | Use sample datasets for development, optimize data loading |
| AWS deployment cost | Low | Low | Use Hugging Face Spaces as primary, AWS optional |
| Team coordination issues | Medium | Medium | Regular meetings, clear task assignments, Git workflow |
| Semester timeline too tight | High | High | Focus on core features first, make advanced features optional |

### 12.2 Contingency Plans

**If YOLOv8 training fails**:
- Use pre-trained YOLO model on similar dataset (COCO)
- Reduce to 10-15 ingredient classes instead of 30
- Focus more on retrieval system evaluation

**If recipe retrieval is poor**:
- Simplify to TF-IDF only (still meets course requirements)
- Use keyword-based search as fallback
- Document why advanced methods struggled

**If deployment blocked**:
- Run locally and record demo video
- Provide Docker container for easy reproduction
- Document deployment steps for future work

---

## 13. Course Concept Integration

### 13.1 Data Structures (Required)

| Concept | Implementation | Location |
|---------|----------------|----------|
| **Bloom Filter** | Duplicate ingredient detection | `src/utils/bloom_filter.py` |
| **Hash Table** | Fast recipe lookup by ID/name | `src/utils/recipe_index.py` |
| **Min Heap** | Top-K recipe ranking | `src/utils/ranking.py` |
| **Graph** | (Optional) Recipe similarity network | `src/utils/recipe_graph.py` |

### 13.2 Information Retrieval (Required)

| Concept | Implementation | Location |
|---------|----------------|----------|
| **TF-IDF** | Keyword-based recipe search | `src/nlp/retriever.py` |
| **Cosine Similarity** | Document similarity scoring | `src/nlp/retriever.py` |
| **Inverted Index** | Ingredient → recipes mapping | `src/utils/recipe_index.py` |
| **Ranking** | Hybrid scoring and sorting | `src/nlp/retriever.py` |

### 13.3 Machine Learning (Core)

| Concept | Implementation | Location |
|---------|----------------|----------|
| **CNNs** | YOLOv8 backbone (ResNet/CSPNet) | `src/vision/detector.py` |
| **Object Detection** | YOLO for ingredient detection | `src/vision/detector.py` |
| **Transfer Learning** | Fine-tune pretrained YOLOv8 | `notebooks/02_model_training.ipynb` |
| **Embeddings** | Sentence-BERT recipe embeddings | `src/nlp/embedder.py` |
| **Clustering** | K-means for recipe grouping | `src/utils/clustering.py` |

### 13.4 Deployment (Required)

| Concept | Implementation | Location |
|---------|----------------|----------|
| **Containerization** | Docker for reproducibility | `Dockerfile` |
| **API Design** | Gradio interface | `app/main.py` |
| **Cloud Deployment** | Hugging Face Spaces | `deployment/README.md` |
| **Version Control** | Git with proper .gitignore | `.gitignore` |

---

## 14. Success Metrics

### 14.1 Technical Metrics

- [ ] **Detection mAP50** ≥ 0.65
- [ ] **Detection Precision** ≥ 0.75
- [ ] **Retrieval Recall@10** ≥ 0.60
- [ ] **Retrieval nDCG@10** ≥ 0.55
- [ ] **End-to-End Relevance** ≥ 0.65
- [ ] **Bloom Filter FPR** < 1%
- [ ] **Min Heap Speedup** > 3x vs full sort

### 14.2 User Experience Metrics

- [ ] Can process multi-photo upload
- [ ] Handles all edge cases gracefully
- [ ] Response time < 5 seconds per query
- [ ] User satisfaction ≥ 4/5

### 14.3 Course Requirements

- [ ] Implemented Bloom filter (data structures)
- [ ] Implemented TF-IDF (information retrieval)
- [ ] Trained ML model (YOLOv8)
- [ ] Deployed application (Hugging Face/Docker)
- [ ] Comprehensive evaluation
- [ ] Well-documented code
- [ ] Team collaboration via Git

---

## 15. Future Enhancements

If time permits, or for future work:

1. **User Accounts**: Save favorite recipes, dietary preferences
2. **Recipe Ratings**: Collect user feedback to improve ranking
3. **Nutritional Info**: Display calories, macros for recipes
4. **Shopping List**: Generate list of missing ingredients
5. **Meal Planning**: Suggest weekly meal plans
6. **Multi-Language**: Support recipes in multiple languages
7. **Voice Input**: "Alexa, what can I cook?"
8. **Mobile App**: Native iOS/Android application
9. **Inventory Tracking**: Track when items expire
10. **Social Features**: Share recipes with friends

---

## 16. Conclusion

Our SmartPantry project demonstrates an end-to-end ML system that:

1. **Solves a Real Problem**: Meal planning and food waste
2. **Integrates Course Concepts**: Data structures, IR, ML, deployment
3. **Handles Edge Cases**: Multi-photo, bad quality, partial matches
4. **Focuses on User Experience**: Clear error messages, editable results
5. **Is Realistically Scoped**: Achievable in one semester

We evolved from an overly ambitious initial proposal to a focused, practical system that we can actually build and learn from. Our design prioritizes:

- **Core functionality** over feature bloat
- **Clear evaluation** over hand-wavy promises
- **Student learning** over production perfection
- **Team collaboration** over individual heroics

By the end of the semester, we will have:
- A working ingredient detection model
- Multiple recipe retrieval methods compared
- Data structures implemented from class
- A deployed demo anyone can try
- Comprehensive evaluation and documentation

This project showcases both our technical skills and our ability to adapt, prioritize, and deliver a complete system.

---

**Document Version**: 1.0  
**Authors**: SmartPantry Team  
**Last Updated**: November 2, 2025

