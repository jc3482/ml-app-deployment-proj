# SmartPantry Project Structure

This document provides a detailed overview of the project organization and architecture.

## Directory Structure

```
ml-app-deployment-proj/
│
├── app/                          # Gradio Application
│   ├── __init__.py               # App package initialization
│   ├── main.py                   # Main Gradio interface
│   ├── static/                   # Static assets (CSS, JS)
│   └── templates/                # HTML templates (if needed)
│
├── data/                         # Data Storage
│   ├── raw/                      # Raw datasets
│   │   ├── food-101/             # Food-101 dataset
│   │   └── recipe1m/             # Recipe1M+ dataset
│   ├── processed/                # Processed data
│   ├── recipes/                  # Recipe database
│   ├── fridge_photos/            # Sample fridge images
│   └── embeddings/               # Cached embeddings and FAISS indices
│
├── deployment/                   # Deployment Configurations
│   ├── huggingface/              # Hugging Face Spaces deployment
│   │   ├── README.md             # HF deployment guide
│   │   └── app.py                # HF-specific app wrapper
│   └── aws/                      # AWS deployment
│       ├── README.md             # AWS deployment guide
│       └── ecs-task-definition.json
│
├── docs/                         # Documentation
│   └── PROJECT_STRUCTURE.md      # This file
│
├── logs/                         # Application logs
│   └── .gitkeep
│
├── models/                       # Model Storage
│   ├── yolo/                     # YOLOv8 weights
│   ├── embeddings/               # Embedding model caches
│   └── checkpoints/              # Training checkpoints
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb # Data analysis
│   ├── 02_model_training.ipynb   # Model training
│   ├── 03_recipe_retrieval.ipynb # Retrieval experiments
│   └── README.md                 # Notebooks documentation
│
├── src/                          # Source Code
│   ├── __init__.py               # Package initialization
│   │
│   ├── vision/                   # Computer Vision Module
│   │   ├── __init__.py
│   │   ├── detector.py           # YOLOv8 ingredient detection
│   │   └── preprocessor.py       # Image preprocessing
│   │
│   ├── nlp/                      # NLP Module
│   │   ├── __init__.py
│   │   ├── embedder.py           # Sentence-BERT/CLIP embeddings
│   │   └── retriever.py          # FAISS-based recipe retrieval
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── helpers.py            # Helper functions
│       ├── metrics.py            # Evaluation metrics
│       └── clustering.py         # Recipe clustering
│
├── tests/                        # Tests
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── test_detector.py          # Detector tests
│   └── test_retriever.py         # Retriever tests
│
├── .gitignore                    # Git ignore rules
├── .gitattributes                # Git LFS configuration
├── config.yaml                   # Configuration file
├── CONTRIBUTING.md               # Contribution guidelines
├── docker-compose.yml            # Docker Compose config
├── Dockerfile                    # Docker image definition
├── LICENSE                       # MIT License
├── Makefile                      # Development commands
├── pytest.ini                    # Pytest configuration
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── setup.sh                      # Setup script
```

## Module Descriptions

### 1. Computer Vision Module (`src/vision/`)

**Purpose**: Detect ingredients from fridge/pantry images using YOLOv8.

#### `detector.py`
- **Class**: `IngredientDetector`
- **Responsibilities**:
  - Load YOLOv8 model
  - Run inference on images
  - Apply NMS and confidence filtering
  - Generate visualizations
  - Batch processing

#### `preprocessor.py`
- **Class**: `ImagePreprocessor`
- **Responsibilities**:
  - Image loading and conversion
  - Resizing and normalization
  - Ingredient name normalization (fuzzy matching)
  - Duplicate removal
  - Batch aggregation

### 2. NLP Module (`src/nlp/`)

**Purpose**: Generate embeddings and retrieve relevant recipes.

#### `embedder.py`
- **Class**: `IngredientEmbedder`
- **Responsibilities**:
  - Load Sentence-BERT or CLIP models
  - Generate embeddings for ingredients
  - Generate embeddings for recipes
  - Batch processing
  - Embedding caching

#### `retriever.py`
- **Class**: `RecipeRetriever`
- **Responsibilities**:
  - Load recipe database
  - Build/load FAISS index
  - Perform similarity search
  - Calculate ingredient overlap
  - Rank and filter recipes
  - Apply user preferences

### 3. Utilities Module (`src/utils/`)

**Purpose**: Shared utilities and helper functions.

#### `helpers.py`
- Configuration loading
- Logging setup
- Text formatting
- Device detection
- Common utilities

#### `metrics.py`
- **Classes**: `DetectionMetrics`, `RetrievalMetrics`
- **Responsibilities**:
  - Calculate detection metrics (precision, recall, mAP)
  - Calculate retrieval metrics (Recall@K, nDCG@K, MRR)
  - Batch evaluation

#### `clustering.py`
- **Class**: `RecipeClustering`
- **Responsibilities**:
  - Cluster recipes by cuisine, difficulty, etc.
  - Support k-means, hierarchical, DBSCAN
  - Generate cluster labels
  - Analyze cluster characteristics

### 4. Application Module (`app/`)

**Purpose**: User-facing Gradio interface.

#### `main.py`
- **Class**: `SmartPantryApp`
- **Responsibilities**:
  - Initialize all components
  - Handle image uploads
  - Process detections
  - Get recipe recommendations
  - Create Gradio UI
  - Launch application

## Data Flow

```
User uploads image
        ↓
[ImagePreprocessor]
    Resize, normalize
        ↓
[IngredientDetector]
    YOLOv8 detection
        ↓
[ImagePreprocessor]
    Normalize names, remove duplicates
        ↓
Ingredient list
        ↓
[IngredientEmbedder]
    Generate embeddings
        ↓
[RecipeRetriever]
    FAISS similarity search
    Calculate ingredient overlap
    Rank recipes
        ↓
[RecipeClustering] (optional)
    Group by cuisine/difficulty
        ↓
Recipe recommendations
        ↓
Display in Gradio UI
```

## Configuration

The `config.yaml` file contains all configurable parameters:

- **Project metadata**: Name, version
- **Paths**: Data, models, logs
- **Detection settings**: Model, thresholds, device
- **Embedding settings**: Model, batch size
- **Retrieval settings**: FAISS index, ranking weights
- **Clustering settings**: Method, features
- **Gradio settings**: Server, UI options
- **Deployment settings**: HF, AWS configs

## Development Workflow

1. **Feature Development**
   - Create feature branch
   - Implement in appropriate module
   - Write unit tests
   - Update documentation

2. **Testing**
   - Run unit tests: `pytest tests/unit/`
   - Run integration tests: `pytest tests/integration/`
   - Check coverage: `pytest --cov=src`

3. **Code Quality**
   - Format: `make format`
   - Lint: `make lint`
   - Type check: `make type-check`

4. **Integration**
   - Create pull request
   - Code review
   - Merge to main

## Deployment Strategies

### 1. Hugging Face Spaces
- Best for: Demo, prototyping
- Cost: Free tier available
- Hardware: CPU, GPU options
- See: `deployment/huggingface/README.md`

### 2. AWS
- Best for: Production, scalability
- Cost: Pay-as-you-go
- Hardware: EC2, ECS, Lambda
- See: `deployment/aws/README.md`

### 3. Docker
- Best for: Local development, portability
- Use: `make docker-build && make docker-run`

## Testing Strategy

### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Fast execution
- Location: `tests/unit/`

### Integration Tests
- Test component interactions
- Use real models (or lightweight versions)
- Slower execution
- Location: `tests/integration/`

### Performance Tests
- Test inference speed
- Test memory usage
- Test scalability
- Location: `tests/performance/`

## Performance Considerations

### Optimization Techniques
1. **Model Optimization**
   - Use smaller YOLO models (nano/small) for CPU
   - Quantization for faster inference
   - ONNX export for production

2. **Caching**
   - Cache embeddings
   - Cache FAISS index
   - Cache recipe metadata

3. **Batch Processing**
   - Process multiple images together
   - Batch embedding generation
   - Vectorized operations

4. **Lazy Loading**
   - Load models only when needed
   - Defer heavy computations

## Monitoring and Logging

### Logs
- Location: `logs/`
- Format: Configurable in `config.yaml`
- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Metrics
- Detection performance (mAP, precision, recall)
- Retrieval performance (Recall@K, nDCG@K)
- System metrics (latency, memory, throughput)

### Tools
- CloudWatch (AWS)
- Weights & Biases (experiment tracking)
- Gradio analytics

## Future Enhancements

1. **Features**
   - Expiration date tracking
   - Nutritional information
   - Meal planning calendar
   - Shopping list generation

2. **Technical**
   - Multi-user support
   - Real-time collaboration
   - Mobile app
   - Voice interface

3. **ML Improvements**
   - Fine-tune on custom fridge dataset
   - Improve ingredient recognition
   - Better recipe ranking
   - Personalized recommendations

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Gradio Documentation](https://www.gradio.app/docs/)

