# SmartPantry: Recipes from Your Fridge Cam

**Course Project**: Machine Learning Application Deployment

## Team Members

- Stacy Che
- Kexin Lyu
- Samantha Wang
- Zexi Wu (Allen)
- Tongrui Zhang (Neil)

## Project Overview

We are building a system that detects ingredients from fridge photos and recommends recipes. This project combines computer vision, information retrieval, and deployment techniques learned in class.

### The Problem

Many people struggle with meal planning and food waste. Our solution helps by:
- Detecting what ingredients you have from a photo
- Finding recipes that match your available ingredients
- Reducing decision fatigue and food waste

## Technical Approach

### 1. Image Recognition (Computer Vision)

**Fridge Ingredient Detection**
- CNN-based models for image classification
- YOLOv8 for object detection
- Dataset: 30 ingredient classes from Roboflow
- Training on 2,895 images

### 2. Recipe Matching (Information Retrieval)

We will implement and compare multiple approaches:

**Method A: TF-IDF with Cosine Similarity**
- Classic information retrieval baseline
- Fast and interpretable
- Good for ingredient matching

**Method B: Embedding-based Search**
- Sentence-BERT for semantic embeddings
- FAISS for approximate nearest neighbor search
- Better at understanding recipe context

**Method C: Hybrid Approach**
- Combine TF-IDF and embeddings
- Weight by ingredient overlap ratio
- Rank by multiple signals

### 3. Data Structures (Course Concepts)

We plan to incorporate:
- **Bloom Filters**: Quick ingredient availability checks
- **Hash Tables**: Fast recipe lookups
- **Min Heap**: Top-K recipe ranking

### 4. Deployment

- **Docker**: Containerized application
- **Hugging Face Spaces**: Demo deployment
- **Gradio**: Web interface

## Project Structure

```
ml-app-deployment-proj/
├── app/                    # Gradio interface
│   └── main.py
├── src/
│   ├── vision/            # CNN and YOLO detection
│   ├── nlp/               # TF-IDF, embeddings, retrieval
│   └── utils/             # Bloom filter, metrics
├── data/
│   ├── fridge_photos_sample/    # Sample images (in Git)
│   └── recipes_sample/          # Sample recipes (in Git)
├── models/                # Trained model weights
├── notebooks/             # Experiments and training
├── tests/                 # Unit tests
└── config.yaml           # Configuration
```

## Quick Start

### Setup

```bash
# Clone repository
git clone <repo-url>
cd ml-app-deployment-proj

# Install dependencies (uses uv for speed)
./setup.sh

# Or manual setup
pip install -e ".[dev]"
```

### Run the App

```bash
# Activate environment
source .venv/bin/activate

# Run Gradio interface
python app/main.py
```

Access at `http://localhost:7860`

## Development Workflow

### 1. Train Detection Model

```bash
# See notebooks/02_model_training.ipynb
jupyter notebook
```

Train YOLOv8 on fridge photos dataset.

### 2. Build Recipe Index

```bash
# Implement in src/nlp/
# - TF-IDF vectorization
# - Embedding generation
# - FAISS index building
```

### 3. Test Integration

```bash
# Run tests
pytest

# Check code quality
make format
make lint
```

## Datasets

### Included in Repository (for testing)

- **Fridge Photos Sample**: 30 images (3.1 MB)
- **Recipes Sample**: 100 recipes (872 KB)

### Full Datasets (download separately)

- **Fridge Photos**: 3,049 images from Roboflow
- **Recipes**: 58,783 recipes with ingredients

See `data/README.md` for download instructions.

## Implementation Plan

### Phase 1: Computer Vision (Weeks 1-4)

- Train YOLOv8 on ingredient detection
- Achieve reasonable accuracy on test set
- Integrate with preprocessing pipeline

### Phase 2: Recipe Retrieval (Weeks 5-8)

- Implement TF-IDF baseline
- Build embedding-based search
- Compare retrieval methods
- Add Bloom filter for quick checks

### Phase 3: Integration & UI (Weeks 9-10)

- Connect detection and retrieval
- Build Gradio interface
- End-to-end testing

### Phase 4: Deployment (Weeks 11-12)

- Docker containerization
- Deploy to Hugging Face Spaces
- Performance testing
- Documentation

## Technical Highlights

### What We'll Implement

**From Class**:
- Bloom filters for ingredient checks
- Hash-based lookups
- Approximate nearest neighbors (FAISS)
- TF-IDF and cosine similarity

**Additional ML**:
- CNN/YOLO for object detection
- Transformer embeddings (Sentence-BERT)
- Hybrid ranking systems

### Evaluation Metrics

**Detection**: Precision, Recall, F1, mAP
**Retrieval**: Recall@K, nDCG@K, MRR

## Commands

```bash
# Development
make run              # Start app
make test             # Run tests
make format           # Format code
make lint             # Check code quality

# Docker
make docker-build     # Build image
make docker-run       # Run container

# Training
jupyter notebook      # Open notebooks
```

## Repository Organization

We keep the full datasets locally (too large for Git):
- Full fridge photos: ~400 MB
- Full recipes: ~254 MB

Sample datasets are in Git for team testing and CI/CD.

## Questions or Issues

- Check `data/README.md` for dataset setup
- Check `CONTRIBUTING.md` for development guidelines
- Open an issue for bugs or questions

## Course Context

This project demonstrates practical ML deployment including:
- Training and evaluating ML models
- Information retrieval techniques
- Data structures for efficiency
- Containerization and deployment
- Team collaboration with Git

---

**Note**: This is a student project for our ML course. We focus on learning and implementing core concepts rather than production-scale deployment.
