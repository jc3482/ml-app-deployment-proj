# SmartPantry: Recipes from Your Fridge Cam

**Course Project**: Machine Learning Application Deployment

## Team Members

- Stacy Che
- Kexin Lyu
- Samantha Wang
- Zexi Wu (Allen)
- Tongrui Zhang (Neil)

## Project Overview

SmartPantry is a complete ML application that detects ingredients from fridge photos and recommends personalized recipes. The system combines computer vision (YOLOv8), information retrieval, and modern deployment practices.

### The Problem

Many people struggle with meal planning and food waste. Our solution helps by:
- Detecting what ingredients you have from a photo
- Finding recipes that match your available ingredients
- Reducing decision fatigue and food waste

## Technical Approach

### 1. Image Recognition (Computer Vision)

**Fridge Ingredient Detection**
- YOLOv8 for object detection
- Trained on 30 ingredient classes from Roboflow dataset
- Canonical ingredient normalization for consistent matching
- Integrated preprocessing pipeline

### 2. Recipe Matching (Information Retrieval)

**Implemented Method: Ingredient Overlap Scoring**
- Set-based ingredient matching
- Counts overlapping ingredients between detected items and recipe requirements
- Simple and fast scoring algorithm
- Supports 13,244+ recipes from recipe dataset

### 3. Application Features

**Core Features:**
- Ingredient Detection: Upload image and detect ingredients
- Recipe Recommendations: Get personalized recipe suggestions based on detected ingredients
- Pantry List: Manually add ingredients not detected in images
- Dietary Filters: Filter recipes by dietary restrictions (vegan, vegetarian, dairy-free, gluten-free)
- History: View past detections and recommendations (last 20 records)
- Persistent Storage: History and pantry data saved to JSON files

**UI/UX:**
- Elegant beige/tan color scheme with serif fonts
- Collapsible sections for better organization
- Responsive layout with full-width design
- Interactive recipe cards with expandable instructions

### 4. Deployment

- Docker: Fully containerized application with Docker Compose
- FastAPI REST API: RESTful endpoints for programmatic access
- Gradio Web Interface: User-friendly web UI
- Health Checks: Docker health check integration

## Project Structure

```
ml-app-deployment-proj/
├── app/                           # Application code
│   ├── main.py                    # Gradio web interface
│   ├── api.py                     # FastAPI REST API
│   └── static/
│       └── style.css              # Custom UI styling
├── src/
│   ├── backend/
│   │   └── recipe_recommender.py  # Unified ML pipeline
│   ├── vision/                    # YOLOv8 detection
│   ├── recipes/                     # Recipe matching
│   └── utils/                     # Utility functions
├── data/
│   ├── canonical_vocab.json       # Ingredient normalization
│   ├── recipes/                   # Recipe dataset
│   ├── history/                   # User data (history, pantry)
│   └── fridge_photos/             # Training images
├── models/
│   └── yolo/                      # YOLOv8 model weights
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose config
└── config.yaml                    # Configuration
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

**Option 1: Direct Python (Development)**
```bash
# Activate environment
source .venv/bin/activate

# Run Gradio interface
python -m app.main
```

**Option 2: Docker (Recommended for Deployment)**
```bash
# Build and start container
docker-compose up -d --build

# View logs
docker-compose logs -f smartpantry

# Stop container
docker-compose down
```

Access at `http://localhost:7860`

For detailed Docker deployment instructions, see [DOCKER.md](DOCKER.md)

### API Access

The application also provides a REST API via FastAPI:

```bash
# Start API server (if not using Docker)
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

**Available Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /detect` - Detect ingredients from image
- `POST /recommend` - Get recipe recommendations

## Features

### Core Functionality

1. **Ingredient Detection**
   - Upload a photo of your fridge/pantry
   - Automatic detection of ingredients using YOLOv8
   - Display detected ingredients in a collapsible section

2. **Recipe Recommendations**
   - Get personalized recipe suggestions based on detected ingredients
   - Configurable number of recommendations (1-20)
   - Recipe cards with:
     - Title and match score
     - Required ingredients
     - Step-by-step instructions (collapsible for long recipes)

3. **Pantry List**
   - Manually add ingredients not detected in images
   - Ingredients are combined with detected ones for recommendations
   - Persistent storage across sessions

4. **Dietary Filters**
   - Filter recipes by dietary restrictions:
     - Vegan
     - Vegetarian
     - Dairy-free
     - Gluten-free

5. **History**
   - View past ingredient detections and recommendations
   - Last 20 records stored
   - Clear history functionality

### User Interface

- Elegant beige/tan color scheme with Georgia serif fonts
- Minimalist design with card-based layout
- Collapsible sections for better organization
- Full-width responsive layout
- Interactive hover effects

## Development Workflow

### Running the Application

```bash
# Option 1: Direct Python
python -m app.main

# Option 2: Docker (Recommended)
docker-compose up -d --build
```

### Testing

```bash
# Run tests
pytest

# Test API endpoints
python test_api.py

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

## Implementation Status

### Phase 1: Computer Vision (Completed)

- Trained YOLOv8 on ingredient detection
- Integrated preprocessing pipeline with canonical normalization
- Real-time ingredient detection from images

### Phase 2: Recipe Retrieval (Completed)

- Implemented ingredient overlap-based recipe matching
- Set-based scoring algorithm
- Supports 13,244+ recipes
- Ingredient overlap ranking

### Phase 3: Integration & UI (Completed)

- Unified RecipeRecommender pipeline connecting detection and retrieval
- Complete Gradio web interface with:
  - Image upload and ingredient detection
  - Recipe recommendations with detailed information
  - Pantry List for manual ingredient input
  - Dietary filters (vegan, vegetarian, dairy-free, gluten-free)
  - History tracking and management
- Elegant UI design with beige/tan theme
- End-to-end testing completed

### Phase 4: Deployment (Completed)

- Docker containerization with Dockerfile and docker-compose.yml
- FastAPI REST API with `/detect`, `/recommend`, `/health` endpoints
- Health checks and logging
- Documentation (README, DOCKER.md)

## Technical Highlights

### What We've Implemented

**Core Technologies:**
- YOLOv8: Real-time object detection for ingredient recognition
- Ingredient Overlap Scoring: Set-based matching algorithm for recipe ranking
- Canonical Ingredient Normalization: Consistent ingredient matching across variations
- Recipe Recommender Pipeline: Unified ML pipeline integrating detection and retrieval

**Application Features:**
- Gradio Web Interface: Beautiful, interactive web UI
- FastAPI REST API: Programmatic access via REST endpoints
- Docker Deployment: Containerized application for easy deployment
- Persistent Storage: JSON-based storage for user history and pantry lists
- Dietary Filtering: Support for multiple dietary restrictions

**Data Structures & Algorithms:**
- Hash-based recipe lookups (dictionary-based recipe storage)
- Top-K ranking for recipe recommendations
- Set intersection for ingredient overlap calculation

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
- Check `DOCKER.md` for deployment instructions
- Open an issue for bugs or questions

## Course Context

This project demonstrates practical ML deployment including:
- Training and evaluating ML models
- Information retrieval techniques
- Data structures for efficiency
- Containerization and deployment
- Team collaboration with Git

---
