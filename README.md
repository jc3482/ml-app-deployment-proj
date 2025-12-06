---
title: SmartPantry
emoji: 🥗
colorFrom: yellow
colorTo: green
sdk: docker
sdk_version: latest
app_file: Dockerfile
pinned: false
license: mit
---

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
- Modern React single-page application
- Elegant beige/tan color scheme with serif fonts
- Collapsible sections for better organization
- Responsive layout with full-width design
- Interactive recipe cards with expandable instructions
- Individual ingredient deletion with × buttons

### 4. Deployment

- **React Frontend**: Modern single-page application with Vite
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **Docker**: Fully containerized application (frontend + backend in one container)
- **Hugging Face Spaces**: Deployed and accessible via public URL
- **Health Checks**: Docker health check integration

## Project Structure

```
ml-app-deployment-proj/
├── app/                           # Application code
│   └── api_extended.py            # FastAPI REST API (with frontend serving)
├── frontend/                      # React frontend
│   ├── src/                       # React source code
│   │   ├── components/            # React components
│   │   ├── services/              # API client
│   │   └── App.jsx                # Main app
│   ├── package.json               # Frontend dependencies
│   └── vite.config.js             # Vite configuration
├── src/
│   ├── backend/
│   │   └── recipe_recommender.py  # Unified ML pipeline
│   ├── vision/                    # YOLOv8 detection
│   ├── recipes/                   # Recipe matching
│   └── utils/                     # Utility functions
├── data/
│   ├── canonical_vocab.json       # Ingredient normalization
│   ├── normalized_recipes.pkl     # Processed recipe cache
│   ├── history/                   # User data (history, pantry)
│   └── fridge_photos/             # Training images
├── models/
│   └── yolo/                      # YOLOv8 model weights
├── recipe_matching_system/        # Advanced matching system
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose config
└── requirements.txt               # Python dependencies
```

## Quick Start

### Local Development

**Prerequisites:**
- Python 3.10+
- Node.js 18+
- npm or yarn

**Step 1: Install Backend Dependencies**
```bash
# Clone repository
git clone <repo-url>
cd ml-app-deployment-proj

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# OR: .venv\Scripts\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt
```

**Step 2: Install Frontend Dependencies**
```bash
cd frontend
npm install
cd ..
```

**Step 3: Run the Application**

Terminal 1 - Start FastAPI Backend:
```bash
uvicorn app.api_extended:app --reload --port 8001
```

Terminal 2 - Start React Frontend:
```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:3000`

### Docker Deployment

**Local Docker (Port 8001):**
```bash
# Build Docker image
docker build -t smartpantry:latest .

# Run container (local development - uses port 8001)
docker run -p 8001:8001 \
  -v $(PWD)/data:/app/data \
  -v $(PWD)/models:/app/models \
  smartpantry:latest

# Or use Docker Compose
docker-compose up -d --build
```

Access at `http://localhost:8001`

**Note**: When deployed to Hugging Face Spaces, the container automatically uses the `PORT` environment variable (typically 7860), not 8001. The Dockerfile handles this automatically.

### Hugging Face Spaces Deployment

**Live Demo:** [https://huggingface.co/spaces/qqmian0820/smartpantryy](https://huggingface.co/spaces/qqmian0820/smartpantryy)

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### API Access

The application provides a REST API via FastAPI:

**Local Development (Port 8001):**
```bash
# Start API server
uvicorn app.api_extended:app --host 0.0.0.0 --port 8001

# Access API documentation
# Swagger UI: http://localhost:8001/docs
# ReDoc: http://localhost:8001/redoc
```

**Hugging Face Spaces:**
- Live API: [https://qqmian0820-smartpantryy.hf.space/docs](https://qqmian0820-smartpantryy.hf.space/docs)
- Health check: [https://qqmian0820-smartpantryy.hf.space/health](https://qqmian0820-smartpantryy.hf.space/health)

**Available Endpoints:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/detect` - Detect ingredients from image
- `POST /api/recommend` - Get recipe recommendations
- `GET /api/pantry/list` - Get pantry list
- `POST /api/pantry/add` - Add ingredients to pantry
- `DELETE /api/pantry/remove/{ingredient}` - Remove ingredient
- `GET /api/history` - Get history
- `POST /api/history/clear` - Clear history

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
# Run tests (if available)
make test

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
- Set-based scoring algorithm with percentage-based ranking (0-100%)
- Supports 13,244+ recipes
- Match score represents percentage of recipe ingredients that match user's available ingredients

### Phase 3: Integration & UI (Completed)

- Unified RecipeRecommender pipeline connecting detection and retrieval
- React frontend with modern UI:
  - Image upload and ingredient detection
  - Recipe recommendations with detailed information
  - Pantry List for manual ingredient input
  - Dietary filters (vegan, vegetarian, dairy-free, gluten-free)
  - History tracking and management
  - Individual ingredient deletion
- Elegant UI design with beige/tan theme
- End-to-end testing completed

### Phase 4: Deployment (Completed)

- Docker containerization with Dockerfile and docker-compose.yml
- FastAPI REST API with comprehensive endpoints (`/api/*`)
- Frontend and backend in single Docker container
- Hugging Face Spaces deployment with optimized health checks
- Health checks optimized for fast startup (non-blocking during model initialization)
- Automatic port configuration: local development uses 8001, deployed Spaces use `PORT` env var (typically 7860)
- Documentation (README, DEPLOYMENT.md, QUICK_DEPLOY.md) 

## Technical Highlights

### What We've Implemented

**Core Technologies:**
- YOLOv8: Real-time object detection for ingredient recognition
- Ingredient Overlap Scoring: Set-based matching algorithm with percentage-based ranking (0-100%)
- Canonical Ingredient Normalization: Consistent ingredient matching across variations
- Recipe Recommender Pipeline: Unified ML pipeline integrating detection and retrieval 

**Application Features:**
- React Frontend: Modern, responsive single-page application with relative API paths
- FastAPI REST API: Comprehensive REST endpoints with `/api` prefix
- Docker Deployment: Containerized application (frontend + backend in one container)
- Hugging Face Spaces: Public deployment with automatic scaling and optimized health checks
- Persistent Storage: JSON-based storage for user history and pantry lists
- Dietary Filtering: Support for multiple dietary restrictions
- Smart Health Checks: Fast, non-blocking health endpoint for reliable deployment status

**Data Structures & Algorithms:**
- Hash-based recipe lookups (dictionary-based recipe storage)
- Top-K ranking for recipe recommendations
- Set intersection for ingredient overlap calculation

## Commands

```bash
# Development (Local - Port 8001)
make run              # Start FastAPI backend (port 8001 for local dev)
make test-api         # Test API endpoints
make test             # Run tests
make format           # Format code
make lint             # Check code quality

# Docker
make docker-build     # Build image
make docker-run       # Run container (local - port 8001)

# Frontend (separate terminal)
cd frontend
npm run dev           # Start React dev server (port 3000)
npm run build         # Build for production

# Training
jupyter notebook      # Open notebooks
```

**Note**: All port 8001 references are for local development. Hugging Face Spaces deployment automatically uses the `PORT` environment variable (typically 7860).

## Repository Organization

We keep the full datasets locally (too large for Git):
- Full fridge photos: ~400 MB
- Full recipes: ~254 MB

Sample datasets are in Git for team testing and CI/CD.

## Questions or Issues

- Check `data/README.md` for dataset setup
- Check `DEPLOYMENT.md` or `QUICK_DEPLOY.md` for Hugging Face deployment instructions
- Check `frontend/README.md` for frontend development
- **Troubleshooting Hugging Face Spaces:**
  - If Space shows "Starting" for a long time, check the Logs tab - the app may already be accessible
  - Health check endpoint (`/health`) returns immediately, even during model initialization
  - Frontend automatically adapts to any port via relative API paths
- Open an issue for bugs or questions

## Course Context

This project demonstrates practical ML deployment including:
- Training and evaluating ML models
- Information retrieval techniques
- Data structures for efficiency
- Containerization and deployment
- Team collaboration with Git

---
