# SmartPantry: Recipes from Your Fridge Cam

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Our team is building an intelligent meal planning application. We use computer vision to identify ingredients from fridge photos. The system then suggests personalized recipes based on available ingredients.

## Team Members

- **Stacy Che**
- **Kexin Lyu**
- **Samantha Wang**
- **Zexi Wu (Allen)**
- **Tongrui Zhang (Neil)**

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Technical Architecture](#technical-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development Roadmap](#development-roadmap)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Problem Statement

Modern food consumption has changed significantly. Fast food costs more and provides less nutrition than cooking with basic ingredients. Many people want to cook at home but face challenges:

- **Meal Planning**: Deciding what to cook takes time and mental energy
- **Ingredient Tracking**: Keeping track of available items is difficult
- **Food Waste**: Ingredients spoil before use
- **Recipe Matching**: Finding recipes for available ingredients is tedious

We identified these problems as opportunities for our project.

## Solution Overview

Our team developed SmartPantry to address these challenges. We combine machine learning and computer vision into a practical cooking assistant.

The workflow is straightforward:

1. **Capture**: User takes a photo of their fridge or pantry
2. **Detect**: Our AI identifies visible ingredients automatically
3. **Suggest**: System recommends recipes based on detected items
4. **Cook**: User follows recipes tailored to available ingredients

Our solution helps reduce food waste and simplifies meal planning. We aim to encourage healthier home-cooked meals.

## Technical Architecture

### Computer Vision Pipeline

- **Image Recognition**: Convolutional Neural Networks (CNNs) and transformer-based architectures for ingredient identification
- **Object Detection**: YOLO (You Only Look Once) for handling cluttered, real-world fridge images with multiple overlapping items
- **Multi-object Recognition**: Simultaneous detection of multiple ingredients in a single photo

### Recipe Matching System

- **Natural Language Processing**: Advanced NLP techniques for recipe analysis and matching
- **Embedding-based Search**: Semantic similarity search to find relevant recipes based on detected ingredients
- **Recipe Database**: Curated collection of recipes with ingredient mappings and nutritional information

### Application & Deployment

- **GUI Application**: User-friendly interface for photo capture, ingredient review, and recipe browsing
- **Containerization**: Docker containers for consistent deployment and scalability
- **Cloud Deployment**: AWS infrastructure for accessibility and performance
- **API Architecture**: RESTful APIs connecting frontend, ML models, and recipe database

## Features

### Current and Planned Features

Our implementation includes:

- **Ingredient Detection**
  - Real-time image analysis
  - Support for common pantry and fridge items
  - Confidence scoring for detected ingredients

- **Recipe Recommendations**
  - Match recipes to available ingredients
  - Filter by dietary preferences and restrictions
  - Sort by match percentage and cooking time

- **User Interface**
  - Simple photo upload and capture
  - Visual ingredient confirmation
  - Interactive recipe browsing
  - Shopping list generation for missing ingredients

- **Future Enhancements**
  - Expiration date tracking
  - Nutritional information display
  - Cooking difficulty ratings
  - User preference learning

## Technology Stack

### Machine Learning & AI
- **Computer Vision**: PyTorch/TensorFlow, OpenCV
- **Object Detection**: YOLO (Ultralytics YOLOv8/v5)
- **NLP**: Transformers, sentence-transformers, spaCy
- **Embeddings**: Vector databases (FAISS, Pinecone, or ChromaDB)

### Backend
- **Framework**: Python (Flask/FastAPI)
- **Database**: PostgreSQL or MongoDB
- **API**: RESTful architecture

### Frontend
- **Framework**: React.js or Streamlit (depending on complexity requirements)
- **UI Components**: Material-UI or Tailwind CSS

### DevOps & Deployment
- **Containerization**: Docker
- **Orchestration**: Docker Compose or Kubernetes
- **Cloud Platform**: AWS (EC2, S3, Lambda, SageMaker)
- **CI/CD**: GitHub Actions

## Project Structure

```
ml-app-deployment-proj/
├── data/                      # Dataset and recipe database
│   ├── raw/                   # Raw ingredient images
│   ├── processed/             # Preprocessed data
│   └── recipes/               # Recipe database
├── models/                    # ML models
│   ├── ingredient_detection/  # YOLO models
│   ├── recipe_matching/       # NLP models
│   └── checkpoints/           # Model weights
├── src/                       # Source code
│   ├── computer_vision/       # CV pipeline
│   ├── nlp/                   # Recipe matching
│   ├── api/                   # Backend API
│   └── frontend/              # GUI application
├── notebooks/                 # Jupyter notebooks for experimentation
├── tests/                     # Unit and integration tests
├── docker/                    # Docker configurations
├── deployment/                # Deployment scripts and configs
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Multi-container setup
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (auto-installed by setup script)
- Docker and Docker Compose (optional, for containerized deployment)
- AWS CLI (optional, for deployment)
- CUDA-capable GPU (recommended for model training)

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ml-app-deployment-proj

# Run automated setup (installs uv and dependencies)
./setup.sh
```

### Manual Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application
python app/main.py
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or use Make
make docker-build
make docker-run

# Access the application at http://localhost:7860
```

## Usage

### Basic Workflow

1. **Launch the Application**
   ```bash
   python src/app.py
   ```

2. **Upload a Photo**
   - Click "Upload Photo" or use the camera feature
   - Select a clear image of your fridge/pantry

3. **Review Detected Ingredients**
   - Verify automatically detected ingredients
   - Add or remove items manually if needed

4. **Browse Recipes**
   - View recommended recipes sorted by match percentage
   - Filter by dietary preferences, cooking time, or difficulty
   - Select a recipe to view detailed instructions

5. **Start Cooking!**
   - Follow step-by-step instructions
   - Generate shopping list for missing ingredients

## Development Roadmap

### Phase 1: Core ML Pipeline (Target: Week 4)
- [x] Project setup and architecture design
- [x] Complete project skeleton with modular structure
- [x] All core modules with placeholder functions
- [ ] Data collection and preprocessing
- [ ] YOLO model training for ingredient detection
- [ ] Initial model evaluation and iteration

### Phase 2: Recipe Matching System (Target: Week 8)
- [ ] Recipe database creation and curation
- [ ] NLP pipeline for recipe embeddings
- [ ] Similarity search implementation
- [ ] Integration testing

### Phase 3: Application Development (Target: Week 10)
- [ ] Backend API development
- [ ] Frontend GUI implementation
- [ ] User authentication and preferences
- [ ] End-to-end testing

### Phase 4: Deployment (Target: Week 12)
- [ ] Docker containerization
- [ ] AWS infrastructure setup
- [ ] CI/CD pipeline configuration
- [ ] Performance optimization and monitoring

### Our MVP Goals

**High Confidence Areas**:
- YOLO-based ingredient identification with reasonable accuracy
- Functional data and NLP pipeline for recipe retrieval
- Working GUI application with core features

**Areas of Concern**:
- AWS deployment coordination among team members
- Real-time performance optimization
- Handling edge cases and diverse ingredient types

## Deployment

### AWS Architecture

- **EC2 Instances**: Application hosting
- **S3 Buckets**: Image storage and model artifacts
- **Lambda Functions**: Serverless API endpoints (optional)
- **SageMaker**: Model training and deployment (optional)
- **RDS/DynamoDB**: Database services

### Deployment Considerations

We face coordination challenges with multiple team members using shared AWS instances. Our approach includes:

- Infrastructure as code (Terraform/CloudFormation)
- Proper access control and resource tagging
- Clear deployment protocols and documentation

### Monitoring

We plan to implement:

- CloudWatch for logging and metrics
- Application performance monitoring
- Cost tracking and optimization

## Contributing

This is our team project for a machine learning course. Team members should follow this workflow:

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and commit: `git commit -m "Add feature description"`
3. Push to the branch: `git push origin feature/your-feature-name`
4. Create a Pull Request for team review

### Coding Standards

We follow these standards:

- PEP 8 for Python code
- Unit tests for new features
- Documentation for functions and classes
- README updates for significant changes

## License

We use the MIT License for this project. See the [LICENSE](LICENSE) file for details.

## Contact

Team members can reach out to each other directly. External questions can be submitted through repository issues.

---

**Note**: We developed this project as our machine learning course final project. It demonstrates practical applications of computer vision, NLP, and cloud deployment technologies.

