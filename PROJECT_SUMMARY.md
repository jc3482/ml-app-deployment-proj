# SmartPantry - Project Summary

## 🎯 Overview

**SmartPantry: Recipes from Your Fridge Cam** is a full-stack AI application that uses computer vision and natural language processing to detect ingredients from fridge photos and recommend personalized recipes.

### Team Members
- Stacy Che
- Kexin Lyu  
- Samantha Wang
- Zexi Wu (Allen)
- Tongrui Zhang (Neil)

## 🏗️ Architecture

### Technology Stack

**Computer Vision**
- YOLOv8 (PyTorch) for ingredient detection
- OpenCV for image preprocessing
- Food-101 dataset for training

**NLP & Retrieval**
- Sentence-BERT / CLIP for embeddings
- FAISS for approximate nearest neighbor search
- Recipe1M+ dataset for recipe database

**Backend**
- Python 3.10+
- Modular architecture with separate vision, NLP, and utils modules

**Frontend**
- Gradio for interactive web interface
- Real-time ingredient detection
- Recipe filtering and recommendations

**Deployment**
- Docker containerization
- Hugging Face Spaces (demo)
- AWS (production-ready)

## 📁 Project Structure

```
ml-app-deployment-proj/
├── app/                    # Gradio application
│   ├── main.py            # Main interface
│   ├── static/            # Static assets
│   └── templates/         # HTML templates
│
├── src/                   # Core modules
│   ├── vision/           # YOLOv8 detection
│   │   ├── detector.py
│   │   └── preprocessor.py
│   ├── nlp/              # Embeddings & retrieval
│   │   ├── embedder.py
│   │   └── retriever.py
│   └── utils/            # Shared utilities
│       ├── helpers.py
│       ├── metrics.py
│       └── clustering.py
│
├── data/                  # Datasets
│   ├── raw/              # Food-101, Recipe1M+
│   ├── processed/        # Processed data
│   ├── recipes/          # Recipe database
│   └── embeddings/       # FAISS indices
│
├── models/               # Model weights
│   ├── yolo/            # YOLOv8 checkpoints
│   └── embeddings/      # Embedding models
│
├── deployment/          # Deployment configs
│   ├── huggingface/    # HF Spaces
│   └── aws/            # AWS EC2/ECS
│
├── notebooks/          # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_recipe_retrieval.ipynb
│
├── tests/             # Test suite
│   ├── unit/
│   └── integration/
│
└── docs/             # Documentation
```

## 🚀 Key Features

### Implemented (Skeleton)

✅ **YOLOv8 Ingredient Detection**
- Multi-object detection in fridge images
- Confidence scoring and NMS
- Visualization of detections
- Batch processing support

✅ **Ingredient Preprocessing**
- Fuzzy name normalization
- Duplicate removal
- Multi-image aggregation
- Blacklist filtering

✅ **Recipe Embeddings**
- Sentence-BERT integration
- CLIP support for multimodal
- Batch embedding generation
- Caching mechanism

✅ **FAISS-based Retrieval**
- Approximate nearest neighbor search
- IVF index support for scalability
- Hybrid ranking (semantic + overlap)
- Filter by cuisine, difficulty, time

✅ **Recipe Clustering**
- K-means clustering
- Hierarchical clustering
- Group by cuisine and difficulty
- Cluster analysis tools

✅ **Gradio Interface**
- Image upload
- Real-time detection
- Recipe recommendations
- Interactive filters

✅ **Evaluation Metrics**
- Detection: Precision, Recall, F1, mAP
- Retrieval: Recall@K, nDCG@K, MRR

### To Be Implemented

⚠️ **Model Training**
- Fine-tune YOLOv8 on Food-101
- Train on custom fridge dataset
- Model optimization and export

⚠️ **Recipe Database**
- Load Recipe1M+ dataset
- Parse and clean recipes
- Generate embeddings
- Build FAISS index

⚠️ **Full Integration**
- Connect all components
- End-to-end testing
- Performance optimization

⚠️ **Deployment**
- Deploy to Hugging Face Spaces
- AWS infrastructure setup
- CI/CD pipeline

## 🔧 Development Workflow

### Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Run application
make run

# 3. Run tests
make test

# 4. Format code
make format
```

### Development Commands

```bash
make help              # Show all commands
make install          # Install dependencies
make test             # Run tests
make test-cov         # Tests with coverage
make lint             # Check code quality
make format           # Format code
make clean            # Clean temp files
make run              # Start app
make docker-build     # Build Docker image
make docker-run       # Run in Docker
make deploy-hf        # Prepare HF deployment
```

## 📊 Configuration

All settings are in `config.yaml`:

```yaml
# Detection
detection:
  model_name: yolov8m
  confidence_threshold: 0.25
  device: cuda

# Embeddings
embeddings:
  model_type: sentence-bert
  model_name: all-MiniLM-L6-v2

# Retrieval
retrieval:
  index_type: IVFFlat
  top_k: 50
  min_ingredient_match: 0.3

# Gradio
gradio:
  server_port: 7860
  share: false
```

## 🧪 Testing

### Test Coverage

```
src/
├── vision/         # TODO: Add tests
├── nlp/           # Basic tests included
└── utils/         # Basic tests included
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific module
pytest tests/test_detector.py

# Integration tests only
pytest -m integration
```

## 📈 Evaluation Metrics

### Detection Metrics
- **Precision**: Accuracy of detections
- **Recall**: Coverage of ingredients
- **F1 Score**: Harmonic mean
- **mAP**: Mean average precision at different IoU thresholds

### Retrieval Metrics
- **Recall@K**: Proportion of relevant recipes in top-K
- **nDCG@K**: Ranking quality
- **MRR**: Mean reciprocal rank
- **MAP**: Mean average precision

## 🚢 Deployment Options

### 1. Hugging Face Spaces (Demo)
```bash
make deploy-hf
git push hf main
```
- **Cost**: Free tier available
- **Hardware**: CPU/GPU options
- **Best for**: Demos, prototypes

### 2. AWS (Production)
```bash
# See deployment/aws/README.md
docker build -t smartpantry .
# Push to ECR and deploy
```
- **Cost**: Pay-as-you-go
- **Hardware**: EC2, ECS, Lambda
- **Best for**: Production, scale

### 3. Local Docker
```bash
make docker-build
make docker-run
```
- **Cost**: Free
- **Hardware**: Your machine
- **Best for**: Development

## 📚 Documentation

- **README.md**: Main project documentation
- **QUICKSTART.md**: 5-minute setup guide
- **CONTRIBUTING.md**: Development guidelines
- **PROJECT_STRUCTURE.md**: Architecture details
- **deployment/*/README.md**: Deployment guides

## 🔮 Future Enhancements

### Short-term (MVP)
1. Complete model training on Food-101
2. Load Recipe1M+ database
3. Build and test full pipeline
4. Deploy demo to HF Spaces

### Medium-term
1. Fine-tune on custom fridge dataset
2. Improve ingredient recognition
3. Add user preferences and history
4. Implement shopping list generation

### Long-term
1. Expiration date tracking
2. Nutritional information
3. Meal planning calendar
4. Mobile app
5. Multi-user support
6. Voice interface

## 📝 Development Phases

### Phase 1: Core ML Pipeline ✅
- [x] Project skeleton
- [ ] YOLOv8 training
- [ ] Recipe embeddings
- [ ] FAISS index

### Phase 2: Integration (Current)
- [ ] End-to-end pipeline
- [ ] Full testing
- [ ] UI refinement
- [ ] Performance optimization

### Phase 3: Deployment
- [ ] HF Spaces deployment
- [ ] AWS infrastructure
- [ ] CI/CD setup
- [ ] Monitoring

### Phase 4: Enhancement
- [ ] User feedback
- [ ] Model improvements
- [ ] New features
- [ ] Scale optimization

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Computer vision with YOLOv8
- ✅ NLP with transformers
- ✅ Vector search with FAISS
- ✅ Full-stack ML application
- ✅ Clean code architecture
- ✅ Testing and evaluation
- ✅ Deployment strategies
- ✅ Team collaboration

## 🤝 Team Collaboration

### Coordination
- Version control with Git
- Feature branches
- Pull request reviews
- Clear documentation

### Communication
- Regular team meetings
- Issue tracking
- Code reviews
- Shared progress updates

## 📄 License

MIT License - See LICENSE file

## 🔗 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gradio](https://www.gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)

## 🆘 Support

For questions or issues:
1. Check documentation in `docs/`
2. Review examples in `notebooks/`
3. Open GitHub issue
4. Contact team members

---

**Status**: Skeleton Complete ✅ | Ready for Development 🚀

Last Updated: November 2025

