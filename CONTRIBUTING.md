# Contributing Guidelines

Quick guide for our team members.

## Team Members

- Stacy Che
- Kexin Lyu
- Samantha Wang
- Zexi Wu (Allen)
- Tongrui Zhang (Neil)

## Workflow

### Branch and Commit

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and commit
git add .
git commit -m "feat: description of changes"

# Push and create PR
git push origin feature/your-feature
```

### Commit Message Format

```
feat: add new feature
fix: bug fix
docs: documentation update
test: add tests
refactor: code refactoring
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Run `make format` before committing

### Testing

```bash
# Run tests
pytest

# Check coverage
pytest --cov=src

# Format code
make format
```

## Project Areas

**Computer Vision**: `src/vision/`
- YOLOv8 detection
- Image preprocessing

**NLP/Retrieval**: `src/nlp/`
- TF-IDF implementation
- Embedding generation
- FAISS indexing

**Data Structures**: `src/utils/`
- Bloom filter
- Evaluation metrics
- Helper functions

**Interface**: `app/`
- Gradio UI
- Integration logic

## Adding Dependencies

```bash
# Edit pyproject.toml
[project]
dependencies = [
    "new-package>=1.0.0",
]

# Install
uv pip install -e ".[dev]"
```

## Common Tasks

### Train Model

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

### Test Retrieval

```python
# In notebooks/03_recipe_retrieval.ipynb
import pandas as pd
from src.nlp.retriever import RecipeRetriever

# Load sample data
df = pd.read_csv('data/recipes_sample/...')
retriever = RecipeRetriever()
# Test implementation
```

### Run App Locally

```bash
source .venv/bin/activate
python app/main.py
```

## Questions

Ask in team chat or open an issue.
