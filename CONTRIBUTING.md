# Contributing to SmartPantry

This document provides guidelines for our team members and external contributors.

## Team Members

- Stacy Che
- Kexin Lyu
- Samantha Wang
- Zexi Wu (Allen)
- Tongrui Zhang (Neil)

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd ml-app-deployment-proj

# Run setup script
chmod +x setup.sh
./setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/bug-description

# For experiments
git checkout -b experiment/experiment-name
```

### Making Changes

We follow this process:

1. Make changes in your branch
2. Write or update tests
3. Ensure all tests pass: `pytest`
4. Format code: `black src/ app/ tests/`
5. Check linting: `flake8 src/ app/ tests/`
6. Update documentation if needed

### Committing Changes

We use conventional commit format:

```bash
# Feature
git commit -m "feat: add ingredient normalization"

# Bug fix
git commit -m "fix: resolve detection threshold issue"

# Documentation
git commit -m "docs: update README with deployment instructions"

# Refactor
git commit -m "refactor: reorganize retrieval module"

# Test
git commit -m "test: add unit tests for embedder"
```

### Pull Requests

Our review process:

1. Push your branch: `git push origin feature/your-feature-name`
2. Create Pull Request on GitHub
3. Request review from at least one team member
4. Address review comments
5. Once approved, merge to main

## Code Style

### Python

We follow these standards:

- PEP 8 style guide
- Type hints where possible
- Maximum line length: 100 characters
- Docstrings for all functions and classes

```python
def process_image(
    image: Image.Image,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Process image and detect ingredients.
    
    Args:
        image: Input PIL Image
        threshold: Confidence threshold
        
    Returns:
        Dictionary with detection results
    """
    # Implementation
    pass
```

### Formatting Tools

```bash
# Format code
black src/ app/ tests/

# Sort imports
isort src/ app/ tests/

# Check linting
flake8 src/ app/ tests/

# Type checking (optional)
mypy src/
```

## Testing

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Name test files as `test_*.py`
- Use descriptive test names

```python
def test_ingredient_detection_returns_correct_format():
    """Test that detector returns results in expected format."""
    detector = IngredientDetector()
    results = detector.detect_ingredients(sample_image)
    
    assert "ingredients" in results
    assert "confidences" in results
    assert isinstance(results["ingredients"], list)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_detector.py

# Run with coverage
pytest --cov=src

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints

### README Updates

- Update README.md when adding new features
- Document new configuration options in config.yaml
- Add examples for new functionality

### Notebooks

- Clear outputs before committing: `jupyter nbconvert --clear-output --inplace *.ipynb`
- Add markdown explanations
- Document experiments and results

## Project Structure

```
ml-app-deployment-proj/
├── app/                    # Gradio application
├── data/                   # Datasets and databases
├── deployment/             # Deployment configs
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── vision/             # Computer vision
│   ├── nlp/                # NLP and retrieval
│   └── utils/              # Utilities
├── tests/                  # Tests
├── config.yaml             # Configuration
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Adding New Features

### 1. Computer Vision Features

- Add new detection methods in `src/vision/`
- Update `IngredientDetector` class
- Add tests in `tests/test_detector.py`
- Update configuration if needed

### 2. NLP/Retrieval Features

- Add new retrieval methods in `src/nlp/`
- Update `RecipeRetriever` class
- Add tests in `tests/test_retriever.py`
- Document in notebooks

### 3. UI Features

- Update `app/main.py`
- Test Gradio interface locally
- Update screenshots/examples

## Common Tasks

### Adding a New Dependency

```bash
# 1. Edit pyproject.toml and add to [project.dependencies]
# or appropriate [project.optional-dependencies] section

# 2. Install the package
uv pip install -e ".[dev]"

# 3. (Optional) Generate requirements.txt for compatibility
make lock
```

**Example:**
```toml
# pyproject.toml
[project]
dependencies = [
    # ... existing deps
    "new-package>=1.0.0",  # Add here
]
```

### Training a New Model

1. Create notebook in `notebooks/`
2. Document training process
3. Save model to `models/`
4. Update config.yaml with model path
5. Add evaluation metrics

### Updating Configuration

1. Edit `config.yaml`
2. Update `src/utils/helpers.py` if needed
3. Document new options in README
4. Update .env.example if environment variables are affected

## Debugging

### Common Issues

**Import Errors**
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**CUDA Out of Memory**
```python
# In config.yaml, reduce batch size
detection:
  batch_size: 4  # Reduce from 8
```

**Gradio Not Loading**
```bash
# Check logs
tail -f logs/app.log

# Try different port
python app/main.py --port 7861
```

## Communication

### Meetings

- Weekly team meetings (TBD)
- Stand-ups: Share progress and blockers
- Code reviews: Timely feedback

### Issues

- Use GitHub Issues for bug reports and feature requests
- Label issues appropriately
- Assign to team members

### Documentation

- Keep README.md up to date
- Document design decisions
- Share knowledge in notebooks

## Questions

If you have questions or need help:
1. Check existing documentation
2. Ask in team chat
3. Create a GitHub issue
4. Reach out to team members

We appreciate your contributions to SmartPantry.

