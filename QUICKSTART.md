# SmartPantry - Quick Start Guide

This guide helps you set up and run SmartPantry in 5 minutes.

## Prerequisites

Our system requires:

- Python 3.10 or higher
- 4GB RAM minimum (we recommend 8GB)
- NVIDIA GPU with CUDA (optional, for faster inference)

## Installation

### Option 1: Automated Setup (Recommended)

We provide an automated setup script:

```bash
# Clone the repository
git clone <repository-url>
cd ml-app-deployment-proj

# Run setup script (installs uv and all dependencies)
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup with uv

For manual installation:

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Create directories
mkdir -p data/{raw,processed,recipes,fridge_photos,embeddings}
mkdir -p models/{yolo,embeddings,checkpoints}
mkdir -p logs
```

## Quick Demo

### 1. Download a Sample Model

```bash
# Download YOLOv8 nano (lightweight for CPU)
make download-models

# Or manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolo/yolov8n.pt
```

### 2. Run the Application

```bash
# Activate environment
source .venv/bin/activate

# Run app
python app/main.py

# Or using Make
make run
```

### 3. Access the Interface

Open your browser and navigate to:
```
http://localhost:7860
```

## Usage

### Basic Workflow

We designed a simple workflow:

1. **Upload Image**: Click "Upload Fridge/Pantry Photo" and select an image
2. **Detect Ingredients**: Click "Detect Ingredients" button
3. **Get Recipes**: Click "Get Recipes!" to see recommendations

### Tips for Best Results

We recommend:

- **Good lighting**: Ensure ingredients are well-lit
- **Clear view**: Arrange items so they are visible
- **Multiple angles**: Try different photos for better coverage
- **Close-ups**: For small items, take closer photos

## Common Issues

### "No module named 'ultralytics'"
```bash
uv pip install ultralytics
# Or reinstall all dependencies
uv pip install -e ".[dev]"
```

### "CUDA out of memory"
Edit `config.yaml`:
```yaml
detection:
  device: cpu  # Change from cuda to cpu
  batch_size: 1  # Reduce batch size
```

### "Port 7860 is already in use"
```bash
# Use different port
GRADIO_SERVER_PORT=7861 python app/main.py
```

### Slow inference on CPU
Our recommendations:
- Use smaller model: `yolov8n.pt` instead of `yolov8m.pt`
- Reduce image size in config
- Consider GPU deployment

## Testing

We provide testing commands:

```bash
# Run tests
make test

# With coverage
make test-cov

# Lint and format
make format
make lint
```

## Development

### Project Structure
```
ml-app-deployment-proj/
├── app/            # Gradio interface
├── src/            # Core modules
│   ├── vision/     # Detection
│   ├── nlp/        # Retrieval
│   └── utils/      # Utilities
├── data/           # Datasets
├── models/         # Model weights
└── tests/          # Test suite
```

### Configuration

Edit `config.yaml` to customize:
- Model paths and parameters
- Detection thresholds
- Retrieval settings
- UI preferences

### Adding Custom Recipes

Create a CSV file at `data/recipes/recipe_database.csv`:
```csv
recipe_id,title,ingredients,instructions,cuisine,difficulty,cooking_time
1,"Pasta Carbonara","pasta,eggs,bacon,cheese","Cook pasta...",Italian,Easy,20
```

## Next Steps

We suggest:

1. **Train Custom Model**: See `notebooks/02_model_training.ipynb`
2. **Build Recipe Database**: Load Recipe1M+ dataset
3. **Deploy**: Follow guides in `deployment/`
4. **Customize**: Modify `config.yaml` for your needs

## Resources

Our documentation includes:

- **Documentation**: See `docs/PROJECT_STRUCTURE.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Deployment**: See `deployment/huggingface/README.md` or `deployment/aws/README.md`
- **Issues**: Report on GitHub

## Example Commands

```bash
# Development
make run-dev              # Run in debug mode
make notebooks            # Launch Jupyter
make clean                # Clean temp files

# Testing
make test                 # Run all tests
make lint                 # Check code quality
make format               # Format code

# Docker
make docker-build         # Build image
make docker-run          # Run container
make docker-compose-up   # Run with compose

# Deployment
make deploy-hf           # Prepare HF deployment
```

## Getting Help

If you encounter issues:

1. Check `README.md` for detailed documentation
2. Review `docs/PROJECT_STRUCTURE.md` for architecture
3. Look at example notebooks in `notebooks/`
4. Open an issue on GitHub
5. Contact team members

---

We hope this helps you get started with SmartPantry!
