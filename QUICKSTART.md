# Quick Start Guide

Get SmartPantry running in 5 minutes.

## Prerequisites

- Python 3.10+
- 4GB RAM (8GB recommended)

## Installation

### Automated Setup

```bash
git clone <repo-url>
cd ml-app-deployment-proj
./setup.sh
```

### Manual Setup

```bash
# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

## Run the App

```bash
source .venv/bin/activate
python app/main.py
```

Open browser to `http://localhost:7860`

## Test with Sample Data

Sample datasets are included:
- 30 fridge images
- 100 recipes

Test immediately without downloading full datasets.

## Common Issues

### "Module not found"
```bash
uv pip install -e ".[dev]"
```

### "CUDA out of memory"
Edit `config.yaml`:
```yaml
detection:
  device: cpu
```

### "Port in use"
```bash
GRADIO_SERVER_PORT=7861 python app/main.py
```

## Next Steps

1. Review `README.md` for project details
2. Explore sample data in `data/`
3. Check `notebooks/` for training examples
4. See `CONTRIBUTING.md` for development workflow

## Quick Commands

```bash
make run        # Start app
make test       # Run tests
make format     # Format code
jupyter notebook # Open notebooks
```
