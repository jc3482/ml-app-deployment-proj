#!/bin/bash
# Setup script for SmartPantry

set -e  # Exit on error

echo "🥗 Setting up SmartPantry..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

echo "Python version: $python_version"

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed"
fi

# Create virtual environment with uv
echo "📦 Creating virtual environment with uv..."
uv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv
echo "📥 Installing dependencies with uv..."
uv pip install -e ".[dev]"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,recipes,fridge_photos,embeddings}
mkdir -p models/{yolo,embeddings,checkpoints}
mkdir -p logs
mkdir -p results

# Download YOLOv8 model (optional)
echo "🤖 Downloading YOLOv8 model (optional, press Ctrl+C to skip)..."
read -p "Download YOLOv8 nano model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Downloading YOLOv8n..."
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolo/yolov8n.pt
    echo "✅ YOLOv8n downloaded!"
fi

# Create .env file from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
# Environment Variables for SmartPantry

APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

YOLO_MODEL_PATH=./models/yolo/yolov8n.pt
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
RECIPE_DB_PATH=./data/recipes/recipe_database.csv

DEVICE=cpu
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

NUM_WORKERS=4
BATCH_SIZE=8
ENABLE_CLUSTERING=true
ENABLE_CACHE=true
EOF
    echo "✅ .env file created!"
else
    echo "⚠️  .env file already exists, skipping..."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the application:"
echo "  python app/main.py"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "📚 Check README.md for more information!"

