# Dockerfile for SmartPantry (Hugging Face Spaces compatible)
# Includes frontend build and static file serving

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including Node.js for frontend build)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code and build it
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install && npm run build
WORKDIR /app

# Copy application code (selective copy for smaller image)
COPY app/ ./app/
COPY src/ ./src/
COPY data/canonical_vocab.json ./data/
COPY recipe_matching_system/ ./recipe_matching_system/ 2>/dev/null || true
COPY pyproject.toml . 2>/dev/null || true

# Copy data files if they exist (use Git LFS for large files)
COPY data/normalized_recipes.pkl ./data/ 2>/dev/null || true
COPY models/yolo/ ./models/yolo/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p logs \
    data/raw \
    data/processed \
    data/recipes \
    data/history \
    models/yolo \
    models/embeddings \
    models/checkpoints

# Expose port (HF Spaces will map this automatically)
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8001

# Health check - check if FastAPI is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the FastAPI application
# Use PORT environment variable (HF Spaces sets this)
CMD uvicorn app.api_extended:app --host 0.0.0.0 --port ${PORT:-8001}

