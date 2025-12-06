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

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY data/canonical_vocab.json ./data/
COPY recipe_matching_system/ ./recipe_matching_system/

# Copy optional files (copy if exists, otherwise skip)
COPY pyproject.toml* ./

# Copy data files (required for deployment)
COPY data/ontology_recipes.pkl ./data/
COPY data/normalized_recipes.json ./data/
COPY data/ontology_recipes.json ./data/

# Copy models directory (will be empty if no models, but that's OK)
COPY models/ ./models/

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
# Don't hardcode PORT - let Hugging Face Spaces set it (usually 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
# PORT will be set by Hugging Face Spaces (typically 7860)
# Use 8001 as fallback for local development

# Health check - check if FastAPI is running
# Use PORT env var (set by HF Spaces, typically 7860) or default to 8001 for local
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Run the FastAPI application
# Use PORT environment variable (HF Spaces sets this)
CMD ["sh", "-c", "uvicorn app.api_extended:app --host 0.0.0.0 --port ${PORT:-8001}"]

