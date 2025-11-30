# Dockerfile for SmartPantry

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project configuration files first (for better Docker layer caching)
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN uv pip install --system -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package itself
RUN uv pip install --system -e . || \
    pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p logs \
    data/raw \
    data/processed \
    data/recipes \
    data/history \
    models/yolo \
    models/embeddings \
    models/checkpoints

# Expose Gradio port (default 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Health check - check if Gradio is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the Gradio application
CMD ["python", "-m", "app.main"]

