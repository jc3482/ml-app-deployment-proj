# Docker Deployment Guide

This guide explains how to build and run the SmartPantry application using Docker.

## Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+ (optional, for docker-compose)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and start the container:**
   ```bash
   docker-compose up -d --build
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

4. **Access the application:**
   - Open your browser and navigate to: `http://localhost:7860`
   - The Gradio interface will be available at this URL

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t smartpantry:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name smartpantry-app \
     -p 7860:7860 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/config.yaml:/app/config.yaml:ro \
     smartpantry:latest
   ```

3. **View logs:**
   ```bash
   docker logs -f smartpantry-app
   ```

4. **Stop the container:**
   ```bash
   docker stop smartpantry-app
   docker rm smartpantry-app
   ```

## Volume Mounts

The following directories are mounted as volumes to persist data:

- `./data` → `/app/data` - Recipe data, canonical vocabulary, and user history
- `./models` → `/app/models` - YOLO models and embeddings
- `./logs` → `/app/logs` - Application logs
- `./config.yaml` → `/app/config.yaml` - Configuration file (read-only)

## Environment Variables

You can customize the application behavior using environment variables:

- `APP_ENV` - Application environment (default: `production`)
- `DEVICE` - Device to use (`cpu` or `cuda`, default: `cpu`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `GRADIO_SERVER_NAME` - Gradio server hostname (default: `0.0.0.0`)
- `GRADIO_SERVER_PORT` - Gradio server port (default: `7860`)

Create a `.env` file in the project root to set these variables:

```env
APP_ENV=production
DEVICE=cpu
LOG_LEVEL=INFO
```

## Health Check

The container includes a health check that verifies the Gradio server is running:

```bash
docker inspect --format='{{.State.Health.Status}}' smartpantry-app
```

## Troubleshooting

### Container won't start

1. Check logs: `docker logs smartpantry-app`
2. Verify data directories exist: `ls -la data/`
3. Ensure required files are present:
   - `data/canonical_vocab.json`
   - `data/normalized_recipes.pkl` (or `data/cached_normalized.csv`)
   - `models/yolo/yolov8n.pt`

### Port already in use

If port 7860 is already in use, change it in `docker-compose.yml`:

```yaml
ports:
  - "7861:7860"  # Use 7861 on host
```

### Out of memory

If the container runs out of memory:

1. Increase Docker memory limit in Docker Desktop settings
2. Use CPU-only mode (set `DEVICE=cpu`)
3. Reduce batch sizes in the application code

### Model files missing

Ensure model files are in the correct locations:

```bash
# Check YOLO model
ls -lh models/yolo/yolov8n.pt

# If missing, download it (the app will download automatically on first run)
```

## Development Mode

For development, you can mount the source code as a volume:

```yaml
volumes:
  - ./app:/app/app
  - ./src:/app/src
```

Then rebuild and restart:

```bash
docker-compose up -d --build
```

## Production Deployment

For production deployment:

1. Use a reverse proxy (nginx, traefik) in front of the container
2. Set up SSL/TLS certificates
3. Configure proper CORS settings
4. Use environment-specific configuration files
5. Set up log rotation
6. Monitor container health and resource usage

## GPU Support

To use GPU acceleration:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

3. Set `DEVICE=cuda` in environment variables

4. Rebuild and restart:
   ```bash
   docker-compose up -d --build
   ```

## Clean Up

To remove all containers, images, and volumes:

```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi smartpantry:latest

# Remove volumes (WARNING: This deletes data)
docker volume prune
```

