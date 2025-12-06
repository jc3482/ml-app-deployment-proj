# Deployment Guide

Simple deployment options for our student project.

## Option 1: Hugging Face Spaces (Recommended)

Free and easy deployment for demos.

### Steps

1. **Prepare the app**
```bash
# Copy main app to root as app.py (HF convention)
cp app/main.py app.py
```

2. **Create Space on Hugging Face**
- Go to https://huggingface.co/spaces
- Create new Space
- Select Gradio SDK
- Choose CPU (free) or GPU (paid)

3. **Push to Space**
```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smartpantry

# Push
git push hf main
```

4. **Requirements**
HF Spaces will use `pyproject.toml` automatically.

### Configuration

For HF Spaces, edit `config.yaml`:
```yaml
detection:
  device: cpu  # Use CPU on free tier
  model_name: yolov8n  # Smaller model
```

## Option 2: Docker (Local or Cloud)

### Build and Run

```bash
# Build image
docker build -t smartpantry .

# Run container
docker run -p 7860:7860 smartpantry

# Or use docker-compose
docker-compose up
```

### Deploy to Cloud

Our Docker image can run on:
- AWS EC2
- Google Cloud Run
- Azure Container Instances
- DigitalOcean

## Testing Deployment

```bash
# Test locally first
python app/main.py

# Verify at http://localhost:7860

# Then deploy to HF Spaces
```

## Notes

- Use sample datasets for demo (included in Git)
- Download full datasets for production
- Model weights can be loaded from Hugging Face Hub
- Keep costs low with free tiers

## Resources

- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Docker Tutorial](https://docs.docker.com/get-started/)

