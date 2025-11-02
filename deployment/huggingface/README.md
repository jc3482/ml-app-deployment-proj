# Hugging Face Spaces Deployment

This directory contains configuration and instructions for deploying SmartPantry to Hugging Face Spaces.

## Prerequisites

1. Hugging Face account
2. Hugging Face CLI installed: `pip install huggingface-hub`
3. Hugging Face token with write access

## Deployment Steps

### 1. Login to Hugging Face

```bash
huggingface-cli login
```

### 2. Create a new Space

Go to https://huggingface.co/spaces and create a new Space:
- Name: `smartpantry`
- SDK: Gradio
- Hardware: CPU basic (can upgrade later if needed)

### 3. Prepare the repository

The Space expects these files in the root:
- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `README.md` - Space documentation
- `.gitattributes` - Git LFS configuration (for large files)

### 4. Copy files for deployment

```bash
# Copy main app as app.py (Spaces convention)
cp app/main.py app.py

# Copy requirements
cp requirements.txt requirements.txt

# Copy config
cp config.yaml config.yaml
```

### 5. Download pre-trained models

```bash
# YOLOv8 model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -O models/yolo/yolov8m.pt

# Or use YOLOv8 nano for faster inference on CPU
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolo/yolov8n.pt
```

### 6. Configure Git LFS for large files

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.faiss"
```

### 7. Push to Hugging Face Space

```bash
# Add Space as remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smartpantry

# Push to Space
git push hf main
```

## Configuration for Spaces

Update `config.yaml` for CPU-optimized inference:

```yaml
detection:
  device: cpu
  model_name: yolov8n  # Use nano model for CPU
  batch_size: 1

embeddings:
  device: cpu
  batch_size: 8

gradio:
  server_name: "0.0.0.0"
  server_port: 7860
  share: false
```

## Hardware Options

- **CPU basic** (Free): 2 vCPU, 16 GB RAM
- **CPU upgrade** ($0.03/hr): 8 vCPU, 32 GB RAM
- **T4 small** ($0.60/hr): 4 vCPU, 15 GB RAM, 16 GB GPU
- **T4 medium** ($1.05/hr): 8 vCPU, 30 GB RAM, 16 GB GPU
- **A10G small** ($1.05/hr): 4 vCPU, 15 GB RAM, 24 GB GPU

## Optimization Tips

1. **Use smaller models**: YOLOv8n instead of YOLOv8m
2. **Reduce batch size**: Set to 1 for CPU inference
3. **Enable caching**: Cache embeddings to reduce compute
4. **Lazy loading**: Load models only when needed
5. **Use Gradio examples**: Pre-cache example results

## Monitoring

Monitor your Space at:
```
https://huggingface.co/spaces/YOUR_USERNAME/smartpantry
```

Check logs for errors and performance metrics.

## Troubleshooting

### Out of Memory
- Use smaller models (YOLOv8n)
- Reduce image size in config
- Upgrade to CPU upgrade or GPU hardware

### Slow Inference
- Upgrade hardware tier
- Enable model caching
- Use batch processing

### Deployment Fails
- Check requirements.txt for incompatible packages
- Ensure all paths in config.yaml are correct
- Verify Git LFS is tracking large files

