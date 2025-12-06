# Deployment Guide

## Docker Deployment

### Local Docker Build and Run

1. **Build the Docker image:**
   ```bash
   docker build -t smartpantry:latest .
   ```

2. **Run with docker-compose:**
   ```bash
   docker-compose up -d
   ```

3. **Or run directly:**
   ```bash
   docker run -d \
     -p 8001:8001 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     --name smartpantry \
     smartpantry:latest
   ```

4. **Check logs:**
   ```bash
   docker logs -f smartpantry
   ```

5. **Access the application:**
   - API: http://localhost:8001
   - Health check: http://localhost:8001/health
   - API docs: http://localhost:8001/docs

## Hugging Face Spaces Deployment

### Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Git LFS**: Install Git LFS for large files
   ```bash
   git lfs install
   ```

### Step 1: Prepare Repository

1. **Ensure all necessary files are committed:**
   ```bash
   git add .
   git commit -m "Prepare for Hugging Face deployment"
   ```

2. **Track large files with Git LFS:**
   ```bash
   git lfs track "*.pkl"
   git lfs track "data/ontology_recipes.*"
   git lfs track "data/normalized_recipes.*"
   git add .gitattributes
   git commit -m "Add Git LFS tracking for large files"
   ```

### Step 2: Create Hugging Face Space

1. **Go to**: https://huggingface.co/spaces
2. **Click "Create new Space"**
3. **Configure:**
   - **Name**: `smartpantry` (or your preferred name)
   - **SDK**: `Docker`
   - **Visibility**: Public or Private
   - **Hardware**: CPU Basic (or GPU if needed)

### Step 3: Push to Hugging Face

1. **Add Hugging Face remote:**
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

2. **Push to Hugging Face:**
   ```bash
   git push hf main
   ```

   **Note**: If you have large files, make sure they're tracked with Git LFS:
   ```bash
   git lfs push hf main
   ```

### Step 4: Verify Deployment

1. **Wait for build** (usually 5-10 minutes)
2. **Check build logs** in the Hugging Face Space interface
3. **Access your app** at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### Important Notes

1. **Data Files**: 
   - `data/ontology_recipes.pkl` (36MB) - Required for recipe matching
   - Use Git LFS for files > 100MB
   - Hugging Face Spaces has a 50GB limit per space

2. **Environment Variables**:
   - `PORT`: Automatically set by Hugging Face (default: 7860)
   - The Dockerfile uses `${PORT:-8001}` as fallback

3. **Health Check**:
   - Hugging Face will automatically check `/health` endpoint
   - Make sure the endpoint returns 200 OK

4. **Build Time**:
   - First build: ~10-15 minutes (installing dependencies)
   - Subsequent builds: ~5-10 minutes (cached layers)

### Troubleshooting

1. **Build fails**:
   - Check build logs in Hugging Face interface
   - Ensure all dependencies are in `requirements.txt`
   - Verify Dockerfile syntax

2. **App doesn't start**:
   - Check logs: `docker logs <container>`
   - Verify PORT environment variable
   - Check health endpoint: `/health`

3. **Large files not uploading**:
   - Ensure Git LFS is installed and configured
   - Check `.gitattributes` file
   - Use `git lfs push` explicitly

4. **Module not found errors**:
   - Verify `recipe_matching_system` is in the repository
   - Check Python path in `api_extended.py`
   - Ensure all imports are correct

## File Checklist for Deployment

- [x] `Dockerfile` - Docker configuration
- [x] `docker-compose.yml` - Docker Compose config
- [x] `requirements.txt` - Python dependencies
- [x] `README.md` - With Hugging Face frontmatter
- [x] `app/api_extended.py` - Main API file
- [x] `src/` - Backend source code
- [x] `recipe_matching_system/` - Recipe matching system
- [x] `frontend/` - Frontend source code
- [x] `data/ontology_recipes.pkl` - Recipe database (use Git LFS)
- [x] `models/yolo/` - YOLO model files (if needed)
- [x] `.dockerignore` - Exclude unnecessary files

