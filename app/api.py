"""
FastAPI REST API for Recipe Recommender
Provides endpoints for ingredient detection and recipe recommendation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
import logging
from typing import Optional, TYPE_CHECKING
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Lazy import to avoid triggering src.__init__.py imports
if TYPE_CHECKING:
    from src.backend.recipe_recommender import RecipeRecommender

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Recipe Recommender API",
    description="API for ingredient detection and recipe recommendation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RecipeRecommender instance (lazy initialization)
_recommender = None


def get_recommender():
    """Get or initialize RecipeRecommender instance."""
    global _recommender
    if _recommender is None:
        # Import here to avoid triggering src.__init__.py imports at module level
        from src.backend.recipe_recommender import RecipeRecommender
        logger.info("Initializing RecipeRecommender...")
        _recommender = RecipeRecommender()
        logger.info("RecipeRecommender initialized successfully")
    return _recommender


# Response models
class HealthResponse(BaseModel):
    status: str
    message: str


class DetectResponse(BaseModel):
    image_path: str
    raw_detections: list[str]
    canonical_ingredients: list[str]


class RecipeRecommendation(BaseModel):
    title: str
    score: float
    normalized_ingredients: list[str]
    cleaned_ingredients: list[str]
    ingredients_raw: str
    instructions: str
    image_name: Optional[str] = None


class RecommendResponse(BaseModel):
    image_path: str
    fridge_items: list[str]
    recommendations: list[RecipeRecommendation]


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "message": "Recipe Recommender API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint (required for Docker).
    Returns API status and basic information.
    """
    try:
        recommender = get_recommender()
        recipe_count = len(recommender.recipe_dict)
        return {
            "status": "healthy",
            "message": f"API is running. Loaded {recipe_count} recipes."
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/detect", response_model=DetectResponse)
async def detect_ingredients(
    file: UploadFile = File(..., description="Image file to detect ingredients from")
):
    """
    Detect ingredients from an uploaded image.
    
    Returns:
        - raw_detections: Raw YOLO detection labels
        - canonical_ingredients: Normalized canonical ingredient names
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_file_path = None
    
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp_file:
            temp_file_path = tmp_file.name
            # Write uploaded content to temp file
            content = await file.read()
            tmp_file.write(content)
        
        logger.info(f"Saved uploaded image to: {temp_file_path}")
        
        # Get recommender and run detection
        recommender = get_recommender()
        
        # Run YOLO detection
        raw_labels = recommender.detector.detect(temp_file_path)
        logger.info(f"Raw YOLO detections: {raw_labels}")
        
        # Normalize to canonical ingredients
        canonical = recommender.preprocessor.normalize(raw_labels)
        logger.info(f"Canonical ingredients: {canonical}")
        
        return {
            "image_path": temp_file_path,
            "raw_detections": raw_labels,
            "canonical_ingredients": canonical
        }
    
    except Exception as e:
        logger.error(f"Error detecting ingredients: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_recipes(
    file: UploadFile = File(..., description="Image file to get recipe recommendations for"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top recipes to return")
):
    """
    Get recipe recommendations based on detected ingredients from an uploaded image.
    
    Args:
        file: Image file to analyze
        top_k: Number of top recipes to return (1-20)
    
    Returns:
        - fridge_items: Detected canonical ingredients
        - recommendations: List of recommended recipes with scores and details
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_file_path = None
    
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as tmp_file:
            temp_file_path = tmp_file.name
            # Write uploaded content to temp file
            content = await file.read()
            tmp_file.write(content)
        
        logger.info(f"Saved uploaded image to: {temp_file_path}")
        
        # Get recommender and run full pipeline
        recommender = get_recommender()
        result = recommender.recommend(temp_file_path, top_k=top_k)
        
        # Convert recommendations to response model
        recommendations = [
            RecipeRecommendation(
                title=rec["title"],
                score=rec["score"],
                normalized_ingredients=rec["normalized_ingredients"],
                cleaned_ingredients=rec["cleaned_ingredients"],
                ingredients_raw=rec["ingredients_raw"],
                instructions=rec["instructions"],
                image_name=rec.get("image_name")
            )
            for rec in result["recommendations"]
        ]
        
        return {
            "image_path": temp_file_path,
            "fridge_items": result["fridge_items"],
            "recommendations": recommendations
        }
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

