"""
Extended FastAPI REST API for Recipe Recommender
Adds pantry management, history, and ingredient removal features.
"""
import sys
from pathlib import Path
# Add project root and recipe_matching_system to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
recipe_matching_path = project_root / "recipe_matching_system"
if str(recipe_matching_path) not in sys.path:
    sys.path.insert(0, str(recipe_matching_path))

import os
import json
import tempfile
import logging
from typing import Optional, List, Dict
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Request, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Lazy import
from src.backend.recipe_recommender import RecipeRecommender
from src.vision.yolo_detector.food_detector import FoodDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartPantry API",
    description="Extended API for ingredient detection, recipe recommendation, pantry and history management",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router for /api prefix
api_router = APIRouter(prefix="/api")

# Global instances
_recommender = None
_preprocessor = None
_detector = None
_history_file = Path("data/history/history.json")
_pantry_file = Path("data/history/pantry.json")
_max_history = 20

# Ensure directories exist
_history_file.parent.mkdir(parents=True, exist_ok=True)


def get_preprocessor():
    """Get or initialize IngredientNormalizer instance."""
    global _preprocessor
    if _preprocessor is None:
        logger.info("Initializing IngredientNormalizer...")
        from recipe_matcher.preprocess import IngredientNormalizer
        _preprocessor = IngredientNormalizer()
        logger.info("IngredientNormalizer initialized")
    return _preprocessor


def get_detector():
    """Get or initialize FoodDetector instance (lightweight, no cache needed)."""
    global _detector
    if _detector is None:
        logger.info("Initializing FoodDetector...")
        _detector = FoodDetector()
        logger.info("FoodDetector initialized")
    return _detector


def get_recommender():
    """Get or initialize RecipeRecommender instance."""
    global _recommender
    if _recommender is None:
        logger.info("Initializing RecipeRecommender...")
        _recommender = RecipeRecommender()
        logger.info(f"RecipeRecommender initialized with {len(_recommender.recipe_dict)} recipes")
    return _recommender


# Request/Response Models
class PantryAddRequest(BaseModel):
    ingredients: List[str]


class PantryResponse(BaseModel):
    pantry: List[str]
    message: str


class HistoryRecord(BaseModel):
    timestamp: str
    ingredients: List[str]
    top_k: int
    recipes: List[Dict]


class HistoryResponse(BaseModel):
    history: List[HistoryRecord]
    count: int


class RecommendRequest(BaseModel):
    pantry_ingredients: Optional[List[str]] = []
    detected_ingredients: Optional[List[str]] = []
    top_k: int = 10
    dietary_filter: Optional[str] = "None"


# API Endpoints

# Root endpoint (for API info)
@app.get("/")
async def root():
    return {"status": "ok", "message": "SmartPantry API is running"}


# Health check (keep at root level for Docker healthcheck)
@app.get("/health")
async def health_check():
    """Fast health check endpoint for Hugging Face Spaces."""
    try:
        # Quick check - don't block on model initialization
        # Just verify the server is running
        if _recommender is None:
            # Server is running but models are still loading
            return {
                "status": "healthy",
                "message": "API is running. Models are initializing...",
                "ready": False
            }
        else:
            # Models are loaded
            return {
                "status": "healthy",
                "message": f"API is running. Loaded {len(_recommender.recipe_dict)} recipes.",
                "ready": True
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Still return 200 to avoid Hugging Face thinking the app is down
        return {
            "status": "healthy",
            "message": f"API is running (initialization in progress: {str(e)})",
            "ready": False
        }


@api_router.post("/detect")
async def detect_ingredients(file: UploadFile = File(...)):
    """Detect ingredients from uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file_path = None
    try:
        suffix = Path(file.filename).suffix if file.filename else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_file_path = tmp_file.name
            content = await file.read()
            tmp_file.write(content)
        
        # Use lightweight components instead of full RecipeRecommender
        detector = get_detector()
        preprocessor = get_preprocessor()
        
        raw_labels = detector.detect(temp_file_path)
        preprocessor = get_preprocessor()
        canonical = preprocessor.normalize_list(raw_labels)
        
        return {
            "raw_detections": raw_labels,
            "canonical_ingredients": canonical
        }
    except Exception as e:
        logger.error(f"Error detecting ingredients: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@api_router.post("/pantry/add")
async def add_to_pantry(request: PantryAddRequest):
    """Add ingredients to pantry."""
    try:
        pantry = load_pantry()
        preprocessor = get_preprocessor()  # Use lightweight preprocessor instead of full recommender
        
        # Normalize and add ingredients
        for ing in request.ingredients:
            # IngredientNormalizer.normalize_ingredient returns a single string
            normalized = preprocessor.normalize_ingredient(ing.strip())
            if normalized:
                pantry.append(normalized)
        
        # Remove duplicates and save
        pantry = sorted(list(set(pantry)))
        save_pantry(pantry)
        
        return PantryResponse(pantry=pantry, message=f"Added {len(request.ingredients)} ingredient(s)")
    except Exception as e:
        logger.error(f"Error adding to pantry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/pantry/list")
async def get_pantry():
    """Get pantry list."""
    pantry = load_pantry()
    return PantryResponse(pantry=pantry, message="Pantry retrieved successfully")


@api_router.delete("/pantry/remove/{ingredient}")
async def remove_from_pantry(ingredient: str):
    """Remove ingredient from pantry."""
    try:
        pantry = load_pantry()
        if ingredient in pantry:
            pantry.remove(ingredient)
            save_pantry(pantry)
            return PantryResponse(pantry=pantry, message=f"Removed '{ingredient}'")
        else:
            raise HTTPException(status_code=404, detail=f"'{ingredient}' not found in pantry")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from pantry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/pantry/clear")
async def clear_pantry():
    """Clear all pantry items."""
    save_pantry([])
    return PantryResponse(pantry=[], message="Pantry cleared")


@api_router.post("/recommend")
async def recommend_recipes(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """Get recipe recommendations."""
    try:
        # Parse request body - handle both JSON and form-data
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # JSON request
            body = await request.json()
            request_obj = RecommendRequest(**body)
        elif "multipart/form-data" in content_type:
            # Form data - get JSON string from form
            form = await request.form()
            request_str = form.get("request")
            if request_str:
                request_data = json.loads(request_str)
                request_obj = RecommendRequest(**request_data)
            else:
                # No request data in form, use defaults
                request_obj = RecommendRequest(
                    detected_ingredients=[],
                    pantry_ingredients=[],
                    top_k=10,
                    dietary_filter="None"
                )
        else:
            # Default
            request_obj = RecommendRequest(
                detected_ingredients=[],
                pantry_ingredients=[],
                top_k=10,
                dietary_filter="None"
            )
        
        recommender = get_recommender()
        all_ingredients = []
        
        # Add detected ingredients from image if provided
        if file:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            temp_file_path = None
            try:
                suffix = Path(file.filename).suffix if file.filename else ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    temp_file_path = tmp_file.name
                    content = await file.read()
                    tmp_file.write(content)
                
                raw_labels = recommender.detector.detect(temp_file_path)
                # Preprocessing is now handled internally by the pipeline in recommend()
                # But we still need to display 'canonical' ingredients for the API response if needed BEFORE pipeline runs
                # Actually, the pipeline will return processed ingredients.
                # We can just accumulate raw labels here.
                all_ingredients.extend(raw_labels)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
        
        # Add provided detected ingredients
        if request_obj.detected_ingredients:
            all_ingredients.extend(request_obj.detected_ingredients)
        
        # Add pantry ingredients
        if request_obj.pantry_ingredients:
            all_ingredients.extend(request_obj.pantry_ingredients)
        
        # Also add saved pantry
        pantry = load_pantry()
        all_ingredients.extend(pantry)
        
        # Remove duplicates
        all_ingredients = sorted(list(set(all_ingredients)))
        
        if not all_ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        # Use the updated recommend() method which handles everything via the pipeline
        # We pass a dummy image path if none exists, or rely on the fact that 
        # we've already gathered ingredients.
        # However, recipe_recommender.recommend() currently expects an image path.
        # Let's adapt it to accept a list of ingredients directly or modify this call.
        
        # Actually, looking at RecipeRecommender.recommend in src/backend/recipe_recommender.py,
        # it strictly requires an image path to start the pipeline.
        # But we already have the ingredients here (all_ingredients).
        # We need to bypass the image detection part of recommend() or update recommend() to accept ingredients.
        
        # OPTION: We can call pipeline.run() directly since we have access to recommender.pipeline
        
        pipeline = recommender.pipeline
        processed_ingredients, ranked_recipes = pipeline.run(all_ingredients, top_k=request_obj.top_k)
        
        # Format output
        top = []
        dietary_keywords = {
            "vegan": ["meat", "beef", "chicken", "pork", "fish", "shrimp", "eggs", "milk", "cheese", "butter"],
            "vegetarian": ["meat", "beef", "chicken", "pork", "fish", "shrimp"],
            "dairy-free": ["milk", "cheese", "butter", "cream", "yogurt"],
            "gluten-free": ["flour", "bread", "pasta", "wheat"],
        }
        dietary_filter = request_obj.dietary_filter

        for r in ranked_recipes:
            # Apply dietary filter
            if dietary_filter and dietary_filter != "None":
                restricted = dietary_keywords.get(dietary_filter.lower(), [])
                # Check against normalized ingredients in the recipe
                recipe_ings = [ing.lower() for ing in r.get("recipe_ingredients", [])]
                if any(restricted_item in ing for ing in recipe_ings for restricted_item in restricted):
                    continue

            top.append({
                "title": r["title"],
                "score": float(r["fuzzy_score"]) * 100,  # Convert 0-1 score to 0-100 percentage
                "normalized_ingredients": r.get("recipe_ingredients", []),
                "cleaned_ingredients": r.get("recipe_ingredients", []),
                "ingredients_raw": r.get("recipe_ingredients", []),
                "instructions": r.get("instructions", ""),
                "image_name": r.get("image_name"),
                "matched_ingredients": r.get("matched", []),  # 已有食材
                "missing_ingredients": r.get("missing", [])  # 缺失食材
            })
        
        # Re-slice top_k after filtering
        top = top[:request_obj.top_k]
        
        # Save to history
        save_history(processed_ingredients, top, request_obj.top_k)
        
        return {
            "fridge_items": processed_ingredients,
            "recommendations": top
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/history")
async def get_history():
    """Get history records."""
    history = load_history()
    return HistoryResponse(history=history, count=len(history))


@api_router.post("/history/clear")
async def clear_history():
    """Clear all history."""
    save_history([], [], 0, clear=True)
    return {"message": "History cleared", "count": 0}


# Helper functions

def load_pantry() -> List[str]:
    """Load pantry from file. If empty or doesn't exist, initialize with default ingredients."""
    # Default pantry ingredients
    default_ingredients = [
        "salt",
        "sea salt",
        "all purpose flour",
        "vegetable oil for frying",
        "olive oil",
        "vegetable oil for grill",
        "vegetable oil",
        "butter",
        "sugar",
        "water",
        "canola oil"
    ]
    
    try:
        if _pantry_file.exists():
            with open(_pantry_file, "r", encoding="utf-8") as f:
                pantry = json.load(f)
                if isinstance(pantry, list) and len(pantry) > 0:
                    return pantry
        
        # If file doesn't exist or is empty, initialize with defaults
        save_pantry(default_ingredients)
        return default_ingredients
    except Exception as e:
        logger.warning(f"Failed to load pantry: {e}")
        # On error, return defaults
        return default_ingredients


def save_pantry(pantry: List[str]):
    """Save pantry to file."""
    try:
        with open(_pantry_file, "w", encoding="utf-8") as f:
            json.dump(pantry, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save pantry: {e}")


def load_history() -> List[Dict]:
    """Load history from file."""
    try:
        if _history_file.exists():
            with open(_history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")
        return []


def save_history(ingredients: List[str], recipes: List[Dict], top_k: int, clear: bool = False):
    """Save history record."""
    try:
        if clear:
            history = []
        else:
            history = load_history()
            record = {
                "timestamp": datetime.now().isoformat(),
                "ingredients": ingredients,
                "top_k": top_k,
                "recipes": [
                    {
                        "title": r["title"],
                        "score": r["score"],
                        "normalized_ingredients": r["normalized_ingredients"]
                    }
                    for r in recipes[:5]
                ]
            }
            history.insert(0, record)
            history = history[:_max_history]
        
        with open(_history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")


# Mount API router first (before static files)
app.include_router(api_router)

# Mount static files (frontend) - must be last to catch all non-API routes
# Use a catch-all route that excludes /api paths
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    from fastapi.responses import FileResponse
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend files, but skip API routes."""
        # Skip API routes - let them be handled by api_router
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Try to serve the file
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        # For SPA routing, serve index.html for all non-API routes
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        
        raise HTTPException(status_code=404, detail="Not found")
    
    logger.info(f"Mounted frontend static files from {frontend_dist}")
else:
    logger.warning(f"Frontend dist directory not found at {frontend_dist}, serving API only")


def main():
    """Entry point for the application."""
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("app.api_extended:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()