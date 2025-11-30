"""
Main Gradio application for SmartPantry.
User interface for ingredient detection and recipe recommendation.
Updated to use RecipeRecommender pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import gradio as gr
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PIL import Image
import numpy as np

from src.backend.recipe_recommender import RecipeRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SmartPantryApp:
    """
    Main application class for SmartPantry.
    Uses the unified RecipeRecommender pipeline.
    """
    
    def __init__(self):
        """Initialize the SmartPantry application."""
        logger.info("Initializing SmartPantry application")
        
        # Initialize RecipeRecommender (unified pipeline)
        logger.info("Loading RecipeRecommender...")
        self.recommender = RecipeRecommender()
        logger.info(f"RecipeRecommender initialized with {len(self.recommender.recipe_dict)} recipes")
        
        # Initialize history storage
        self.history_dir = Path("data/history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "history.json"
        self.max_history = 20  # Keep last 20 records
        
        # Initialize pantry list storage
        self.pantry_file = self.history_dir / "pantry.json"
        self.pantry_list = self._load_pantry()
        
        # Store current recommendations
        self.current_recommendations = []
        
        # Dietary filter keywords
        self.dietary_keywords = {
            "vegan": ["meat", "beef", "chicken", "pork", "fish", "shrimp", "eggs", "milk", "cheese", "butter", "cream", "yogurt"],
            "vegetarian": ["meat", "beef", "chicken", "pork", "fish", "shrimp"],
            "dairy-free": ["milk", "cheese", "butter", "cream", "yogurt"],
            "gluten-free": ["flour", "bread", "pasta", "wheat"],
        }
        
        logger.info("SmartPantry application initialized successfully")
    
    def _load_pantry(self) -> List[str]:
        """Load pantry list from storage."""
        try:
            if self.pantry_file.exists():
                with open(self.pantry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Failed to load pantry: {e}")
            return []
    
    def _save_pantry(self, pantry: List[str]):
        """Save pantry list to storage."""
        try:
            with open(self.pantry_file, "w", encoding="utf-8") as f:
                json.dump(pantry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save pantry: {e}")
    
    def _check_dietary_compliance(self, recipe_ingredients: List[str], dietary_filter: str) -> bool:
        """Check if recipe complies with dietary filter."""
        if not dietary_filter or dietary_filter == "None":
            return True
        
        restricted = self.dietary_keywords.get(dietary_filter.lower(), [])
        recipe_ingredients_lower = [ing.lower() for ing in recipe_ingredients]
        
        for restricted_item in restricted:
            if any(restricted_item in ing for ing in recipe_ingredients_lower):
                return False
        return True
    
    def process_image(
        self,
        image: Image.Image,
    ) -> str:
        """
        Process uploaded image and detect ingredients.
        
        Args:
            image: Input image from user
            
        Returns:
            Formatted ingredient text
        """
        logger.info("Processing uploaded image")
        
        if image is None:
            return "Please upload an image first."
        
        try:
            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path, "JPEG")
            
            try:
                # Use RecipeRecommender's detector and preprocessor
                # Run YOLO detection
                raw_labels = self.recommender.detector.detect(temp_path)
                logger.info(f"Raw YOLO detections: {raw_labels}")
                
                # Normalize to canonical ingredients
                canonical = self.recommender.preprocessor.normalize(raw_labels)
                logger.info(f"Canonical ingredients: {canonical}")
                
                # Format output - collapsible with elegant styling
                if canonical:
                    ingredients_html = "".join(
                        f"<li>{ing}</li>"
                        for ing in canonical
                    )
                    count = len(canonical)
                    ingredient_text = f"""
                    <details style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 0; font-family: Georgia, serif; margin-bottom: 1em;'>
                        <summary style='padding: 1.5em; cursor: pointer; font-weight: 400; color: #5a4a3a; font-size: 1.1em; letter-spacing: 0.02em; list-style: none; user-select: none;'>
                            Detected Ingredients ({count} items)
                        </summary>
                        <div style='padding: 0 1.5em 1.5em 1.5em; border-top: 1px solid #e8ddd4; margin-top: 0.5em;'>
                            <ul style='list-style: none; padding: 0; margin: 0;'>
                                {ingredients_html}
                            </ul>
                        </div>
                    </details>
                    """
                else:
                    ingredient_text = "<div style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; font-family: Georgia, serif; color: #8b7355; font-style: italic;'>No ingredients detected. Please try another image.</div>"
                
                return ingredient_text
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error: {str(e)}"
    
    def _save_history(self, ingredients: List[str], recipes: List[Dict], top_k: int):
        """Save detection and recommendation to history."""
        try:
            # Load existing history
            history = []
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            
            # Create new record (don't save image path as it's temporary)
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
                    for r in recipes[:5]  # Save top 5 only
                ]
            }
            
            # Add to history
            history.insert(0, record)
            
            # Keep only last N records
            history = history[:self.max_history]
            
            # Save back
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved history record: {len(history)} total records")
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    def _load_history(self) -> List[Dict]:
        """Load history records."""
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")
            return []
    
    def _format_pantry_display(self) -> str:
        """Format pantry list for display - collapsible."""
        count = len(self.pantry_list)
        
        if not self.pantry_list:
            return f"""
            <details style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 0; font-family: Georgia, serif; margin-bottom: 1em;'>
                <summary style='padding: 1.5em; cursor: pointer; font-weight: 400; color: #5a4a3a; font-size: 1.1em; letter-spacing: 0.02em; list-style: none; user-select: none;'>
                    My Pantry (0 items)
                </summary>
                <div style='padding: 0 1.5em 1.5em 1.5em; border-top: 1px solid #e8ddd4; margin-top: 0.5em;'>
                    <p style='color: #8b7355; font-style: italic; margin: 0;'>Your pantry is empty. Add ingredients above.</p>
                </div>
            </details>
            """
        
        ingredients_html = "".join(
            f"<li>{ing}</li>"
            for ing in sorted(self.pantry_list)
        )
        return f"""
        <details style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 0; font-family: Georgia, serif; margin-bottom: 1em;'>
            <summary style='padding: 1.5em; cursor: pointer; font-weight: 400; color: #5a4a3a; font-size: 1.1em; letter-spacing: 0.02em; list-style: none; user-select: none;'>
                My Pantry ({count} items)
            </summary>
            <div style='padding: 0 1.5em 1.5em 1.5em; border-top: 1px solid #e8ddd4; margin-top: 0.5em;'>
                <ul style='list-style: none; padding: 0; margin: 0;'>
                    {ingredients_html}
                </ul>
            </div>
        </details>
        """
    
    def _add_to_pantry(self, new_ingredients: str) -> Tuple[str, str]:
        """Add ingredients to pantry list."""
        if not new_ingredients or not new_ingredients.strip():
            return self._format_pantry_display(), ""
        
        # Parse comma-separated ingredients
        ingredients = [ing.strip() for ing in new_ingredients.split(",") if ing.strip()]
        
        # Normalize and add to pantry
        for ing in ingredients:
            normalized = self.recommender.preprocessor.normalize([ing])
            if normalized:
                self.pantry_list.extend(normalized)
        
        # Remove duplicates and save
        self.pantry_list = sorted(list(set(self.pantry_list)))
        self._save_pantry(self.pantry_list)
        
        return self._format_pantry_display(), ""
    
    def _clear_pantry(self) -> str:
        """Clear pantry list."""
        self.pantry_list = []
        self._save_pantry(self.pantry_list)
        return self._format_pantry_display()
    
    def get_history_display(self) -> str:
        """Format history for display - collapsible."""
        history = self._load_history()
        count = len(history)
        
        if not history:
            return """
            <details style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 0; font-family: Georgia, serif; margin-bottom: 1em;'>
                <summary style='padding: 1.5em; cursor: pointer; font-weight: 400; color: #5a4a3a; font-size: 1.1em; letter-spacing: 0.02em; list-style: none; user-select: none;'>
                    History (0 items)
                </summary>
                <div style='padding: 0 1.5em 1.5em 1.5em; border-top: 1px solid #e8ddd4; margin-top: 0.5em;'>
                    <p style='color: #8b7355; font-style: italic; margin: 0;'>No history yet. Start detecting ingredients to see your history here.</p>
                </div>
            </details>
            """
        
        output_lines = ["<details style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 0; font-family: Georgia, serif; margin-bottom: 1em;'>"]
        output_lines.append(f"<summary style='padding: 1.5em; cursor: pointer; font-weight: 400; color: #5a4a3a; font-size: 1.1em; letter-spacing: 0.02em; list-style: none; user-select: none;'>History ({count} items)</summary>")
        output_lines.append("<div style='padding: 0 1.5em 1.5em 1.5em; border-top: 1px solid #e8ddd4; margin-top: 0.5em;'>")
        
        for i, record in enumerate(history[:10], 1):  # Show last 10
            dt = datetime.fromisoformat(record["timestamp"])
            time_str = dt.strftime("%Y-%m-%d %H:%M")
            
            output_lines.append(f"<div style='border-bottom: 1px solid #e8ddd4; padding: 1em 0; margin-bottom: 1em;'>")
            output_lines.append(f"<p style='margin: 0 0 0.5em 0; color: #8b7355; font-size: 0.9em;'>{time_str}</p>")
            output_lines.append(f"<p style='margin: 0 0 0.5em 0; color: #3d3d3d;'><strong>Ingredients:</strong> {', '.join(record['ingredients'][:10])}</p>")
            if record.get("recipes"):
                top_recipe = record["recipes"][0]
                output_lines.append(f"<p style='margin: 0; color: #5a4a3a; font-style: italic;'>Top: {top_recipe['title']}</p>")
            output_lines.append("</div>")
        
        output_lines.append("</div>")
        output_lines.append("</details>")
        return "\n".join(output_lines)
    
    def _clear_history(self) -> str:
        """Clear all history records."""
        try:
            if self.history_file.exists():
                with open(self.history_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
            logger.info("History cleared")
        except Exception as e:
            logger.warning(f"Failed to clear history: {e}")
        return self.get_history_display()
    
    def process_multiple_images(
        self,
        images: List[Image.Image],
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Process multiple images and combine detected ingredients.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (ingredient_text, combined_annotated_image)
        """
        if not images or all(img is None for img in images):
            return "Please upload at least one image.", None
        
        all_ingredients = []
        annotated_images = []
        
        for idx, image in enumerate(images):
            if image is None:
                continue
            
            try:
                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    temp_path = tmp_file.name
                    image.save(temp_path, "JPEG")
                
                try:
                    # Get YOLO results
                    results = self.recommender.detector.model(temp_path)[0]
                    
                    # Get raw labels
                    raw_labels = [
                        self.recommender.detector.class_names[int(b.cls)]
                        for b in results.boxes
                        if float(b.conf) >= self.recommender.detector.conf
                    ]
                    
                    # Normalize to canonical ingredients
                    canonical = self.recommender.preprocessor.normalize(raw_labels)
                    all_ingredients.extend(canonical)
                    
                    # Create annotated image
                    annotated_img = results.plot()
                    annotated_images.append(Image.fromarray(annotated_img))
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            except Exception as e:
                logger.error(f"Error processing image {idx+1}: {e}")
        
        # Remove duplicates
        unique_ingredients = sorted(list(set(all_ingredients)))
        
        # Format output
        if unique_ingredients:
            ingredients_html = "".join(
                f"<li>{ing}</li>"
                for ing in unique_ingredients
            )
            ingredient_text = f"""
            <div style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; font-family: Georgia, serif;'>
                <p style='margin: 0 0 1em 0; font-weight: 400; color: #5a4a3a; font-size: 1.1em; letter-spacing: 0.02em;'>Detected Ingredients ({len(images)} images)</p>
                <ul style='list-style: none; padding: 0; margin: 0;'>
                    {ingredients_html}
                </ul>
            </div>
            """
            # Return first annotated image (or combine them if needed)
            combined_image = annotated_images[0] if annotated_images else None
        else:
            ingredient_text = "<div style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; font-family: Georgia, serif; color: #8b7355; font-style: italic;'>No ingredients detected from the uploaded images.</div>"
            combined_image = None
        
        return ingredient_text, combined_image
    
    def get_recipe_recommendations(
        self,
        image: Optional[Image.Image],
        pantry_list: str,
        top_k: int = 10,
        dietary_filter: str = "None",
    ) -> str:
        """
        Get recipe recommendations based on uploaded image and/or pantry list.
        
        Args:
            image: Input image from user (optional)
            pantry_list: Comma-separated string of pantry ingredients
            top_k: Maximum number of recipes to return
            dietary_filter: Dietary restriction filter (None, vegan, vegetarian, dairy-free, gluten-free)
            
        Returns:
            Formatted recipe recommendations
        """
        logger.info(f"Getting recommendations (top_k={top_k}, dietary_filter={dietary_filter})")
        
        # Combine ingredients from image and pantry list
        all_ingredients = []
        
        if image is not None:
            try:
                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    temp_path = tmp_file.name
                    image.save(temp_path, "JPEG")
                
                try:
                    # Get YOLO detection
                    raw_labels = self.recommender.detector.detect(temp_path)
                    canonical = self.recommender.preprocessor.normalize(raw_labels)
                    all_ingredients.extend(canonical)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Error detecting from image: {e}")
        
        # Add pantry list ingredients
        if pantry_list and pantry_list.strip():
            # Parse comma-separated ingredients
            pantry_items = [item.strip() for item in pantry_list.split(",") if item.strip()]
            # Also add from saved pantry list
            pantry_items.extend(self.pantry_list)
            
            # Normalize pantry ingredients
            normalized_pantry = []
            for item in pantry_items:
                if item.strip():
                    normalized = self.recommender.preprocessor.normalize([item.strip()])
                    normalized_pantry.extend(normalized)
            all_ingredients.extend(normalized_pantry)
        
        # Remove duplicates
        all_ingredients = sorted(list(set(all_ingredients)))
        
        if not all_ingredients:
            return "Please upload an image or add ingredients to your pantry list."
        
        try:
            # Create a temporary image for the recommendation pipeline
            # (RecipeRecommender expects an image path, so we'll use a dummy approach)
            # Actually, we need to modify the approach - let's match recipes directly
            scored_recipes = []
            for title, meta in self.recommender.recipe_dict.items():
                # Check dietary filter
                if not self._check_dietary_compliance(meta["normalized"], dietary_filter):
                    continue
                
                score = self.recommender.matcher.match(all_ingredients, meta["normalized"])
                
                scored_recipes.append({
                    "title": title,
                    "score": float(score),
                    "normalized_ingredients": meta["normalized"],
                    "cleaned_ingredients": meta["cleaned"],
                    "ingredients_raw": meta["ingredients_raw"],
                    "instructions": meta["instructions"],
                    "image_name": meta.get("image_name")
                })
            
            # Sort by score
            ranked = sorted(scored_recipes, key=lambda x: x["score"], reverse=True)
            top = ranked[:top_k]
            
            # Store current recommendations
            self.current_recommendations = top.copy()
            
            # Format as result dict
            result = {
                "fridge_items": all_ingredients,
                "recommendations": top
            }
            
            # Format output
            if not result.get("recommendations"):
                filter_msg = f" with {dietary_filter} filter" if dietary_filter != "None" else ""
                return f"No recipes found matching your ingredients{filter_msg}. Try adding more ingredients or changing the dietary filter!"
            
            # Header with elegant beige styling
            output_lines = [f"<div style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; margin-bottom: 2em; font-family: Georgia, serif;'>"]
            output_lines.append(f"<h2 style='margin: 0; color: #2d2d2d; border: none; padding: 0; font-weight: 400;'>Found {len(result['recommendations'])} Recipe Recommendations</h2>")
            output_lines.append(f"<p style='margin: 0.5em 0 0 0; color: #5a4a3a;'><strong>Detected Ingredients:</strong> {', '.join(result['fridge_items'])}</p>")
            if dietary_filter != "None":
                output_lines.append(f"<p style='margin: 0.5em 0 0 0; color: #5a4a3a;'><strong>Dietary Filter:</strong> {dietary_filter}</p>")
            output_lines.append(f"</div>")
            
            for i, rec in enumerate(result["recommendations"], 1):
                # Recipe box with elegant styling
                output_lines.append(f"<div class='recipe-box' style='margin-bottom: 2em; position: relative;'>")
                output_lines.append(f"<h3 style='margin: 0 0 0.5em 0; color: #2d2d2d; border-bottom: none; padding-bottom: 0; font-weight: 400;'>")
                output_lines.append(f"{i}. {rec['title']}")
                output_lines.append(f"<span class='score-badge'>{rec['score']:.1f}</span>")
                output_lines.append(f"</h3>")
                
                # Ingredients with elegant formatting
                ingredients_list = ', '.join(rec['normalized_ingredients'])
                output_lines.append(f"<p style='color: #3d3d3d; margin: 1em 0;'><strong style='color: #5a4a3a;'>Ingredients:</strong> <span style='color: #8b7355;'>{ingredients_list}</span></p>")
                
                # Collapsible instructions with beige theme
                instructions = rec['instructions']
                if len(instructions) > 300:
                    preview = instructions[:300] + "..."
                    output_lines.append(f"<p style='color: #3d3d3d; margin: 1em 0 0.5em 0;'><strong style='color: #5a4a3a;'>Instructions:</strong></p>")
                    output_lines.append(f"<details>")
                    output_lines.append(f"<summary>{preview}</summary>")
                    output_lines.append(f"<div style='background: #faf8f5; border: 1px solid #e8ddd4; border-radius: 4px; padding: 1.2em; margin-top: 0.5em; color: #3d3d3d; line-height: 1.8; white-space: pre-wrap; font-family: Georgia, serif;'>{instructions}</div>")
                    output_lines.append(f"</details>")
                else:
                    output_lines.append(f"<p style='color: #3d3d3d; margin: 1em 0 0.5em 0;'><strong style='color: #5a4a3a;'>Instructions:</strong></p>")
                    output_lines.append(f"<div style='background: #faf8f5; border: 1px solid #e8ddd4; border-radius: 4px; padding: 1.2em; color: #3d3d3d; line-height: 1.8; white-space: pre-wrap; font-family: Georgia, serif;'>{instructions}</div>")
                
                output_lines.append(f"</div>")
            
            result_text = "\n".join(output_lines)
            
            # Save to history
            self._save_history(
                result['fridge_items'],
                result['recommendations'],
                top_k
            )
            
            return result_text
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Load CSS from external file
        css_path = Path(__file__).parent / "static" / "style.css"
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                custom_css = f.read()
        else:
            # Fallback CSS if file doesn't exist
            custom_css = "/* CSS file not found */"
        
        with gr.Blocks(
            title="SmartPantry - Recipe Recommender",
        ) as interface:
            
            # Inject custom CSS
            gr.HTML(f"<style>{custom_css}</style>")
            
            # Header Section - Elegant beige/tan style
            gr.HTML("""
                <div style="text-align: center; padding: 3em 0; background: #f5f1eb; margin-bottom: 2em; border-bottom: 1px solid #d4c4b0;">
                    <h1 style="margin: 0; color: #8b7355; font-family: Georgia, serif; font-weight: 300; letter-spacing: 0.1em;">SmartPantry</h1>
                    <p style="font-size: 0.9em; color: #5a4a3a; margin-top: 0.5em; letter-spacing: 0.2em; text-transform: uppercase; font-family: Georgia, serif;">AI-Powered Recipe Recommender</p>
                </div>
            """)
            
            gr.Markdown(
                """
                <div style="background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; margin-bottom: 2em; font-family: Georgia, serif; color: #3d3d3d; line-height: 1.8;">
                    <h3 style="color: #5a4a3a; margin-top: 0;">How It Works</h3>
                    <p style="margin: 0;">Upload a photo of your fridge or pantry, and our AI will detect ingredients and recommend personalized recipes based on what you have available.</p>
                </div>
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Image upload section
                    gr.Markdown("### Upload Image")
                    image_input = gr.Image(
                        label="",
                        type="pil",
                        height=400,
                    )
                    
                    # Action buttons
                    with gr.Row():
                        detect_button = gr.Button(
                            "Detect Ingredients",
                            variant="primary",
                            scale=1
                        )
                        recommend_button = gr.Button(
                            "Get Recommendations",
                            variant="primary",
                            scale=1
                        )
                    
                    # Settings section
                    with gr.Column():
                        gr.Markdown("### Settings")
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Recipes",
                        )
                        dietary_filter = gr.Dropdown(
                            choices=["None", "vegan", "vegetarian", "dairy-free", "gluten-free"],
                            value="None",
                            label="Dietary Filter",
                        )
                
                with gr.Column(scale=1):
                    # Detection results section
                    gr.Markdown("### Detected Ingredients")
                    ingredient_output = gr.Markdown(
                        value="<div style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; font-family: Georgia, serif; color: #8b7355; font-style: italic;'>Upload an image and click 'Detect Ingredients' to see detected items.</div>",
                        elem_classes=["markdown-text"]
                    )
                    
                    # Pantry List section - moved to right column
                    gr.Markdown("### My Pantry List")
                    pantry_input = gr.Textbox(
                        label="Enter ingredients",
                        placeholder="Enter ingredients not detected in the image (e.g., milk, eggs, flour). These will be combined with detected ingredients.",
                        lines=2,
                    )
                    pantry_display = gr.Markdown(
                        value=self._format_pantry_display(),
                        elem_classes=["markdown-text"]
                    )
                    with gr.Row():
                        add_to_pantry_button = gr.Button("Add to Pantry", variant="primary", scale=1)
                        clear_pantry_button = gr.Button("Clear Pantry", variant="secondary", scale=1)
            
            # Recipe recommendations section
            gr.Markdown("---")
            gr.Markdown("### Recipe Recommendations")
            recipe_output = gr.Markdown(
                value="<div style='background: #ffffff; border: 1px solid #d4c4b0; border-radius: 4px; padding: 1.5em; font-family: Georgia, serif; color: #8b7355; font-style: italic;'>Upload an image and click 'Get Recommendations' to see recipe suggestions.</div>",
                elem_classes=["markdown-text"]
            )
            
            # History section
            gr.Markdown("---")
            gr.Markdown("### History")
            with gr.Row():
                with gr.Column(scale=3):
                    history_output = gr.Markdown(
                        value=self.get_history_display(),
                        elem_classes=["markdown-text"]
                    )
                with gr.Column(scale=1):
                    with gr.Column():
                        refresh_history_button = gr.Button(
                            "Refresh History",
                            variant="primary",
                        )
                        clear_history_button = gr.Button(
                            "Clear History",
                            variant="secondary",
                        )
            
            # Event handlers
            detect_button.click(
                fn=self.process_image,
                inputs=[image_input],
                outputs=[ingredient_output],
            )
            
            
            add_to_pantry_button.click(
                fn=self._add_to_pantry,
                inputs=[pantry_input],
                outputs=[pantry_display, pantry_input],
            )
            
            clear_pantry_button.click(
                fn=self._clear_pantry,
                outputs=[pantry_display],
            )
            
            refresh_history_button.click(
                fn=lambda: self.get_history_display(),
                outputs=history_output,
            )
            
            clear_history_button.click(
                fn=self._clear_history,
                outputs=history_output,
            )
            
            recommend_button.click(
                fn=self.get_recipe_recommendations,
                inputs=[image_input, pantry_input, top_k, dietary_filter],
                outputs=recipe_output,
            ).then(
                fn=lambda: self.get_history_display(),
                outputs=history_output,
            )
        
        return interface
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
    ):
        """
        Launch the Gradio application.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
        """
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=False,
        )


def main():
    """Main entry point for the application."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize and launch app
    app = SmartPantryApp()
    app.launch()


if __name__ == "__main__":
    main()
