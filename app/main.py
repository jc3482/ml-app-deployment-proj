"""
Main Gradio application for SmartPantry.
User interface for ingredient detection and recipe recommendation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import gradio as gr
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from src.vision.detector import IngredientDetector
from src.vision.preprocessor import ImagePreprocessor
from src.nlp.embedder import IngredientEmbedder
from src.nlp.retriever import RecipeRetriever
from src.utils.helpers import load_config, setup_logging, format_recipe_card
from src.utils.clustering import RecipeClustering

# Setup logging
logger = setup_logging(level="INFO", log_file="logs/app.log")


class SmartPantryApp:
    """
    Main application class for SmartPantry.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the SmartPantry application.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing SmartPantry application")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("SmartPantry application initialized successfully")
    
    def _initialize_components(self):
        """
        Initialize all application components.
        
        TODO: Load actual models
        - Initialize YOLOv8 detector
        - Load embedding model
        - Load recipe database and FAISS index
        - Initialize clustering
        """
        detection_config = self.config.get("detection", {})
        embeddings_config = self.config.get("embeddings", {})
        retrieval_config = self.config.get("retrieval", {})
        clustering_config = self.config.get("clustering", {})
        
        # Initialize detector
        self.detector = IngredientDetector(
            model_path=self.config["paths"].get("yolo_weights", "yolov8m.pt"),
            confidence_threshold=detection_config.get("confidence_threshold", 0.25),
            iou_threshold=detection_config.get("iou_threshold", 0.45),
            device=detection_config.get("device", "cuda"),
            image_size=detection_config.get("image_size", 640),
        )
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            target_size=(detection_config.get("image_size", 640),
                        detection_config.get("image_size", 640)),
            normalization_method=self.config["ingredients"].get("normalization_method", "fuzzy"),
            fuzzy_threshold=self.config["ingredients"].get("fuzzy_threshold", 85),
        )
        
        # Initialize embedder
        self.embedder = IngredientEmbedder(
            model_type=embeddings_config.get("model_type", "sentence-bert"),
            model_name=embeddings_config.get("model_name", "all-MiniLM-L6-v2"),
            device=embeddings_config.get("device", "cuda"),
            max_length=embeddings_config.get("max_length", 128),
            normalize=embeddings_config.get("normalize", True),
        )
        
        # Initialize retriever
        self.retriever = RecipeRetriever(
            recipe_database_path=self.config["paths"].get("recipes_db"),
            index_path=None,  # Will be built if not exists
            index_type=retrieval_config.get("index_type", "IVFFlat"),
            metric=retrieval_config.get("metric", "cosine"),
            top_k=retrieval_config.get("top_k", 50),
            min_ingredient_match=retrieval_config.get("min_ingredient_match", 0.3),
        )
        
        # Initialize clustering (optional)
        if clustering_config.get("enable", False):
            self.clustering = RecipeClustering(
                method=clustering_config.get("method", "kmeans"),
                n_clusters=clustering_config.get("n_clusters", 5),
                features=clustering_config.get("features", ["cuisine", "difficulty"]),
            )
        else:
            self.clustering = None
        
        logger.info("All components initialized")
    
    def process_image(
        self,
        image: Image.Image,
        show_visualization: bool = True,
    ) -> Tuple[str, Optional[Image.Image], List[str]]:
        """
        Process uploaded image and detect ingredients.
        
        Args:
            image: Input image from user
            show_visualization: Whether to return visualization
            
        Returns:
            Tuple of (ingredient_text, visualization_image, ingredient_list)
            
        TODO: Implement full pipeline
        - Run YOLO detection
        - Normalize ingredient names
        - Remove duplicates
        - Create visualization
        """
        logger.info("Processing uploaded image")
        
        try:
            # Run detection
            detections = self.detector.detect_ingredients(image, visualize=show_visualization)
            
            # Normalize ingredient names
            ingredients = self.preprocessor.normalize_ingredient_names(
                detections.get("ingredients", [])
            )
            
            # Remove duplicates
            detections["ingredients"] = ingredients
            detections = self.preprocessor.remove_duplicates(detections)
            
            # Format output
            ingredient_list = detections.get("ingredients", [])
            if ingredient_list:
                ingredient_text = "**Detected Ingredients:**\n" + "\n".join(
                    f"- {ing} ({conf:.2%})"
                    for ing, conf in zip(
                        ingredient_list,
                        detections.get("confidences", [1.0] * len(ingredient_list))
                    )
                )
            else:
                ingredient_text = "No ingredients detected. Please try another image."
            
            visualization = detections.get("visualization") if show_visualization else None
            
            return ingredient_text, visualization, ingredient_list
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error: {str(e)}", None, []
    
    def get_recipe_recommendations(
        self,
        ingredients: List[str],
        max_results: int = 10,
        cuisine_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        max_cooking_time: Optional[int] = None,
    ) -> str:
        """
        Get recipe recommendations based on detected ingredients.
        
        Args:
            ingredients: List of detected ingredient names
            max_results: Maximum number of recipes to return
            cuisine_filter: Optional cuisine filter
            difficulty_filter: Optional difficulty filter
            max_cooking_time: Optional maximum cooking time
            
        Returns:
            Formatted recipe recommendations
            
        TODO: Implement recommendation pipeline
        - Embed ingredients
        - Retrieve similar recipes
        - Apply filters
        - Rank recipes
        - Format output
        """
        logger.info(f"Getting recommendations for {len(ingredients)} ingredients")
        
        if not ingredients:
            return "Please upload an image first to detect ingredients."
        
        try:
            # Embed ingredients
            ingredient_text = ", ".join(ingredients)
            query_embedding = self.embedder.embed_ingredients(ingredient_text)
            
            # Build filters
            filters = {}
            if cuisine_filter and cuisine_filter != "Any":
                filters["cuisine"] = [cuisine_filter]
            if difficulty_filter and difficulty_filter != "Any":
                filters["difficulty"] = [difficulty_filter]
            if max_cooking_time:
                filters["max_cooking_time"] = max_cooking_time
            
            # Retrieve recipes
            recipes = self.retriever.retrieve_recipes(
                query_embedding=query_embedding,
                detected_ingredients=ingredients,
                top_k=max_results,
                filters=filters if filters else None,
            )
            
            # Rank recipes
            ranked_recipes = self.retriever.rank_recipes(recipes)
            
            # Cluster recipes (optional)
            if self.clustering and len(ranked_recipes) > 0:
                clustered = self.clustering.cluster_recipes(ranked_recipes)
                # For now, just use ranked recipes
                # TODO: Organize by clusters
            
            # Format output
            if not ranked_recipes:
                return "No recipes found matching your ingredients. Try adding more ingredients!"
            
            output_lines = [f"# Found {len(ranked_recipes)} Recipes\n"]
            
            for i, recipe in enumerate(ranked_recipes[:max_results], 1):
                output_lines.append(f"## {i}. {format_recipe_card(recipe)}")
                output_lines.append("")  # Empty line between recipes
            
            return "\n".join(output_lines)
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return f"Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        gradio_config = self.config.get("gradio", {})
        
        with gr.Blocks(
            title=gradio_config.get("title", "SmartPantry"),
            theme=gradio_config.get("theme", "default"),
        ) as interface:
            
            gr.Markdown(
                f"""
                # {gradio_config.get('title', 'SmartPantry')}
                {gradio_config.get('description', 'Upload a photo of your fridge and get personalized recipe recommendations!')}
                
                ### How to use:
                1. Upload a photo of your fridge or pantry
                2. Review detected ingredients
                3. Adjust filters (optional)
                4. Get recipe recommendations
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Image upload
                    image_input = gr.Image(
                        label="Upload Fridge/Pantry Photo",
                        type="pil",
                        height=400,
                    )
                    
                    # Detection button
                    detect_button = gr.Button("Detect Ingredients", variant="primary")
                    
                    # Filters
                    gr.Markdown("### Filters (Optional)")
                    
                    cuisine_filter = gr.Dropdown(
                        choices=["Any"] + self.config.get("clustering", {}).get("cuisine_types", []),
                        value="Any",
                        label="Cuisine",
                    )
                    
                    difficulty_filter = gr.Dropdown(
                        choices=["Any"] + self.config.get("clustering", {}).get("difficulty_levels", []),
                        value="Any",
                        label="Difficulty",
                    )
                    
                    max_time = gr.Slider(
                        minimum=10,
                        maximum=180,
                        value=60,
                        step=10,
                        label="Max Cooking Time (minutes)",
                    )
                    
                    max_recipes = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Recipes",
                    )
                    
                    # Recommend button
                    recommend_button = gr.Button("Get Recipes", variant="primary")
                
                with gr.Column(scale=1):
                    # Detection results
                    gr.Markdown("### Detected Ingredients")
                    ingredient_output = gr.Markdown()
                    detection_viz = gr.Image(label="Detection Visualization")
            
            # Recipe recommendations
            gr.Markdown("---")
            gr.Markdown("### Recipe Recommendations")
            recipe_output = gr.Markdown()
            
            # Store ingredients in state
            ingredients_state = gr.State([])
            
            # Event handlers
            detect_button.click(
                fn=self.process_image,
                inputs=[image_input],
                outputs=[ingredient_output, detection_viz, ingredients_state],
            )
            
            recommend_button.click(
                fn=self.get_recipe_recommendations,
                inputs=[
                    ingredients_state,
                    max_recipes,
                    cuisine_filter,
                    difficulty_filter,
                    max_time,
                ],
                outputs=recipe_output,
            )
            
            # Examples (optional)
            gr.Markdown("---")
            gr.Markdown("### 📝 Examples")
            gr.Markdown(
                "Try uploading a photo of your fridge! If you don't have one, "
                "you can use stock photos of refrigerators or ingredient layouts."
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
        gradio_config = self.config.get("gradio", {})
        
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        
        interface.launch(
            share=share or gradio_config.get("share", False),
            server_name=server_name or gradio_config.get("server_name", "0.0.0.0"),
            server_port=server_port or gradio_config.get("server_port", 7860),
            debug=gradio_config.get("debug", False),
        )


def main():
    """Main entry point for the application."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize and launch app
    app = SmartPantryApp(config_path="config.yaml")
    app.launch()


if __name__ == "__main__":
    main()

