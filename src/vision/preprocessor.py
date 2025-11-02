"""
Image preprocessing and ingredient name normalization.
"""

import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple
import numpy as np
from PIL import Image
import cv2
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing and ingredient name normalization.
    
    Features:
    - Image loading and format conversion
    - Resizing and augmentation
    - Ingredient name normalization (fuzzy matching)
    - Duplicate removal and aggregation
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        normalization_method: str = "fuzzy",
        fuzzy_threshold: int = 85,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            normalization_method: Method for ingredient name normalization
            fuzzy_threshold: Threshold for fuzzy string matching
        """
        self.target_size = target_size
        self.normalization_method = normalization_method
        self.fuzzy_threshold = fuzzy_threshold
        
        # Ingredient vocabulary (will be loaded from file or database)
        self.ingredient_vocab = self._load_ingredient_vocabulary()
        
        logger.info("ImagePreprocessor initialized")
    
    def _load_ingredient_vocabulary(self) -> List[str]:
        """
        Load ingredient vocabulary for normalization.
        
        TODO: Load from file or database
        - Common ingredient names
        - Synonyms and variations
        - Categorizations
        
        Returns:
            List of standardized ingredient names
        """
        # TODO: Load from actual vocabulary file
        # Placeholder vocabulary
        vocab = [
            "milk", "eggs", "cheese", "butter", "yogurt",
            "chicken", "beef", "pork", "fish", "tofu",
            "tomato", "lettuce", "carrot", "onion", "garlic",
            "apple", "banana", "orange", "strawberry", "grape",
            "bread", "rice", "pasta", "flour", "sugar",
        ]
        
        return vocab
    
    def load_image(
        self,
        image_source: Union[str, Path, Image.Image, np.ndarray],
    ) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            image_source: Image path, PIL Image, or numpy array
            
        Returns:
            PIL Image object
        """
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        elif isinstance(image_source, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
        
        return image
    
    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        resize: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image
            resize: Whether to resize image
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image as numpy array
            
        TODO: Implement preprocessing pipeline
        - Load image
        - Resize to target size
        - Normalize pixel values
        - Convert to appropriate format
        """
        # Load image
        img = self.load_image(image)
        
        # Resize if needed
        if resize:
            img = img.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize if needed
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def normalize_ingredient_names(
        self,
        ingredients: List[str],
    ) -> List[str]:
        """
        Normalize ingredient names using fuzzy matching.
        
        Args:
            ingredients: List of raw ingredient names from detection
            
        Returns:
            List of normalized ingredient names
            
        TODO: Implement normalization strategies
        - Fuzzy string matching against vocabulary
        - Handle plurals and variations
        - Map to canonical names
        """
        normalized = []
        
        for ingredient in ingredients:
            if self.normalization_method == "fuzzy":
                # Fuzzy match against vocabulary
                match, score = process.extractOne(
                    ingredient.lower(),
                    self.ingredient_vocab,
                    scorer=fuzz.ratio,
                )
                
                if score >= self.fuzzy_threshold:
                    normalized.append(match)
                else:
                    # Keep original if no good match found
                    normalized.append(ingredient.lower())
            
            elif self.normalization_method == "exact":
                # Exact matching
                ingredient_lower = ingredient.lower()
                if ingredient_lower in self.ingredient_vocab:
                    normalized.append(ingredient_lower)
                else:
                    normalized.append(ingredient_lower)
            
            else:
                # No normalization
                normalized.append(ingredient.lower())
        
        return normalized
    
    def remove_duplicates(
        self,
        detections: Dict,
        threshold: float = 0.9,
    ) -> Dict:
        """
        Remove duplicate ingredient detections.
        
        Args:
            detections: Detection results with ingredients and confidences
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            Deduplicated detection results
            
        TODO: Implement deduplication logic
        - Check for duplicate ingredient names
        - Compare bounding boxes for spatial duplicates
        - Keep detection with highest confidence
        """
        ingredients = detections.get("ingredients", [])
        confidences = detections.get("confidences", [])
        
        # Simple deduplication by name
        seen = {}
        unique_ingredients = []
        unique_confidences = []
        
        for ing, conf in zip(ingredients, confidences):
            if ing not in seen or conf > seen[ing]:
                seen[ing] = conf
        
        for ing, conf in seen.items():
            unique_ingredients.append(ing)
            unique_confidences.append(conf)
        
        detections["ingredients"] = unique_ingredients
        detections["confidences"] = unique_confidences
        
        return detections
    
    def aggregate_ingredients(
        self,
        batch_detections: List[Dict],
    ) -> Dict:
        """
        Aggregate ingredients from multiple images.
        
        Args:
            batch_detections: List of detection results from multiple images
            
        Returns:
            Aggregated detection results
            
        TODO: Implement aggregation strategy
        - Combine ingredients from all images
        - Handle duplicates across images
        - Compute aggregate confidence scores
        """
        all_ingredients = []
        all_confidences = []
        
        for detection in batch_detections:
            all_ingredients.extend(detection.get("ingredients", []))
            all_confidences.extend(detection.get("confidences", []))
        
        # Create aggregated result
        aggregated = {
            "ingredients": all_ingredients,
            "confidences": all_confidences,
        }
        
        # Remove duplicates
        aggregated = self.remove_duplicates(aggregated)
        
        return aggregated
    
    def apply_filters(
        self,
        ingredients: List[str],
        blacklist: List[str] = None,
    ) -> List[str]:
        """
        Apply filtering to ingredient list.
        
        Args:
            ingredients: List of ingredient names
            blacklist: List of ingredients to exclude
            
        Returns:
            Filtered ingredient list
        """
        if blacklist is None:
            blacklist = []
        
        filtered = [ing for ing in ingredients if ing.lower() not in blacklist]
        
        return filtered

