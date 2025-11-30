"""
Image preprocessing and ingredient name normalization.
Simplified version for compatibility with app/main.py
"""

import json
import os
from rapidfuzz import process, fuzz
import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple
import numpy as np
from PIL import Image
import cv2

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
        fuzzy_threshold: int = 75,
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

        # Load unified canonical ingredient vocabulary
        self.ingredient_vocab = self._load_ingredient_vocabulary()

        logger.info("ImagePreprocessor initialized")

    # ------------------------------------------------------------
    # Load canonical vocabulary
    # ------------------------------------------------------------
    def _load_ingredient_vocabulary(self):
        """
        Load canonical ingredient vocabulary stored in:
        project_root/data/canonical_vocab.json

        Returns:
            List of normalized ingredient tokens
        """
        vocab_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "data", "canonical_vocab.json"
        )

        vocab_path = os.path.abspath(vocab_path)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"canonical_vocab.json not found at {vocab_path}"
            )

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # clean + standardize
        vocab = [v.lower().strip() for v in vocab if isinstance(v, str)]

        return vocab

    # ------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------
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
        """
        img = self.load_image(image)

        if resize:
            img = img.resize(self.target_size, Image.LANCZOS)

        img_array = np.array(img)

        if normalize:
            img_array = img_array.astype(np.float32) / 255.0

        return img_array

    # ------------------------------------------------------------
    # Ingredient normalization
    # ------------------------------------------------------------
    def normalize_ingredient_names(
        self,
        ingredients: List[str],
    ) -> List[str]:
        """
        Normalize ingredient names using fuzzy matching into canonical vocab.

        Args:
            ingredients: List of raw ingredient names from detection

        Returns:
            List of normalized ingredient names belonging ONLY to canonical vocab
        """
        normalized = []

        for ingredient in ingredients:
            if ingredient is None or not isinstance(ingredient, str):
                continue

            raw = ingredient.lower().strip()
            if not raw:
                continue

            # fuzzy normalization
            if self.normalization_method == "fuzzy":
                match, score = process.extractOne(
                    raw,
                    self.ingredient_vocab,
                    scorer=fuzz.ratio,
                )

                # accept only if confident match
                if match is not None and score >= self.fuzzy_threshold:
                    normalized.append(match)
                # else: skip low-confidence noise

            # exact matching mode
            elif self.normalization_method == "exact":
                raw = raw.lower()
                if raw in self.ingredient_vocab:
                    normalized.append(raw)
                # else skip

            # fallback
            else:
                continue

        return normalized

    # ------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------
    def remove_duplicates(
        self,
        detections: Dict,
    ) -> Dict:
        """
        Remove duplicate ingredient detections by keeping highest confidence.
        """
        ingredients = detections.get("ingredients", [])
        confidences = detections.get("confidences", [])

        seen = {}
        for ing, conf in zip(ingredients, confidences):
            if ing not in seen or conf > seen[ing]:
                seen[ing] = conf

        dedup_ing = list(seen.keys())
        dedup_conf = list(seen.values())

        return {"ingredients": dedup_ing, "confidences": dedup_conf}

    # ------------------------------------------------------------
    # Aggregation from multiple images
    # ------------------------------------------------------------
    def aggregate_ingredients(
        self,
        batch_detections: List[Dict],
    ) -> Dict:
        """
        Aggregate ingredient detections from multiple images
        and deduplicate.
        """
        all_ingredients = []
        all_confidences = []

        for detection in batch_detections:
            all_ingredients.extend(detection.get("ingredients", []))
            all_confidences.extend(detection.get("confidences", []))

        aggregated = {
            "ingredients": all_ingredients,
            "confidences": all_confidences,
        }

        return self.remove_duplicates(aggregated)

    # ------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------
    def apply_filters(
        self,
        ingredients: List[str],
        blacklist: List[str] = None,
    ) -> List[str]:
        """
        Apply blacklist filtering to ingredient list.
        """
        if blacklist is None:
            blacklist = []

        blacklist = {b.lower().strip() for b in blacklist}
        return [ing for ing in ingredients if ing.lower() not in blacklist]
