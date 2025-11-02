"""
Helper utilities for configuration, logging, and common operations.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger("SmartPantry")
    logger.info(f"Logging initialized at level {level}")
    
    return logger


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories from config.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    
    directories = [
        paths.get("data_root"),
        paths.get("raw_data"),
        paths.get("processed_data"),
        paths.get("recipes_db"),
        paths.get("fridge_photos"),
        paths.get("embeddings_cache"),
        paths.get("models_root"),
        paths.get("logs"),
        paths.get("results"),
    ]
    
    for directory in directories:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)


def format_ingredients_list(ingredients: list) -> str:
    """
    Format ingredient list for display.
    
    Args:
        ingredients: List of ingredient names
        
    Returns:
        Formatted string
    """
    if not ingredients:
        return "No ingredients"
    
    return ", ".join(ingredients)


def format_recipe_card(recipe: Dict) -> str:
    """
    Format recipe as a readable card.
    
    Args:
        recipe: Recipe dictionary
        
    Returns:
        Formatted recipe string
    """
    lines = []
    lines.append(f"📖 {recipe.get('title', 'Untitled Recipe')}")
    lines.append(f"🍴 Cuisine: {recipe.get('cuisine', 'Unknown')}")
    lines.append(f"⏱️  Time: {recipe.get('cooking_time', '?')} minutes")
    lines.append(f"📊 Difficulty: {recipe.get('difficulty', 'Unknown')}")
    
    if "final_score" in recipe:
        lines.append(f"⭐ Match Score: {recipe['final_score']:.2%}")
    
    if "matched_ingredients" in recipe and recipe["matched_ingredients"]:
        lines.append(f"✅ Have: {format_ingredients_list(recipe['matched_ingredients'])}")
    
    if "missing_ingredients" in recipe and recipe["missing_ingredients"]:
        lines.append(f"❌ Need: {format_ingredients_list(recipe['missing_ingredients'])}")
    
    return "\n".join(lines)


def parse_ingredient_string(ingredient_str: str) -> list:
    """
    Parse ingredient string into list.
    
    Args:
        ingredient_str: Comma or newline separated ingredient string
        
    Returns:
        List of ingredients
    """
    if not ingredient_str:
        return []
    
    # Try comma separation first
    if "," in ingredient_str:
        ingredients = [ing.strip() for ing in ingredient_str.split(",")]
    # Try newline separation
    elif "\n" in ingredient_str:
        ingredients = [ing.strip() for ing in ingredient_str.split("\n")]
    else:
        ingredients = [ingredient_str.strip()]
    
    # Remove empty strings
    ingredients = [ing for ing in ingredients if ing]
    
    return ingredients


def calculate_match_percentage(matched: int, total: int) -> float:
    """
    Calculate match percentage.
    
    Args:
        matched: Number of matched items
        total: Total number of items
        
    Returns:
        Match percentage (0-100)
    """
    if total == 0:
        return 0.0
    
    return (matched / total) * 100


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    import torch
    
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    return info


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ["project", "paths", "detection", "embeddings", "retrieval"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    return True

