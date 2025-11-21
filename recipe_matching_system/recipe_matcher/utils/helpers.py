"""
Data loading and saving utilities.
Handles file I/O for raw and processed recipe datasets.
"""
import json
import pandas as pd
import ast
from pathlib import Path


# Path Configuration
DATA_DIR = Path("data")
RAW_CSV_PATH = DATA_DIR / "recipe_dataset_raw.csv"
NORMALIZED_JSON_PATH = DATA_DIR / "normalized_recipes.json"
NORMALIZED_CSV_PATH = DATA_DIR / "normalized_recipes.csv"
ONTOLOGY_JSON_PATH = DATA_DIR / "ontology_recipes.json"
ONTOLOGY_CSV_PATH = DATA_DIR / "ontology_recipes.csv"


# =============================================================================
# Load Raw Recipe Dataset
# =============================================================================

def load_raw_recipes(path=None):
    """
    Load raw recipe dataset from CSV.
    
    Args:
        path: Optional path to CSV file. Uses default if None.
    
    Returns:
        DataFrame with original columns (lowercase with underscores)
    """
    if path is None:
        path = RAW_CSV_PATH
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Raw recipe file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Standardize column names: lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(':', '')
    
    return df


# =============================================================================
# Load Normalized Recipe Dataset
# =============================================================================

def load_normalized_recipes(path=None):
    """
    Load normalized recipe dataset.
    Auto-detects JSON or CSV format.
    
    Args:
        path: Optional path. Auto-detects if None.
    
    Returns:
        DataFrame with 'normalized_ingredients' column as list
    """
    # Auto-detect
    if path is None:
        if NORMALIZED_JSON_PATH.exists():
            path = NORMALIZED_JSON_PATH
        elif NORMALIZED_CSV_PATH.exists():
            path = NORMALIZED_CSV_PATH
        else:
            raise FileNotFoundError(
                f"Normalized recipe dataset not found. "
                f"Expected: {NORMALIZED_JSON_PATH} or {NORMALIZED_CSV_PATH}"
            )
    
    path = Path(path)
    
    # Load based on file type
    if path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Convert normalized_ingredients from string to list if needed
    if "normalized_ingredients" in df.columns:
        if isinstance(df["normalized_ingredients"].iloc[0], str):
            try:
                df["normalized_ingredients"] = df["normalized_ingredients"].apply(ast.literal_eval)
            except:
                print("Warning: Could not parse normalized_ingredients column")
    
    return df


# =============================================================================
# Load Ontology-Processed Recipe Dataset
# =============================================================================

def load_ontology_recipes(path=None):
    """
    Load ontology-processed recipe dataset.
    
    Args:
        path: Optional path. Auto-detects if None.
    
    Returns:
        DataFrame with 'ontology_ingredients' column
    """
    if path is None:
        if ONTOLOGY_JSON_PATH.exists():
            path = ONTOLOGY_JSON_PATH
        elif ONTOLOGY_CSV_PATH.exists():
            path = ONTOLOGY_CSV_PATH
        else:
            raise FileNotFoundError(
                f"Ontology recipe dataset not found. "
                f"Expected: {ONTOLOGY_JSON_PATH} or {ONTOLOGY_CSV_PATH}"
            )
    
    path = Path(path)
    
    if path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    # Convert ontology_ingredients from string to list if needed
    if "ontology_ingredients" in df.columns:
        if isinstance(df["ontology_ingredients"].iloc[0], str):
            try:
                df["ontology_ingredients"] = df["ontology_ingredients"].apply(ast.literal_eval)
            except:
                print("Warning: Could not parse ontology_ingredients column")
    
    return df


# =============================================================================
# Save Datasets
# =============================================================================

def save_normalized_recipes(df, save_json=True, save_csv=True):
    """
    Save normalized recipe dataset.
    
    Args:
        df: DataFrame with normalized_ingredients column
        save_json: Save JSON format
        save_csv: Save CSV format
    
    Returns:
        Dict with paths to saved files
    """
    saved_files = {}
    DATA_DIR.mkdir(exist_ok=True)
    
    if save_json:
        df.to_json(NORMALIZED_JSON_PATH, orient='records', indent=2)
        saved_files['json'] = str(NORMALIZED_JSON_PATH)
        print(f"Saved: {NORMALIZED_JSON_PATH}")
    
    if save_csv:
        df.to_csv(NORMALIZED_CSV_PATH, index=False)
        saved_files['csv'] = str(NORMALIZED_CSV_PATH)
        print(f"Saved: {NORMALIZED_CSV_PATH}")
    
    return saved_files


def save_ontology_recipes(df, save_json=True, save_csv=True):
    """
    Save ontology-processed recipe dataset.
    
    Args:
        df: DataFrame with ontology_ingredients column
        save_json: Save JSON format
        save_csv: Save CSV format
    
    Returns:
        Dict with paths to saved files
    """
    saved_files = {}
    DATA_DIR.mkdir(exist_ok=True)
    
    if save_json:
        df.to_json(ONTOLOGY_JSON_PATH, orient='records', indent=2)
        saved_files['json'] = str(ONTOLOGY_JSON_PATH)
        print(f"Saved: {ONTOLOGY_JSON_PATH}")
    
    if save_csv:
        df.to_csv(ONTOLOGY_CSV_PATH, index=False)
        saved_files['csv'] = str(ONTOLOGY_CSV_PATH)
        print(f"Saved: {ONTOLOGY_CSV_PATH}")
    
    return saved_files


def save_json(data, path):
    """
    Save data to JSON file.
    
    Args:
        data: Python object (list, dict, etc.)
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved: {path}")
