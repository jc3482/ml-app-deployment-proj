"""
Smart Refrigerator Food Detection Model Training
Professional YOLOv8 model training using Roboflow dataset

SETUP:
    1. Set your Roboflow API key as environment variable:
       export ROBOFLOW_API_KEY='your_api_key_here'
    
    OR pass it directly when running:
       python food_model_trainer.py --api-key your_api_key_here

USAGE:
    python food_model_trainer.py
"""

import os
import yaml
import glob
import shutil
import torch
from pathlib import Path


# ============================================
# Configuration
# ============================================

ROBOFLOW_CONFIG = {
    'api_key': os.getenv('ROBOFLOW_API_KEY', ''),  # Set via environment variable
    'workspace': 'northumbria-university-newcastle',
    'project': 'smart-refrigerator-zryjr',
    'version': 2
}

TRAINING_CONFIG = {
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'name': 'smart_fridge_v1',
    'project': 'runs/detect',
    'patience': 20,
    'save': True,
    'workers': 2,
    'optimizer': 'AdamW',
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'plots': True,
    'verbose': True
}


# ============================================
# Step 1: Download Dataset from Roboflow
# ============================================

def download_dataset(api_key=None):
    """
    Download dataset from Roboflow platform
    
    Args:
        api_key: Roboflow API key (optional, will use environment variable if not provided)
    
    Returns:
        Dataset object with location information
    """
    print("=" * 70)
    print("STEP 1: Downloading Dataset from Roboflow")
    print("=" * 70)
    
    from roboflow import Roboflow
    
    # Use provided API key or environment variable
    key = api_key or ROBOFLOW_CONFIG['api_key']
    
    if not key:
        raise ValueError(
            "Roboflow API key not found. Please either:\n"
            "  1. Set environment variable: export ROBOFLOW_API_KEY='your_key'\n"
            "  2. Pass api_key parameter: download_dataset(api_key='your_key')"
        )
    
    rf = Roboflow(api_key=key)
    project = rf.workspace(ROBOFLOW_CONFIG['workspace']).project(ROBOFLOW_CONFIG['project'])
    version = project.version(ROBOFLOW_CONFIG['version'])
    dataset = version.download("yolov8")
    
    print(f"Dataset downloaded successfully")
    print(f"Location: {dataset.location}")
    
    return dataset


# ============================================
# Step 2: Verify Dataset Structure
# ============================================

def verify_dataset(dataset):
    """
    Verify dataset structure and display statistics
    
    Args:
        dataset: Downloaded dataset object
    
    Returns:
        Tuple of (dataset_path, data_yaml_path, data_config)
    """
    print("\n" + "=" * 70)
    print("STEP 2: Verifying Dataset Structure")
    print("=" * 70)
    
    dataset_path = dataset.location
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset Information:")
    print(f"  Path: {dataset_path}")
    print(f"  Number of classes: {data_config.get('nc', 'Unknown')}")
    print(f"  Class names: {data_config.get('names', [])}")
    
    # Count samples
    train_images = glob.glob(f"{dataset_path}/train/images/*.jpg") + \
                   glob.glob(f"{dataset_path}/train/images/*.png")
    val_images = glob.glob(f"{dataset_path}/valid/images/*.jpg") + \
                 glob.glob(f"{dataset_path}/valid/images/*.png")
    test_images = glob.glob(f"{dataset_path}/test/images/*.jpg") + \
                  glob.glob(f"{dataset_path}/test/images/*.png") if os.path.exists(f"{dataset_path}/test") else []
    
    print(f"\nSample Statistics:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} images")
    
    return dataset_path, data_yaml_path, data_config


# ============================================
# Step 3: Train Model
# ============================================

def train_model(data_yaml_path, model_name='yolov8s.pt'):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml_path: Path to dataset configuration file
        model_name: Pretrained model to use
    
    Returns:
        Trained model object
    """
    print("\n" + "=" * 70)
    print("STEP 3: Training YOLOv8 Model")
    print("=" * 70)
    
    from ultralytics import YOLO
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    
    # Load pretrained model
    model = YOLO(model_name)
    
    # Update training config with data and device
    train_config = TRAINING_CONFIG.copy()
    train_config['data'] = data_yaml_path
    train_config['device'] = device
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Image size: {train_config['imgsz']}")
    print(f"  Batch size: {train_config['batch']}")
    print(f"  Optimizer: {train_config['optimizer']}")
    print(f"  Initial learning rate: {train_config['lr0']}")
    
    print("\nStarting training...")
    print("=" * 70)
    
    # Train model
    results = model.train(**train_config)
    
    print("\nTraining completed successfully")
    
    return model


# ============================================
# Step 4: Validate Model
# ============================================

def validate_model(model_path='runs/detect/smart_fridge_v1/weights/best.pt'):
    """
    Validate trained model on validation set
    
    Args:
        model_path: Path to trained model weights
    
    Returns:
        Validation metrics
    """
    print("\n" + "=" * 70)
    print("STEP 4: Validating Model")
    print("=" * 70)
    
    from ultralytics import YOLO
    
    # Load best model
    model = YOLO(model_path)
    
    # Validate on validation set
    metrics = model.val()
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


# ============================================
# Step 5: Save Detector Class
# ============================================

def save_detector_class(output_dir='.'):
    """
    Save FoodDetector wrapper class
    
    Args:
        output_dir: Directory to save the file
    """
    print("\n" + "=" * 70)
    print("STEP 5: Saving FoodDetector Class")
    print("=" * 70)
    
    detector_code = '''"""
Food Detector Class
High-level wrapper for food ingredient detection using YOLOv8
"""

from ultralytics import YOLO
from pathlib import Path


class FoodDetector:
    """
    Smart Refrigerator Food Detector
    
    Example:
        detector = FoodDetector('best.pt', conf_threshold=0.85)
        ingredients = detector.detect('image.jpg')
    """
    
    def __init__(self, model_path='best.pt', conf_threshold=0.85):
        """
        Initialize the food detector
        
        Args:
            model_path: Path to YOLOv8 model weights (.pt file)
            conf_threshold: Confidence threshold for detection (0-1)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        
        print(f"Model loaded: {model_path}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Confidence threshold: {conf_threshold}")
    
    def detect(self, image_path, conf_threshold=None):
        """
        Detect food ingredients and return unique names
        
        Args:
            image_path: Path to image file
            conf_threshold: Optional confidence threshold override
        
        Returns:
            list: Sorted list of unique ingredient names
                  Example: ['Apple', 'Banana', 'Tomato']
        """
        conf = conf_threshold or self.conf_threshold
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        detected = [self.class_names[int(box.cls[0])] for box in results[0].boxes]
        return sorted(list(set(detected)))
    
    def detect_with_confidence(self, image_path, conf_threshold=None):
        """
        Detect food ingredients with confidence scores
        
        Args:
            image_path: Path to image file
            conf_threshold: Optional confidence threshold override
        
        Returns:
            list: List of tuples (ingredient_name, confidence)
                  Example: [('Apple', 0.95), ('Banana', 0.92)]
        """
        conf = conf_threshold or self.conf_threshold
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        detected = [(self.class_names[int(box.cls[0])], float(box.conf[0]))
                   for box in results[0].boxes]
        return sorted(detected, key=lambda x: x[1], reverse=True)
    
    def detect_with_count(self, image_path, conf_threshold=None):
        """
        Detect food ingredients with count statistics
        
        Args:
            image_path: Path to image file
            conf_threshold: Optional confidence threshold override
        
        Returns:
            dict: Dictionary mapping ingredient names to counts
                  Example: {'Apple': 2, 'Banana': 1}
        """
        conf = conf_threshold or self.conf_threshold
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        counts = {}
        for box in results[0].boxes:
            name = self.class_names[int(box.cls[0])]
            counts[name] = counts.get(name, 0) + 1
        return counts
'''
    
    output_path = Path(output_dir) / 'food_detector.py'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(detector_code)
    
    print(f"Saved: {output_path}")


# ============================================
# Main Execution
# ============================================

def main():
    """
    Main training pipeline
    """
    import sys
    
    print("\n" + "=" * 70)
    print("SMART REFRIGERATOR FOOD DETECTION TRAINING")
    print("=" * 70)
    
    # Check for API key from command line arguments
    api_key = None
    if '--api-key' in sys.argv:
        idx = sys.argv.index('--api-key')
        if idx + 1 < len(sys.argv):
            api_key = sys.argv[idx + 1]
    
    # Install roboflow if needed
    try:
        import roboflow
    except ImportError:
        print("\nInstalling roboflow package...")
        os.system('pip install roboflow -q')
    
    # Install ultralytics if needed
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics package...")
        os.system('pip install ultralytics -q')
    
    # Step 1: Download dataset
    try:
        dataset = download_dataset(api_key)
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    # Step 2: Verify dataset
    dataset_path, data_yaml_path, data_config = verify_dataset(dataset)
    
    # Step 3: Train model
    model = train_model(data_yaml_path)
    
    # Step 4: Validate model
    best_model_path = 'runs/detect/smart_fridge_v1/weights/best.pt'
    metrics = validate_model(best_model_path)
    
    # Step 5: Save detector class
    save_detector_class()
    
    # Copy best model to current directory
    if Path(best_model_path).exists():
        shutil.copy(best_model_path, './best.pt')
        print(f"\nModel copied to: ./best.pt")
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - best.pt (trained model)")
    print("  - food_detector.py (detector class)")
    print("\nUsage:")
    print("  from food_detector import FoodDetector")
    print("  detector = FoodDetector('best.pt')")
    print("  result = detector.detect('image.jpg')")


if __name__ == "__main__":
    main()
