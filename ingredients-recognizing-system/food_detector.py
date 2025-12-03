"""
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
