from ultralytics import YOLO
import os

class FoodDetector:
    def __init__(self, weights_path="best.pt", conf_threshold=0.6):
        here = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(here, weights_path)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

        self.model = YOLO(weights_path)
        self.class_names = self.model.names
        self.conf = conf_threshold

    def detect(self, image_path):
        results = self.model(image_path)[0]
        names = [
            self.class_names[int(b.cls)]
            for b in results.boxes
            if float(b.conf) >= self.conf
        ]
        return names

    def detect_with_confidence(self, image_path):
        results = self.model(image_path)[0]
        data = [
            (self.class_names[int(b.cls)], float(b.conf))
            for b in results.boxes
            if float(b.conf) >= self.conf
        ]
        return data

    def detect_with_count(self, image_path):
        detections = self.detect(image_path)
        freq = {}
        for d in detections:
            freq[d] = freq.get(d, 0) + 1
        return freq
