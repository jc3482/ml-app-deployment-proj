"""
Unit tests for ingredient detector.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.detector import IngredientDetector


class TestIngredientDetector:
    """Test cases for IngredientDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return IngredientDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.25,
            device="cpu",
        )
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.confidence_threshold == 0.25
        assert detector.device == "cpu"
    
    def test_get_model_info(self, detector):
        """Test model info retrieval."""
        info = detector.get_model_info()
        assert "device" in info
        assert "confidence_threshold" in info
        assert info["device"] == "cpu"
    
    # TODO: Add more tests
    # - test_detect_ingredients
    # - test_detect_batch
    # - test_filter_detections
    # - test_visualize_detections

