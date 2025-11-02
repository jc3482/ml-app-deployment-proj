# Fridge Photos Sample Dataset

## Overview

This is a small sample dataset for testing and development. We included only 30 images to keep the repository size manageable for GitHub.

## Sample Statistics

- **Train**: 20 images
- **Valid**: 5 images
- **Test**: 5 images
- **Total**: 30 images
- **Size**: ~3.1 MB
- **Classes**: 30 ingredient types

## Purpose

Our team uses this sample for:

- Quick testing of code without downloading the full dataset
- CI/CD pipeline testing
- Verifying the data pipeline works correctly
- Demo and development purposes

## Full Dataset

The complete dataset contains 3,049 images:

### Option 1: Download from Roboflow (Recommended)

```bash
# Download the full dataset
# Visit: https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr/dataset/2
# Export as YOLOv8 format
# Extract to data/fridge_photos/
```

### Option 2: Download from Team Storage

Contact team members for access to the full dataset (~400MB).

### Full Dataset Stats
- Train: 2,895 images
- Valid: 103 images
- Test: 51 images
- Total: 3,049 images

## Usage

### For Testing
```python
from ultralytics import YOLO

# Quick test with sample data
model = YOLO('yolov8n.pt')
results = model.train(
    data='data/fridge_photos_sample/data.yaml',
    epochs=5,  # Just a few epochs for testing
    imgsz=640,
    batch=4
)
```

### For Production Training

Download the full dataset and update the path:

```python
# Use full dataset for actual training
model.train(
    data='data/fridge_photos/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

## Dataset Classes

The dataset includes 30 common fridge ingredients:

apple, banana, beef, blueberries, bread, butter, carrot, cheese, chicken, chicken_breast, chocolate, corn, eggs, flour, goat_cheese, green_beans, ground_beef, ham, heavy_cream, lime, milk, mushrooms, onion, potato, shrimp, spinach, strawberries, sugar, sweet_potato, tomato

## License

CC BY 4.0 - See Roboflow link for details

## Notes

- This sample represents the full dataset structure
- Label format is standard YOLO format
- All coordinates are normalized (0-1 range)
- Use this for development, full dataset for final training

