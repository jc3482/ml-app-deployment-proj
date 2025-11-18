# Custom Object Detector Implementation

This directory contains a custom object detection model implementation using:
- **Backbone**: Pre-trained ResNet50 (ImageNet)
- **Neck**: Feature Pyramid Network (FPN)
- **Head**: Custom detection head (classification + regression + objectness)

## Architecture Overview

```
Input Image (3, 640, 640)
    ↓
ResNet50 Backbone
    ↓
Multi-scale Features [feat2, feat3, feat4]
    ↓
Feature Pyramid Network (FPN)
    ↓
Fused Multi-scale Features
    ↓
Detection Head
    ├── Classification Branch (30 classes)
    ├── Regression Branch (bbox: x, y, w, h)
    └── Objectness Branch (confidence)
```

## File Structure

- `backbone.py`: ResNet50 backbone with multi-scale feature extraction
- `neck.py`: Feature Pyramid Network for feature fusion
- `head.py`: Detection head with classification, regression, and objectness branches
- `detector.py`: Complete detector model combining all components
- `dataset.py`: YOLO format dataset loader
- `transforms.py`: Data augmentation and preprocessing
- `loss.py`: Detection loss function (classification + regression + objectness)
- `trainer.py`: Training loop and utilities
- `utils.py`: NMS, IoU, mAP calculation, visualization
- `train_example.py`: Example training script

## Quick Start

### 1. Install Dependencies

Make sure you have all required packages:
```bash
pip install torch torchvision albumentations opencv-python pillow pyyaml tqdm tensorboard
```

### 2. Prepare Data

Ensure your data is in YOLO format:
```
data/fridge_photos/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 3. Train the Model

Run the training script:
```bash
python src/vision/train_example.py
```

Or use the trainer directly:
```python
from src.vision.trainer import train_from_config

train_from_config("config.yaml", device="cuda")
```

### 4. Use the Trained Model

```python
from src.vision.detector import CustomDetector
import torch

# Load model
model = CustomDetector(num_classes=30)
checkpoint = torch.load("models/checkpoints/best.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
image = torch.randn(1, 3, 640, 640)  # Your image
cls_pred, reg_pred, obj_pred = model(image)
```

## Training Configuration

You can customize training by modifying the config dictionary in `train_example.py`:

```python
config = {
    "data": {
        "yaml_path": "data/fridge_photos/data.yaml",
        "img_size": 640,
    },
    "model": {
        "num_classes": 30,
        "backbone_pretrained": True,
        "fpn_out_channels": 256,
        "head_hidden_channels": 256,
    },
    "training": {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "cls_weight": 1.0,    # Classification loss weight
        "reg_weight": 5.0,    # Regression loss weight
        "obj_weight": 1.0,    # Objectness loss weight
    },
}
```

## Model Details

### Backbone (ResNet50)
- Pre-trained on ImageNet
- Extracts features at 3 scales:
  - Layer 2: stride 8, 512 channels
  - Layer 3: stride 16, 1024 channels
  - Layer 4: stride 32, 2048 channels

### Feature Pyramid Network (FPN)
- Fuses multi-scale features
- Output: 256 channels at each scale
- Top-down pathway with lateral connections

### Detection Head
- Shared feature extraction (2 conv layers)
- 3 branches:
  - Classification: 30 classes
  - Regression: 4 values (x, y, w, h)
  - Objectness: 1 value (confidence)

### Loss Function
- Classification Loss: Cross-entropy (or Focal Loss)
- Regression Loss: Smooth L1 Loss
- Objectness Loss: Binary Cross-entropy
- Total: weighted sum of all three

## Training Tips

1. **Start Small**: Test with a few epochs first to verify everything works
2. **Monitor Loss**: Use TensorBoard to visualize training progress
3. **Adjust Learning Rate**: If loss doesn't decrease, try lower LR (0.0001)
4. **Batch Size**: Adjust based on GPU memory (8GB GPU → batch_size=4)
5. **Data Augmentation**: Already included in training transforms

## Evaluation

The model outputs predictions at 3 scales. To get final detections:
1. Apply sigmoid to objectness predictions
2. Apply softmax to classification predictions
3. Filter by confidence threshold
4. Apply NMS to remove duplicates
5. Convert to image coordinates

See `utils.py` for NMS and evaluation functions.

## Next Steps

- [ ] Implement full post-processing pipeline
- [ ] Complete mAP calculation
- [ ] Add anchor-based detection option
- [ ] Implement Focal Loss for better handling of class imbalance
- [ ] Add more data augmentation strategies
- [ ] Implement model export (ONNX, TorchScript)

## Notes

- This is a simplified implementation for educational purposes
- For production use, consider using established frameworks (mmdetection, detectron2)
- The current implementation uses anchor-free approach (simpler but may be less accurate)
- Full post-processing and evaluation are placeholders and need to be completed

