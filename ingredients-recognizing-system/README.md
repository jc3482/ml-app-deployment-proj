# Smart Refrigerator Food Detection Model Training

Professional YOLOv8 model training using Roboflow dataset.

## Security: API Key Setup

**IMPORTANT:** Never hardcode your API key in the code!

### Method 1: Environment Variable (Recommended)

```bash
# Set API key as environment variable
export ROBOFLOW_API_KEY='your_api_key_here'

# Then run the training
python food_model_trainer.py
```

### Method 2: Command Line Argument

```bash
python food_model_trainer.py --api-key your_api_key_here
```

### Method 3: .env File (Optional)

Create a `.env` file:
```
ROBOFLOW_API_KEY=your_api_key_here
```

Then load it before running:
```bash
source .env
python food_model_trainer.py
```

**Remember to add `.env` to your `.gitignore` file!**

## Installation

```bash
pip install roboflow ultralytics
```

## Usage

### Training

```bash
# Set your API key
export ROBOFLOW_API_KEY='your_api_key_here'

# Run training
python food_model_trainer.py
```

### Using the Trained Model

After training completes, you'll have:
- `best.pt` - Trained model weights
- `food_detector.py` - Detector wrapper class

```python
from food_detector import FoodDetector

# Initialize detector
detector = FoodDetector('best.pt', conf_threshold=0.85)

# Detect ingredients
result = detector.detect('image.jpg')
print(result)  # ['Apple', 'Banana', 'Tomato']

# With confidence scores
result = detector.detect_with_confidence('image.jpg')
for name, conf in result:
    print(f"{name}: {conf:.1%}")

# With counts
result = detector.detect_with_count('image.jpg')
for name, count in result.items():
    print(f"{name}: {count} items")
```

## Training Configuration

The model uses the following default configuration:
- **Model**: YOLOv8s (Small)
- **Epochs**: 100
- **Image size**: 640x640
- **Batch size**: 16
- **Optimizer**: AdamW
- **Patience**: 20 (early stopping)

## Output Files

After training:
```
./
├── best.pt                          # Trained model weights
├── food_detector.py                 # Detector class
└── runs/detect/smart_fridge_v1/     # Training outputs
    ├── weights/
    │   ├── best.pt                  # Best weights
    │   └── last.pt                  # Last weights
    └── results.png                  # Training curves
```

## Security Best Practices

1. ✅ **Never commit API keys to Git**
2. ✅ **Use environment variables**
3. ✅ **Add `.env` to `.gitignore`**
4. ✅ **Rotate keys periodically**
5. ✅ **Use different keys for dev/prod**

## Example .gitignore

```
# API Keys
.env
*.key
*_key.txt

# Model weights (optional)
*.pt
runs/

# Python
__pycache__/
*.pyc
```

## Troubleshooting

### API Key Not Found Error

If you see:
```
Error: Roboflow API key not found...
```

Make sure you've set the environment variable:
```bash
echo $ROBOFLOW_API_KEY  # Should print your key
```

If empty, set it:
```bash
export ROBOFLOW_API_KEY='your_api_key_here'
```

### GPU Not Available

The script will automatically use CPU if GPU is not available, but training will be slower.

## License

MIT License
