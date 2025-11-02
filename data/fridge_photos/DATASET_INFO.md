# Smart Refrigerator Dataset

## Overview

We obtained this dataset from Roboflow Universe. It contains labeled images of common fridge items in YOLOv8 format.

## Dataset Details

- **Source**: Northumbria University Newcastle
- **Format**: YOLOv8
- **License**: CC BY 4.0
- **URL**: https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr/dataset/2

## Dataset Statistics

- **Total Classes**: 30 ingredients
- **Train Images**: 2,895
- **Validation Images**: 103
- **Test Images**: 51
- **Total Images**: 3,049

## Classes (30 items)

The dataset includes these common fridge ingredients:

1. apple
2. banana
3. beef
4. blueberries
5. bread
6. butter
7. carrot
8. cheese
9. chicken
10. chicken_breast
11. chocolate
12. corn
13. eggs
14. flour
15. goat_cheese
16. green_beans
17. ground_beef
18. ham
19. heavy_cream
20. lime
21. milk
22. mushrooms
23. onion
24. potato
25. shrimp
26. spinach
27. strawberries
28. sugar
29. sweet_potato
30. tomato

## Directory Structure

```
data/fridge_photos/
├── train/
│   ├── images/     (2,895 images)
│   └── labels/     (2,895 labels)
├── valid/
│   ├── images/     (103 images)
│   └── labels/     (103 labels)
├── test/
│   ├── images/     (51 images)
│   └── labels/     (51 labels)
└── data.yaml       (Configuration file)
```

## Label Format

Labels are in YOLO format (one `.txt` file per image):
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0-1 range).

## Usage for Training

Our team can use this dataset to train YOLOv8:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # Start with pretrained model

# Train the model
model.train(
    data='data/fridge_photos/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='fridge_ingredient_detector'
)
```

## Next Steps

1. Review sample images to understand the data quality
2. Start with a small training run to verify setup
3. Fine-tune hyperparameters based on validation results
4. Use the trained model in our SmartPantry application

## Notes

- This dataset covers common fridge items but may not include all possible ingredients
- We may need to augment with additional data for edge cases
- The dataset is well-balanced for common items
- Consider combining with Food-101 for broader ingredient recognition

