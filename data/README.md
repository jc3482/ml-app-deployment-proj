# Data Directory

This directory contains datasets for our SmartPantry project.

## Current Datasets

### fridge_photos_sample/ (Included in Git)
- **Size**: 3.1 MB
- **Images**: 30 (20 train, 5 valid, 5 test)
- **Purpose**: Testing and development
- **Status**: Included in repository

### fridge_photos/ (Not in Git)
- **Size**: ~400 MB
- **Images**: 3,049 (2,895 train, 103 valid, 51 test)
- **Purpose**: Full ingredient detection training
- **Status**: Download separately (see instructions below)

### recipes_sample/ (Included in Git)
- **Size**: 872 KB
- **Recipes**: 100 recipes with ingredients and instructions
- **Images**: 50 food images
- **Purpose**: Testing recipe retrieval pipeline
- **Status**: Included in repository

### recipes/ (Not in Git)
- **Size**: ~254 MB
- **Recipes**: 58,783 recipes
- **Images**: 13,582 food images
- **Purpose**: Full recipe database
- **Status**: Download separately (contact team)

## Setup Instructions

### For Development and Testing

The sample dataset is already included. You can start testing immediately:

```bash
cd data/fridge_photos_sample
# Sample data ready to use!
```

### For Full Training

Download the complete dataset:

1. **Visit Roboflow**:
   - URL: https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr/dataset/2
   - Select YOLOv8 format
   - Download the dataset

2. **Extract to correct location**:
   ```bash
   cd data/fridge_photos
   unzip smart\ refrigerator.v2i.yolov8.zip
   ```

3. **Verify**:
   ```bash
   ls fridge_photos/
   # Should see: train/ valid/ test/ data.yaml
   ```

## Dataset Structure

Both datasets follow this structure:

```
dataset/
├── train/
│   ├── images/     # Training images
│   └── labels/     # YOLO format labels (.txt)
├── valid/
│   ├── images/     # Validation images
│   └── labels/     # YOLO format labels (.txt)
├── test/
│   ├── images/     # Test images
│   └── labels/     # YOLO format labels (.txt)
└── data.yaml       # Dataset configuration
```

## Alternative Datasets (Optional)

### Recipe1M+ (Alternative)
- Location: `data/raw/recipe1m/`
- Purpose: Alternative large-scale recipe database
- Status: Optional (we use Food Ingredients dataset instead)

### Food-101 (Optional)
- Location: `data/raw/food-101/`
- Purpose: Additional ingredient recognition training
- Status: Optional supplement to fridge dataset

## Git Ignore Rules

We configured `.gitignore` to:
- **Include**: Sample dataset (3.1 MB)
- **Exclude**: Full dataset (400 MB)
- **Exclude**: Raw datasets
- **Exclude**: Processed files and embeddings

## Storage Guidelines

For team members:
- Keep the full dataset locally in `fridge_photos/`
- Share via cloud storage if needed (Dropbox, Google Drive, etc.)
- Don't commit the full dataset to Git
- The sample dataset is sufficient for code development and testing

## Questions

Contact team members if you need:
- Access to the full dataset
- Help with dataset setup
- Additional dataset formats

