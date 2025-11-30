"""
Utility script to inspect class names inside a YOLO model.
Run this to list all ingredient labels the YOLO model can detect.
"""

import os
from ultralytics import YOLO


def main():
    # Resolve path to best.pt relative to this script
    here = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(here, "best.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"best.pt not found at: {weights_path}")

    print(f"Loading YOLO model from: {weights_path}")

    # Load model
    model = YOLO(weights_path)

    # Display class names
    print("\n=== YOLO Model Class Names ===")
    for idx, name in model.names.items():
        print(f"{idx}: {name}")

    print("\nTotal classes:", len(model.names))
    print("===============================")


if __name__ == "__main__":
    main()
