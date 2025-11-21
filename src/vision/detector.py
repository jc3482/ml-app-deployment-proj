"""
Complete custom detector model combining backbone, neck, and head.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from PIL import Image

from .utils import visualize_detections
from .transforms import get_val_transforms, get_inference_transforms


class CustomDetector(nn.Module):
    """
    Custom multi-label ingredient classifier using ResNet50.
    Simplified from object detector for stability and performance.
    """

    def __init__(
        self,
        num_classes: int = 30,
        class_names: Optional[List[str]] = None,
        backbone_pretrained: bool = True,
        # Legacy parameters for backward compatibility
        fpn_out_channels: int = 256,
        head_hidden_channels: int = 256,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        image_size: int = 640,
    ):
        """
        Initialize custom classifier.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # Use torchvision ResNet50 directly
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V2 if backbone_pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace final FC layer for multi-label classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))

        # Freeze all except Layer 4 and FC for fine-tuning
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    def train(self, mode: bool = True):
        """
        Set training mode.
        Overridden to keep frozen layers in eval mode.
        """
        super().train(mode)

        # Keep frozen layers in eval mode (BN stats)
        for name, module in self.backbone.named_modules():
            if (
                "layer4" not in name
                and "fc" not in name
                and not isinstance(module, type(self.backbone))
            ):
                module.eval()

        return self

    def forward(
        self, x: torch.Tensor, targets: Optional[Dict] = None  # noqa: ARG002
    ) -> torch.Tensor:
        """
        Forward pass: Image -> Logits
        """
        return self.backbone(x)

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        # Legacy args
        nms_threshold: float = 0.45,
        max_detections: int = 50,
    ) -> List[Dict]:
        """
        Predict ingredients in input images.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)

        results = []
        for p in probs:
            # Filter by threshold
            mask = p > conf_threshold
            indices = torch.where(mask)[0]
            scores = p[mask]
            results.append({"labels": indices, "scores": scores})
        return results

    def detect_ingredients(
        self,
        image: Union[str, Image.Image, np.ndarray],
        visualize: bool = True,
        conf_threshold: Optional[float] = None,
    ) -> Dict:
        """
        End-to-end detection pipeline for a single image.
        """
        # 1. Load Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        # Keep original image for visualization (H, W, 3)
        orig_image = np.array(image)

        # 2. Preprocess
        transforms = get_inference_transforms(img_size=self.image_size)
        # transforms expects numpy array
        transformed = transforms(image=orig_image)
        img_tensor = transformed["image"]

        # Convert to tensor (H, W, C) -> (1, C, H, W)
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)

        img_tensor = img_tensor.unsqueeze(0).float().to(self.device)

        # 3. Predict
        threshold = conf_threshold if conf_threshold is not None else self.confidence_threshold
        detections = self.predict(img_tensor, conf_threshold=threshold)

        # Unpack single image results
        result = detections[0]
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()

        # 4. Map Labels to Names
        ingredient_names = []
        if self.class_names:
            for label in labels:
                if 0 <= label < len(self.class_names):
                    ingredient_names.append(self.class_names[int(label)])
                else:
                    ingredient_names.append(f"Class {label}")
        else:
            ingredient_names = [f"Class {l}" for l in labels]

        # 5. Visualization (Just return original image, no boxes)
        # Optionally overlay text list
        viz_image = None
        if visualize:
            viz_image = orig_image.copy()
            # Could add text overlay here if needed

        return {
            "ingredients": ingredient_names,
            "confidences": scores.tolist(),
            "boxes": [],  # No boxes for classification
            "visualization": viz_image,
        }

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "type": "Multi-label Classifier",
            "backbone": "ResNet50",
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


# Alias for backward compatibility
IngredientDetector = CustomDetector
