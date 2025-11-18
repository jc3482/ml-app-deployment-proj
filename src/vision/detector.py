"""
Complete custom detector model combining backbone, neck, and head.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union

from .backbone import ResNetBackbone, FeaturePyramidNetwork
from .head import DetectionHead


class CustomDetector(nn.Module):
    """
    Custom object detector using ResNet backbone + FPN + Detection Head.

    Architecture:
    1. ResNet50 backbone (pretrained) -> multi-scale features
    2. Feature Pyramid Network -> fused multi-scale features
    3. Detection Head -> class, bbox, objectness predictions
    """

    def __init__(
        self,
        num_classes: int = 30,
        backbone_pretrained: bool = True,
        fpn_out_channels: int = 256,
        head_hidden_channels: int = 256,
        # Legacy parameters for backward compatibility
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        image_size: int = 640,
    ):
        """
        Initialize custom detector.

        Args:
            num_classes: Number of object classes (30 for ingredients)
            backbone_pretrained: Whether to use pretrained ResNet weights
            fpn_out_channels: Output channels for FPN
            head_hidden_channels: Hidden channels in detection head
            model_path: Legacy parameter (ignored, kept for compatibility)
            confidence_threshold: Legacy parameter (stored for later use)
            iou_threshold: Legacy parameter (stored for later use)
            device: Legacy parameter (stored for later use)
            image_size: Legacy parameter (stored for later use)
        """
        super().__init__()
        self.num_classes = num_classes

        # Store legacy parameters for compatibility
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size

        # Backbone: ResNet50
        self.backbone = ResNetBackbone(pretrained=backbone_pretrained)
        backbone_channels = self.backbone.get_out_channels()

        # Neck: Feature Pyramid Network
        self.neck = FeaturePyramidNetwork(
            in_channels_list=backbone_channels, out_channels=fpn_out_channels
        )

        # Head: Detection head
        self.head = DetectionHead(
            in_channels=fpn_out_channels,
            num_classes=num_classes,
            num_anchors=1,  # Anchor-free approach
            hidden_channels=head_hidden_channels,
        )

        # Feature map strides (relative to input image)
        # These correspond to the strides of layer2, layer3, layer4
        self.strides = [8, 16, 32]

    def forward(
        self, x: torch.Tensor, targets: Optional[Dict] = None  # noqa: ARG002
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through the detector.

        Args:
            x: Input images tensor of shape (B, 3, H, W)
            targets: Optional ground truth targets for training

        Returns:
            Tuple of (cls_pred, reg_pred, obj_pred):
            - cls_pred: List of classification predictions for each scale
            - reg_pred: List of regression predictions for each scale
            - obj_pred: List of objectness predictions for each scale
        """
        # Extract features using backbone
        backbone_features = self.backbone(x)

        # Fuse features using FPN
        fpn_features = self.neck(backbone_features)

        # Generate predictions using detection head
        cls_pred, reg_pred, obj_pred = self.head(fpn_features)

        return cls_pred, reg_pred, obj_pred

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 50,
    ) -> List[Dict]:
        """
        Predict objects in input images.

        Args:
            x: Input images tensor of shape (B, 3, H, W)
            conf_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image

        Returns:
            List of detection dictionaries, one per image
        """
        self.eval()
        with torch.no_grad():
            cls_pred, reg_pred, obj_pred = self.forward(x)

        # Process predictions (will be implemented in utils)
        # This is a placeholder - actual post-processing in utils.py
        detections = self._post_process(
            cls_pred,
            reg_pred,
            obj_pred,
            conf_threshold,
            nms_threshold,
            max_detections,
            x.shape[2:],  # Original image size
        )

        return detections

    def _post_process(
        self,
        cls_pred: List[torch.Tensor],  # noqa: ARG002
        reg_pred: List[torch.Tensor],  # noqa: ARG002
        obj_pred: List[torch.Tensor],  # noqa: ARG002
        conf_threshold: float,  # noqa: ARG002
        nms_threshold: float,  # noqa: ARG002
        max_detections: int,  # noqa: ARG002
        img_size: Tuple[int, int],  # noqa: ARG002
    ) -> List[Dict]:
        """
        Post-process predictions to get final detections.
        This is a simplified version - full implementation in utils.py

        Args:
            cls_pred: Classification predictions
            reg_pred: Regression predictions
            obj_pred: Objectness predictions
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            max_detections: Max detections per image
            img_size: Original image size (H, W)

        Returns:
            List of detection dictionaries
        """
        # This will be implemented using utils functions
        # For now, return empty list
        return []

    def get_model_info(self) -> Dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone": "ResNet50",
            "neck": "FPN",
            "head": "Custom Detection Head",
        }


# Alias for backward compatibility
IngredientDetector = CustomDetector
