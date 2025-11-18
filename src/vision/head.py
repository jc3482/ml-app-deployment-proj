"""
Detection head for object detection.
Predicts class probabilities, bounding box coordinates, and objectness scores.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class DetectionHead(nn.Module):
    """
    Detection head that predicts:
    - Class probabilities (num_classes)
    - Bounding box coordinates (4: x, y, w, h)
    - Objectness score (1: confidence)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1,
        hidden_channels: int = 256,
    ):
        """
        Initialize detection head.

        Args:
            in_channels: Number of input channels from FPN
            num_classes: Number of object classes
            num_anchors: Number of anchors per location (default 1 for anchor-free)
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Classification branch
        self.cls_conv = nn.Conv2d(
            hidden_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Regression branch (bounding box: x, y, w, h)
        self.reg_conv = nn.Conv2d(hidden_channels, num_anchors * 4, kernel_size=3, padding=1)

        # Objectness branch (confidence score)
        self.obj_conv = nn.Conv2d(hidden_channels, num_anchors * 1, kernel_size=3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for detection head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Initialize classification head with bias for better training
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_conv.bias, bias_value)
        nn.init.constant_(self.obj_conv.bias, bias_value)

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through detection head.

        Args:
            features: List of feature maps from FPN at different scales

        Returns:
            Tuple of (cls_pred, reg_pred, obj_pred):
            - cls_pred: Classification predictions (B, num_anchors*num_classes, H, W)
            - reg_pred: Regression predictions (B, num_anchors*4, H, W)
            - obj_pred: Objectness predictions (B, num_anchors*1, H, W)
        """
        cls_outputs = []
        reg_outputs = []
        obj_outputs = []

        for feat in features:
            # Shared feature extraction
            shared = self.shared_conv(feat)

            # Branch predictions
            cls_pred = self.cls_conv(shared)
            reg_pred = self.reg_conv(shared)
            obj_pred = self.obj_conv(shared)

            cls_outputs.append(cls_pred)
            reg_outputs.append(reg_pred)
            obj_outputs.append(obj_pred)

        return cls_outputs, reg_outputs, obj_outputs


class AnchorGenerator:
    """
    Simple anchor generator for anchor-based detection.
    Currently using anchor-free approach, but this can be extended.
    """

    def __init__(self, scales: List[float] = None, aspect_ratios: List[float] = None):
        """
        Initialize anchor generator.

        Args:
            scales: List of anchor scales
            aspect_ratios: List of aspect ratios
        """
        self.scales = scales or [1.0]
        self.aspect_ratios = aspect_ratios or [1.0]

    def generate_anchors(self, feature_map_size: Tuple[int, int], stride: int):
        """
        Generate anchors for a feature map.

        Args:
            feature_map_size: (H, W) of feature map
            stride: Stride of feature map relative to input image

        Returns:
            Anchors tensor of shape (H*W*num_anchors, 4)
        """
        # For anchor-free approach, we'll use grid centers
        # This is a placeholder for future anchor-based implementation
        H, W = feature_map_size
        anchors = []

        for h in range(H):
            for w in range(W):
                # Center coordinates in feature map space
                cx = (w + 0.5) * stride
                cy = (h + 0.5) * stride
                anchors.append([cx, cy, stride, stride])

        return torch.tensor(anchors, dtype=torch.float32)
