"""
Backbone network for feature extraction.
Uses pre-trained ResNet50 as feature extractor.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNetBackbone(nn.Module):
    """
    ResNet50 backbone for feature extraction.
    Removes the final classification layers and returns multi-scale features.
    """

    def __init__(self, pretrained: bool = True, freeze_bn: bool = False):
        """
        Initialize ResNet backbone.

        Args:
            pretrained: Whether to use ImageNet pretrained weights
            freeze_bn: Whether to freeze batch normalization layers
        """
        super().__init__()

        # Load pretrained ResNet50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)

        # Extract layers for feature extraction
        # We'll use conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # Output: 256 channels, stride 4
        self.layer2 = resnet.layer2  # Output: 512 channels, stride 8
        self.layer3 = resnet.layer3  # Output: 1024 channels, stride 16
        self.layer4 = resnet.layer4  # Output: 2048 channels, stride 32

        # Freeze batch norm if requested
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through ResNet backbone.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of feature maps at different scales:
            - layer2: (B, 512, H/8, W/8) - for medium objects
            - layer3: (B, 1024, H/16, W/16) - for large objects
            - layer4: (B, 2048, H/32, W/32) - for very large objects
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat2 = self.layer2(x)  # stride 8
        feat3 = self.layer3(feat2)  # stride 16
        feat4 = self.layer4(feat3)  # stride 32

        # Return multi-scale features
        return [feat2, feat3, feat4]

    def get_out_channels(self) -> List[int]:
        """Get output channel numbers for each feature level."""
        return [512, 1024, 2048]


class FeaturePyramidNetwork(nn.Module):
    """
    Simplified Feature Pyramid Network for multi-scale feature fusion.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        """
        Initialize FPN.

        Args:
            in_channels_list: List of input channel numbers for each level
            out_channels: Output channel number for all levels
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # Top-down pathway (upsampling and fusion)
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through FPN.

        Args:
            features: List of feature maps from backbone [feat2, feat3, feat4]

        Returns:
            List of fused feature maps at different scales
        """
        # Apply lateral connections
        laterals = []
        for i, feat in enumerate(features):
            laterals.append(self.lateral_convs[i](feat))

        # Top-down pathway
        # Start from the top level (smallest spatial size)
        fpn_features = [laterals[-1]]  # Start with top level

        # Build top-down features
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample previous level
            upsampled = nn.functional.interpolate(
                fpn_features[0], size=laterals[i].shape[2:], mode="nearest"
            )
            # Add lateral connection
            fused = upsampled + laterals[i]
            fpn_features.insert(0, fused)

        # Apply final conv layers
        output_features = []
        for i, feat in enumerate(fpn_features):
            output_features.append(self.fpn_convs[i](feat))

        return output_features
