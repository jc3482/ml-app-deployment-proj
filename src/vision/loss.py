"""
Loss functions for ingredient classification.
Simplified from detection loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class DetectionLoss(nn.Module):
    """
    Multi-label classification loss using BCEWithLogitsLoss.
    """

    def __init__(
        self,
        num_classes: int = 30,
        pos_weight: float = 5.0,
        **kwargs: Any,
    ):
        """
        Initialize loss.

        Args:
            num_classes: Number of classes
            pos_weight: Weight for positive samples (to handle imbalance)
        """
        super().__init__()
        # We use a fixed positive weight to balance the fact that most ingredients are absent
        # If an image has 3 ingredients out of 30, negative/positive ratio is 9:1
        self.pos_weight_val = pos_weight

        # Will be initialized in forward when device is known
        self.criterion = None

    def forward(
        self,
        pred: torch.Tensor,
        target_unused: Any,
        target_unused2: Any,
        targets: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            pred: Predicted logits (B, num_classes)
            targets: Dictionary containing 'targets' (B, num_classes)

        Returns:
            Dictionary with 'loss'
        """
        device = pred.device

        # Initialize criterion if needed
        if self.criterion is None:
            pos_weight = torch.tensor([self.pos_weight_val], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        logits = pred
        gt_labels = targets["targets"].to(device)

        loss = self.criterion(logits, gt_labels)

        return {
            "loss": loss,
            "cls_loss": loss,
            "reg_loss": torch.tensor(0.0, device=device),
            "obj_loss": torch.tensor(0.0, device=device),
        }
