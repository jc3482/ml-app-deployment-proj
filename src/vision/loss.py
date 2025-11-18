"""
Loss functions for object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DetectionLoss(nn.Module):
    """
    Detection loss combining classification, regression, and objectness losses.

    Loss components:
    1. Classification loss: Cross-entropy for class prediction
    2. Regression loss: Smooth L1 or IoU loss for bbox coordinates
    3. Objectness loss: Binary cross-entropy for object confidence
    """

    def __init__(
        self,
        num_classes: int = 30,
        img_size: int = 640,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
        obj_weight: float = 1.0,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Initialize detection loss.

        Args:
            num_classes: Number of object classes
            img_size: Input image size
            cls_weight: Weight for classification loss
            reg_weight: Weight for regression loss
            obj_weight: Weight for objectness loss
            use_focal_loss: Whether to use focal loss for classification
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Feature map strides
        self.strides = [8, 16, 32]

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        # Use larger beta for SmoothL1Loss to be more like L2 for small errors, L1 for large errors
        # This helps prevent regression loss from exploding
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none", beta=1.0)

    def forward(
        self,
        cls_pred: List[torch.Tensor],
        reg_pred: List[torch.Tensor],
        obj_pred: List[torch.Tensor],
        targets: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.

        Args:
            cls_pred: List of classification predictions for each scale
            reg_pred: List of regression predictions for each scale
            obj_pred: List of objectness predictions for each scale
            targets: Dictionary containing:
                - boxes: List of bbox tensors (one per image)
                - labels: List of label tensors (one per image)
                - images: Batched image tensor

        Returns:
            Dictionary with loss components and total loss
        """
        device = cls_pred[0].device
        batch_size = cls_pred[0].shape[0]

        # Initialize loss accumulators
        cls_loss_sum = torch.tensor(0.0, device=device)
        reg_loss_sum = torch.tensor(0.0, device=device)
        obj_loss_sum = torch.tensor(0.0, device=device)
        num_positives = 0

        # Process each scale
        num_scales = len(cls_pred)
        for scale_idx, (cls_p, reg_p, obj_p) in enumerate(zip(cls_pred, reg_pred, obj_pred)):
            _stride = self.strides[scale_idx]  # noqa: F841
            B, _, H, W = cls_p.shape

            # Reshape predictions
            # cls_p: (B, num_classes, H, W) -> (B, H, W, num_classes)
            cls_p = cls_p.permute(0, 2, 3, 1).contiguous()
            # reg_p: (B, 4, H, W) -> (B, H, W, 4)
            reg_p = reg_p.permute(0, 2, 3, 1).contiguous()
            # obj_p: (B, 1, H, W) -> (B, H, W, 1)
            obj_p = obj_p.permute(0, 2, 3, 1).contiguous()

            # Build target tensors for this scale
            cls_target = torch.zeros((B, H, W, self.num_classes), device=device)
            reg_target = torch.zeros((B, H, W, 4), device=device)
            obj_target = torch.zeros((B, H, W, 1), device=device)

            # Match targets to grid cells
            for b in range(batch_size):
                boxes = targets["boxes"][b]  # (N, 4) in normalized YOLO format
                labels = targets["labels"][b]  # (N,)

                if len(boxes) == 0:
                    continue

                # Convert boxes to grid coordinates
                for box, label in zip(boxes, labels):
                    x_center, y_center, _width, _height = box  # noqa: F841

                    # Find corresponding grid cell
                    grid_x = int(x_center * W)
                    grid_y = int(y_center * H)

                    # Clamp to valid range
                    grid_x = max(0, min(W - 1, grid_x))
                    grid_y = max(0, min(H - 1, grid_y))

                    # Set targets
                    cls_target[b, grid_y, grid_x, label] = 1.0
                    reg_target[b, grid_y, grid_x] = box
                    obj_target[b, grid_y, grid_x, 0] = 1.0
                    num_positives += 1

            # Compute losses
            # Classification loss (only on positive cells)
            pos_mask = obj_target.squeeze(-1) > 0.5  # (B, H, W)
            if pos_mask.sum() > 0:
                cls_p_flat = cls_p[pos_mask]  # (N_pos, num_classes)
                cls_t_flat = cls_target[pos_mask]  # (N_pos, num_classes)

                if self.use_focal_loss:
                    cls_loss = self._focal_loss(cls_p_flat, cls_t_flat)
                else:
                    # Convert to class indices for cross-entropy
                    cls_indices = cls_t_flat.argmax(dim=1)
                    cls_loss = self.ce_loss(cls_p_flat, cls_indices)

                # Normalize by number of positive samples in this scale
                cls_loss_sum += cls_loss.mean() * self.cls_weight

            # Regression loss (only on positive cells)
            if pos_mask.sum() > 0:
                reg_p_flat = reg_p[pos_mask]  # (N_pos, 4)
                reg_t_flat = reg_target[pos_mask]  # (N_pos, 4)
                reg_loss = self.smooth_l1_loss(reg_p_flat, reg_t_flat).mean(dim=1)
                # Clip regression loss to prevent explosion
                reg_loss = torch.clamp(reg_loss, max=10.0)
                reg_loss_sum += reg_loss.mean() * self.reg_weight

            # Objectness loss (all cells)
            obj_loss = self.bce_loss(obj_p, obj_target)
            obj_loss_sum += obj_loss.mean() * self.obj_weight

        # Normalize losses by number of scales to get average loss per scale
        cls_loss_sum = cls_loss_sum / num_scales
        reg_loss_sum = reg_loss_sum / num_scales
        obj_loss_sum = obj_loss_sum / num_scales

        # Total loss
        total_loss = cls_loss_sum + reg_loss_sum + obj_loss_sum

        # Check for NaN or Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf detected in loss, returning zero loss")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            cls_loss_sum = torch.tensor(0.0, device=device)
            reg_loss_sum = torch.tensor(0.0, device=device)
            obj_loss_sum = torch.tensor(0.0, device=device)

        return {
            "loss": total_loss,
            "cls_loss": cls_loss_sum,
            "reg_loss": reg_loss_sum,
            "obj_loss": obj_loss_sum,
        }

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for classification.

        Args:
            pred: Predicted logits (N, num_classes)
            target: Target one-hot encoding (N, num_classes)

        Returns:
            Focal loss values
        """
        # Convert to probabilities
        probs = F.softmax(pred, dim=1)

        # Get probability of correct class
        pt = (probs * target).sum(dim=1)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.focal_gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(pred, target.argmax(dim=1), reduction="none")

        # Apply focal weight
        focal_loss = self.focal_alpha * focal_weight * ce_loss

        return focal_loss
