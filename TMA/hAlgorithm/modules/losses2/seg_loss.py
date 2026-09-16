"""Segmentation Loss for multi-class semantic segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss


class SegmentationLoss(Loss):
    """Multi-class segmentation loss (CrossEntropy + optional Dice)."""
    
    def __init__(
        self,
        loss_weight=1.0,
        use_dice=False,
        dice_weight=1.0,
        ce_weight=1.0,
        ignore_index=255,
        class_weights=None,
    ):
        super(SegmentationLoss, self).__init__(loss_weight=loss_weight)
        
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, pred, target, name=None, **kwargs):
        """
        Compute segmentation loss.
        
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] class indices
            
        Returns:
            loss: scalar tensor
        """
        loss_weight = self.get_loss_weight(name)
        if loss_weight == 0:
            return 0 * torch.sum(pred)
        
        # Move class_weights to same device as pred
        class_weights = self.class_weights
        if class_weights is not None and class_weights.device != pred.device:
            class_weights = class_weights.to(pred.device)
        
        # CrossEntropy loss
        ce_loss = F.cross_entropy(
            pred, target,
            weight=class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        total_loss = ce_loss * self.ce_weight
        
        # Optional Dice loss
        if self.use_dice:
            dice_loss = self.dice_loss(pred, target)
            total_loss = total_loss + dice_loss * self.dice_weight
        
        return total_loss * loss_weight
    
    def dice_loss(self, pred, target):
        """
        Compute Dice loss for multi-class segmentation.
        
        Args:
            pred: [B, C, H, W] logits
            target: [B, H, W] class indices
            
        Returns:
            loss: scalar tensor
        """
        pred = F.softmax(pred, dim=1)  # [B, C, H, W]
        num_classes = pred.shape[1]
        
        # Convert target to one-hot
        target_onehot = F.one_hot(
            target.clamp(0, num_classes - 1),  # Clamp ignore_index
            num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Create valid mask (ignore ignore_index)
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()  # [B, 1, H, W]
        
        # Compute Dice coefficient per class
        dice_losses = []
        for c in range(num_classes):
            pred_c = pred[:, c:c+1] * valid_mask  # [B, 1, H, W]
            target_c = target_onehot[:, c:c+1] * valid_mask  # [B, 1, H, W]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            dice_losses.append(1.0 - dice)
        
        return torch.stack(dice_losses).mean()
