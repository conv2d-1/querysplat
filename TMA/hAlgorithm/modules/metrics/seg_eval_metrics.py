"""Segmentation Evaluation Metrics."""
import torch
import numpy as np
from .mask_eval_metrics import intersect_and_union, total_intersect_and_union


class SegEvalMetrics:
    """Segmentation evaluation metrics (mIoU, pixel accuracy)."""
    
    def __init__(self, num_classes=3, ignore_index=255, metrics=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = metrics or ['miou', 'pixel_acc']
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.total_area_intersect = torch.zeros((self.num_classes,), dtype=torch.float64)
        self.total_area_union = torch.zeros((self.num_classes,), dtype=torch.float64)
        self.total_area_pred = torch.zeros((self.num_classes,), dtype=torch.float64)
        self.total_area_label = torch.zeros((self.num_classes,), dtype=torch.float64)
    
    def update(self, pred, target):
        """Update metrics with batch predictions and targets.
        
        Args:
            pred: [B, C, H, W] logits or [B, H, W] class indices
            target: [B, H, W] class indices
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)  # [B, H, W]
        
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        for p, t in zip(pred, target):
            area_intersect, area_union, area_pred, area_label = intersect_and_union(
                p, t, self.num_classes, self.ignore_index
            )
            self.total_area_intersect += area_intersect
            self.total_area_union += area_union
            self.total_area_pred += area_pred
            self.total_area_label += area_label
    
    def compute(self):
        """Compute final metrics."""
        results = {}
        
        # Per-class IoU
        iou = self.total_area_intersect / (self.total_area_union + 1e-10)
        
        if 'miou' in self.metrics:
            results['miou'] = iou.mean().item()
        
        if 'iou_per_class' in self.metrics:
            for i in range(self.num_classes):
                results[f'iou_class_{i}'] = iou[i].item()
        
        if 'pixel_acc' in self.metrics:
            pixel_acc = self.total_area_intersect.sum() / (self.total_area_label.sum() + 1e-10)
            results['pixel_acc'] = pixel_acc.item()
        
        if 'dice' in self.metrics:
            dice = 2 * self.total_area_intersect / (self.total_area_pred + self.total_area_label + 1e-10)
            results['dice'] = dice.mean().item()
        
        return results
    
    def __call__(self, outputs, batch):
        """Evaluate single batch."""
        pred = outputs.get('seg_pred')
        target = batch.get('semantic_target')
        
        if pred is None or target is None:
            return {}
        
        self.update(pred, target)
        return self.compute()
