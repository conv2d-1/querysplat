"""Simple Segmentation Pipeline - inherits from base Pipeline."""
import logging
import numpy as np
import torch
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.utils import instantiate_from_config


class SegPipeline(Pipeline):
    """Simple pipeline for semantic segmentation training.
    
    Reuses existing Base model structure (rgb_encoder + normal_head),
    only adds segmentation-specific loss computation.
    """

    def __init__(
        self,
        target_seg_name="masks",
        target_labels_name="labels",
        seg_loss=None,
        task_weight=None,
        background_class=0,  # Class index for background (no objects)
        **kwargs,
    ):
        super(SegPipeline, self).__init__(**kwargs)
        
        self.target_seg_name = target_seg_name
        self.target_labels_name = target_labels_name
        self.background_class = background_class
        
        self.seg_loss = instantiate_from_config(seg_loss)
        self.task_weight = task_weight or dict(seg=1.0)

    def forward(self, data, return_loss=True, **kwargs):
        """Forward pass with optional loss computation."""
        # Get input image
        rgb = data["image"]
        b, c, h, w = rgb.shape
        
        # Prepare meta_data for model (DPT head needs input dimensions)
        meta_data = data.get("meta_data", {})
        if "input_width" not in meta_data:
            meta_data["input_width"] = [w] * b
        if "input_height" not in meta_data:
            meta_data["input_height"] = [h] * b
        
        # Forward through model (encoder + head)
        # Base model returns dict with 'normal' key (we repurpose for segmentation)
        results = self.model(rgb=rgb, meta_data=meta_data)
        
        # Get segmentation prediction (from normal_head output)
        # Shape: [B, num_classes, H, W]
        seg_pred = results.get("normal", None)
        
        if seg_pred is None:
            raise ValueError("Model did not output segmentation prediction")
        
        output = dict(seg_pred=seg_pred)
        
        if return_loss:
            losses = self.compute_loss(seg_pred, data, h, w)
            output["losses"] = losses
        
        return output

    def compute_loss(self, seg_pred, data, h, w):
        """Compute segmentation losses."""
        losses = dict()
        
        # If no seg_loss configured, skip supervision for this task
        if self.seg_loss is None:
            return losses
        
        # Get ground truth
        # Note: empty masks/labels list is valid (means all background)
        target_masks = data.get(self.target_seg_name)
        target_labels = data.get(self.target_labels_name)
        
        if target_masks is None or target_labels is None:
            raise ValueError(
                f"Missing GT data - masks={target_masks is not None}, labels={target_labels is not None}. "
                f"Available keys: {list(data.keys())}"
            )
        
        # Convert instance masks + labels to semantic segmentation target
        # Empty masks -> all background (handled in instance_to_semantic)
        target_semantic = self.instance_to_semantic(target_masks, target_labels, h, w)
        
        # Resize prediction to match target if needed
        if seg_pred.shape[-2:] != target_semantic.shape[-2:]:
            seg_pred = F.interpolate(
                seg_pred, size=target_semantic.shape[-2:],
                mode="bilinear", align_corners=False
            )
        
        # Compute loss using configured loss function
        seg_loss_val = self.seg_loss(seg_pred, target_semantic)
        losses["seg_loss"] = seg_loss_val * self.task_weight.get("seg", 1.0)
        
        return losses

    def instance_to_semantic(self, instance_masks, labels, h, w):
        """Convert instance masks + labels to semantic segmentation target.
        
        Args:
            instance_masks: List of [list of masks per sample] or tensor
            labels: List of [list of labels per sample]
            h, w: target height and width
            
        Returns:
            semantic: [B, H, W] semantic segmentation target (class indices)
        """
        b = len(labels)  # Batch size
        device = self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize with background class (not ignore_index=255)
        # This handles "no objects in scene" correctly as valid background
        semantic = torch.full((b, h, w), self.background_class, dtype=torch.long, device=device)
        
        # Process each sample in batch
        for bi in range(b):
            sample_masks = instance_masks[bi]  # List of masks for this sample
            sample_labels = labels[bi]  # List of labels for this sample
            
            # Empty masks = all background (already initialized above)
            if not isinstance(sample_masks, list) or len(sample_masks) == 0:
                continue
            
            # Batch convert all masks to tensor
            masks_tensor_list = []
            for mask in sample_masks:
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
                masks_tensor_list.append(mask)
            
            # Stack all masks: [N, H_orig, W_orig]
            masks_stacked = torch.stack(masks_tensor_list, dim=0).float()
            
            # Batch resize all masks at once if needed
            if masks_stacked.shape[-2:] != (h, w):
                masks_stacked = F.interpolate(
                    masks_stacked.unsqueeze(1),  # [N, 1, H_orig, W_orig]
                    size=(h, w),
                    mode="nearest"
                ).squeeze(1)  # [N, H, W]
            
            # Move to device
            masks_stacked = masks_stacked.to(device).long()
            labels_tensor = torch.tensor(sample_labels, device=device, dtype=torch.long)
            
            # Apply masks: later instances overwrite earlier
            for ni in range(len(sample_labels)):
                mask = masks_stacked[ni] > 0
                semantic[bi][mask] = labels_tensor[ni]
        
        return semantic

    def train_step(self, batch):
        """Training step for segmentation."""
        self.train()
        
        # Forward pass
        output = self.forward(batch, return_loss=True)
        
        # Extract losses
        losses = output.get("losses", {})
        
        if not losses:
            # Should not happen in training - dataset should skip samples without GT
            raise ValueError("No losses computed in train_step - check dataset filtering")
        
        # Sum all losses (all should have gradients)
        total_loss = sum(losses.values())
        
        # Loss dict for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_dict["total_loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict

    def inference(self, data, **kwargs):
        """Inference without loss computation."""
        return self.forward(data, return_loss=False, **kwargs)

    def visualize(self, data, output, save_dir=None, **kwargs):
        """Visualize segmentation results."""
        import cv2
        import numpy as np
        import os
        
        seg_pred = output["seg_pred"]
        # Get class predictions
        seg_pred_class = seg_pred.argmax(dim=1)  # [B, H, W]
        
        # Color map for visualization
        colors = [
            (255, 0, 0),    # Class 0: Red
            (0, 255, 0),    # Class 1: Green
            (0, 0, 255),    # Class 2: Blue
            (255, 255, 0),  # Class 3: Yellow
            (255, 0, 255),  # Class 4: Magenta
        ]
        
        vis_results = []
        for bi in range(seg_pred_class.shape[0]):
            pred = seg_pred_class[bi].cpu().numpy()
            h, w = pred.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            
            for ci, color in enumerate(colors):
                vis[pred == ci] = color
            
            vis_results.append(vis)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"seg_pred_{bi}.png"), vis[..., ::-1])
        
        return vis_results
