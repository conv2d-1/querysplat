import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional

from hAlgorithm.modules.models2.head.vggt_dpt_head import VGGTDPTHead


class MotionHeadV2(VGGTDPTHead):
    """
    Motion Head V2 for Scene Flow Prediction.
    
    Simplified version without query_times input.
    Always uses frame 0 as the reference frame and predicts flow to all other frames.
    
    Output shape: [B, 1, N, 3, H, W] where:
        - B: batch size
        - 1: single query (always frame 0)
        - N: number of frames
        - 3: flow dimensions (dx, dy, dz)
        - H, W: spatial dimensions
    
    Can initialize from a pre-trained global points head (VGGTDPTHead) via:
    - pretrain: path to motion head checkpoint
    - load_from_glb_points_head: path to global points head checkpoint (adapts weights)
    """

    def __init__(
        self,
        dim_in: int = 1024,
        output_dim: int = 3,
        load_from_glb_points_head: str = None,
        **kwargs
    ):
        super().__init__(dim_in=dim_in, output_dim=output_dim, **kwargs)

        # Load weights from global points head (adapts architecture differences)
        if load_from_glb_points_head is not None:
            self._load_from_glb_points_head(load_from_glb_points_head)

    def _load_from_glb_points_head(self, ckpt_path: str):
        """
        Load weights from a pre-trained global points head (VGGTDPTHead).
        
        Adapts for architecture differences:
        - output_conv2[-1] has different output_dim (4 vs 3), skip this layer
        
        Args:
            ckpt_path: Path to checkpoint containing 'glb_points_head' state dict
        """
        logging.info(f"MotionHeadV2: Loading weights from global points head: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Extract glb_points_head weights - try multiple prefixes
        possible_prefixes = [
            "model.glb_points_head.",
            "glb_points_head.",
            "module.glb_points_head.",
            "module.model.glb_points_head.",
        ]
        
        glb_head_state = {}
        
        for prefix in possible_prefixes:
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    new_key = k[len(prefix):]
                    glb_head_state[new_key] = v
            if glb_head_state:
                logging.info(f"MotionHeadV2: Found glb_points_head weights with prefix '{prefix}'")
                break
        
        if not glb_head_state:
            # Last resort: check if state_dict directly contains DPT keys
            dpt_keys = ['projects.', 'resize_layers.', 'scratch.']
            has_dpt_keys = any(any(k.startswith(dk) for dk in dpt_keys) for k in state_dict.keys())
            
            if has_dpt_keys:
                glb_head_state = state_dict
                logging.info("MotionHeadV2: Using state_dict directly (contains DPT keys without prefix)")
            else:
                logging.warning("MotionHeadV2: Could not find glb_points_head weights in checkpoint!")
                logging.warning(f"MotionHeadV2: Available top-level keys: {set(k.split('.')[0] for k in state_dict.keys())}")
                return
        
        # Adapt state dict for MotionHeadV2
        adapted_state = {}
        skipped_keys = []
        loaded_keys = []
        
        for key, value in glb_head_state.items():
            # Skip output_conv2 final layer (different output_dim: 4 vs 3)
            if "output_conv2.2" in key:
                skipped_keys.append(f"{key} (output_dim mismatch)")
                continue
            
            # Copy all other weights directly
            adapted_state[key] = value
            loaded_keys.append(key)
        
        # Load adapted weights
        missing_keys, unexpected_keys = self.load_state_dict(adapted_state, strict=False)
        
        logging.info(f"MotionHeadV2: Loaded {len(loaded_keys)} keys from glb_points_head")
        logging.info(f"MotionHeadV2: Skipped keys: {skipped_keys}")
        if missing_keys:
            # Filter out expected missing keys (output_conv2.2)
            expected_missing = [k for k in missing_keys if 'output_conv2.2' in k]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]
            if unexpected_missing:
                logging.warning(f"MotionHeadV2: Unexpected missing keys: {unexpected_missing}")
        if unexpected_keys:
            logging.warning(f"MotionHeadV2: Unexpected keys in checkpoint: {unexpected_keys}")

    def _activate_head_stub(self, x):
        """
        Helper to handle activation logic.
        """
        # Scene Flow predicts offset (dx, dy, dz) which can be negative
        pred = x
        if self.activation == "tanh":
            pred = torch.tanh(x)
        elif self.activation == "sigmoid":
            pred = torch.sigmoid(x)
        
        # Dummy confidence map (all 1s)
        conf = torch.ones_like(x[:, :1])
        
        return pred, conf

    def _forward_impl(
        self,
        aggregated_tokens_list,
        patch_h=None,
        patch_w=None,
        patch_start_idx=None,
        view_start_idx=None,
        view_end_idx=None,
        meta_data=None,
    ):
        """
        Core forward pass through DPT layers.
        """
        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]

            # --- Handling Input Dimensions ---
            if x.ndim == 3:  # [B, L, C]
                if patch_start_idx is not None:
                    x = x[:, patch_start_idx:, :]
                
                # Standard LayerNorm (no time conditioning)
                x = self.norm(x)
                
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            elif x.ndim == 4:  # [B, C, H, W]
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                
                x_flat = self.norm(x_flat)
                
                x = x_flat.transpose(1, 2).view(B, C, H, W)
            
            # --- DPT Projections ---
            x = self.projects[dpt_idx](x)
            
            H_input = patch_h * self.patch_size
            W_input = patch_w * self.patch_size
            if self.pos_embed:
                x = self._apply_pos_embed(x, W_input, H_input)

            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1

        # --- Fusion ---
        out = self.scratch_forward(out)
        
        target_h = int(patch_h * self.patch_size / self.down_ratio)
        target_w = int(patch_w * self.patch_size / self.down_ratio)
        
        out = F.interpolate(
            out,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W_input, H_input)

        if self.feature_only:
            return out

        # --- Output Head ---
        out2 = self.scratch.output_conv2(out)
        
        preds, conf = self._activate_head_stub(out2)

        # Format output
        if preds.ndim == 4 and preds.shape[1] != self.scratch.output_conv2[-1].out_channels:
            preds = preds.permute(0, 3, 1, 2).contiguous()

        return dict(scene_flow=preds, flow_confidence=conf)

    def forward(
        self,
        features,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        """
        Forward pass for scene flow prediction.
        
        Uses frame 0 as the reference frame and predicts flow to all other frames.
        
        Args:
            features: List of feature tensors from backbone, each [B*N, L, C] or [B, N, L, C]
            patch_h: Height in patches
            patch_w: Width in patches
            meta_data: Dictionary containing 'frames' (number of frames N)
            patch_start_idx: Optional start index for patch tokens
            
        Returns:
            dict with:
                - scene_flow: [B, 1, N, 3, H, W] - flow from frame 0 to all frames
                - flow_confidence: [B, 1, N, 1, H, W] - confidence map
        """
        # 1. Resolve dimensions from metadata
        N = meta_data["frames"][0]
        if isinstance(N, torch.Tensor):
            N = N.item()
        
        # Infer batch size from features
        feat_sample = features[0]
        if feat_sample.ndim == 4:  # [B, N, L, C]
            B = feat_sample.shape[0]
        else:  # [B*N, L, C]
            B = feat_sample.shape[0] // N
        
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size
            if isinstance(patch_w, torch.Tensor):
                patch_w = patch_w.item()
            if isinstance(patch_h, torch.Tensor):
                patch_h = patch_h.item()

        # 2. Prepare features - flatten to [B*N, L, C] if needed
        features_flat_list = []
        for feat in features:
            if feat.ndim == 4:  # [B, N, L, C]
                feat_flat = feat.view(B * N, *feat.shape[2:])
            else:  # Already [B*N, L, C]
                feat_flat = feat
            features_flat_list.append(feat_flat)

        # 3. Run forward pass through DPT
        outputs = self._forward_impl(
            aggregated_tokens_list=features_flat_list,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_start_idx=patch_start_idx,
        )

        # 4. Format outputs
        scene_flow_flat = outputs['scene_flow']  # [B*N, C, H, W]
        H_img, W_img = scene_flow_flat.shape[-2:]
        C_out = scene_flow_flat.shape[1]
        
        # Reshape to [B, N, C, H, W]
        scene_flow = scene_flow_flat.view(B, N, C_out, H_img, W_img)
        
        # Add query dimension (always 1, since frame 0 is the only reference)
        # Output shape: [B, 1, N, C, H, W]
        scene_flow = scene_flow.unsqueeze(1)
        
        outputs['scene_flow'] = scene_flow

        if 'flow_confidence' in outputs:
            flow_conf = outputs['flow_confidence'].view(B, N, -1, H_img, W_img)
            outputs['flow_confidence'] = flow_conf.unsqueeze(1)
        
        return outputs
