import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Optional, Tuple

from hAlgorithm.modules.models2.head.vggt_dpt_head import VGGTDPTHead


# ==========================================
# 1. 辅助模块 (Time Embedding & AdaLN)
# ==========================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes scalar time steps into high-dimensional embeddings.
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_steps: [B, M] normalized time values in [0, 1].
        Returns:
            embedding: [B, M, dim]
        """
        # Ensure time_steps is float
        time_steps = time_steps.float()
        
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        
        # [B, M, 1] * [Half] -> [B, M, Half]
        args = time_steps.unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
            
        return embedding


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Replaces standard LayerNorm to inject time information (gamma, beta) predicted from time_emb.
    """
    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        # elementwise_affine=False because we predict scale/shift dynamically
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * channels)
        )
        
        # Zero initialization ensures it acts like Identity Norm initially
        nn.init.zeros_(self.time_mlp[1].weight)
        nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch_Total, L, Channels]
            time_emb: [Batch_Total, Time_Dim]
        """
        # 1. Standard Normalization
        x_norm = self.norm(x)
        
        # 2. Predict Scale & Shift
        # [Batch, Time_Dim] -> [Batch, 2*C] -> [Batch, 1, 2*C]
        style = self.time_mlp(time_emb).unsqueeze(1)
        gamma, beta = style.chunk(2, dim=-1)
        
        # 3. Modulate
        return x_norm * (1 + gamma) + beta


# ==========================================
# 2. Motion Head 实现
# ==========================================

class MotionHead(VGGTDPTHead):
    """
    Motion Head for Scene Flow or Dynamic Pointmap Prediction.
    Inherits from VGGTDPTHead.
    
    Can initialize from a pre-trained global points head (VGGTDPTHead) via:
    - pretrain: path to motion head checkpoint
    - load_from_glb_points_head: path to global points head checkpoint (adapts weights)
    
    Prediction Types:
    - "scene_flow": Predicts 3D displacement vectors (dx, dy, dz) from source to target frame
    - "dynamic_pointmap": Predicts absolute 3D positions (X, Y, Z) at each target frame
    """

    # Class-level constant for valid prediction types
    VALID_PREDICTION_TYPES = ("scene_flow", "dynamic_pointmap")

    def __init__(
        self,
        dim_in: int = 1024,
        time_embed_dim: int = 256,
        output_dim: int = 3,
        load_from_glb_points_head: str = None,
        prediction_type: str = "scene_flow",  # "scene_flow" or "dynamic_pointmap"
        **kwargs
    ):
        super().__init__(dim_in=dim_in, output_dim=output_dim, **kwargs)

        # Validate prediction_type
        if prediction_type not in self.VALID_PREDICTION_TYPES:
            raise ValueError(
                f"prediction_type must be one of {self.VALID_PREDICTION_TYPES}, "
                f"got '{prediction_type}'"
            )
        self.prediction_type = prediction_type
        logging.info(f"MotionHead initialized with prediction_type='{prediction_type}'")

        self.time_embed_dim = time_embed_dim
        self.time_encoder = SinusoidalTimeEmbedding(time_embed_dim)
        
        if hasattr(self, 'norm'):
            del self.norm 
        self.adaln = AdaLN(channels=dim_in, time_embed_dim=time_embed_dim)
        
        # Load weights from global points head (adapts architecture differences)
        if load_from_glb_points_head is not None:
            self._load_from_glb_points_head(load_from_glb_points_head)

    def _load_from_glb_points_head(self, ckpt_path: str):
        """
        Load weights from a pre-trained global points head (VGGTDPTHead).
        
        Adapts for architecture differences:
        - norm (LayerNorm) -> adaln.norm (LayerNorm)  
        - output_conv2[-1] has different output_dim (4 vs 3), skip this layer
        
        Args:
            ckpt_path: Path to checkpoint containing 'glb_points_head' state dict
        """
        logging.info(f"MotionHead: Loading weights from global points head: {ckpt_path}")
        
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
                logging.info(f"MotionHead: Found glb_points_head weights with prefix '{prefix}'")
                break
        
        if not glb_head_state:
            # Last resort: check if state_dict directly contains DPT keys
            dpt_keys = ['projects.', 'resize_layers.', 'scratch.']
            has_dpt_keys = any(any(k.startswith(dk) for dk in dpt_keys) for k in state_dict.keys())
            
            if has_dpt_keys:
                glb_head_state = state_dict
                logging.info("MotionHead: Using state_dict directly (contains DPT keys without prefix)")
            else:
                logging.warning("MotionHead: Could not find glb_points_head weights in checkpoint!")
                logging.warning(f"MotionHead: Available top-level keys: {set(k.split('.')[0] for k in state_dict.keys())}")
                return
        
        # Adapt state dict for MotionHead
        adapted_state = {}
        skipped_keys = []
        loaded_keys = []
        
        for key, value in glb_head_state.items():
            # Skip keys that don't exist in MotionHead or have size mismatch
            
            # 1. Adapt norm -> adaln.norm
            if key.startswith("norm."):
                new_key = "adaln." + key
                adapted_state[new_key] = value
                loaded_keys.append(f"{key} -> {new_key}")
                continue
            
            # 2. Skip output_conv2 final layer (different output_dim: 4 vs 3)
            if "output_conv2.2" in key:
                skipped_keys.append(f"{key} (output_dim mismatch)")
                continue
            
            # 3. Copy all other weights directly
            adapted_state[key] = value
            loaded_keys.append(key)
        
        # Load adapted weights
        missing_keys, unexpected_keys = self.load_state_dict(adapted_state, strict=False)
        
        logging.info(f"MotionHead: Loaded {len(loaded_keys)} keys from glb_points_head")
        logging.info(f"MotionHead: Skipped keys: {skipped_keys}")
        if missing_keys:
            # Filter out expected missing keys (time_encoder, adaln.time_mlp, output_conv2.2)
            expected_missing = [k for k in missing_keys if 
                               'time_encoder' in k or 
                               'time_mlp' in k or 
                               'output_conv2.2' in k]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]
            if unexpected_missing:
                logging.warning(f"MotionHead: Unexpected missing keys: {unexpected_missing}")
        if unexpected_keys:
            logging.warning(f"MotionHead: Unexpected keys in checkpoint: {unexpected_keys}")

    def _activate_head_stub(self, x):
        """
        [RESTORED] Helper to handle activation logic.
        Separating this makes it easier to change activation functions later.
        """
        # 1. Prediction Branch
        # Scene Flow usually predicts offset (dx, dy, dz) which can be negative, 
        # so we typically use Identity (no activation) or Tanh (if normalized).
        pred = x
        if self.activation == "tanh":
            pred = torch.tanh(x)
        elif self.activation == "sigmoid":
            pred = torch.sigmoid(x)
            
        # 2. Confidence Branch
        # If the head doesn't explicitly predict confidence channels, 
        # we return a dummy confidence map (all 1s).
        # Note: If you want real confidence, you'd need to change output_conv2 
        # to output (output_dim + 1) channels and split them here.
        conf = torch.ones_like(x[:, :1]) 
        
        return pred, conf

    def _forward_impl(
        self,
        aggregated_tokens_list,
        time_emb=None,
        patch_h=None,
        patch_w=None,
        patch_start_idx=None,
        view_start_idx=None, 
        view_end_idx=None,
        meta_data=None,
    ):
        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]

            # --- Handling Input Dimensions ---
            if x.ndim == 3: # [B, L, C]
                if patch_start_idx is not None:
                    x = x[:, patch_start_idx:, :]
                
                if time_emb is not None:
                    x = self.adaln(x, time_emb)
                else:
                    x = self.adaln.norm(x)
                
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            elif x.ndim == 4: # [B, C, H, W]
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                
                if time_emb is not None:
                    x_flat = self.adaln(x_flat, time_emb)
                else:
                    x_flat = self.adaln.norm(x_flat)
                
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
        
        # [CALLING RESTORED FUNCTION]
        preds, conf = self._activate_head_stub(out2)

        # Format output
        if preds.ndim == 4 and preds.shape[1] != self.scratch.output_conv2[-1].out_channels:
             preds = preds.permute(0, 3, 1, 2).contiguous()

        # Output key depends on prediction_type
        if self.prediction_type == "dynamic_pointmap":
            return dict(dynamic_pointmap=preds, pointmap_confidence=conf)
        else:
            return dict(scene_flow=preds, flow_confidence=conf)

    def forward(
        self,
        features, 
        query_times, 
        patch_h=None,
        patch_w=None,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        # 1. Resolve core dimensions from metadata and query inputs
        # B, M are determined by the query_times shape; N is from metadata.
        """
        Output shape: [B, query_num, frame_num, 3, H, W]
        """
        B, M = query_times.shape
        N = meta_data["frames"][0]
        
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        # 2. Generate and broadcast time embeddings
        # Encode time: [B, M] -> [B, M, T_dim]
        time_emb = self.time_encoder(query_times)
        
        # Broadcast time query to all N frames: [B, M, T_dim] -> [B, M, N, T_dim] -> [B*M*N, T_dim]
        time_emb_expanded = time_emb.unsqueeze(2).expand(B, M, N, -1).reshape(B * M * N, -1)

        # 3. Broadcast backbone features to match query times
        features_expanded_list = []
        
        for feat in features:
            # Ensure feature map is viewed as [B, N, ...] irrespective of upstream formatting ([B, N, ...] or [B*N, ...])
            if feat.shape[0] == B and feat.shape[1] == N:
                feat_bn = feat
            else:
                # Handle flattened batch dimension [B*N, ...] -> [B, N, ...]
                feat_bn = feat.view(B, N, *feat.shape[1:])
            
            # Expand features M times for each query: [B, N, ...] -> [B, M, N, ...]
            expand_shape = [B, M, N] + list(feat_bn.shape[2:])
            feat_m = feat_bn.unsqueeze(1).expand(*expand_shape)
            
            # Flatten batch dimensions for processing: [B*M*N, ...]
            features_expanded_list.append(feat_m.reshape(B * M * N, *feat_bn.shape[2:]))

        # 4. Run internal forward pass (DPT + AdaLN)
        outputs = self._forward_impl(
            aggregated_tokens_list=features_expanded_list,
            time_emb=time_emb_expanded,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_start_idx=patch_start_idx,
        )

        # 5. Format outputs based on prediction_type
        if self.prediction_type == "dynamic_pointmap":
            output_flat = outputs['dynamic_pointmap']
            H_img, W_img = output_flat.shape[-2:]
            C_out = output_flat.shape[1]
            
            # Reshape flat output to structured format: [B, M, N, C, H, W]
            dynamic_pointmap = output_flat.view(B, M, N, C_out, H_img, W_img)
            outputs['dynamic_pointmap'] = dynamic_pointmap
            
            if 'pointmap_confidence' in outputs:
                outputs['pointmap_confidence'] = outputs['pointmap_confidence'].view(B, M, N, -1, H_img, W_img)
        else:
            scene_flow_flat = outputs['scene_flow']
            H_img, W_img = scene_flow_flat.shape[-2:]
            C_out = scene_flow_flat.shape[1]
            
            # Reshape flat output to structured format: [B, M, N, C, H, W]
            scene_flow = scene_flow_flat.view(B, M, N, C_out, H_img, W_img)
            outputs['scene_flow'] = scene_flow

            if 'flow_confidence' in outputs:
                outputs['flow_confidence'] = outputs['flow_confidence'].view(B, M, N, -1, H_img, W_img)
        
        return outputs