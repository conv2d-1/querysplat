"""
VDPM model wrapper for integration with internal framework.
Video Depth Prediction Model for multi-frame 3D reconstruction.
"""

import logging
from typing import Dict, List, Tuple, Union
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import PyTorchModelHubMixin
from einops import repeat

from hAlgorithm.modules.models2.external.vdpm.dpm.model import VDPM as VDPMModel
from hAlgorithm.modules.models2.external.vdpm.dpm.aggregator import Aggregator
from hAlgorithm.modules.models2.external.vdpm.dpm.decoder import Decoder
from hAlgorithm.modules.models2.external.vdpm.vggt.heads.camera_head import CameraHead
from hAlgorithm.modules.models2.external.vdpm.vggt.heads.dpt_head import DPTHead
from hAlgorithm.modules.models2.external.vdpm.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from hAlgorithm.modules.models2.external.vdpm.vggt.utils.rotation import quat_to_mat


# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def freeze_all_params(modules):
    """Freeze all parameters in the given modules."""
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


class VDPM(nn.Module, PyTorchModelHubMixin):
    """
    VDPM model wrapper for video depth prediction with multi-frame reconstruction.
    
    This model outputs:
    - pts3d: 3D pointmap in camera frame per timestep
    - pts3d_cam: 3D pointmap in camera frame (same as pts3d for VDPM)
    - conf: Confidence scores per pointmap
    - extrinsics: Camera extrinsics (world-to-camera) per frame
    - intrinsics: Camera intrinsics per frame
    - scene_flow: Scene flow vectors (computed from temporal pointmaps)
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        decoder_depth: int = 4,
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = None,
    ):
        """
        Initialize VDPM model.

        Args:
            img_size: Input image size (default 518 for DINOv2)
            patch_size: Patch size for ViT encoder
            embed_dim: Embedding dimension
            decoder_depth: Depth of the decoder
            pretrained_checkpoint_path: Path to pretrained checkpoint
            load_specific_pretrained_submodules: Whether to load specific submodules only
            specific_pretrained_submodules: List of submodules to load
        """
        super().__init__()

        # Store config
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_depth = decoder_depth
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules

        # Create config namespace for compatibility with original VDPM model
        cfg = SimpleNamespace()
        cfg.model = SimpleNamespace()
        cfg.model.decoder_depth = decoder_depth

        # Initialize components
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.decoder = Decoder(
            cfg,
            dim_in=2 * embed_dim,
            embed_dim=embed_dim,
            depth=decoder_depth
        )
        self.point_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)

        # Freeze patch embedding (same as original)
        self.set_freeze()

        # Load pretrained weights
        self._load_pretrained_weights()

        # Runtime settings
        self.memory_efficient_inference: bool = False
        self.use_amp: bool = True

    def set_freeze(self):
        """Freeze patch embedding layer."""
        to_be_frozen = [self.aggregator.patch_embed]
        freeze_all_params(to_be_frozen)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint."""
        if self.pretrained_checkpoint_path is not None:
            if self.pretrained_checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                ckpt = load_file(self.pretrained_checkpoint_path)
                strict = False
            else:
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False, map_location='cpu')
                strict = True

            if not self.load_specific_pretrained_submodules:
                logging.info(f"Loading pretrained VDPM weights from {self.pretrained_checkpoint_path} ...")
                # Handle different checkpoint formats
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                
                # Filter out unwanted heads from VGGT checkpoint
                exclude = ["depth_head", "track_head"]
                state_dict = {k: v for k, v in state_dict.items() if k.split('.')[0] not in exclude}
                
                logging.info(self.load_state_dict(state_dict, strict=strict))
            else:
                logging.info(f"Loading pretrained VDPM weights for specific submodules: {self.specific_pretrained_submodules} ...")
                filtered_ckpt = {}
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                for ckpt_key, ckpt_value in state_dict.items():
                    for submodule in self.specific_pretrained_submodules:
                        if ckpt_key.startswith(submodule):
                            filtered_ckpt[ckpt_key] = ckpt_value
                logging.info(self.load_state_dict(filtered_ckpt, strict=False))

    def load_state_dict(self, ckpt, strict=True, **kw):
        """Override load_state_dict to filter out unwanted heads."""
        # Filter out depth_head and track_head which are not needed
        exclude = ["depth_head", "track_head"]
        ckpt = {k: v for k, v in ckpt.items() if k.split('.')[0] not in exclude}
        return super().load_state_dict(ckpt, strict=strict, **kw)

    def forward(
        self,
        rgb,
        prompt_depth=None,
        ray_directions=None,
        camera_quats=None,
        camera_trans=None,
        meta_data=None,
        **kwargs,
    ):
        """
        Forward pass of VDPM model.

        Args:
            rgb: Input images, shape (B, V, C, H, W) where V = num_frames
            prompt_depth: Optional prompt depth (not used in VDPM)
            ray_directions: Optional ray directions (not used in VDPM)
            camera_quats: Optional camera quaternions (not used in VDPM)
            camera_trans: Optional camera translations (not used in VDPM)
            meta_data: Optional metadata

        Returns:
            Dictionary containing:
            - depth: Pointmap in camera frame (B, V, H, W, 3)
            - confidence: Confidence scores (B, V, H, W)
            - global_points: Pointmap in world frame (B, V, H, W, 3)
            - scene_flow: Scene flow vectors (B, V, H, W, 3)
            - cam_quats: Camera quaternions (B, V, 4)
            - cam_trans: Camera translations (B, V, 3)
            - intrinsics: Camera intrinsics (B, V, 3, 3)
            - extrinsics: Camera extrinsics (B, V, 3, 4)
        """
        batch_size_per_view, num_views, _, height, width = rgb.shape
        img_shape = (int(height), int(width))

        # VDPM expects images in format [B, S, C, H, W] with values in [0, 1]
        # Our input is in [-1, 1], convert to [0, 1]
        images = (rgb + 1) * 0.5

        # Run inference using the model's internal inference method
        autocast_amp = torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16)

        with autocast_amp:
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        S = images.shape[1]  # num_views/frames

        # === VDPM Dynamic Pointmap (for scene flow and visualization) ===
        # This gives tracked points in frame 0's camera coordinates
        dynamic_pointmaps = []
        ones = torch.ones(batch_size_per_view, S, dtype=torch.int64, device=images.device)

        # Per-timestep pointmap prediction
        for time_ in range(S):
            cond_view_idxs = ones * time_

            with autocast_amp:
                decoded_tokens = self.decoder(
                    images, aggregated_tokens_list, patch_start_idx, cond_view_idxs
                )
            
            padded_decoded_tokens = [None] * len(aggregated_tokens_list)
            for idx, layer_idx in enumerate(self.point_head.intermediate_layer_idx):
                padded_decoded_tokens[layer_idx] = decoded_tokens[idx]

            with torch.cuda.amp.autocast(enabled=False):
                pts3d, pts3d_conf = self.point_head(
                    padded_decoded_tokens, images, patch_start_idx
                )

            dynamic_pointmaps.append({
                "pts3d": pts3d,
                "conf": pts3d_conf
            })

        # Camera pose prediction
        with autocast_amp:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]  # Use last iteration

        # Convert pose encoding to extrinsics and intrinsics
        # pose_enc shape: [B, S, 9] = [T(3), quat(4), fov(2)]
        HW = dynamic_pointmaps[0]["pts3d"].shape[2:4]  # (H, W)
        
        with torch.cuda.amp.autocast(enabled=False):
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_enc.float(), HW, build_intrinsics=True
            )

        # Extract camera quaternions and translations from pose encoding
        cam_trans = pose_enc[..., :3]  # [B, S, 3]
        cam_quats = pose_enc[..., 3:7]  # [B, S, 4] - XYZW order

        # Build full extrinsics matrices [B, S, 4, 4]
        extrinsics_full = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).unsqueeze(0).unsqueeze(0)
        extrinsics_full = extrinsics_full.expand(batch_size_per_view, S, 4, 4).clone()
        extrinsics_full[..., :3, :] = extrinsics  # [B, S, 3, 4]

        # Compute world-to-camera transformation (inverse of camera-to-world)
        # VDPM outputs camera-to-world (c2w), we need world-to-camera (w2c)
        c2w = extrinsics_full
        w2c = torch.linalg.inv(c2w)

        # === Process Dynamic Pointmap (for scene flow) ===
        # Stack pointmaps: pts3d_all[t] is the dynamic pointmap at timestep t
        # Shape: [S, B, V=1, H, W, 3] -> we need [B, S, H, W, 3]
        # IMPORTANT: Dynamic pointmap is in FRAME 0's CAMERA COORDINATE SYSTEM
        # This means pts3d_all[t] represents where frame 0's points are at time t,
        # expressed in frame 0's camera coordinates.
        dynamic_pts3d_all = torch.stack([pm["pts3d"][:, 0] for pm in dynamic_pointmaps], dim=1)  # [B, S, H, W, 3]
        dynamic_conf_all = torch.stack([pm["conf"][:, 0] for pm in dynamic_pointmaps], dim=1)  # [B, S, H, W]

        # VDPM Scene Flow Computation:
        # Since dynamic pointmap is already in frame 0's camera coordinates,
        # scene flow is simply the difference between pointmaps at different times.
        # For static objects: dynamic_pts3d_all[t] == dynamic_pts3d_all[0], so scene_flow = 0 (correct!)
        # For moving objects: dynamic_pts3d_all[t] != dynamic_pts3d_all[0], scene_flow shows motion
        dynamic_pts3d_ref = dynamic_pts3d_all[:, 0:1, ...].expand_as(dynamic_pts3d_all)  # Reference frame (t=0)
        scene_flow = dynamic_pts3d_all - dynamic_pts3d_ref  # [B, S, H, W, 3] in frame 0's camera coordinates

        # === Process Pointmap for visualization ===
        # For point cloud visualization, transform dynamic pointmap to world coordinates
        # Use frame 0's c2w since all dynamic pointmaps are in frame 0's camera coordinates
        c2w_ref = c2w[:, 0:1, :, :]  # [B, 1, 4, 4] - frame 0's c2w
        
        pts3d_homo = torch.cat([
            dynamic_pts3d_all, 
            torch.ones(*dynamic_pts3d_all.shape[:-1], 1, device=dynamic_pts3d_all.device, dtype=dynamic_pts3d_all.dtype)
        ], dim=-1)  # [B, S, H, W, 4]
        
        B, S, H, W, _ = pts3d_homo.shape
        pts3d_homo_flat = pts3d_homo.view(B, S, -1, 4).transpose(-1, -2)  # [B, S, 4, H*W]
        
        # Apply SAME c2w (frame 0's) to all frames for world coordinates
        # This ensures static objects stay static in world coordinates
        c2w_ref_expanded = c2w_ref.expand(B, S, 4, 4)  # [B, S, 4, 4]
        pts3d_world_flat = torch.matmul(c2w_ref_expanded, pts3d_homo_flat)  # [B, S, 4, H*W]
        pts3d_world = pts3d_world_flat[:, :, :3, :].transpose(-1, -2).view(B, S, H, W, 3)
        
        # === Transform to per-frame camera coordinates for "depth" output ===
        # depth[t] should be in frame t's camera coordinates, not frame 0's
        # Transform: pts_cam_t = w2c[t] @ c2w[0] @ pts_cam_0
        # relative_transform[t] = w2c[t] @ c2w[0] transforms from frame 0's camera to frame t's camera
        relative_transform = torch.matmul(w2c, c2w_ref.expand(B, S, 4, 4))  # [B, S, 4, 4]
        pts3d_per_frame_cam_flat = torch.matmul(relative_transform, pts3d_homo_flat)  # [B, S, 4, H*W]
        pts3d_per_frame_cam = pts3d_per_frame_cam_flat[:, :, :3, :].transpose(-1, -2).view(B, S, H, W, 3)

        # Convert quaternions from XYZW to WXYZ format for compatibility
        # VDPM uses XYZW (scalar-last), Any4D style uses WXYZ (scalar-first)
        # cam_quats is [B, S, 4] in XYZW format
        cam_quats_wxyz = cam_quats[..., [3, 0, 1, 2]]  # Convert to WXYZ

        # Construct output dictionary matching Any4D format
        total = {
            "depth": pts3d_per_frame_cam,  # [B, S, H, W, 3] - pointmap in each frame's camera coordinates
            "confidence": dynamic_conf_all,  # [B, S, H, W] - confidence
            "global_points": pts3d_world,  # [B, S, H, W, 3] - pointmap in world frame (for visualization)
            "global_confidence": dynamic_conf_all,  # [B, S, H, W]
            "scene_flow": scene_flow,  # [B, S, H, W, 3] - scene flow in frame 0's camera coords
            "dynamic_pointmap": dynamic_pts3d_all,  # [B, S, H, W, 3] - dynamic pointmap in frame 0's camera coords
            "dynamic_confidence": dynamic_conf_all,  # [B, S, H, W] - dynamic pointmap confidence
            "cam_quats": cam_quats_wxyz,  # [B, S, 4] - camera quaternions (WXYZ)
            "cam_trans": cam_trans,  # [B, S, 3] - camera translations
            "intrinsics": intrinsics,  # [B, S, 3, 3] - camera intrinsics
            "extrinsics": w2c[:, :, :3, :],  # [B, S, 3, 4] - camera extrinsics (w2c, world-to-camera)
            "extrinsics_full": w2c,  # [B, S, 4, 4] - full extrinsics matrix (w2c)
            "c2w": c2w,  # [B, S, 4, 4] - camera-to-world matrix
            "metric_scaling_factor": torch.ones(batch_size_per_view, S, 1, device=rgb.device),  # Placeholder
        }

        return total
