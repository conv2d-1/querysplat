import logging
import time

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.sdk.base import MVBase

logger = logging.getLogger(__name__)


class MOVIES(MVBase):
    # Encoder module names that are frozen when freeze_encoders_for_motion=True.
    _ENCODER_MODULE_NAMES = (
        "rgb_encoder",
        "fuse_encoder",
        "camera_encoder",
        "ray_in_world_encoder",
        "time_encoder",
    )

    def __init__(
        self,
        time_encoder=None,
        appearance_head=None,
        motion_head=None,
        # Optional motion-conditioned Gaussian attribute predictor for 4DGS.
        # When None (default) the forward pass is 100% identical to the
        # original MOVIES pipeline — no overhead, no side effects.
        gaussian_head=None,
        # [NEW] Camera Token Encoder
        camera_token_encoder=None,
        # When True, all backbone/fusion encoders are frozen (eval + no grad)
        # each forward pass so that only the motion_head receives gradient
        # updates.  Mirrors the freeze_encoders_for_motion flag in SparseMotion.
        freeze_encoders_for_motion: bool = False,
        **kwargs,
    ):
        # Populate Base.freeze_modules before super().__init__() so that the
        # freeze list is ready before any module is registered.
        if freeze_encoders_for_motion and "freeze_modules" not in kwargs:
            kwargs["freeze_modules"] = list(self._ENCODER_MODULE_NAMES)
        super(MOVIES, self).__init__(**kwargs)
        self.freeze_encoders_for_motion = freeze_encoders_for_motion

        # ... (Existing encoders) ...
        self.time_encoder = self._instantiate_and_register(time_encoder, "time_encoder")
        self.appearance_head = self._instantiate_and_register(appearance_head, "appearance_head")
        self.motion_head = self._instantiate_and_register(motion_head, "motion_head")
        # Dynamic Gaussian head — optional, None by default.
        self.gaussian_head = self._instantiate_and_register(gaussian_head, "gaussian_head")
        # [NEW] Instantiate Camera Token Encoder
        self.camera_token_encoder = self._instantiate_and_register(camera_token_encoder, "camera_token_encoder")

    def aggregator(self, rgb, scale, prompt_depth, intrinsics, ray_directions, w2c, ray_world, meta_data, time_idx, prompt_extra=None, **kwargs):
        
        if rgb.ndim == 5:
            frame_num, view_num = meta_data["frames"][0], meta_data["views"][0]
            b, n, c, h, w = rgb.shape
            rgb = rgb.view(b * n, c, h, w)
            
            def _reshape_if_exists(x, shape_suffix):
                return x.view(b * n, *shape_suffix) if x is not None else None

            intrinsics = _reshape_if_exists(intrinsics, (3, 3))
            w2c = _reshape_if_exists(w2c, (4, 4))
            ray_world = _reshape_if_exists(ray_world, ray_world.shape[-3:] if ray_world is not None else ())

        # ------------------------------------------------------------------
        # Strategy 1: Plücker Embedding Generation
        # ------------------------------------------------------------------
        prompt_ray_in_world = None
        if self.ray_in_world_encoder is not None and intrinsics is not None and w2c is not None:
            prompt_ray_in_world = self.ray_in_world_encoder(
                ray_directions=ray_world,  # 实际上被 Encoder 忽略，传 None 也行
                intrinsics=intrinsics, 
                w2c=w2c, 
                meta_data=meta_data
            )

        # ------------------------------------------------------------------
        # Backbone Feature Extraction
        # ------------------------------------------------------------------
        if self.rgb_encoder is not None:
            patch_features = self.rgb_encoder(rgb, meta_data=meta_data)
        else:
            # rgb is already [B*N, C, H, W] after the ndim==5 reshape above,
            # or was already 4-D on entry — no further view() needed.
            patch_features = rgb

        # [Fix 2] REMOVED FLATTENING HERE
        # We pass spatial features [B*N, C, H, W] directly to fuse_encoder.
        # This makes adding prompt_ray_in_world (which is spatial) much easier inside the encoder.
        # patch_features_tokens = patch_features.flatten(2).transpose(1, 2) <--- Deleted

        # ------------------------------------------------------------------
        # Strategy 2 & Time: Token Preparation
        # ------------------------------------------------------------------
        # [Fix 1] Initialize to None to avoid UnboundLocalError
        camera_token = None
        time_token = None

        # Camera Token
        if self.camera_token_encoder is not None and intrinsics is not None and w2c is not None:
            # [B*N, 1, C]
            camera_token = self.camera_token_encoder(intrinsics, w2c)

        # Time Token
        if self.time_encoder is not None:
            if time_idx is not None:
                time_val = time_idx.view(-1)
            elif meta_data is not None and "time_idx" in meta_data:
                # 使用 metadata 中的真实时间戳
                time_val = meta_data["time_idx"].view(-1) # [B*N]
                print('Using time_idx from meta_data')
            else:
                # 如果没有 time_idx，假设时间位于中间 (0.5)
                # rgb 此时已经是 reshape 过的 [B*N, C, H, W]
                B_N = rgb.shape[0]
                time_val = torch.full((B_N,), 0.5, device=rgb.device, dtype=torch.float32)
                print('Using time_idx from assumption')

            # [B*N, 1, C] (Assuming your TimeEncoder returns [B*N, C])
            time_token = self.time_encoder(time_val)
            
            # Ensure shape is [B*N, 1, C]
            if time_token.dim() == 2:
                time_token = time_token.unsqueeze(1)

        # ------------------------------------------------------------------
        # Fuse Encoder (The Core MoVieS Logic)
        # ------------------------------------------------------------------
        if self.fuse_encoder is not None:
            # We pass spatial patch_features here!
            patch_features, pos, patch_start_idx = self.fuse_encoder(
                patch_tokens=patch_features, # Passing [B, C, H, W]
                prompt_depth=prompt_depth,
                prompt_ray=None, # Assuming prompt_ray is not used or derived from ray_directions externally
                prompt_ray_in_world=prompt_ray_in_world, # Spatial Prompt
                prompt_extra=prompt_extra,
                scale=scale,
                intrinsics=intrinsics,
                w2c=w2c,
                meta_data=meta_data,
                camera_token=camera_token, # Pre-calculated token
                time_token=time_token,     # Pre-calculated token
            )
        else:
            # Fallback if no fuse encoder (unlikely for MoVieS)
            # Manual flatten if needed for output
            if patch_features.ndim == 4:
                patch_features = patch_features.flatten(2).transpose(1, 2)
            pos = None
            patch_start_idx = 0

        return patch_features, pos, patch_start_idx
    
    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        ray_world=None,
        query_points=None, # For tracking
        query_times=None,  # [NEW] Explicit query times for inference
        time_idx=None,
        meta_data=None,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        b, n, c, h, w = rgb.shape

        # 1. Run Aggregator (TimeFuseEncoder inside)
        # Output: patch_features is a list of tensors (multi-scale)
        patch_features, pos, patch_start_idx = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, 
            intrinsics=intrinsics, ray_directions=ray_directions, 
            w2c=w2c, ray_world=ray_world, meta_data=meta_data, time_idx=time_idx,
            # Pass extra prompts if needed (e.g. from kwargs)
            **kwargs
        )

        results = dict()

        # ------------------------------------------------------------------
        # 2. Run Heads
        # ------------------------------------------------------------------
        
        # ... (Existing heads: Depth, Points, Camera, Track, Normal - Keep same) ...
        if self.depth_head is not None:
            depth_results = self.depth_head(patch_features, patch_start_idx=patch_start_idx, meta_data=meta_data)
            for key, val in depth_results.items():
                if val is not None: results[key] = val.view(b, n, *val.shape[-3:])

        if self.glb_points_head is not None:
            glb_results = self.glb_points_head(patch_features, patch_start_idx=patch_start_idx, meta_data=meta_data)
            for key, val in glb_results.items():
                if val is not None: results["global_" + key] = val.view(b, n, *val.shape[-3:])

        if self.camera_head is not None:
            feat = patch_features[-1] if isinstance(patch_features, (list, tuple)) else patch_features
            # Handle both 3D [B*S, L, C] and 4D [B, S, L, C] feature shapes
            if feat.ndim == 3:
                # Reshape from [B*S, L, C] to [B, S, L, C]
                feat = feat.view(b, n, *feat.shape[1:])
            # Select first patch_start_idx tokens from all views: [B, S, patch_start_idx, C]
            camera_tokens = [feat[:, :, :patch_start_idx, :]]
            results["pose_enc"] = self.camera_head(camera_tokens, meta_data=meta_data)

        if self.track_head is not None and query_points is not None:
            track, track_vis, track_conf = self.track_head(patch_features, query_points=query_points, meta_data=meta_data)
            results.update({"track": track, "track_vis": track_vis, "track_confidence": track_conf})

        if self.normal_head is not None:
            normal_results = self.normal_head(patch_features, patch_start_idx=patch_start_idx, meta_data=meta_data)
            for key, val in normal_results.items():
                if val is not None: results[key] = val.view(b, n, *val.shape[1:])

        # ------------------------------------------------------------------
        # Motion Head
        # ------------------------------------------------------------------
        if self.motion_head is not None:
            # Get query_times from parameter or meta_data
            if query_times is not None:
                target_query_times = query_times
            elif meta_data is not None and "query_times" in meta_data:
                target_query_times = meta_data["query_times"]
            else:
                logger.warning("Motion head requires query_times but none provided.")
                raise ValueError("query_times is required for motion head. Pass it as parameter or in meta_data['query_times'].")
            
            # Ensure shape is [B, M]
            if target_query_times.numel() == b * n:
                target_query_times = target_query_times.reshape(b, n)
            
            # Ensure device is correct
            if target_query_times.device != rgb.device:
                target_query_times = target_query_times.to(rgb.device)

            motion_results = self.motion_head(
                features=patch_features, 
                query_times=target_query_times, 
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            
            for key, val in motion_results.items():
                if val is not None:
                    results[key] = val

        # ------------------------------------------------------------------
        # Dynamic Gaussian Head (optional — only runs when configured)
        # Requires scene_flow from the motion head; silently skipped otherwise.
        # Existing training pipelines that do not configure gaussian_head are
        # completely unaffected: this block has zero cost when the head is None.
        # ------------------------------------------------------------------
        if self.gaussian_head is not None and "scene_flow" in results:
            # Reuse src_frame_idx chosen by the motion head so both heads
            # operate on the same reference frame within each training step.
            motion_src_frame = results.get("src_frame_idx")

            gs_results = self.gaussian_head(
                features=patch_features,
                scene_flow=results["scene_flow"],
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
                src_frame_idx=motion_src_frame,
            )

            for key, val in gs_results.items():
                if val is not None:
                    results[key] = val

        # ------------------------------------------------------------------
        # [NEW] Appearance Head
        # ------------------------------------------------------------------
        if self.appearance_head is not None:
            app_results = self.appearance_head(
                features=patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
                **kwargs 
            )
            
            for key, val in app_results.items():
                if val is not None:
                    # 尝试恢复 Batch 和 Frame 维度
                    if val.ndim >= 2 and val.shape[0] == b * n:
                         results[key] = val.view(b, n, *val.shape[1:])
                    elif val.ndim >= 2 and val.shape[0] == b and val.shape[1] == n:
                         results[key] = val
                    else:
                         # 其他情况直接返回 (可能是 Global feature)
                         results[key] = val

        return results