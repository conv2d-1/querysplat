import logging

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.sdk.base import MVBase
from hAlgorithm.utils import instantiate_from_config


def _compute_frame_timestamps(n: int, b: int, device) -> torch.Tensor:
    """Compute normalized timestamps in [0, 1] for n frames, shape [B, N]."""
    if n <= 1:
        return torch.zeros(b, 1, device=device, dtype=torch.float32)
    return torch.linspace(0, 1, n, device=device, dtype=torch.float32).unsqueeze(0).expand(b, -1)


class MVBaseMotion(MVBase):
    """
    Multi-View Base model with Motion Head support.
    
    Extends MVBase with a motion head that predicts scene flow.
    Uses MotionHeadV2 which doesn't require query_times input
    (always uses frame 0 as reference).
    
    Output includes:
        - scene_flow: [B, 1, N, 3, H, W] - flow from frame 0 to all frames
    """

    def __init__(
        self,
        motion_head=None,
        **kwargs,
    ):
        super(MVBaseMotion, self).__init__(**kwargs)

        # Motion Head (V2 - no query_times needed)
        self.motion_head = self._instantiate_and_register(motion_head, "motion_head")

    def forward(
        self,
        rgb,
        scale=None,
        prompt_depth=None,
        intrinsics=None,
        ray_directions=None,
        w2c=None,
        c2w=None,
        ray_world=None,
        query_points=None,
        meta_data=None,
        return_features=False,
        **kwargs,
    ):
        if self.training:
            self.freeze()

        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
        else:
            n = 1
            b, c, h, w = rgb.shape

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth, intrinsics=intrinsics,
            ray_directions=ray_directions, w2c=w2c, c2w=c2w, ray_world=ray_world, rgb_mask=None, meta_data=meta_data
        )

        results = dict()

        # Depth Head
        if self.depth_head is not None:
            depth_results = self.depth_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                prompt_depth=prompt_depth,
                meta_data=meta_data,
            )
            for key, val in depth_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        # Global Points Head
        if self.glb_points_head is not None:
            glb_depth_results = self.glb_points_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in glb_depth_results.items():
                results["global_" + key] = val.view(b, n, *val.shape[-3:])

        # Camera Head
        if self.camera_head is not None:
            if isinstance(patch_features, (list, tuple)):
                camera_tokens = [patch_features[-1][:, :, :patch_start_idx]]
            else:
                camera_tokens = [patch_features[:, :, :patch_start_idx]]

            pose_enc = self.camera_head(
                camera_tokens,
                meta_data=meta_data,
            )
            results["pose_enc"] = pose_enc

        # Track Head
        if self.track_head is not None and query_points is not None:
            track, track_vis, track_confidence = self.track_head(
                patch_features,
                query_points=query_points,
                meta_data=meta_data,
            )
            results["track"] = track
            results["track_vis"] = track_vis
            results["track_confidence"] = track_confidence

        # Normal Head
        if self.normal_head is not None:
            normal_results = self.normal_head(
                patch_features,
                patch_start_idx=patch_start_idx,
                meta_data=meta_data,
            )
            for key, val in normal_results.items():
                results[key] = val.view(b, n, *val.shape[-3:])

        # Motion Head
        if self.motion_head is not None:
            motion_kwargs = dict(patch_start_idx=patch_start_idx, meta_data=meta_data)
            # MotionHead / MotionHead4RC require query_times; MotionHeadV2 does not.
            # We detect this via the presence of 'time_embed_dim' attribute.
            if hasattr(self.motion_head, 'time_embed_dim'):
                query_times = torch.zeros(b, 1, device=rgb.device, dtype=torch.float32)
                motion_kwargs['query_times'] = query_times
            motion_results = self.motion_head(patch_features, **motion_kwargs)

            for key, val in motion_results.items():
                if val is not None:
                    results[key] = val

        if return_features:
            results["patch_features"] = patch_features

        return results


class MVBaseMotionWithConcatTime(MVBaseMotion):
    """
    MVBaseMotion variant with MoVieS-style concat time token.

    Instead of adding the time embedding onto the camera token (add-style), this class
    injects the time token as an **independent concat slot** inside the DA3 fuse_encoder,
    exactly mirroring the MoVieS design in MoviesMVEncoder.

    Sequence layout inside fuse_encoder:
        [cls(→cam_token at alt_start) | time_token | patch_tokens...]
    patch_start_idx is automatically bumped from 1 to 2 by DinoVisionTransformer.

    Config example:
        type='hAlgorithm.modules.models2.sdk.base_motion.MVBaseMotionWithConcatTime'
        time_encoder=dict(
            type='hAlgorithm.modules.models2.encoder.time_encoder.TimeTokenEncoder',
            embed_dim=1536,          # must match fuse_encoder embed_dim
            use_mlp_projection=True,
        )
    """

    def __init__(self, time_encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.time_encoder = self._instantiate_and_register(time_encoder, "time_encoder")

    def _get_time_token(self, b, n, device, meta_data):
        """Override: produce [B*N, 1, C] time token for concat injection in fuse_encoder."""
        if self.time_encoder is None:
            return None

        if "time_idx" in meta_data:
            ts = meta_data["time_idx"].float()
            frame_timestamps = ts.view(b, n)
        else:
            frame_timestamps = _compute_frame_timestamps(n, b, device)

        # time_encoder: [B, N] -> [B, N, 1, C] -> reshape to [B*N, 1, C]
        time_token = self.time_encoder(frame_timestamps)  # [B, N, 1, C]
        time_token = time_token.view(b * n, 1, -1)        # [B*N, 1, C]
        return time_token


class MVBaseMotionWithTime(MVBaseMotion):
    """
    MVBaseMotion variant that injects per-frame time embeddings into the backbone.

    The time token is added to the camera token before it is passed into the DA3
    fuse_encoder, so the backbone attention sees temporal ordering information.
    This is the DA3 analogue of the MoVieS time-token design.

    Config example (time_encoder):
        time_encoder=dict(
            type='hAlgorithm.modules.models2.encoder.time_encoder.TimeTokenEncoder',
            embed_dim=1536,          # must match camera_encoder dim_out
            use_mlp_projection=True,
        )
    """

    def __init__(self, time_encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.time_encoder = self._instantiate_and_register(time_encoder, "time_encoder")

    def _modify_cam_token(self, cam_token, b, n, device, meta_data):
        """Override: add sinusoidal time embeddings to the camera token."""
        if self.time_encoder is None:
            return cam_token

        # Build normalized frame timestamps [B, N] in [0, 1]
        if "time_idx" in meta_data:
            # Use dataset-provided timestamps if available (shape [B*N] or [B, N])
            ts = meta_data["time_idx"].float()
            frame_timestamps = ts.view(b, n)
        else:
            frame_timestamps = _compute_frame_timestamps(n, b, device)

        # time_encoder: [B, N] -> [B, N, 1, embed_dim] -> [B, N, embed_dim]
        time_emb = self.time_encoder(frame_timestamps).squeeze(-2)

        if cam_token is not None:
            return cam_token + time_emb
        return time_emb
