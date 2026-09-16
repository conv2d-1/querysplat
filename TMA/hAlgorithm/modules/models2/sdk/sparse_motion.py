"""
SparseMotion Model — D4RT Unified Encoder-Decoder
===================================================
Implements the D4RT encode/decode architecture:

  Encoder (run once):
    Video → ViT backbone → fuse_encoder → Global Scene Representation F
    + Dense heads (depth, global points, camera pose)

  Decoder (run per query batch):
    Query q = (u, v, t_src, t_tgt, t_cam) + local RGB patch
    → cross-attention into F → predicted 3D position

The encoder is shared across all tasks; the decoder is lightweight and
can process arbitrary numbers of queries independently.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from hAlgorithm.utils import instantiate_from_config

logger = logging.getLogger(__name__)


def _build_module(cfg) -> Optional[nn.Module]:
    """Instantiate a module from config dict, or return None."""
    return instantiate_from_config(cfg) if isinstance(cfg, dict) else None


class SparseMotion(nn.Module):
    """D4RT-style unified model with encode/decode split.

    Components:
      - rgb_encoder: ViT backbone (spatio-temporal patches)
      - fuse_encoder: feature fusion with prompt encodings
      - time_encoder / camera_encoder / ray_in_world_encoder: prompt modules
      - depth_head / glb_points_head / camera_head: dense task heads
      - sparse_motion_head: D4RT query decoder

    The ``encode()`` method computes the Global Scene Representation once.
    The ``decode_queries()`` method decodes arbitrary queries against it.
    The ``forward()`` method chains both for standard training.
    """

    def __init__(
        self,
        rgb_encoder=None,
        fuse_encoder=None,
        time_encoder=None,
        camera_encoder=None,
        ray_in_world_encoder=None,
        depth_head=None,
        glb_points_head=None,
        camera_head=None,
        sparse_motion_head=None,
        freeze_encoders_for_motion: bool = False,
        encoder_input_mode: str = "flat",
        backbone_time_mode: str = "token",
        **kwargs,
    ):
        super().__init__()
        self.rgb_encoder = _build_module(rgb_encoder)
        self.fuse_encoder = _build_module(fuse_encoder)
        self.time_encoder = _build_module(time_encoder)
        self.camera_encoder = _build_module(camera_encoder)
        self.ray_in_world_encoder = _build_module(ray_in_world_encoder)

        self.depth_head = _build_module(depth_head)
        self.glb_points_head = _build_module(glb_points_head)
        self.camera_head = _build_module(camera_head)
        self.sparse_motion_head = _build_module(sparse_motion_head)

        self.freeze_encoders_for_motion = freeze_encoders_for_motion
        # Names of encoder sub-modules that are frozen when
        # freeze_encoders_for_motion=True.  Kept as a class-level constant so
        # that subclasses can override it without touching __init__.
        self._encoder_module_names = (
            "rgb_encoder",
            "fuse_encoder",
            "camera_encoder",
            "ray_in_world_encoder",
            "time_encoder",
        )

        if encoder_input_mode not in ("flat", "structured"):
            raise ValueError(
                f"encoder_input_mode must be one of ('flat', 'structured'), got {encoder_input_mode!r}"
            )
        if backbone_time_mode not in ("token", "camera_add"):
            raise ValueError(
                f"backbone_time_mode must be one of ('token', 'camera_add'), got {backbone_time_mode!r}"
            )
        self.encoder_input_mode = encoder_input_mode
        self.backbone_time_mode = backbone_time_mode

    # ─────────────────────────────────────────────────────────────────────
    # Encoder freeze helper
    # ─────────────────────────────────────────────────────────────────────

    def _freeze_encoders(self) -> None:
        """Set all backbone/fusion encoders to eval() + requires_grad=False.

        Called at the top of every training forward pass when
        ``freeze_encoders_for_motion=True`` so that gradient updates flow
        exclusively through the sparse_motion_head.
        """
        for name in self._encoder_module_names:
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    # ─────────────────────────────────────────────────────────────────────
    # Feature extraction (aligned with MOVIES pipeline)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_flat_token_layout(x: Optional[torch.Tensor], B: int, V: int) -> Optional[torch.Tensor]:
        """Normalize token-like tensors to [B*V, T, C]."""
        if x is None:
            return None
        if x.ndim == 4 and x.shape[0] == B and x.shape[1] == V:
            return x.reshape(B * V, *x.shape[2:])
        if x.ndim == 3 and x.shape[0] == B and x.shape[1] == V:
            return x.reshape(B * V, 1, x.shape[-1])
        if x.ndim == 2:
            return x.unsqueeze(1)
        return x

    @staticmethod
    def _to_structured_token_layout(x: Optional[torch.Tensor], B: int, V: int) -> Optional[torch.Tensor]:
        """Normalize token-like tensors to [B, V, C] for DA3-style camera tokens."""
        if x is None:
            return None
        if x.ndim == 4 and x.shape[0] == B and x.shape[1] == V:
            return x.squeeze(2) if x.shape[2] == 1 else x
        if x.ndim == 3 and x.shape[0] == B and x.shape[1] == V:
            return x
        if x.ndim == 3 and x.shape[0] == B * V:
            return x.squeeze(1).reshape(B, V, -1) if x.shape[1] == 1 else x.reshape(B, V, *x.shape[1:])
        if x.ndim == 2 and x.shape[0] == B * V:
            return x.reshape(B, V, -1)
        return x

    def _extract_features(
        self,
        rgb: torch.Tensor,
        scale: Optional[torch.Tensor],
        prompt_depth: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
        w2c: Optional[torch.Tensor],
        meta_data: Optional[Dict],
        time_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[torch.Tensor], int]:
        """Run backbone + fusion encoder → scene features.

        Returns:
            (patch_features, positional_info, patch_start_idx)
        """
        B, V, C, H, W = rgb.shape

        use_structured = self.encoder_input_mode == "structured"
        rgb_flat = rgb.view(B * V, C, H, W)
        intr_flat = intrinsics.view(B * V, 3, 3) if intrinsics is not None else None
        w2c_flat = w2c.view(B * V, 4, 4) if w2c is not None else None

        # Backbone input
        if self.rgb_encoder is not None:
            backbone_input = rgb if use_structured else rgb_flat
            features = self.rgb_encoder(backbone_input, meta_data=meta_data)
        else:
            features = rgb if use_structured else rgb_flat

        # Prompt encodings
        prompt_ray = None
        if self.ray_in_world_encoder and intrinsics is not None and w2c is not None:
            if use_structured:
                prompt_ray = self.ray_in_world_encoder(
                    ray_directions=None, intrinsics=intrinsics, w2c=w2c, meta_data=meta_data,
                )
            else:
                prompt_ray = self.ray_in_world_encoder(
                    ray_directions=None, intrinsics=intr_flat, w2c=w2c_flat, meta_data=meta_data,
                )

        camera_token = None
        if self.camera_encoder and intrinsics is not None and w2c is not None:
            if use_structured:
                try:
                    camera_token = self.camera_encoder(
                        w2c=w2c, intrinsics=intrinsics, meta_data=meta_data, scale=scale,
                    )
                except TypeError:
                    camera_token = self.camera_encoder(intr_flat, w2c_flat)
                camera_token = self._to_structured_token_layout(camera_token, B, V)
            else:
                try:
                    camera_token = self.camera_encoder(intr_flat, w2c_flat)
                except TypeError:
                    camera_token = self.camera_encoder(
                        w2c=w2c, intrinsics=intrinsics, meta_data=meta_data, scale=scale,
                    )
                camera_token = self._to_flat_token_layout(camera_token, B, V)

        time_token = None
        if self.time_encoder:
            if time_idx is not None:
                time_values = time_idx.view(B, V) if (use_structured or self.backbone_time_mode == "camera_add") else time_idx.view(-1)
            else:
                shape = (B, V) if (use_structured or self.backbone_time_mode == "camera_add") else (B * V,)
                time_values = torch.full(shape, 0.5, device=rgb.device, dtype=torch.float32)

            time_token = self.time_encoder(time_values)

            if self.backbone_time_mode == "camera_add":
                time_emb = self._to_structured_token_layout(time_token, B, V)
                if time_emb is not None and time_emb.ndim == 4 and time_emb.shape[2] == 1:
                    time_emb = time_emb.squeeze(2)
                if camera_token is None:
                    camera_token = time_emb
                elif time_emb is not None:
                    camera_token = camera_token + time_emb
                time_token = None
            else:
                time_token = self._to_flat_token_layout(time_token, B, V)

        # Fusion
        if self.fuse_encoder is not None:
            features, pos, start_idx = self.fuse_encoder(
                features,
                prompt_depth=prompt_depth,
                prompt_ray_in_world=prompt_ray,
                scale=scale,
                intrinsics=intrinsics if use_structured else intr_flat,
                w2c=w2c if use_structured else w2c_flat,
                meta_data=meta_data,
                camera_token=camera_token,
                cam_token=camera_token,
                time_token=time_token,
                **kwargs,
            )
            # Ensure BV-first layout for features
            # fuse_encoder may return List[Tensor] when hooks is set (multi-scale)
            if isinstance(features, (list, tuple)):
                # Reshape each scale: [B, V, P, C] -> [B*V, P, C]
                reshaped = []
                for feat in features:
                    if feat.ndim == 4 and feat.shape[0] == B:
                        feat = feat.reshape(B * V, *feat.shape[2:])
                    reshaped.append(feat)
                features = reshaped
            elif isinstance(features, torch.Tensor) and features.ndim == 4 and features.shape[0] == B:
                features = features.reshape(B * V, *features.shape[2:])
        else:
            if features.ndim == 4:
                features = features.flatten(2).transpose(1, 2)
            pos, start_idx = None, 0

        return features, pos, start_idx

    # ─────────────────────────────────────────────────────────────────────
    # Dense head helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _restore_view(x: Optional[torch.Tensor], B: int, V: int) -> Optional[torch.Tensor]:
        """Reshape [B*V, ...] → [B, V, ...] if needed."""
        if x is None:
            return None
        if x.ndim >= 2 and x.shape[0] == B and x.shape[1] == V:
            return x
        return x.view(B, V, *x.shape[1:])

    @staticmethod
    def _extract_head_output(out: Optional[Dict]) -> Optional[torch.Tensor]:
        """Extract the primary prediction from a head output dict."""
        if out is None:
            return None
        for key in ('depth', 'points', 'pointmap'):
            if (val := out.get(key)) is not None:
                return val
        return None

    def _run_dense_heads(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        patch_start_idx: int,
        meta_data: Optional[Dict],
        B: int,
        V: int,
    ) -> Dict[str, torch.Tensor]:
        """Run all dense heads and return results dict."""
        results = {}
        rv = lambda x: self._restore_view(x, B, V)

        if self.depth_head is not None:
            out = self.depth_head(features, patch_start_idx=patch_start_idx, meta_data=meta_data)
            results['depth'] = rv(self._extract_head_output(out))
            results['confidence'] = rv(out.get('confidence'))

        if self.glb_points_head is not None:
            out = self.glb_points_head(features, patch_start_idx=patch_start_idx, meta_data=meta_data)
            results['global_points'] = rv(self._extract_head_output(out))
            results['global_confidence'] = rv(out.get('confidence'))

        if self.camera_head is not None and patch_start_idx > 0:
            feat = features[-1] if isinstance(features, (list, tuple)) else features
            if feat.ndim == 3:
                feat = feat.view(B, V, *feat.shape[1:])
            results['pose_enc'] = self.camera_head(
                [feat[:, :, :patch_start_idx, :]], meta_data=meta_data,
            )

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Encode / Decode split (D4RT core design)
    # ─────────────────────────────────────────────────────────────────────

    def encode(
        self,
        rgb: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        w2c: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        meta_data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        """Encode video → Global Scene Representation + dense head outputs.

        Run this ONCE per video. The returned dict is passed to
        ``decode_queries()`` for arbitrary query batches.
        """
        if self.training and self.freeze_encoders_for_motion:
            self._freeze_encoders()

        B, V, _, H, W = rgb.shape

        features, pos, start_idx = self._extract_features(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth,
            intrinsics=intrinsics, w2c=w2c, meta_data=meta_data,
            time_idx=time_idx, **kwargs,
        )

        results = self._run_dense_heads(features, start_idx, meta_data, B, V)

        return {
            'patch_features': features,
            'pos': pos,
            'patch_start_idx': start_idx,
            'results': results,
            'B': B, 'V': V, 'H': H, 'W': W,
        }

    def decode_queries(
        self,
        cached: Dict,
        rgb: torch.Tensor,
        motion_queries: Dict[str, torch.Tensor],
        img_size: Tuple[int, int],
        meta_data: Optional[Dict] = None,
    ) -> Optional[torch.Tensor]:
        """Decode queries against cached scene representation.

        This is the lightweight D4RT decoder — no re-encoding needed.
        """
        if self.sparse_motion_head is None:
            return None
        return self.sparse_motion_head(
            features=cached['patch_features'],
            rgb_images=rgb,
            queries=motion_queries,
            img_size=img_size,
            patch_start_idx=cached['patch_start_idx'],
            meta_data=meta_data,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Full forward (training — backward compatible)
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        rgb: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        prompt_depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        w2c: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        meta_data: Optional[Dict] = None,
        motion_queries: Optional[Dict[str, torch.Tensor]] = None,
        return_cached: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Full forward: encode + (optionally) decode queries.

        Args:
            return_cached: If True, include encoder features and
                patch_start_idx in the results dict under private keys
                ``_cached_features`` and ``_cached_patch_start_idx``.
                This is used by per-target-frame inference in the pipeline
                so that the encoder runs through DDP's ``__call__`` while
                the sparse head can be called separately afterward.
        """
        cached = self.encode(
            rgb=rgb, scale=scale, prompt_depth=prompt_depth,
            intrinsics=intrinsics, w2c=w2c, time_idx=time_idx,
            meta_data=meta_data, **kwargs,
        )
        results = cached['results']

        if return_cached:
            results['_cached_features'] = cached['patch_features']
            results['_cached_patch_start_idx'] = cached['patch_start_idx']

        if self.sparse_motion_head is not None and motion_queries is not None:
            H_orig = kwargs.get('H_orig', cached['H'])
            W_orig = kwargs.get('W_orig', cached['W'])
            results['sparse_motion_pred'] = self.decode_queries(
                cached=cached, rgb=rgb, motion_queries=motion_queries,
                img_size=(H_orig, W_orig), meta_data=meta_data,
            )

        return results