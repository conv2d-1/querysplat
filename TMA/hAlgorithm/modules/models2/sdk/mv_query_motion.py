"""MVQueryMotion — DA3 ViT-G encoder + sparse motion head.

Bridges the MVBase2 / MVQuery encoder architecture (DinoV2 ViT-G with
DA3-style multi-view cross-attention) with the CrossAttnAdaLNFlowHead
sparse motion decoder, providing a SparseMotion-compatible forward interface
so that SparseMotionPipeline can drive training without modification.

Design rationale
----------------
* ``MVBase2.aggregator()`` already returns
  ``(features_list, pos, patch_start_idx, prompt_depth)`` where
  ``features_list`` is a ``List[Tensor[B*N, prefix+P, C]]`` — exactly the
  format that ``CrossAttnAdaLNFlowHead.build_pyramid()`` expects.
* The only dimensional adaptation required in the config is
  ``MotionQueryAggregator.feature_dim = 3072`` (ViT-G embed dim) instead of
  ``2048`` (ViT-L + MoviesMV).
* ``time_idx`` from the pipeline is passed through to ``meta_data`` for
  bookkeeping; the backbone itself does not need it because temporal context
  is encoded implicitly by the DA3 multi-view cross-attention layers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from hAlgorithm.modules.models2.sdk.base import MVBase2
from hAlgorithm.utils import instantiate_from_config

logger = logging.getLogger(__name__)


class MVQueryMotion(MVBase2):
    """DA3 ViT-G encoder combined with a sparse-motion cross-attention decoder.

    Inherits the full ``MVBase2`` aggregator pipeline (camera encoder, ray
    encoder, DA3 fuse_encoder) and adds a ``sparse_motion_head`` module
    (typically ``CrossAttnAdaLNFlowHead``) that consumes the same multi-scale
    patch features.

    The public API mirrors ``SparseMotion`` so that ``SparseMotionPipeline``
    can drive training and inference without any pipeline-side changes:

    * ``encode()``        — run backbone once, return cached scene representation
    * ``decode_queries()``— decode motion queries against the cached features
    * ``forward()``       — chain encode + decode (standard training path)

    Args:
        sparse_motion_head: Config dict for the motion decoder
            (e.g. ``CrossAttnAdaLNFlowHead``).
        freeze_encoders_for_motion: If True, all encoder sub-modules are set
            to ``eval()`` and ``requires_grad=False`` during every training
            forward, confining gradient updates to ``sparse_motion_head``
            alone.  Set to False to fine-tune the backbone jointly.
        **kwargs: Forwarded to ``MVBase2`` (fuse_encoder, camera_encoder,
            depth_head, glb_points_head, camera_head, freeze_modules, …).
    """

    # Sub-module names that are frozen when freeze_encoders_for_motion=True.
    _ENCODER_MODULE_NAMES: Tuple[str, ...] = (
        "fuse_encoder",
        "rgb_encoder",
        "camera_encoder",
        "ray_encoder",
        "ray_in_world_encoder",
        "depth_encoder",
        "extra_encoder",
    )

    def __init__(
        self,
        sparse_motion_head: Optional[dict] = None,
        freeze_encoders_for_motion: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sparse_motion_head = self._instantiate_and_register(
            sparse_motion_head, "sparse_motion_head"
        )
        self.freeze_encoders_for_motion = freeze_encoders_for_motion

    # ─────────────────────────────────────────────────────────────────────
    # Encoder freeze helper
    # ─────────────────────────────────────────────────────────────────────

    def _freeze_encoders(self) -> None:
        """Freeze all encoder sub-modules (eval + no grad)."""
        for name in self._ENCODER_MODULE_NAMES:
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    # ─────────────────────────────────────────────────────────────────────
    # SparseMotion-compatible encode / decode split
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
        """Run the backbone encoder once and return a cached scene representation.

        Stores the multi-scale patch features, positional info, patch start
        index, and dense head outputs so that ``decode_queries()`` can be
        called repeatedly for different query batches without re-encoding.

        Args:
            rgb:          Input images, shape ``[B, V, C, H, W]``.
            scale:        Per-scene normalisation scale.
            prompt_depth: Optional depth prompt tensor.
            intrinsics:   Camera intrinsics ``[B, V, 3, 3]``.
            w2c:          World-to-camera transforms ``[B, V, 4, 4]``.
            time_idx:     Per-frame time normalised to ``[0, 1]`` — unused by
                          the DA3 backbone but stored in ``meta_data`` for the
                          motion decoder.
            meta_data:    Auxiliary metadata dict.

        Returns:
            Dict with keys: ``patch_features``, ``pos``, ``patch_start_idx``,
            ``results`` (dense head outputs), ``B``, ``V``, ``H``, ``W``.
        """
        if self.training and self.freeze_encoders_for_motion:
            self._freeze_encoders()

        B, V, C, H, W = rgb.shape

        # Store time_idx in meta_data so the motion decoder can access it.
        if time_idx is not None and meta_data is not None:
            meta_data["time_idx"] = time_idx

        patch_features, pos, patch_start_idx, prompt_depth = self.aggregator(
            rgb=rgb,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            ray_directions=None,
            w2c=w2c,
            c2w=None,
            ray_world=None,
            meta_data=meta_data,
        )

        # Run dense task heads (depth, global points, camera pose).
        results = self.decoder(
            b=B,
            n=V,
            patch_features=patch_features,
            pos=pos,
            patch_start_idx=patch_start_idx,
            prompt_depth=prompt_depth,
            query_points=None,
            meta_data=meta_data,
        )

        return {
            "patch_features": patch_features,
            "pos": pos,
            "patch_start_idx": patch_start_idx,
            "results": results,
            "B": B,
            "V": V,
            "H": H,
            "W": W,
        }

    def decode_queries(
        self,
        cached: Dict,
        rgb: torch.Tensor,
        motion_queries: Dict[str, torch.Tensor],
        img_size: Tuple[int, int],
        meta_data: Optional[Dict] = None,
    ) -> Optional[torch.Tensor]:
        """Decode motion queries against a cached scene representation.

        This is the lightweight per-query-batch path: encoder features are
        reused from ``encode()`` so that decoding a new set of queries costs
        only the sparse-head compute.

        Args:
            cached:         Output of :meth:`encode`.
            rgb:            Original input images (may be used for local patch
                            features inside the head).
            motion_queries: Dict with ``'uv'``, ``'tgt_frame_idx'``, etc.
            img_size:       ``(H_orig, W_orig)`` of the input images.
            meta_data:      Auxiliary metadata dict.

        Returns:
            Sparse motion head output (dict or tensor), or ``None`` if no head.
        """
        if self.sparse_motion_head is None:
            return None
        return self.sparse_motion_head(
            features=cached["patch_features"],
            rgb_images=rgb,
            queries=motion_queries,
            img_size=img_size,
            patch_start_idx=cached["patch_start_idx"],
            meta_data=meta_data,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Full forward (used by SparseMotionPipeline.train_step)
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
        H_orig: Optional[int] = None,
        W_orig: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Full encode → (optionally) decode forward pass.

        Mirrors ``SparseMotion.forward()`` so that ``SparseMotionPipeline``
        can drive this model without modification.

        Args:
            return_cached: If ``True``, include encoder features and
                ``patch_start_idx`` in the result dict under private keys
                ``_cached_features`` and ``_cached_patch_start_idx`` for
                per-target-frame inference in the pipeline.
            motion_queries: If provided, run the sparse motion head and store
                the output under ``'sparse_motion_pred'``.
        """
        cached = self.encode(
            rgb=rgb,
            scale=scale,
            prompt_depth=prompt_depth,
            intrinsics=intrinsics,
            w2c=w2c,
            time_idx=time_idx,
            meta_data=meta_data,
        )
        results = cached["results"]

        if return_cached:
            results["_cached_features"] = cached["patch_features"]
            results["_cached_patch_start_idx"] = cached["patch_start_idx"]

        if self.sparse_motion_head is not None and motion_queries is not None:
            H_eff = H_orig if H_orig is not None else rgb.shape[-2]
            W_eff = W_orig if W_orig is not None else rgb.shape[-1]
            results["sparse_motion_pred"] = self.decode_queries(
                cached=cached,
                rgb=rgb,
                motion_queries=motion_queries,
                img_size=(H_eff, W_eff),
                meta_data=meta_data,
            )

        return results
