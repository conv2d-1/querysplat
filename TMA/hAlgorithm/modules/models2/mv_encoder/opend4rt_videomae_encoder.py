"""Open-d4rt VideoMAE encoder wrapper compatible with MVBase2 ``fuse_encoder``."""

from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.mv_encoder.opend4rt_backbone import VideoPatchTransformerEncoder
from hAlgorithm.modules.models2.mv_encoder.opend4rt_weight_loader import load_opend4rt_encoder_checkpoint

_LOGGER = logging.getLogger(__name__)


class OpenD4RTVideoEncoder(nn.Module):
    """Drop-in ``fuse_encoder`` using Open-d4rt's VideoMAE ViT-g backbone.

    Input: ``(B, N, C, H, W)`` clip tensor (same layout as ``DinoV2`` fuse path).
    Output: list of ``num_out_layers`` feature maps, each ``(B, N, L, C)`` where
    ``L = patch_h * patch_w``, plus ``pos=None`` and ``patch_start_idx=0``.
    """

    def __init__(
        self,
        hidden_dim: int = 1408,
        num_layers: int = 40,
        num_heads: int = 16,
        mlp_ratio: float = 6144.0 / 1408.0,
        patch_size_t_h_w: Sequence[int] = (2, 16, 16),
        attention_pattern: str = "interleaved_local_framewise_and_global",
        max_tokens: int = 6144,
        in_channels: int = 3,
        num_out_layers: int = 4,
        replicate_single_layer: bool = True,
        patch_size: int = 16,
        pretrain: str | None = None,
        pretrain_strict: bool = False,
        normalize_input: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            _LOGGER.debug("OpenD4RTVideoEncoder ignoring unused kwargs: %s", sorted(kwargs))

        patch_t, patch_h, patch_w = (int(v) for v in patch_size_t_h_w)
        self.patch_size = int(patch_size)
        self.patch_size_t = patch_t
        self.hidden_dim = int(hidden_dim)
        self.num_out_layers = max(1, int(num_out_layers))
        self.replicate_single_layer = bool(replicate_single_layer)
        self.normalize_input = bool(normalize_input)

        self.encoder = VideoPatchTransformerEncoder(
            in_channels=in_channels,
            hidden_dim=self.hidden_dim,
            patch_size_t_h_w=(patch_t, patch_h, patch_w),
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=float(mlp_ratio),
            max_tokens=max_tokens,
            attention_pattern=attention_pattern,
        )

        if replicate_single_layer:
            self.out_projs = None
        else:
            self.out_projs = nn.ModuleList(
                [nn.Identity() if i == 0 else nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_out_layers)]
            )

        if pretrain is not None:
            missing, unexpected = load_opend4rt_encoder_checkpoint(
                self.encoder,
                pretrain,
                strict=pretrain_strict,
            )
            _LOGGER.info(
                "OpenD4RTVideoEncoder loaded pretrain from %s (missing=%d unexpected=%d)",
                pretrain,
                len(missing),
                len(unexpected),
            )

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return x
        return (x + 1.0) * 0.5

    def _pad_temporal(self, video: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad ``T`` to a multiple of ``patch_size_t`` by repeating the last frame."""
        b, t, c, h, w = video.shape
        rem = t % self.patch_size_t
        if rem == 0:
            return video, t
        pad = self.patch_size_t - rem
        tail = video[:, -1:].expand(b, pad, c, h, w)
        return torch.cat([video, tail], dim=1), t

    def _tokens_to_per_frame(self, encoded: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Map joint spatiotemporal tokens back to per-frame patch tokens."""
        grid = self.encoder.last_grid_shape
        if grid is None:
            raise RuntimeError("Encoder grid shape unavailable; call encoder forward first.")
        tp, hp, wp = grid
        b, _, dim = encoded.shape
        tokens = encoded.view(b, tp, hp, wp, dim)

        frame_tokens: List[torch.Tensor] = []
        for frame_idx in range(num_frames):
            t_idx = min(frame_idx // self.patch_size_t, tp - 1)
            frame_tokens.append(tokens[:, t_idx].reshape(b, hp * wp, dim))
        return torch.stack(frame_tokens, dim=1)

    def forward(self, x: torch.Tensor, **kwargs):
        if x.ndim != 5:
            raise ValueError(f"OpenD4RTVideoEncoder expects (B, N, C, H, W), got {x.shape}")

        x = self._maybe_normalize(x)
        original_frames = x.shape[1]
        video, valid_frames = self._pad_temporal(x)

        encoded = self.encoder(video)
        per_frame = self._tokens_to_per_frame(encoded, valid_frames)

        if self.out_projs is None:
            outputs = [per_frame for _ in range(self.num_out_layers)]
        else:
            outputs = [proj(per_frame) for proj in self.out_projs]

        return outputs, None, 0
