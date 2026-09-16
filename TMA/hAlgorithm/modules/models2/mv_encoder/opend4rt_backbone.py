"""Open-d4rt VideoMAE encoder backbone (vendored minimal port)."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("sinusoidal_position_embedding requires even dim")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _normalize_attention_pattern(pattern: str | None) -> str:
    raw = (pattern or "global").strip().lower()
    aliases = {
        "interleaved_local_framewise_and_global": "interleaved_local_global",
        "interleaved_local_framewise_global": "interleaved_local_global",
        "interleaved_local_and_global": "interleaved_local_global",
        "global": "global",
        "full_global": "global",
    }
    return aliases.get(raw, raw)


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention + MLP."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1) -> None:
        super().__init__()
        ff_dim = int(math.ceil(hidden_dim * mlp_ratio))
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_attn(tokens)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        x = tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class VideoPatchTransformerEncoder(nn.Module):
    """Patchify video then encode tokens with interleaved local/global attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size_t_h_w: tuple[int, int, int],
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        max_tokens: int = 4096,
        attention_pattern: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.patch_size_t = int(patch_size_t_h_w[0])
        self.patch_size_h = int(patch_size_t_h_w[1])
        self.patch_size_w = int(patch_size_t_h_w[2])
        self.attention_pattern = _normalize_attention_pattern(attention_pattern)
        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size_t_h_w,
            stride=patch_size_t_h_w,
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.block_modes = self._build_block_modes(num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.last_grid_shape: tuple[int, int, int] | None = None

    def _build_block_modes(self, num_layers: int) -> list[Literal["local", "global"]]:
        if self.attention_pattern == "interleaved_local_global":
            return ["local" if (i % 2 == 0) else "global" for i in range(num_layers)]
        return ["global"] * num_layers

    def _token_cap(self, x: torch.Tensor) -> torch.Tensor:
        b, c, tp, hp, wp = x.shape
        token_count = tp * hp * wp
        if token_count <= self.max_tokens:
            return x
        scale = math.sqrt(self.max_tokens / float(token_count))
        out_h = max(1, int(round(hp * scale)))
        out_w = max(1, int(round(wp * scale)))
        return F.adaptive_avg_pool3d(x, output_size=(tp, out_h, out_w))

    def forward(self, video_b_t_c_h_w: torch.Tensor, extra_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if video_b_t_c_h_w.ndim != 5:
            raise ValueError(f"Expected video tensor with ndim=5, got {video_b_t_c_h_w.shape}")
        x = video_b_t_c_h_w.permute(0, 2, 1, 3, 4)
        x = self.patch_embed(x)
        x = self._token_cap(x)

        b, c, tp, hp, wp = x.shape
        self.last_grid_shape = (tp, hp, wp)
        video_tokens = x.flatten(2).transpose(1, 2)
        token_count = video_tokens.shape[1]
        pos = sinusoidal_position_embedding(token_count, self.hidden_dim, video_tokens.device)
        video_tokens = video_tokens + pos.unsqueeze(0)

        spatial_tokens = hp * wp
        for mode, block in zip(self.block_modes, self.blocks):
            if mode == "local":
                local = video_tokens.reshape(b, tp, spatial_tokens, c).reshape(b * tp, spatial_tokens, c)
                local = block(local)
                video_tokens = local.reshape(b, tp, spatial_tokens, c).reshape(b, tp * spatial_tokens, c)
                continue

            if extra_tokens is None:
                video_tokens = block(video_tokens)
                continue

            merged = torch.cat([video_tokens, extra_tokens], dim=1)
            merged = block(merged)
            video_tokens = merged[:, :token_count]

        encoded = self.final_norm(video_tokens)
        return encoded
