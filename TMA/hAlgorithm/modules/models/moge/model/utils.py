import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint

    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__

        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= "2.0", "SDPA requires PyTorch 2.0 or later"

    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)  # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module


def wrap_dinov2_attention_with_flash_attention2(module: nn.Module):
    try:
        from flash_attn import flash_attn_func
    except ImportError as e:
        raise ImportError(
            "Cannot import flash_attn_func from flash_attn. Please ensure that the latest version of flash-attn is installed."
        ) from e

    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            if attn_bias is not None:
                raise NotImplementedError(
                    "Flash Attention2 currently does not support the attn_bias parameter"
                )

            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)

            softmax_scale = 1.0 / math.sqrt(q.size(-1))

            x = flash_attn_func(q, k, v, dropout_p=0.0, causal=False, softmax_scale=softmax_scale)
            x = self.proj(x.reshape(B, N, C))
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module


def wrap_dinov2_attention_with_flash_attention3(module: nn.Module):
    try:
        from flash_attn_interface import flash_attn_func as flash_attn3
    except ImportError as e:
        raise ImportError(
            "Cannot import flash_attn_func from flash_attn_interface. Please ensure that the latest version of flash-attn is installed."
        ) from e

    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            if attn_bias is not None:
                raise NotImplementedError(
                    "Flash Attention3 currently does not support the attn_bias parameter"
                )

            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)

            softmax_scale = 1.0 / math.sqrt(q.size(-1))

            x = flash_attn3(q, k, v, causal=False, softmax_scale=softmax_scale)[0]
            x = self.proj(x.reshape(B, N, C))
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module
