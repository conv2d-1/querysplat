import logging

# import torch
import torch.nn as nn

from .activation import DynamicTanh, RMSNorm
from .attention import (
    AdaptiveLinearAttention,
    AdaptiveWindowAttention,
    GroupQueryAttention,
    LinearAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
)

# import math


# import torch.nn.functional as F



def replace_transformer_blocks(model_config, blocks):
    attention_type = model_config.get("attention_type", None)
    num_groups = model_config.get("num_groups", 4)
    latent_dim = model_config.get("latent_dim", 64)
    norm_type = model_config.get("norm_type", None)
    norm_config = model_config.get("norm_config", {})
    hooks = model_config.get("hooks", None)

    # Add debug logging
    logging.info(f"CustomViT: Initializing with config: {model_config}")

    # Replace normalization layers
    for i, block in enumerate(blocks):
        hidden_size = block.norm1.normalized_shape[0]

        if norm_type is None:
            pass
        elif norm_type.lower() == "rmsnorm":
            logging.info(f"Block {i}: Replacing with RMSNorm")
            block.norm1 = RMSNorm(hidden_size)
            block.norm2 = RMSNorm(hidden_size)
        elif norm_type.lower() == "dynamictanh":
            logging.info(f"Block {i}: Replacing with DynamicTanh")
            init_alpha = norm_config.get("init_alpha", 1.0)
            block.norm1 = DynamicTanh(hidden_size, init_alpha=init_alpha)
            block.norm2 = DynamicTanh(hidden_size, init_alpha=init_alpha)
        elif norm_type.lower() == "layernorm":
            logging.info(f"Block {i}: Using LayerNorm")
            block.norm1 = nn.LayerNorm(hidden_size)
            block.norm2 = nn.LayerNorm(hidden_size)

    # Configure attention mechanism based on attention_type
    if attention_type in ["WindowAttention", "LinearAttention"]:
        # These types use adapter pattern to modify the original attention layer
        if attention_type == "WindowAttention":
            # Validate window configuration parameters
            window_config = model_config.get("window_config", {})
            if not window_config or "input_resolution" not in window_config:
                raise ValueError(
                    "window_config with input_resolution is required for WindowAttention"
                )

            # Configure AdaptiveWindowAttention for each block
            for i, block in enumerate(blocks):
                if hooks is None or i in hooks:
                    block.attn = AdaptiveWindowAttention(
                        original_attn=block.attn,
                        input_resolution=window_config["input_resolution"],
                        window_size=window_config.get("window_size", 8),
                        shift_size=window_config.get("shift_size", 0),
                        with_cls_token=window_config.get("with_cls_token", True),
                        with_multi_frames=window_config.get("with_multi_frames", False),
                    )
        else:  # LinearAttention
            # Configure AdaptiveLinearAttention for each block
            for block in blocks:
                block.attn = AdaptiveLinearAttention(original_attn=block.attn)
    elif attention_type is not None:
        # Other attention types require complete replacement of original attention layers
        attention_mapping = {
            "GroupQueryAttention": GroupQueryAttention,
            "MultiQueryAttention": MultiQueryAttention,
            "MultiHeadLatentAttention": MultiHeadLatentAttention,
            "LinearAttention": LinearAttention,
        }

        if attention_type not in attention_mapping:
            raise ValueError(
                f"Invalid attention_type. Choose among: {', '.join(attention_mapping.keys())}, "
                "WindowAttention, AdaptiveLinearAttention"
            )

        attention_module = attention_mapping[attention_type]

        # Replace attention layer for each block
        for i, block in enumerate(blocks):
            if hooks is None or i in hooks:
                # Get parameters from original attention layer
                hidden_size = (
                    block.attn.qkv.in_features
                    if hasattr(block.attn, "qkv")
                    else block.attn.q_proj.in_features
                )
                num_heads = block.attn.num_heads

                # Configure parameters based on attention type
                attn_params = {"hidden_size": hidden_size, "num_heads": num_heads}

                if attention_type == "GroupQueryAttention":
                    attn_params["num_kv_heads"] = num_heads // num_groups
                elif attention_type == "MultiQueryAttention":
                    attn_params["num_kv_heads"] = 1  # MQA uses single KV head
                elif attention_type == "MultiHeadLatentAttention":
                    attn_params["latent_dim"] = latent_dim

                # Create and replace attention layer
                block.attn = attention_module(**attn_params)
