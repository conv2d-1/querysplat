import logging

import torch
import torch.nn as nn

from hAlgorithm.modules.models.transformer.activation import DynamicTanh, RMSNorm
from hAlgorithm.modules.models.transformer.attention import (
    AdaptiveLinearAttention,
    AdaptiveWindowAttention,
    GroupQueryAttention,
    LinearAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
)
from hAlgorithm.modules.models.transformer.utils import (
    merge_lora_weights,
    replace_linear_layers,
    replace_mlp_layers,
)

# import math


# import torch.nn.functional as F



class CustomViT(nn.Module):
    def __init__(self, blocks, original_model, **model_config):
        """
        A configurable Vision Transformer (ViT) that supports customizable attention mechanisms,
        normalization layers, and optimization methods.

        This class provides a flexible way to adapt ViT models by allowing:
        1. Different attention mechanisms (GroupQuery, MultiQuery, MultiHeadLatent, etc.)
        2. Various normalization strategies (LayerNorm, RMSNorm, DynamicTanh)
        3. Optimization techniques like LoRA
        4. MLP replacement with KAN (Knowledge-Augmented Network)

        Args:
            original_model: The base ViT model to be configured
            model_config: A dictionary containing all model configurations:
                - attention_type: The attention mechanism to use (default: 'GroupQueryAttention')
                - num_groups: Number of groups for GroupQueryAttention (default: 4)
                - latent_dim: Number of latent queries for MultiHeadLatentAttention (default: 64)
                - norm_type: Type of normalization to use ('layernorm', 'rmsnorm', or 'DynamicTanh') (default: 'layernorm')
                - norm_config: Configuration for the normalization layer (default: {})
                - window_config: Configuration for WindowAttention (if used)
                - lora_config: Configuration for LoRA (if used)
                - mlp_config: Configuration for MLP replacement:
                    - act_init: List of two activation functions for KAN ["identity", "gelu"]
                    - target_modules: List of module names to replace (if None, replace all MLPs)
                    - drop: Dropout rate (default: 0.0)
                    - bias: Whether to use bias (default: True)
        """
        super(CustomViT, self).__init__()
        self.original_model = original_model

        # Extract configuration with defaults
        self.attention_type = model_config.get("attention_type", "GroupQueryAttention")
        self.num_groups = model_config.get("num_groups", 4)
        self.latent_dim = model_config.get("latent_dim", 64)
        self.norm_type = model_config.get("norm_type", "layernorm")
        self.norm_config = model_config.get("norm_config", {})

        # Add debug logging
        logging.info(f"CustomViT: Initializing with config: {model_config}")

        # Replace normalization layers
        for i, block in enumerate(self.original_model.blocks):
            hidden_size = block.norm1.normalized_shape[0]

            if self.norm_type.lower() == "rmsnorm":
                logging.info(f"Block {i}: Replacing with RMSNorm")
                block.norm1 = RMSNorm(hidden_size)
                block.norm2 = RMSNorm(hidden_size)
            elif self.norm_type.lower() == "dynamictanh":
                logging.info(f"Block {i}: Replacing with DynamicTanh")
                init_alpha = self.norm_config.get("init_alpha", 1.0)
                block.norm1 = DynamicTanh(hidden_size, init_alpha=init_alpha)
                block.norm2 = DynamicTanh(hidden_size, init_alpha=init_alpha)
            else:  # default to LayerNorm
                logging.info(f"Block {i}: Using LayerNorm")
                block.norm1 = nn.LayerNorm(hidden_size)
                block.norm2 = nn.LayerNorm(hidden_size)

        # Configure attention mechanism based on attention_type
        if self.attention_type in ["WindowAttention", "LinearAttention"]:
            # These types use adapter pattern to modify the original attention layer
            if self.attention_type == "WindowAttention":
                # Validate window configuration parameters
                window_config = model_config.get("window_config", {})
                if not window_config or "input_resolution" not in window_config:
                    raise ValueError(
                        "window_config with input_resolution is required for WindowAttention"
                    )

                # Configure AdaptiveWindowAttention for each block
                for block in self.original_model.blocks:
                    block.attn = AdaptiveWindowAttention(
                        original_attn=block.attn,
                        input_resolution=window_config["input_resolution"],
                        window_size=window_config.get("window_size", 8),
                        shift_size=window_config.get("shift_size", 0),
                    )
            else:  # LinearAttention
                # Configure AdaptiveLinearAttention for each block
                for block in self.original_model.blocks:
                    block.attn = AdaptiveLinearAttention(original_attn=block.attn)
        else:
            # Other attention types require complete replacement of original attention layers
            attention_mapping = {
                "GroupQueryAttention": GroupQueryAttention,
                "MultiQueryAttention": MultiQueryAttention,
                "MultiHeadLatentAttention": MultiHeadLatentAttention,
                "LinearAttention": LinearAttention,
            }

            if self.attention_type not in attention_mapping:
                raise ValueError(
                    f"Invalid attention_type. Choose among: {', '.join(attention_mapping.keys())}, "
                    "WindowAttention, AdaptiveLinearAttention"
                )

            attention_module = attention_mapping[self.attention_type]

            # Replace attention layer for each block
            for block in self.original_model.blocks:
                # Get parameters from original attention layer
                hidden_size = (
                    block.attn.qkv.in_features
                    if hasattr(block.attn, "qkv")
                    else block.attn.q_proj.in_features
                )
                num_heads = block.attn.num_heads

                # Configure parameters based on attention type
                attn_params = {"hidden_size": hidden_size, "num_heads": num_heads}

                if self.attention_type == "GroupQueryAttention":
                    attn_params["num_kv_heads"] = num_heads // self.num_groups
                elif self.attention_type == "MultiQueryAttention":
                    attn_params["num_kv_heads"] = 1  # MQA uses single KV head
                elif self.attention_type == "MultiHeadLatentAttention":
                    attn_params["latent_dim"] = self.latent_dim

                # Create and replace attention layer
                block.attn = attention_module(**attn_params)

        # Replace MLP layers with KAN if configured
        if "mlp_config" in model_config and model_config["mlp_config"] is not None:
            logging.info(f"Replacing MLP layers with KAN, config: {model_config['mlp_config']}")

            # Count parameters before replacement
            n_params_before = sum(p.numel() for p in self.original_model.parameters())

            self.original_model = replace_mlp_layers(
                self.original_model, model_config["mlp_config"]
            )

            # Count parameters after replacement
            n_params_after = sum(p.numel() for p in self.original_model.parameters())
            logging.info(
                f"Parameters after MLP replacement: {n_params_before:,} -> {n_params_after:,} ({n_params_after-n_params_before:,} difference)"
            )

        # Apply LoRA if configured
        if "lora_config" in model_config and model_config["lora_config"] is not None:
            logging.info(f"Applying LoRA with config: {model_config['lora_config']}")

            # Count parameters before LoRA
            n_params_before = sum(p.numel() for p in self.original_model.parameters())

            self.original_model = replace_linear_layers(
                self.original_model, model_config["lora_config"]
            )

            # Count parameters after LoRA
            n_params_after = sum(p.numel() for p in self.original_model.parameters())
            logging.info(
                f"Parameters after LoRA: {n_params_before:,} -> {n_params_after:,} ({n_params_after-n_params_before:,} added)"
            )

    def forward(self, *args, **kwargs):
        return self.original_model(*args, **kwargs)

    def merge_lora_weights(self):
        """Merge LoRA weights into the original model if LoRA is applied."""
        if hasattr(self, "original_model"):
            self.original_model = merge_lora_weights(self.original_model)
            logging.info("Successfully merged LoRA weights into the original model")
        return self


if __name__ == "__main__":
    """
    Test cases for different configurations of CustomViT
    """
    # Test DynamicTanh
    dyt = DynamicTanh(dim=256, init_alpha=1.0).cuda()
    test_input = torch.randn(32, 197, 256).cuda()
    output = dyt(test_input)
    print(f"DyT output shape: {output.shape}")

    # Test different attention types
    test_configs = [
        {"attention_type": "GroupQueryAttention"},
        {"attention_type": "MultiQueryAttention"},
        {"attention_type": "MultiHeadLatentAttention"},
        {"attention_type": "LinearAttention"},
        {
            "attention_type": "WindowAttention",
            "window_config": {"input_resolution": (48, 32), "window_size": 8, "shift_size": 4},
        },
    ]

    for config in test_configs:
        # 1. Load pretrained ViT model
        model = torch.hub.load(
            "hAlgorithm/modules/models/facebookresearch_dinov2_main",
            "dinov2_vitb14",
            source="local",
            pretrained=False,
        )

        # 2. Create configurable ViT model
        configurable_model = CustomViT(model, **config).cuda()

        # 3. Forward pass
        input_tensor = torch.randn(1, 3, 224 * 3, 224 * 2).cuda()
        output = configurable_model(input_tensor, [2, 5, 8, 11], return_class_token=True)
        print(f'{config["attention_type"]} Forward Pass')
