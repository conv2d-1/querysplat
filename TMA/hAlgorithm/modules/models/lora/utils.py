import logging

import torch.nn as nn

from .layers import LoRALinear, LowRankLinear


def merge_lora_weights(module):
    """Recursively merge LoRA weights in the model."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge_weights())
        else:
            merge_lora_weights(child)
    return module


def restore_from_lora(module, config):
    """Merge LoRA weights and restore model to original structure.

    Args:
        module (nn.Module): Model with LoRA layers
        config (dict): LoRA configuration

    Returns:
        nn.Module: Restored model
    """
    if config is None or config.get("type") != "lora":
        return module

    logging.info("Starting to merge LoRA weights and restore model structure...")
    module = merge_lora_weights(module)
    logging.info("Successfully merged LoRA weights and restored original model structure")
    return module


def should_apply_lora(name, target_modules):
    """Check if LoRA should be applied to the module."""
    if target_modules is None:
        return True
    return any(target in name for target in target_modules)


def replace_linear_layers(module, config):
    """Replace linear layers with LoRA or low-rank versions based on config.

    Args:
        module (nn.Module): Module to modify
        config (dict): Configuration with type, rank, alpha, and target_modules
    """
    if config is None:
        return module

    lora_type = config.get("type", None)
    if lora_type is None:
        return module

    rank = config.get("rank", 8)
    alpha = config.get("alpha", 16)
    target_modules = config.get("target_modules", None)

    def _replace_linear(module, name=""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and should_apply_lora(full_name, target_modules):
                if lora_type == "lora":
                    new_module = LoRALinear(child, rank=rank, alpha=alpha)
                    logging.info(
                        f"Replaced Linear layer '{full_name}' with LoRA (rank={rank}, alpha={alpha})"
                    )
                elif lora_type == "low_rank":
                    new_module = LowRankLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        r=rank,
                        bias=(child.bias is not None),
                    )
                    logging.info(f"Replaced Linear layer '{full_name}' with LowRank (rank={rank})")
                setattr(module, child_name, new_module)
            else:
                _replace_linear(child, full_name)

    _replace_linear(module)
    return module


# Backward compatibility functions
def replace_linear_with_lora(module, rank=8, alpha=16):
    """Legacy function for LoRA replacement."""
    return replace_linear_layers(
        module, {"type": "lora", "rank": rank, "alpha": alpha, "target_modules": None}
    )


def replace_linear_with_low_rank(module, r=8):
    """Legacy function for low-rank replacement."""
    return replace_linear_layers(module, {"type": "low_rank", "rank": r, "target_modules": None})
