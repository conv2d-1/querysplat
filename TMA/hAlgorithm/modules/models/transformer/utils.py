import logging

import torch
import torch.nn as nn

# import math



def replace_mlp_layers(module, config):
    """Replace MLP layers with KAN (Knowledge-Augmented Network) based on config.

    Args:
        module (nn.Module): Module to modify
        config (dict): Configuration with:
            - act_init: List of two activation functions for KAN ["identity", "gelu"]
            - target_modules: List of module names to replace (if None, replace all MLPs)
            - drop: Dropout rate (default: 0.0)
            - bias: Whether to use bias (default: True)

    Returns:
        nn.Module: Modified module with KAN layers
    """
    if config is None:
        return module

    act_init = config.get("act_init", ["identity", "gelu"])
    target_modules = config.get("target_modules", None)
    drop = config.get("drop", 0.0)
    bias = config.get("bias", True)

    def should_replace_mlp(name, target_modules):
        """Check if MLP should be replaced based on module name."""
        if target_modules is None:
            return True
        return any(target in name for target in target_modules)

    def _replace_mlp(module, name=""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is an MLP module (typically has fc1/fc2 or similar structure)
            is_mlp = (
                isinstance(child, nn.Sequential)
                or (hasattr(child, "fc1") and hasattr(child, "fc2"))
                or (hasattr(child, "linear1") and hasattr(child, "linear2"))
            )

            if is_mlp and should_replace_mlp(full_name, target_modules):
                # Get the input and output dimensions from the original MLP
                if hasattr(child, "fc1"):
                    in_features = child.fc1.in_features
                    hidden_features = child.fc1.out_features
                    out_features = child.fc2.out_features
                elif hasattr(child, "linear1"):
                    in_features = child.linear1.in_features
                    hidden_features = child.linear1.out_features
                    out_features = child.linear2.out_features
                else:
                    # For Sequential, try to find the first and last Linear layers
                    linear_layers = [m for m in child.modules() if isinstance(m, nn.Linear)]
                    if len(linear_layers) >= 2:
                        in_features = linear_layers[0].in_features
                        hidden_features = linear_layers[0].out_features
                        out_features = linear_layers[-1].out_features
                    else:
                        continue

                # Create KAN layer
                from .mlp import KAN

                new_module = KAN(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act_cfg=dict(type="KAT", act_init=act_init),
                    bias=bias,
                    drop=drop,
                )

                logging.info(
                    f"Replaced MLP layer '{full_name}' with KAN "
                    f"(in={in_features}, hidden={hidden_features}, out={out_features})"
                )

                setattr(module, child_name, new_module)
            else:
                _replace_mlp(child, full_name)

    _replace_mlp(module)
    return module


if __name__ == "__main__":
    # Example usage of replace_mlp_layers
    class ExampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 15)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    # Create model and test input
    model = ExampleModel()
    x = torch.randn(5, 10)

    # Test MLP replacement
    config = {
        "act_init": ["identity", "gelu"],
        "target_modules": ["linear1"],
        "drop": 0.1,
        "bias": True,
    }

    model_with_kan = replace_mlp_layers(model, config)
    output = model_with_kan(x)
    print("Output shape:", output.shape)
