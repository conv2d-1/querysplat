import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA (Low-Rank Adaptation) applied."""

    def __init__(self, linear_layer, rank=8, alpha=16):
        """Initialize LoRA layer.

        Args:
            linear_layer (nn.Linear): Original linear layer
            rank (int): Rank of low-rank approximation
            alpha (int): Scaling factor
        """
        super().__init__()
        # Original layer
        self.linear_layer = linear_layer
        self.linear_layer.weight.requires_grad = False
        if self.linear_layer.bias is not None:
            self.linear_layer.bias.requires_grad = False

        # LoRA hyperparameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """Forward pass combining original layer with LoRA adaptation."""
        original_out = self.linear_layer(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return original_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the original linear layer."""
        if not isinstance(self.linear_layer, nn.Linear):
            raise TypeError("linear_layer must be nn.Linear")

        with torch.no_grad():
            lora_weights = self.lora_B.weight @ self.lora_A.weight * self.scaling
            self.linear_layer.weight.data += lora_weights
            self.linear_layer.weight.requires_grad = True
            if self.linear_layer.bias is not None:
                self.linear_layer.bias.requires_grad = True

        return self.linear_layer


class LowRankLinear(nn.Module):
    """Linear layer implemented with low-rank decomposition."""

    def __init__(self, in_features, out_features, r=8, bias=True):
        """Initialize low-rank linear layer.

        Args:
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension
            r (int): Rank of decomposition
            bias (bool): Whether to include bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Low-rank decomposition parameters
        self.A = nn.Parameter(torch.empty(in_features, r))
        self.B = nn.Parameter(torch.empty(r, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """Forward pass using low-rank decomposition."""
        return x @ self.A @ self.B + (self.bias if self.bias is not None else 0)
