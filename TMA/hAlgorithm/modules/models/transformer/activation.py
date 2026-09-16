# import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    A simplified version of LayerNorm that only performs scaling,
    using RMS statistics instead of mean and variance.

    Args:
        dim (int): The dimension to normalize over
        eps (float): Small constant for numerical stability (default: 1e-6)
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # Trainable scaling parameter

    def forward(self, x):
        """
        Forward pass of RMSNorm.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of the same shape
        """
        # Calculate Root Mean Square (RMS)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma  # Scale the normalized input with learned gamma parameter


class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) normalization layer.
    Applies learnable tanh activation with scaling and bias.
    Can be used as a replacement for LayerNorm.

    Args:
        dim (int): The dimension to normalize over
        init_alpha (float): Initial value for the alpha parameter (default: 1.0)
    """

    def __init__(self, dim, init_alpha=1.0):
        super().__init__()

        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)  # Global scaling
        self.gamma = nn.Parameter(torch.ones(dim))  # Per-channel scaling
        self.beta = nn.Parameter(torch.zeros(dim))  # Per-channel bias

    def forward(self, x):
        """
        Forward pass of DyT.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of the same shape
        """
        # Apply scaled tanh
        x = torch.tanh(self.alpha * x)

        # Apply per-channel affine transform
        # Ensure gamma and beta are properly broadcast
        if x.dim() == 2:
            x = self.gamma * x + self.beta
        else:
            x = self.gamma.view(1, 1, -1) * x + self.beta.view(1, 1, -1)

        return x
