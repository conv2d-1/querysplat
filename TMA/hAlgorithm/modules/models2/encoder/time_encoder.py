import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Tuple

class TimeTokenEncoder(nn.Module):
    """
    TimeTokenEncoder implements the sinusoidal time encoding described in the MoVieS paper.

    "To inform the model that input images originate from a temporally ordered video, 
    we additionally encode each timestamp ti ∈ [0, 1] using sinusoidal positional encoding [11] 
    to produce a time token, which is then concatenated with the aforementioned image and camera tokens."

    Implementation Details:
    1. Takes normalized time t in [0, 1].
    2. Applies Sinusoidal Positional Encoding.
    3. Optionally projects it via an MLP to match the backbone's embedding dimension perfectly 
       and learn semantic alignment (common practice in 'Token' generation).
    """

    def __init__(
        self, 
        embed_dim: int = 1024, 
        max_period: float = 10000.0, 
        use_mlp_projection: bool = True
    ):
        """
        Args:
            embed_dim (int): The dimension of the output token (must match backbone embed_dim).
            max_period (float): The max period (temperature) for PE calculations. 
                                Controls the frequency bands.
            use_mlp_projection (bool): If True, adds a Linear+GELU+Linear projection 
                                       after raw PE to refine the token representation.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.use_mlp_projection = use_mlp_projection

        # Cache for frequencies to avoid recomputing
        # We calculate half_dim because we concat sin and cos
        half_dim = embed_dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=half_dim, dtype=torch.float32)
                / half_dim
            ),
        )

        # Optional: MLP to map raw PE to a learnable semantic space
        # This is crucial if 'embed_dim' is very large or if the backbone requires
        # the time token to have specific feature distributions.
        if self.use_mlp_projection:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # Initialize MLP to be close to identity or small to start
            nn.init.xavier_uniform_(self.mlp[0].weight)
            nn.init.xavier_uniform_(self.mlp[2].weight)

    def forward(self, time_stamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_stamps (torch.Tensor): Normalized time values.
                Shape options:
                - [B] (Batch size, assuming 1 frame per batch item)
                - [B, N] (Batch size, N frames per video)
                
        Returns:
            torch.Tensor: The generated Time Tokens.
                Shape: Matches input spatial dims with added embedding dim.
                - If input [B, N], output [B, N, 1, C] (ready for concat with image tokens)
        """
        # 1. Input Validation
        if time_stamps.min() < 0 or time_stamps.max() > 1.0 + 1e-5:
            logging.warning(
                f"TimeTokenEncoder: Input times are outside [0, 1]. Range: [{time_stamps.min():.2f}, {time_stamps.max():.2f}]"
            )

        # 2. Sinusoidal Encoding
        # time_stamps: [B, N] -> [B, N, 1]
        args = time_stamps.unsqueeze(-1).float() * self.freqs
        
        # [B, N, half_dim] -> [B, N, embed_dim]
        # Concatenate sin and cos
        pe = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dimensions just in case (padding)
        if self.embed_dim % 2 == 1:
            pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)

        # 3. MLP Projection (Refinement)
        if self.use_mlp_projection:
            time_token = self.mlp(pe)
        else:
            time_token = pe

        # 4. Formatting for Concatenation
        # Usually Image Tokens are [B, N, L, C] (L=Patches)
        # We want Time Token to be [B, N, 1, C] so it acts like an extra patch token.
        
        # Current shape is [B, N, C] (assuming input was [B, N])
        # Unsqueeze the "Token" dimension
        time_token = time_token.unsqueeze(-2) # [B, N, 1, C]

        return time_token

# --- Usage Example (Self-contained test) ---
if __name__ == "__main__":
    # Hyperparameters
    B, N = 2, 16  # Batch size 2, 16 Frames
    C = 1024      # Embedding Dimension (ViT-Large)

    # 1. Instantiate Encoder
    time_encoder = TimeTokenEncoder(embed_dim=C, use_mlp_projection=True)

    # 2. Create dummy normalized timestamps [0, 1]
    # e.g., 16 frames linearly spaced from 0 to 1
    raw_time = torch.linspace(0, 1, N).unsqueeze(0).repeat(B, 1) # [B, N]
    
    # 3. Forward pass
    time_tokens = time_encoder(raw_time)

    print(f"Input Shape: {raw_time.shape}")     # [2, 16]
    print(f"Output Shape: {time_tokens.shape}") # [2, 16, 1, 1024]
    
    # 4. Simulation of Concatenation with Image Tokens
    # Assume we have 256 patches per image
    num_patches = 16 * 16
    image_tokens = torch.randn(B, N, num_patches, C)
    
    # Concatenate along the token dimension (dim=2)
    # [B, N, 1, C] + [B, N, 256, C] -> [B, N, 257, C]
    combined_tokens = torch.cat([time_tokens, image_tokens], dim=2)
    
    print(f"Combined Shape: {combined_tokens.shape}")
    print("Verification: Time Token successfully prepended/concatenated.")