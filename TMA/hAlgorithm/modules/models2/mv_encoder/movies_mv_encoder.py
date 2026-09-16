import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from hAlgorithm.modules.models2.mv_encoder.vggt_mv_encoder import MVEncoder
from .blocks import slice_expand_and_flatten


class MoviesMVEncoder(MVEncoder):
    """
    MoVieS MoviesMVEncoder.
    
    Extends MVEncoder to support:
    1. Spatial Fusion: Adding Plücker embeddings (prompt_ray_in_world) and others.
    2. Dynamic Tokens: Concatenating external Camera and Time tokens.
    3. Consistency: Maintains the exact attention loop and output format of MVEncoder.
    """

    def __init__(self, normalized_shape=None, add_cam_token_to_static=False, **kwargs):
        super(MoviesMVEncoder, self).__init__(**kwargs)
        self.add_cam_token_to_static = add_cam_token_to_static

        # Fusion Norm Layer (Ref: RayMVEncoder)
        if normalized_shape is not None:
            self.fusion_norm_layer = nn.LayerNorm(normalized_shape=normalized_shape, eps=1e-6)
        else:
            self.fusion_norm_layer = None

    def forward(
        self,
        patch_tokens,
        meta_data,
        # --- Fusion Arguments ---
        prompt_ray_in_world=None,
        prompt_depth=None,
        prompt_ray=None,
        prompt_extra=None,
        # --- Global Tokens ---
        camera_token=None,
        time_token=None,
        # --- Control Flags ---
        memory_efficient_infer=False,
        **kwargs
    ):
        # ------------------------------------------------------------------
        # Part 1: Spatial Fusion (Ref: RayMVEncoder)
        # ------------------------------------------------------------------
        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        # Helper to ensure alignment (in case prompts are [B*N, C, H, W])
        def _align_add(feat, prompt):
            if prompt is None:
                return feat
            # If prompt is spatial [B*N, C, H, W], flatten to [B*N, L, C]
            if prompt.ndim == 4:
                prompt = prompt.flatten(2).transpose(1, 2)
            return feat + prompt

        # Apply prompts
        patch_tokens = _align_add(patch_tokens, prompt_ray_in_world)
        patch_tokens = _align_add(patch_tokens, prompt_depth)
        patch_tokens = _align_add(patch_tokens, prompt_ray)

        if prompt_extra is not None:
            for p in prompt_extra:
                if p.ndim == 2: # Broadcast [B*N, C]
                    patch_tokens = patch_tokens + p.unsqueeze(1)
                else:
                    patch_tokens = _align_add(patch_tokens, p)

        # Normalize
        if self.fusion_norm_layer is not None:
            patch_tokens = self.fusion_norm_layer(patch_tokens)

        # ------------------------------------------------------------------
        # Part 2: Token Preparation (Overriding MVEncoder logic)
        # ------------------------------------------------------------------
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        BS, P, C = patch_tokens.shape
        S = frame_num * view_num
        B = BS // S

        # List to collect all token types
        tokens_list = []

        # A. Camera Token
        # Accept cam_token from MVBase.aggregator() as alias for camera_token.
        # CameraEnc outputs [B, S, C]; reshape to [B*S, 1, C] for sequence concat.
        if camera_token is None:
            cam = kwargs.get("cam_token")
            if cam is not None:
                if cam.dim() == 3 and cam.shape[0] != BS:
                    camera_token = cam.reshape(-1, cam.shape[-1]).unsqueeze(1)
                elif cam.dim() == 2:
                    camera_token = cam.unsqueeze(1)
                else:
                    camera_token = cam

        if camera_token is not None:
            if camera_token.dim() == 2: camera_token = camera_token.unsqueeze(1)
            if self.add_cam_token_to_static:
                static_cam = slice_expand_and_flatten(self.camera_token, B, S)
                camera_token = static_cam + camera_token
            tokens_list.append(camera_token)
        else:
            static_cam = slice_expand_and_flatten(self.camera_token, B, S)
            tokens_list.append(static_cam)

        # B. Time Token (New in MoVieS)
        if time_token is not None:
            if time_token.dim() == 2: time_token = time_token.unsqueeze(1)
            tokens_list.append(time_token)

        # C. Register Tokens (Inherited from MVEncoder)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, S)
            tokens_list.append(register_token)

        # D. Patch Tokens
        tokens_list.append(patch_tokens)

        # Concatenate all
        tokens = torch.cat(tokens_list, dim=1)

        # Calculate patch_start_idx dynamically based on how many special tokens we added
        # This ensures RoPE is applied correctly only to image tokens
        self.patch_start_idx = tokens.shape[1] - P

        # ------------------------------------------------------------------
        # Part 3: Positional Embedding (RoPE)
        # ------------------------------------------------------------------
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, patch_h, patch_w, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # Do not use position embedding for special tokens
            # Shift valid pos by 1 (0 reserved for special)
            pos = pos + 1
            # Create dummy pos (zeros) for special tokens
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        if memory_efficient_infer:
            del patch_tokens
            if 'camera_token' in locals(): del camera_token
            if 'time_token' in locals(): del time_token
            if 'register_token' in locals(): del register_token

        # Update P (Total sequence length)
        _, P, C = tokens.shape

        # ------------------------------------------------------------------
        # Part 4: Attention Loop (Exact copy of MVEncoder logic)
        # ------------------------------------------------------------------
        frame_idx = 0
        global_idx = 0
        output_list = []

        bar = range(self.depth) if not memory_efficient_infer else tqdm(range(self.depth), desc="mv encoder depth")
        for di in bar:
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, memory_efficient_infer=memory_efficient_infer
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, memory_efficient_infer=memory_efficient_infer
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # Concat frame and global intermediates, [B x S x P x 2C]
            # Adhering to MVEncoder output format
            if (self.hooks is not None and di in self.hooks) or (self.hooks is None and di == self.depth - 1):
                output_list.append(torch.cat([frame_intermediates, global_intermediates], dim=-1))
        
        if memory_efficient_infer:
            del frame_intermediates
            del global_intermediates

        # Return standard MVEncoder tuple
        return output_list, pos, self.patch_start_idx