"""
Conditional 4D Motion Decoder (4RC-style).

Implements the cross-attention based motion decoder from:
    "4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere"
    (https://arxiv.org/abs/2602.10094)

Key design:
    1. Per-scale cross-attention between source and target frame tokens
    2. Time conditioning via AdaLN on the self-attention path
    3. Bottleneck projection (dim_in → cross_attn_dim) to reduce parameters
    4. DPT multi-scale upsampling for dense displacement prediction
    5. (v4) Zero-init for cross-attn output and FFN output, plus residual skip
       from original pre-bottleneck features, so the head starts identical
       to the baseline DPT head and cross-attention grows in gradually.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.head.vggt_dpt_head import VGGTDPTHead

logger = logging.getLogger(__name__)


# ==========================================================================
# Auxiliary Modules
# ==========================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Encodes scalar time steps into high-dimensional embeddings."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time_steps: torch.Tensor) -> torch.Tensor:
        time_steps = time_steps.float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_steps.device)
        args = time_steps.unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1
            )
        return embedding


class AdaLN4RC(nn.Module):
    """Adaptive Layer Normalization with zero-init projection."""

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm = self.norm(x)
        if cond is None:
            return x_norm
        scale_shift = self.proj(cond)
        gamma, beta = scale_shift.chunk(2, dim=-1)
        return x_norm * (1.0 + gamma) + beta


class MotionHead4RCBlock(nn.Module):
    """Single transformer block for 4RC-style motion decoding.

    Two ordering modes (controlled by ``cross_before_self``):

    * ``False`` (original / v1–v2):
        AdaLN(t) → Self-Attn → Cross-Attn → FFN
        Self-attention runs on copies of the source frame before cross-frame
        interaction, making it identical across all target-frame replicas and
        therefore redundant.

    * ``True`` (v3+, recommended):
        Cross-Attn → AdaLN(t)+Self-Attn → FFN
        Source tokens first absorb target-frame context, then communicate
        among themselves conditioned on the target time.  Each replica is
        already differentiated when self-attention runs, making it meaningful.
    """

    def __init__(
        self,
        d_model: int = 1536,
        nhead: int = 16,
        d_cond: int = 1536,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_before_self: bool = False,
        zero_init: bool = True,
    ):
        super().__init__()
        self.cross_before_self = cross_before_self

        # Cross-attention
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            batch_first=True, dropout=dropout,
        )

        # Self-attention (with AdaLN time conditioning)
        self.adaln_self = AdaLN4RC(d_model, d_cond)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            batch_first=True, dropout=dropout,
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

        if zero_init:
            # Zero-init output projections so the block acts as identity at
            # initialization.  The residual connections then preserve the
            # original features and cross-attention grows in gradually.
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        z_q: torch.Tensor,
        t_tau: Optional[torch.Tensor],
        z_tau: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_before_self:
            # Step 1: cross-attention — absorb target context first
            cross_out, _ = self.cross_attn(
                query=self.norm_cross(z_q), key=z_tau, value=z_tau,
            )
            z_q = z_q + cross_out
            # Step 2: time-conditioned self-attention — tokens are now differentiated.
            # AdaLN is computed once and shared across Q/K/V to avoid triple redundant
            # forward passes through LayerNorm + Linear.
            z_q_norm = self.adaln_self(z_q, t_tau)
            sa_out, _ = self.self_attn(query=z_q_norm, key=z_q_norm, value=z_q_norm)
            z_q = z_q + sa_out
        else:
            # Original order (v1/v2 backward-compatible)
            z_q_norm = self.adaln_self(z_q, t_tau)
            sa_out, _ = self.self_attn(query=z_q_norm, key=z_q_norm, value=z_q_norm)
            z_q = z_q + sa_out
            cross_out, _ = self.cross_attn(
                query=self.norm_cross(z_q), key=z_tau, value=z_tau,
            )
            z_q = z_q + cross_out

        z_q = z_q + self.mlp(self.norm_ffn(z_q))
        return z_q


# ==========================================================================
# Main Head
# ==========================================================================

class MotionHead4RC(VGGTDPTHead):
    """
    Conditional 4D Motion Decoder following the 4RC paper.

    Cross-attention is applied at every backbone scale with independent
    per-scale blocks.  A bottleneck projection (dim_in → cross_attn_dim)
    keeps the attention parameter count manageable.

    Forward contract:
        Input:  features (multi-scale list), query_times [B, M], meta_data
        Output: dict with 'scene_flow' [B, 1, N, 3, H, W]
    """

    VALID_PREDICTION_TYPES = ("scene_flow", "dynamic_pointmap")

    def __init__(
        self,
        dim_in: int = 2048,
        cross_attn_dim: int = 0,
        num_cross_attn_layers: int = 1,
        nhead: int = 16,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        use_time_condition: bool = True,
        output_dim: int = 3,
        prediction_type: str = "scene_flow",
        random_src_frame: bool = True,
        cross_before_self: bool = False,
        separate_qkv_proj: bool = False,
        zero_init_cross_attn: bool = True,
        pretrain: str = None,
        **kwargs,
    ):
        super().__init__(dim_in=dim_in, output_dim=output_dim, **kwargs)

        if prediction_type not in self.VALID_PREDICTION_TYPES:
            raise ValueError(
                f"prediction_type must be one of {self.VALID_PREDICTION_TYPES}, "
                f"got '{prediction_type}'"
            )
        self.prediction_type = prediction_type
        self.random_src_frame = random_src_frame
        self.time_embed_dim = time_embed_dim
        self.use_time_condition = use_time_condition
        self.cross_before_self = cross_before_self
        self.separate_qkv_proj = separate_qkv_proj
        self.zero_init_cross_attn = zero_init_cross_attn

        # Bottleneck: 0 means no projection (use dim_in directly)
        d_attn = cross_attn_dim if cross_attn_dim > 0 else dim_in
        self.d_attn = d_attn
        self.use_bottleneck = (d_attn != dim_in)

        # Time encoding → project to cross_attn_dim
        self.time_encoder = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, d_attn),
            nn.GELU(),
            nn.Linear(d_attn, d_attn),
        )

        num_scales = len(self.intermediate_layer_idx)

        # Per-scale bottleneck projections.
        # When separate_qkv_proj=True, query (source) and key/value (target) use
        # independent projections, which is more appropriate for cross-attention
        # where the two sides play different roles.
        if self.use_bottleneck:
            self.q_projs = nn.ModuleList([
                nn.Linear(dim_in, d_attn) for _ in range(num_scales)
            ])
            if self.separate_qkv_proj:
                self.kv_projs = nn.ModuleList([
                    nn.Linear(dim_in, d_attn) for _ in range(num_scales)
                ])
            else:
                # Shared projection for backward compatibility (v1/v2)
                self.kv_projs = self.q_projs
            self.output_projs = nn.ModuleList([
                nn.Linear(d_attn, dim_in) for _ in range(num_scales)
            ])
            # Legacy alias kept for checkpoint compatibility
            self.input_projs = self.q_projs

        # Per-scale cross-attention blocks (independent weights per scale)
        self.per_scale_blocks = nn.ModuleList([
            nn.ModuleList([
                MotionHead4RCBlock(
                    d_model=d_attn, nhead=nhead,
                    d_cond=d_attn, mlp_ratio=mlp_ratio,
                    cross_before_self=cross_before_self,
                    zero_init=zero_init_cross_attn,
                )
                for _ in range(num_cross_attn_layers)
            ])
            for _ in range(num_scales)
        ])

        if self.use_bottleneck and zero_init_cross_attn:
            # Zero-init bottleneck output projections.  Combined with the
            # residual skip added in forward(), the head is equivalent to the
            # baseline DPT head at initialization and cross-attention is
            # introduced gradually as training proceeds.
            for proj in self.output_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

        # Per-scale output normalization
        self.cross_attn_out_norms = nn.ModuleList([
            nn.LayerNorm(dim_in, eps=1e-6) for _ in range(num_scales)
        ])

        if pretrain is not None:
            self._load_pretrain(pretrain)

        total_attn_params = sum(
            p.numel() for s in self.per_scale_blocks for p in s.parameters()
        )
        logger.info(
            f"MotionHead4RC: {num_scales} scales x {num_cross_attn_layers} blocks, "
            f"dim_in={dim_in}, cross_attn_dim={d_attn}, nhead={nhead}, "
            f"attn_params={total_attn_params/1e6:.1f}M, "
            f"prediction_type={prediction_type}, "
            f"random_src_frame={random_src_frame}, "
            f"use_time_condition={use_time_condition}, "
            f"cross_before_self={cross_before_self}, "
            f"separate_qkv_proj={separate_qkv_proj}, "
            f"zero_init_cross_attn={zero_init_cross_attn}"
        )

    def _load_pretrain(self, path: str):
        logger.info(f"MotionHead4RC: Loading pretrain from {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            logger.info(f"MotionHead4RC: Missing keys (expected for new modules): "
                        f"{[k for k in missing if 'cross_attn' in k or 'time_' in k][:5]}...")
        if unexpected:
            logger.warning(f"MotionHead4RC: Unexpected keys: {unexpected[:5]}...")

    def _build_target_time_token(
        self,
        meta_data,
        device: torch.device,
        B: int,
        N: int,
    ) -> Optional[torch.Tensor]:
        """Build target-frame time tokens for AdaLN conditioning."""
        if not self.use_time_condition:
            return None

        time_idx = meta_data.get("time_idx") if meta_data is not None else None
        if time_idx is None:
            time_idx = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
        elif time_idx.shape[1] != N:
            time_idx = time_idx[:, :N]

        target_time_emb = self.time_encoder(time_idx)                          # [B, N, T]
        target_time_emb = target_time_emb.reshape(B * N, -1)                  # [B*N, T]
        weight_dtype = next(self.time_proj.parameters()).dtype
        return self.time_proj(target_time_emb.to(weight_dtype)).unsqueeze(1)   # [B*N, 1, d_attn]

    # ------------------------------------------------------------------
    # Feature shape normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _to_4d(feat, B, N):
        """Normalize a feature tensor to [B, N, L, C] regardless of input shape."""
        if feat.ndim == 4:
            if feat.shape[0] == B and feat.shape[1] == N:
                return feat
            return feat.view(B, N, feat.shape[-2], feat.shape[-1])
        return feat.view(B, N, feat.shape[1], feat.shape[2])

    # ------------------------------------------------------------------
    # DPT forward with cross-attention refined features at ALL scales
    # ------------------------------------------------------------------
    def _dpt_upsample(self, refined_per_scale, patch_h: int, patch_w: int):
        """Run DPT projection + fusion on cross-attention refined features."""
        out = []

        for dpt_idx in range(len(self.intermediate_layer_idx)):
            x = refined_per_scale[dpt_idx]
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)

            H_input = patch_h * self.patch_size
            W_input = patch_w * self.patch_size
            if self.pos_embed:
                x = self._apply_pos_embed(x, W_input, H_input)

            x = self.resize_layers[dpt_idx](x)
            out.append(x)

        out = self.scratch_forward(out)

        target_h = int(patch_h * self.patch_size / self.down_ratio)
        target_w = int(patch_w * self.patch_size / self.down_ratio)
        out = F.interpolate(
            out, size=(target_h, target_w), mode="bilinear", align_corners=True,
        )

        if self.pos_embed:
            H_input = patch_h * self.patch_size
            W_input = patch_w * self.patch_size
            out = self._apply_pos_embed(out, W_input, H_input)

        return out

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def _activate_output(self, out):
        pred = out
        if self.activation == "tanh":
            pred = torch.tanh(out)
        elif self.activation == "sigmoid":
            pred = torch.sigmoid(out)
        conf = torch.ones_like(out[:, :1])
        return pred, conf

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------
    def forward(
        self,
        features,
        query_times,
        meta_data=None,
        patch_start_idx=None,
        patch_h=None,
        patch_w=None,
        src_frame_idx=None,
        **kwargs,
    ):
        B = query_times.shape[0]
        N = int(meta_data["frames"][0])

        if patch_h is None or patch_w is None:
            patch_w = int(meta_data["input_width"][0]) // self.patch_size
            patch_h = int(meta_data["input_height"][0]) // self.patch_size

        features_4d = [self._to_4d(f, B, N) for f in features]

        # ---- 1. Source frame selection ----
        if src_frame_idx is not None:
            pass
        elif self.training and self.random_src_frame:
            src_frame_idx = torch.randint(0, N, (1,)).item()
        else:
            src_frame_idx = 0

        # ---- 2. Time token ----
        target_time_token = self._build_target_time_token(
            meta_data=meta_data, device=features_4d[0].device, B=B, N=N,
        )

        # ---- 3. Per-scale cross-attention ----
        refined_per_scale = []
        for scale_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            feat_bn = features_4d[layer_idx]              # [B, N, L_total, C]

            if patch_start_idx is not None and patch_start_idx > 0:
                feat_bn = feat_bn[:, :, patch_start_idx:, :]

            L = feat_bn.shape[2]
            C = feat_bn.shape[-1]

            z_q = feat_bn[:, src_frame_idx, :, :]         # [B, L, C]
            z_q = z_q.unsqueeze(1).expand(B, N, L, C).reshape(B * N, L, C)
            z_tau = feat_bn.reshape(B * N, L, C)

            # Project down to cross_attn_dim if bottleneck is enabled.
            # q_projs and kv_projs are the same object when separate_qkv_proj=False
            # (backward-compatible with v1/v2), or independent when True (v3+).
            if self.use_bottleneck:
                # Keep original dim_in features as a residual so that at
                # initialization (when output_projs is zero-init) the head
                # falls back to passing the original features through DPT,
                # matching baseline behavior before cross-attention learns.
                z_residual = z_q                              # [B*N, L, dim_in]
                z_q = self.q_projs[scale_idx](z_q)           # dim_in → d_attn
                z_tau = self.kv_projs[scale_idx](z_tau)       # dim_in → d_attn

            for block in self.per_scale_blocks[scale_idx]:
                z_q = block(z_q, target_time_token, z_tau)

            # Project back to dim_in for DPT, adding residual skip.
            if self.use_bottleneck:
                z_q = self.output_projs[scale_idx](z_q) + z_residual

            refined_per_scale.append(
                self.cross_attn_out_norms[scale_idx](z_q)
            )

        # ---- 4. DPT upsampling ----
        dpt_out = self._dpt_upsample(
            refined_per_scale=refined_per_scale,
            patch_h=patch_h, patch_w=patch_w,
        )

        if self.feature_only:
            return dpt_out

        # ---- 5. Output projection ----
        out2 = self.scratch.output_conv2(dpt_out)
        preds, conf = self._activate_output(out2)

        H_img, W_img = preds.shape[-2:]
        C_out = preds.shape[1]

        preds = preds.view(B, N, C_out, H_img, W_img).unsqueeze(1)
        conf = conf.view(B, N, -1, H_img, W_img).unsqueeze(1)

        if self.prediction_type == "dynamic_pointmap":
            return dict(
                dynamic_pointmap=preds, pointmap_confidence=conf,
                src_frame_idx=src_frame_idx,
            )
        return dict(
            scene_flow=preds, flow_confidence=conf,
            src_frame_idx=src_frame_idx,
        )
