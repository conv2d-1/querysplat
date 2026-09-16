"""Decode sparse-pair Gaussian predictions (dev_lx sharp-zero-init or legacy Envision)."""
from __future__ import annotations

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.gaussians.sharp_zero_init_core import (
    SharpZeroInitGaussianAdapter,
    initialize_sparse_query_gaussians,
    sharp_adapter_param_dim,
    sharp_init_to_sparse_head_tensors,
)
from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import (
    normalize_quaternions,
    rgb_to_sh_dc,
    squeeze_query_points,
)


class SparsePairGaussianAdapter(nn.Module):
    """Map raw head outputs to Gaussian attributes.

    ``mode="sharp_zero_init"`` (dev_lx ffgs_sv aligned):
        * Initialize scales as ``z/fx, z/fy, min(z/fx,z/fy)`` from ref-frame means.
        * Apply zero-init residual offsets with multiplicative scale factor in ``[min_rate, max_rate]``.

    ``mode="envision"`` (legacy):
        Depth-conditioned bounded scale in log space.
    """

    def __init__(
        self,
        sh_degree: int = 0,
        mode: str = "sharp_zero_init",
        min_scale_rate: float = 0.0,
        max_scale_rate: float = 10.0,
        init_opacity: float = 0.5,
        sharp_zero_activation: str = "sigmoid",
        param_factors: dict | None = None,
        init_scale_px: float = 1.0,
        # legacy envision knobs
        depth_condition_scale: bool = True,
        scale_min: float = 0.5,
        scale_max: float = 15.0,
        scale_log_bias: float = -4.0,
        focal_scale_multiplier: float = 0.1,
        use_rgb_color_anchor: bool = True,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.d_sh = (sh_degree + 1) ** 2
        self.mode = mode
        self.init_opacity = init_opacity
        self.init_scale_px = init_scale_px
        self.depth_condition_scale = depth_condition_scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_log_bias = scale_log_bias
        self.focal_scale_multiplier = focal_scale_multiplier
        self.use_rgb_color_anchor = use_rgb_color_anchor

        if mode == "sharp_zero_init":
            self.sharp_adapter = SharpZeroInitGaussianAdapter(
                sh_degree=sh_degree,
                min_scale_rate=min_scale_rate,
                max_scale_rate=max_scale_rate,
                activation=sharp_zero_activation,
                param_factors=param_factors,
            )
        else:
            self.sharp_adapter = None
            sh_mask = torch.ones((self.d_sh,), dtype=torch.float32)
            for degree in range(1, sh_degree + 1):
                sh_mask[degree ** 2 : (degree + 1) ** 2] = 0.1 * (0.25 ** degree)
            self.register_buffer("sh_mask", sh_mask, persistent=False)

    @property
    def raw_param_dim(self) -> int:
        if self.mode == "sharp_zero_init":
            return sharp_adapter_param_dim(self.sh_degree)
        return 1 + 3 + 4 + 3 * self.d_sh

    def forward(
        self,
        raw_attr: torch.Tensor,
        rgb_anchor: torch.Tensor | None = None,
        depth_q: torch.Tensor | None = None,
        focal_mult: torch.Tensor | None = None,
        means_ref: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        ref_frame: int = 0,
    ) -> dict:
        if self.mode == "sharp_zero_init":
            if means_ref is None or intrinsics is None:
                raise ValueError(
                    "sharp_zero_init adapter requires means_ref and intrinsics",
                )
            means_ref = squeeze_query_points(means_ref.float())
            gs_init = initialize_sparse_query_gaussians(
                means_ref=means_ref,
                rgb=rgb_anchor,
                intrinsics=intrinsics,
                ref_frame=ref_frame,
                sh_degree=self.sh_degree,
                init_opacity=self.init_opacity,
                init_scale_px=self.init_scale_px,
            )
            decoded = self.sharp_adapter(gs_init, raw_attr)
            out = sharp_init_to_sparse_head_tensors(decoded)
            if "gs_offset_xyz" in decoded:
                out["gs_offset_xyz"] = decoded["gs_offset_xyz"]
            return out

        opacity = raw_attr[..., :1].sigmoid()
        scale_raw = raw_attr[..., 1:4]
        rotation = normalize_quaternions(raw_attr[..., 4:8])

        if self.depth_condition_scale and depth_q is not None:
            depth_q = depth_q.float().clamp(min=1e-3)
            bounded = self.scale_min + (self.scale_max - self.scale_min) * scale_raw.sigmoid()
            if focal_mult is None:
                focal_mult = torch.ones_like(depth_q[..., :1])
            else:
                while focal_mult.dim() < depth_q.dim():
                    focal_mult = focal_mult.unsqueeze(-1)
            scale_metric = bounded * depth_q * focal_mult
            scale = torch.log(scale_metric.clamp(min=1e-6))
        else:
            scale = scale_raw + self.scale_log_bias

        sh_raw = raw_attr[..., 8:]
        sh = sh_raw.reshape(*sh_raw.shape[:-1], 3, self.d_sh).permute(0, 1, 3, 2)
        sh = sh * self.sh_mask.view(1, 1, -1, 1)
        if self.use_rgb_color_anchor and rgb_anchor is not None:
            sh = sh.clone()
            sh[..., 0, :] = rgb_to_sh_dc(rgb_anchor) + sh[..., 0, :]

        return dict(
            gs_opacity=opacity,
            gs_scale=scale,
            gs_rotation=rotation,
            gs_sh=sh,
        )
