"""Shared sharp-zero-init Gaussian logic (aligned with origin/dev_lx ffgs_mv).

Used by dense FFGS (:mod:`sharp_zero_init`) and sparse pair 4DGS adapters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.pipelines2.utils.sparse_pair_gaussian_utils import rgb_to_sh_dc


def inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    return torch.log(tensor / (1.0 - tensor))


def softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x.abs())


def softsign_to_range(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    half = (max_val - min_val) * 0.5
    mid = (max_val + min_val) * 0.5
    return mid + half * softsign(x)


def opa_logit_init_softsign(opacities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = opacities.clamp(eps, 1.0 - eps)
    y = 2.0 * p - 1.0
    return y / (1.0 - y.abs())


def get_scale_activation_constant(max_scale: float, min_scale: float) -> tuple[float, float]:
    """Sigmoid scale constants (legacy ffgs_sv path)."""
    constant_a = (max_scale - min_scale) / (1 - min_scale) / (max_scale - 1)
    constant_b = inverse_sigmoid(
        torch.tensor((1.0 - min_scale) / (max_scale - min_scale))
    ).item()
    return constant_a, constant_b


def get_softsign_scale_constant(max_scale: float, min_scale: float) -> tuple[float, float]:
    """Match dev_lx ``get_softsign_scale_constant`` (ffgs_mv)."""
    half = (max_scale - min_scale) * 0.5
    mid = (max_scale + min_scale) * 0.5
    target = (1.0 - mid) / half
    if abs(target) >= 1.0:
        raise ValueError(
            f"min_scale ({min_scale}) and max_scale ({max_scale}) must satisfy "
            f"min_scale < 1 < max_scale; got softsign target={target}."
        )
    constant_b = target / (1.0 - abs(target))
    constant_a = (1.0 + abs(constant_b)) ** 2 / half
    return constant_a, constant_b


def build_sh_mask(sh_degree: int) -> torch.Tensor:
    sh_mask = torch.ones(((sh_degree + 1) ** 2,), dtype=torch.float32)
    for degree in range(1, sh_degree + 1):
        sh_mask[degree ** 2 : (degree + 1) ** 2] = 0.1 * (0.25 ** degree)
    return sh_mask


def sharp_adapter_param_dim(sh_degree: int) -> int:
    d_sh = (sh_degree + 1) ** 2
    return 1 + 3 + 3 + 4 + 3 * d_sh


class SharpZeroInitGaussianAdapter(nn.Module):
    """dev_lx ``GaussianAdapter`` / ``GaussianAdapterMV`` for flat query sets ``[B, Q, ...]``."""

    def __init__(
        self,
        sh_degree: int = 0,
        min_scale_rate: float = 0.0,
        max_scale_rate: float = 10.0,
        activation: str = "sigmoid",
        param_factors: dict | None = None,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.d_sh = (sh_degree + 1) ** 2
        self.min_scale_rate = min_scale_rate
        self.max_scale_rate = max_scale_rate
        if activation not in ("sigmoid", "softsign"):
            raise ValueError(f"Unsupported sharp_zero activation: {activation}")
        self.activation = activation
        self.param_factors = dict(param_factors or {})
        self.register_buffer("sh_mask", build_sh_mask(sh_degree), persistent=False)

    @property
    def d_in(self) -> int:
        return sharp_adapter_param_dim(self.sh_degree)

    def forward(self, gs_init: dict, gs_offset: torch.Tensor) -> dict:
        """Apply zero-init residual offsets (dev_lx ``GaussianAdapter`` math)."""
        means = gs_init["means"]
        harmonics = gs_init["harmonics"]
        opacities = gs_init["opacities"]
        scales = gs_init["scales"]
        rotations = gs_init["rotations"]

        opa_offset, xyz_offset, scale_offset, quat_offset, sh_offset = torch.split(
            gs_offset,
            [1, 3, 3, 4, 3 * self.d_sh],
            dim=-1,
        )
        if "xyz" in self.param_factors:
            xyz_offset = xyz_offset * self.param_factors["xyz"]
        if "opacities" in self.param_factors:
            opa_offset = opa_offset * self.param_factors["opacities"]
        if "scales" in self.param_factors:
            scale_offset = scale_offset * self.param_factors["scales"]
        if "rotations" in self.param_factors:
            quat_offset = quat_offset * self.param_factors["rotations"]
        if "harmonics" in self.param_factors:
            sh_offset = sh_offset * self.param_factors["harmonics"]

        if self.activation == "softsign":
            constant_a, constant_b = get_softsign_scale_constant(
                self.max_scale_rate, self.min_scale_rate,
            )
            scale_factor = softsign_to_range(
                constant_a * scale_offset + constant_b,
                self.min_scale_rate,
                self.max_scale_rate,
            )
            opa_logit_init = opa_logit_init_softsign(opacities)
            opacities_updated = softsign_to_range(
                opa_logit_init + opa_offset.squeeze(-1), 0.0, 1.0,
            )
        else:
            constant_a, constant_b = get_scale_activation_constant(
                self.max_scale_rate, self.min_scale_rate,
            )
            scale_factor = (self.max_scale_rate - self.min_scale_rate) * torch.sigmoid(
                constant_a * scale_offset + constant_b
            ) + self.min_scale_rate
            opacities_updated = torch.sigmoid(
                inverse_sigmoid(opacities) + opa_offset.squeeze(-1),
            )
        scales_updated = scales * scale_factor

        quat_updated = F.normalize(rotations + quat_offset, dim=-1)
        harmonics_offset = sh_offset.view(*sh_offset.shape[:-1], 3, self.d_sh)
        harmonics_offset = harmonics_offset * self.sh_mask.view(1, 1, 1, -1)
        harmonics_updated = harmonics + harmonics_offset
        means_updated = means + xyz_offset

        return dict(
            means=means_updated,
            harmonics=harmonics_updated,
            opacities=opacities_updated,
            scales=scales_updated,
            rotations=quat_updated,
            gs_offset_xyz=xyz_offset,
        )


def initialize_sparse_query_gaussians(
    means_ref: torch.Tensor,
    rgb: torch.Tensor | None,
    intrinsics: torch.Tensor,
    ref_frame: int,
    sh_degree: int = 0,
    init_opacity: float = 0.5,
    init_scale_px: float = 1.0,
) -> dict:
    """Sparse-query ``GaussianInitializer`` using ref-camera ``means`` and ``z/fx`` scales.

    Args:
        means_ref: ``[B, Q, 3]`` positions in the ref-camera frame (``warp3d``).
        rgb: ``[B, Q, 3]`` RGB in ``[0, 1]`` for SH DC, or None.
        intrinsics: ``[B, N, 3, 3]`` or ``[B, 3, 3]``.
        ref_frame: Index of the reference view for intrinsics.
    """
    means = means_ref.float()
    b, q, _ = means.shape
    device, dtype = means.device, means.dtype
    d_sh = (sh_degree + 1) ** 2

    z = means[..., 2].clamp(min=1e-3)
    if intrinsics.dim() == 4:
        k = intrinsics[:, ref_frame].float()
    else:
        k = intrinsics.float()
    fx = k[:, 0, 0].clamp(min=1e-6)
    fy = k[:, 1, 1].clamp(min=1e-6)
    sx = z / fx.unsqueeze(-1) * init_scale_px
    sy = z / fy.unsqueeze(-1) * init_scale_px
    sz = torch.minimum(sx, sy)
    scales = torch.stack([sx, sy, sz], dim=-1)

    rotations = torch.zeros(b, q, 4, device=device, dtype=dtype)
    rotations[..., 0] = 1.0

    opacities = torch.full((b, q), init_opacity, device=device, dtype=dtype)

    harmonics = torch.zeros(b, q, 3, d_sh, device=device, dtype=dtype)
    if rgb is not None:
        dc = rgb_to_sh_dc(rgb.float())
        harmonics[..., 0] = dc

    return dict(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )


def sharp_init_to_sparse_head_tensors(decoded: dict) -> dict:
    """Convert adapter output to ``SparsePairDynamicGaussianHead`` keys."""
    scales = decoded["scales"].clamp(min=1e-6)
    return dict(
        gs_opacity=decoded["opacities"].unsqueeze(-1),
        gs_scale=torch.log(scales),
        gs_rotation=decoded["rotations"],
        gs_sh=decoded["harmonics"].permute(0, 1, 3, 2).contiguous(),
    )
