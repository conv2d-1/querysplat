from dataclasses import dataclass
from typing import Dict

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from hAlgorithm.modules.models.gaussiansv2.decoder.cuda_splatting import (
    DepthRenderingMode,
    render_cuda,
    render_depth_cuda,
)
from hAlgorithm.modules.utils.gaussians.types import GaussiansV2 as Gaussians


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None
    radii: Float[Tensor, "batch view num_gaussians"]
    visibility_filter: Float[Tensor, "batch view num_gaussians"]


class DecoderSplattingCUDA(nn.Module):
    background_color: Float[Tensor, "3"]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = "depth",
    ) -> DecoderOutput:  # 确保返回的是字典
        b, v, _, _ = extrinsics.shape
        all_colors = []
        all_depths = []
        all_radii = []
        all_visibility_filter = []
        for i in range(b):
            color, depth, radii, visibility_filter = render_cuda(
                extrinsics[i],
                intrinsics[i],
                near[i],
                far[i],
                image_shape,
                repeat(self.background_color, "c -> v c", v=v),
                gaussians[i],
            )
            all_colors.append(color)
            all_depths.append(depth)
            all_radii.append(radii)
            all_visibility_filter.append(visibility_filter)

        torch.cuda.empty_cache()

        color = torch.stack(all_colors, dim=0)
        depth = torch.stack(all_depths, dim=0)

        radii = torch.stack(all_radii, dim=0)
        visibility_filter = torch.stack(all_visibility_filter, dim=0)

        output = DecoderOutput(
            color=color,
            depth=None if depth_mode is None else depth,
            radii=radii,
            visibility_filter=visibility_filter,
        )

        return output
