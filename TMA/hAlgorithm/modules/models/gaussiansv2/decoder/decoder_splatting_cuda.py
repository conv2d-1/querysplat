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
        for i in range(b):
            color = render_cuda(
                extrinsics[i],
                intrinsics[i],
                near[i],
                far[i],
                image_shape,
                repeat(self.background_color, "c -> v c", v=v),
                gaussians[i],
            )
            all_colors.append(color)

            if depth_mode is not None:
                depth = render_depth_cuda(
                    extrinsics[i],
                    intrinsics[i],
                    near[i],
                    far[i],
                    image_shape,
                    gaussians[i],
                    depth_mode,
                )
                all_depths.append(depth)
        torch.cuda.empty_cache()

        color = torch.stack(all_colors, dim=0)
        if depth_mode is not None:
            depth = torch.stack(all_depths, dim=0)

        output = DecoderOutput(
            color=color,
            depth=None if depth_mode is None else depth,
        )

        return output
