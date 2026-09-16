from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_2d_blocks import DownBlock2D, UNetMidBlock2D, UpBlock2D


class UNet2DModel(nn.Module):
    r"""
    Parameters:
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                num_layers=layers_per_block,
                resnet_eps=norm_eps,
                resnet_groups=norm_num_groups,
                resnet_pre_norm=True,
                output_scale_factor=1.0,
                add_downsample=not is_final_block,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            dropout=dropout,
            resnet_eps=norm_eps,
            output_scale_factor=mid_block_scale_factor,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpBlock2D(
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                resolution_idx=None,
                dropout=dropout,
                num_layers=layers_per_block + 1,
                resnet_eps=norm_eps,
                resnet_groups=norm_num_groups,
                resnet_pre_norm=True,
                output_scale_factor=1.0,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = (
            norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample: torch.Tensor, **kwargs):

        H, W = sample.shape[-2:]
        stride = 2 ** len(self.down_blocks)
        if (H % stride > 0) or (W % stride > 0):
            new_H = (H // stride) * stride
            new_w = (W // stride) * stride
            sample = F.interpolate(sample, (new_H, new_w), mode="bilinear", align_corners=True)

        # 1. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 2. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample)
            down_block_res_samples += res_samples

        # 3. mid
        sample = self.mid_block(sample)

        # 4. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples)

        # 5. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if (H % stride > 0) or (W % stride > 0):
            sample = F.interpolate(sample, (H, W), mode="bilinear", align_corners=True)

        return sample


# if __name__ == "__main__":

#     model = UNet2DModel()
#     sample = model(torch.ones([1, 3, 448, 672]))
#     print(sample.shape)
