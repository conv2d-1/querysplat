"""
Modified From https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/dpt_temporal.py
"""

import torch.nn as nn
from easydict import EasyDict

from ..motion_module.motion_module import TemporalModule
from .blocks import _make_fusion_block, _make_scratch


class PromptDinov2TemporalDecoder(nn.Module):
    def __init__(
        self,
        features,
        prompt_inchannel,
        in_channels,
        use_bn=False,
        prompt_enc_deg=None,
        prompt_fusion="add",
        prompt_cfg=None,
        prompt_flag=None,
        fusion_out_cfg=None,
        enable_temporal_module=True,
        num_frames=32,
        pe="ape",
        **kwargs,
    ):
        """
        Initializes the PromptFast3RDinoDecoder module.

        Parameters:
        - features: Number of features for the scratch layers.
        - prompt_inchannel: Number of input channels for prompt features.
        - in_channels: List of input channel sizes for each layer.
        - use_bn: Boolean flag indicating whether to use batch normalization.
        - prompt_enc_deg: Degree for positional encoding.
        - prompt_fusion: Method to fuse prompt features ('add', 'concat', etc.).
        - prompt_cfg: Configuration dictionary for prompt processing.
        - prompt_flag: List of boolean flags indicating whether to use prompt features at each stage.
        - fusion_out_cfg: Configuration for output convolution in fusion blocks.
        """
        super().__init__()

        self.features = features
        self.prompt_inchannel = prompt_inchannel
        self.in_channels = in_channels
        self.use_bn = use_bn
        self.prompt_enc_deg = prompt_enc_deg
        self.prompt_fusion = prompt_fusion
        self.prompt_cfg = prompt_cfg
        self.fusion_out_cfg = fusion_out_cfg
        self.prompt_flag = prompt_flag if prompt_flag is not None else [True, True, True, True]

        # Initialize scratch layers for feature refinement
        self.scratch = _make_scratch(
            self.in_channels,
            self.features,
            groups=1,
            expand=False,
        )

        # Set stem transpose to None (optional component)
        self.scratch.stem_transpose = None

        # Initialize refinenet layers with optional prompt features
        self.scratch.refinenet1 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[0] else 0,
            sin_enc=self.prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[1] else 0,
            sin_enc=self.prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[2] else 0,
            sin_enc=self.prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[3] else 0,
            sin_enc=self.prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )

        self.enable_temporal_module = enable_temporal_module

        if self.enable_temporal_module:
            assert num_frames > 0
            motion_module_kwargs = EasyDict(
                num_attention_heads=8,
                num_transformer_block=1,
                num_attention_blocks=2,
                temporal_max_len=num_frames,
                zero_initialize=True,
                pos_embedding_type=pe,
            )

            self.motion_modules = nn.ModuleList(
                [
                    TemporalModule(in_channels=in_channels[2], **motion_module_kwargs),
                    TemporalModule(in_channels=in_channels[3], **motion_module_kwargs),
                    TemporalModule(in_channels=features, **motion_module_kwargs),
                    TemporalModule(in_channels=features, **motion_module_kwargs),
                ]
            )

    def forward(self, rgb_features, prompt_features=None, meta_data=None):
        """
        Forward pass through the model.

        Parameters:
        - rgb_features: List of RGB feature tensors from different layers.
        - prompt_features: Optional tensor containing prompt features.

        Returns:
        - path_1: Processed feature tensor after all refinement stages.
        """
        if meta_data is not None:
            frame_num = meta_data["frames"][0]
            view_num = meta_data["views"][0]
            frame_length = frame_num * view_num
        else:
            raise ValueError("meta_data is None in PromptDinov2TemporalDecoder!")

        layer_1, layer_2, layer_3, layer_4 = rgb_features

        B, T = layer_1.shape[0] // frame_length, frame_length

        if self.enable_temporal_module:
            layer_3 = (
                self.motion_modules[0](
                    layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None
                )
                .permute(0, 2, 1, 3, 4)
                .flatten(0, 1)
            )
            layer_4 = (
                self.motion_modules[1](
                    layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None
                )
                .permute(0, 2, 1, 3, 4)
                .flatten(0, 1)
            )

        # Apply reduction networks on RGB features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Refine features using refinenet layers, optionally incorporating prompt features
        path_4 = self.scratch.refinenet4(
            layer_4_rn,
            size=layer_3_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[3] else None,
        )
        if self.enable_temporal_module:
            path_4 = (
                self.motion_modules[2](
                    path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None
                )
                .permute(0, 2, 1, 3, 4)
                .flatten(0, 1)
            )

        path_3 = self.scratch.refinenet3(
            path_4,
            layer_3_rn,
            size=layer_2_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[2] else None,
        )
        if self.enable_temporal_module:
            path_3 = (
                self.motion_modules[3](
                    path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None
                )
                .permute(0, 2, 1, 3, 4)
                .flatten(0, 1)
            )

        path_2 = self.scratch.refinenet2(
            path_3,
            layer_2_rn,
            size=layer_1_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[1] else None,
        )
        path_1 = self.scratch.refinenet1(
            path_2,
            layer_1_rn,
            prompt_depth=prompt_features if self.prompt_flag[0] else None,
        )

        return path_1
