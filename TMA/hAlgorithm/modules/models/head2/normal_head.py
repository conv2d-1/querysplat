import torch
import torch.nn as nn
from typing import *
import functools
import torch.nn.functional as F

from hAlgorithm.modules.models.decoder.blocks import _make_fusion_block, _make_scratch
from hAlgorithm.modules.models.vggt.heads.utils import create_uv_grid, position_grid_to_embed
from hAlgorithm.utils import instantiate_from_config

from ..head.blocks import Exp, InverseLog, MogeActivation


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        in_norm: Literal["group_norm", "layer_norm", "instance_norm", "none"] = "layer_norm",
        hidden_norm: Literal["group_norm", "layer_norm", "instance_norm"] = "group_norm",
    ):
        super(ResidualConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = nn.ReLU
        elif activation == "leaky_relu":
            activation_cls = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        elif activation == "silu":
            activation_cls = nn.SiLU
        elif activation == "elu":
            activation_cls = nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.Sequential(
            (
                nn.GroupNorm(in_channels // 32, in_channels)
                if in_norm == "group_norm"
                else (
                    nn.GroupNorm(1, in_channels)
                    if in_norm == "layer_norm"
                    else (
                        nn.InstanceNorm2d(in_channels)
                        if in_norm == "instance_norm"
                        else nn.Identity()
                    )
                )
            ),
            activation_cls(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            (
                nn.GroupNorm(hidden_channels // 32, hidden_channels)
                if hidden_norm == "group_norm"
                else (
                    nn.GroupNorm(1, hidden_channels)
                    if hidden_norm == "layer_norm"
                    else (
                        nn.InstanceNorm2d(hidden_channels)
                        if hidden_norm == "instance_norm"
                        else nn.Identity()
                    )
                )
            ),
            activation_cls(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class Resampler(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: Literal[
            "pixel_shuffle",
            "nearest",
            "bilinear",
            "conv_transpose",
            "pixel_unshuffle",
            "avg_pool",
            "max_pool",
        ],
        scale_factor: int = 2,
    ):
        if type_ == "pixel_shuffle":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels * (scale_factor**2),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            for i in range(1, scale_factor**2):
                self[0].weight.data[i :: scale_factor**2] = self[0].weight.data[
                    0 :: scale_factor**2
                ]
                self[0].bias.data[i :: scale_factor**2] = self[0].bias.data[0 :: scale_factor**2]
        elif type_ in ["nearest", "bilinear"]:
            nn.Sequential.__init__(
                self,
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode=type_,
                    align_corners=False if type_ == "bilinear" else None,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "conv_transpose":
            nn.Sequential.__init__(
                self,
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == "pixel_unshuffle":
            nn.Sequential.__init__(
                self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(
                    in_channels * (scale_factor**2),
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "avg_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == "max_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f"Unsupported resampler type: {type_}")


class ConvStack(nn.Module):
    def __init__(
        self,
        dim_in: List[Optional[int]],
        dim_res_blocks: List[int],
        dim_out: List[Optional[int]],
        resamplers: Union[
            Literal[
                "pixel_shuffle",
                "nearest",
                "bilinear",
                "conv_transpose",
                "pixel_unshuffle",
                "avg_pool",
                "max_pool",
            ],
            List,
        ],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_in_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "layer_norm",
        res_block_hidden_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "group_norm",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
    ):
        super().__init__()
        self.input_blocks = nn.ModuleList(
            [
                (
                    nn.Conv2d(dim_in_, dim_res_block_, kernel_size=1, stride=1, padding=0)
                    if dim_in_ is not None
                    else nn.Identity()
                )
                for dim_in_, dim_res_block_ in zip(
                    dim_in if isinstance(dim_in, Sequence) else itertools.repeat(dim_in),
                    dim_res_blocks,
                )
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler)
                for i, (dim_prev, dim_succ, resampler) in enumerate(
                    zip(
                        dim_res_blocks[:-1],
                        dim_res_blocks[1:],
                        (
                            resamplers
                            if isinstance(resamplers, Sequence)
                            else itertools.repeat(resamplers)
                        ),
                    )
                )
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        ResidualConvBlock(
                            dim_res_block_,
                            dim_res_block_,
                            dim_times_res_block_hidden * dim_res_block_,
                            activation=activation,
                            in_norm=res_block_in_norm,
                            hidden_norm=res_block_hidden_norm,
                        )
                        for _ in range(
                            num_res_blocks[i]
                            if isinstance(num_res_blocks, list)
                            else num_res_blocks
                        )
                    )
                )
                for i, dim_res_block_ in enumerate(dim_res_blocks)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                (
                    nn.Conv2d(dim_res_block_, dim_out_, kernel_size=1, stride=1, padding=0)
                    if dim_out_ is not None
                    else nn.Identity()
                )
                for dim_out_, dim_res_block_ in zip(
                    dim_out if isinstance(dim_out, Sequence) else itertools.repeat(dim_out),
                    dim_res_blocks,
                )
            ]
        )

    def forward(self, in_features: List[torch.Tensor]):
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


class NormalConvStack(ConvStack):
    def __init__(self, patch_size=14, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def forward(
        self,
        features,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        return_dict=False,
        meta_data=None,
        patch_start_idx=None,
        additiontal_features=None,
        **kwargs,
    ):
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        feature_name = "refine_features"
        if additiontal_features is not None and feature_name in additiontal_features:
            features_in = additiontal_features[feature_name][:4]
        else:
            features_in = []
            for x in features:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = (
                    x.permute(0, 2, 1)
                    .reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
                    .contiguous()
                )
                features_in.append(x)

        out_features = super().forward(features_in)
        normal = out_features[-1]
        normal = F.interpolate(
            normal,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )
        normal = F.normalize(normal, dim=1)
        out = dict(normal=normal)
        return out


from ..aggregator.fuse_encoder import FuseEncoder


class DPTNormalHead(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        patch_size=14,
        features=256,
        features2=32,
        output_act="sigmoid",
        pred_confidence=False,
        pred_mask=False,
        pred_gradient=False,
        pred_prompt_confidence=False,
        final_outchannel=3,
        interpolate_out_cfg=None,
        interpolate_out_size=None,
        use_bn=False,
        use_dino_clstoken=False,
        prompt_inchannel=0,
        prompt_fusion_mode="add",
        prompt_project_cfg=None,
        prompt_flag=None,
        prompt_patch_add=False,
        prompt_patch_size=7,
        refinenet_out_cfg=None,
        return_features=False,
        features_only=False,
        return_refine_features=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.patch_size = patch_size
        self.features = features
        self.features_2 = features2
        self.final_outchannel = final_outchannel
        self.interpolate_out_cfg = interpolate_out_cfg
        self.interpolate_out_size = interpolate_out_size
        self.return_features = return_features

        self.use_bn = use_bn
        self.use_dino_clstoken = use_dino_clstoken
        self.prompt_inchannel = prompt_inchannel
        self.prompt_fusion_mode = prompt_fusion_mode
        self.prompt_project_cfg = prompt_project_cfg
        self.refinenet_out_cfg = refinenet_out_cfg
        self.prompt_flag = prompt_flag if prompt_flag is not None else [True, True, True, True]
        self.features_only = features_only
        self.return_refine_features = return_refine_features

        self.build_act_postprocess()
        self.build_dpt_adapter()

        if output_act == "sigmoid":
            self.act_func = nn.Sigmoid()
        elif output_act == "relu":
            self.act_func = nn.ReLU(True)
        elif output_act == "exp":
            self.act_func = Exp()
        elif output_act == "inverse_log":
            self.act_func = InverseLog()
        elif output_act == "moge":
            self.act_func = MogeActivation()
        else:
            self.act_func = nn.Identity()

        self.output_conv1 = nn.Conv2d(
            self.features,
            self.features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.interpolate_out_cfg is not None:
            self.interpolate_out_cfg["input_channel"] = self.features // 2
            self.interpolate_out_conv = instantiate_from_config(self.interpolate_out_cfg)

        if not features_only:
            self.prompt_patch_add = prompt_patch_add
            if self.prompt_patch_add:
                self.fuse_layer = FuseEncoder(
                    in_channels[-1],
                    use_cls_token=False,
                    with_layer_norm=True,
                    fuse_depth=True,
                    fuse_ray=False,
                    fuse_extra=False,
                    prompt_depth_in_channel=prompt_inchannel,
                    prompt_depth_patch_size=prompt_patch_size,
                )

            self.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    self.features // 2,
                    self.features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    self.features_2, self.final_outchannel, kernel_size=1, stride=1, padding=0
                ),
                self.act_func,
            )

            self.pred_confidence = pred_confidence
            if self.pred_confidence:
                self.confidence_conv = self.build_output_conv(self.features, self.features_2, 1)

            self.pred_mask = pred_mask
            if self.pred_mask:
                self.mask_conv = self.build_output_conv(self.features, self.features_2, 1)

            self.pred_gradient = pred_gradient
            if self.pred_gradient:
                self.gradient_conv = self.build_output_conv(self.features, self.features_2, 2)

            self.pred_prompt_confidence = pred_prompt_confidence
            if self.pred_prompt_confidence:
                self.prompt_confidence_conv = self.build_output_conv(
                    self.features, self.features_2, 1
                )

    def build_act_postprocess(self):
        if isinstance(self.in_channels, (list, tuple)):
            self.projects = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for in_channel, out_channel in zip(self.in_channels, self.mid_channels)
                ]
            )
        else:
            self.projects = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for out_channel in self.mid_channels
                ]
            )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.mid_channels[0],
                    out_channels=self.mid_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.mid_channels[1],
                    out_channels=self.mid_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=self.mid_channels[3],
                    out_channels=self.mid_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if self.use_dino_clstoken:
            self.readout_projects = nn.ModuleList()
            for i in range(len(self.projects)):
                if isinstance(self.in_channels, (list, tuple)):
                    self.readout_projects.append(
                        nn.Sequential(
                            nn.Linear(2 * self.in_channels[i], self.in_channels[i]), nn.GELU()
                        )
                    )
                else:
                    self.readout_projects.append(
                        nn.Sequential(nn.Linear(2 * self.in_channels, self.in_channels), nn.GELU())
                    )

    def build_dpt_adapter(self):
        # 初始化用于特征细化的scratch层
        self.scratch = _make_scratch(
            self.mid_channels,
            self.features,
            groups=1,
            expand=False,
        )

        # 设置stem transpose为None（可选组件）
        self.scratch.stem_transpose = None

        # 根据是否使用提示特征初始化refinenet层
        self.scratch.refinenet1 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[1] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[2] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[3] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )

    def build_output_conv(self, ch1, ch2, outch, last_act=None):
        if last_act == "relu":
            act = nn.ReLU(True)
        elif last_act == "exp":
            act = Exp()
        else:
            act = nn.Identity()
        conv = nn.Sequential(
            nn.Conv2d(
                ch1 // 2,
                ch2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(ch2, outch, kernel_size=1, stride=1, padding=0),
            # nn.Sigmoid(),  # do sigmoid outside
            act,
        )
        return conv

    def forward(
        self,
        features,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        return_dict=False,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        if self.prompt_patch_add and prompt_features is not None:
            features = self.fuse_layer(
                patch_features=features,
                prompt_depth=prompt_features,
                meta_data=meta_data,
            )
        out = []
        for i, x in enumerate(features):

            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            if self.use_dino_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            elif isinstance(x, (list, tuple)):
                x = x[0]

            if x.ndim == 4:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # 对RGB特征应用降维网络
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 使用refinenet层细化特征，可选择性地结合提示特征
        path_4 = self.scratch.refinenet4(
            layer_4_rn,
            size=layer_3_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[3] else None,
        )
        path_3 = self.scratch.refinenet3(
            path_4,
            layer_3_rn,
            size=layer_2_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[2] else None,
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

        out = self.output_conv1(path_1)

        if self.interpolate_out_size is not None:
            out = F.interpolate(
                out,
                self.interpolate_out_size,
                mode="bilinear",
                align_corners=True,
            )
        else:
            out = F.interpolate(
                out,
                (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
                mode="bilinear",
                align_corners=True,
            )

        if self.interpolate_out_cfg is not None:
            out = self.interpolate_out_conv(out)

        if self.features_only:
            return out

        normal = self.output_conv2(out)
        normal = F.normalize(normal, dim=1)
        return_dict = dict(normal=normal)

        if self.pred_confidence:
            confidence = self.confidence_conv(out)
            return_dict["confidence"] = confidence

        if self.pred_mask:
            mask = self.mask_conv(out)
            return_dict["mask"] = mask

        if self.pred_gradient:
            gradient = self.gradient_conv(out)
            return_dict["gradient"] = gradient

        if self.pred_prompt_confidence:
            prompt_confidence = self.prompt_confidence_conv(out)
            return_dict["prompt_confidence"] = prompt_confidence

        if self.return_features:
            return_dict["features"] = out

        if self.return_refine_features:
            return_dict["refine_features"] = [path_4, path_3, path_2, path_1, out]

        return return_dict
