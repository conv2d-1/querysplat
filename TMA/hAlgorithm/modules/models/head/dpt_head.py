import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.decoder.blocks import _make_fusion_block, _make_scratch
from hAlgorithm.modules.models.vggt.heads.utils import create_uv_grid, position_grid_to_embed
from hAlgorithm.utils import instantiate_from_config

from .blocks import Exp, InverseLog, MogeActivation


class DPTHead(nn.Module):
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

    def build_output_conv(self, ch1, ch2, outch):
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

        pointmap = self.output_conv2(out)
        return_dict = dict(pointmap=pointmap)

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


class DPTHeadV2(nn.Module):
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
        mask_cls_num=1,
        pred_gradient=False,
        pred_normal=False,
        pred_prompt_confidence=False,
        final_outchannel=3,
        interpolate_out_cfg=None,
        interpolate_out_size=None,
        return_features=False,
        use_bn=False,
        prompt_inchannel=0,
        prompt_fusion_mode="add",
        prompt_project_cfg=None,
        prompt_flag=None,
        refinenet_out_cfg=None,
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
        self.mask_cls_num = mask_cls_num

        self.use_bn = use_bn
        self.prompt_inchannel = prompt_inchannel
        self.prompt_fusion_mode = prompt_fusion_mode
        self.prompt_project_cfg = prompt_project_cfg
        self.refinenet_out_cfg = refinenet_out_cfg
        self.prompt_flag = prompt_flag if prompt_flag is not None else [True, True, True, True]

        self.build_dpt_adapter()

        if output_act == "sigmoid":
            act_func = nn.Sigmoid()
        elif output_act == "relu":
            act_func = nn.ReLU(True)
        elif output_act == "exp":
            act_func = Exp()
        elif output_act == "inverse_log":
            act_func = InverseLog()
        else:
            act_func = nn.Identity()

        self.output_conv1 = nn.Conv2d(
            self.features,
            self.features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
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
            nn.Conv2d(self.features_2, self.final_outchannel, kernel_size=1, stride=1, padding=0),
            act_func,
        )

        if self.interpolate_out_cfg is not None:
            self.interpolate_out_cfg["input_channel"] = self.features // 2
            self.interpolate_out_conv = instantiate_from_config(self.interpolate_out_cfg)

        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.confidence_conv = self.build_output_conv(self.features, self.features_2, 1)

        self.pred_mask = pred_mask
        if self.pred_mask:
            self.mask_conv = self.build_output_conv(
                self.features, self.features_2, self.mask_cls_num
            )

        self.pred_gradient = pred_gradient
        if self.pred_gradient:
            self.gradient_conv = self.build_output_conv(self.features, self.features_2, 2)

        self.pred_normal = pred_normal
        if self.pred_normal:
            self.normal_conv = self.build_output_conv(self.head_features_1, self.head_features_2, 3)

        self.pred_prompt_confidence = pred_prompt_confidence
        if self.pred_prompt_confidence:
            self.prompt_confidence_conv = self.build_output_conv(self.features, self.features_2, 1)

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

    def build_output_conv(self, ch1, ch2, outch):
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
        )
        return conv

    def forward(
        self,
        rgb_features,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        return_dict=False,
        meta_data=None,
    ):
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        layer_1, layer_2, layer_3, layer_4 = rgb_features

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

        pointmap = self.output_conv2(out)
        return_dict = dict(pointmap=pointmap)

        if self.pred_confidence:
            confidence = self.confidence_conv(out)
            return_dict["confidence"] = confidence

        if self.pred_mask:
            mask = self.mask_conv(out)
            return_dict["mask"] = mask

        if self.pred_gradient:
            gradient = self.gradient_conv(out)
            return_dict["gradient"] = gradient

        if self.pred_normal:
            normal = self.normal_conv(out)
            return_dict["normal"] = normal

        if self.pred_prompt_confidence:
            prompt_confidence = self.prompt_confidence_conv(out)
            return_dict["prompt_confidence"] = prompt_confidence

        if self.return_features:
            return_dict["features"] = out

        return return_dict
