import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.decoder.blocks import _make_fusion_block, _make_scratch
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
        depth_channel=1,
        depth_act="sigmoid",
        prompt_inchannel=0,
        prompt_fusion_mode="add",
        prompt_project_cfg=None,
        prompt_flag=None,
        interp_refinenet_cfg=None,
        pred_confidence=False,
        pred_normal=False,
        pred_motion_mask=False,
        pred_invalid_mask=False,
        pred_ray=False,
        pred_ray_in_world=False,
        return_features=False,
        features_only=False,
        chunk_size=0,
        apply_zero_module=True,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.patch_size = patch_size
        self.features = features
        self.features_2 = features2
        self.depth_channel = depth_channel

        self.prompt_inchannel = prompt_inchannel
        self.prompt_fusion_mode = prompt_fusion_mode
        self.prompt_project_cfg = prompt_project_cfg
        self.prompt_flag = prompt_flag if prompt_flag is not None else [True, True, True, True]
        self.apply_zero_module = apply_zero_module

        self.interp_refinenet_cfg = interp_refinenet_cfg

        self.return_features = return_features
        self.features_only = features_only
        self.chunk_size = chunk_size

        self.build_act_postprocess()
        self.build_dpt_adapter()

        if depth_act == "sigmoid":
            self.depth_act = nn.Sigmoid()
        elif depth_act == "relu":
            self.depth_act = nn.ReLU(True)
        elif depth_act == "exp":
            self.depth_act = Exp()
        elif depth_act == "inverse_log":
            self.depth_act = InverseLog()
        elif depth_act == "moge":
            self.depth_act = MogeActivation()
        else:
            self.depth_act = nn.Identity()

        self.output_conv1 = nn.Conv2d(
            self.features,
            self.features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.interp_refinenet_cfg is not None:
            self.interp_refinenet_cfg["input_channel"] = self.features // 2
            self.interp_refinenet = instantiate_from_config(self.interp_refinenet_cfg)

        if not features_only:
            if self.depth_channel > 0:
                self.output_conv2 = nn.Sequential(
                    nn.Conv2d(
                        self.features // 2,
                        self.features_2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(True),
                    nn.Conv2d(self.features_2, self.depth_channel, kernel_size=1, stride=1, padding=0),
                    self.depth_act,
                )

            self.pred_confidence = pred_confidence
            if self.pred_confidence:
                self.confidence_conv = self.build_output_conv(self.features, self.features_2, 1)

            self.pred_normal = pred_normal
            if self.pred_normal:
                self.normal_conv = self.build_output_conv(self.features, self.features_2, 3)

            self.pred_motion_mask = pred_motion_mask
            if self.pred_motion_mask:
                self.motion_mask_conv = self.build_output_conv(self.features, self.features_2, 1)

            self.pred_invalid_mask = pred_invalid_mask
            if self.pred_invalid_mask:
                self.invalid_mask_conv = self.build_output_conv(self.features, self.features_2, 1)

            self.pred_ray = pred_ray
            if self.pred_ray:
                self.ray_conv = self.build_output_conv(self.features, self.features_2, 3)
            
            self.pred_ray_in_world = pred_ray_in_world
            if self.pred_ray_in_world:
                self.ray_in_world_conv = self.build_output_conv(self.features, self.features_2, 6)

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
            use_bn=False,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.interp_refinenet_cfg,
            apply_zero_module=self.apply_zero_module,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            self.features,
            use_bn=False,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[1] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.interp_refinenet_cfg,
            apply_zero_module=self.apply_zero_module,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            self.features,
            use_bn=False,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[2] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.interp_refinenet_cfg,
            apply_zero_module=self.apply_zero_module,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            self.features,
            use_bn=False,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[3] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.interp_refinenet_cfg,
            apply_zero_module=self.apply_zero_module,
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
        )
        return conv

    def _forward_impl(
        self,
        features,
        prompt_depth=None,
        num_views=None,
        patch_h=None,
        patch_w=None,
        patch_start_idx=None,
        view_start_idx=None,
        view_end_idx=None,
    ):
        out = []
        for i, x in enumerate(features):
            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            if view_start_idx is not None and view_end_idx is not None:
                x = x[:, view_start_idx:view_end_idx]

            if x.ndim == 4:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

            x = x.permute(0, 2, 1).contiguous().reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        if prompt_depth is not None and prompt_depth.ndim == 3:
            prompt_depth = prompt_depth.permute(0, 2, 1).view(-1, prompt_depth.shape[2], patch_h, patch_w)

        if prompt_depth is not None and view_start_idx is not None and view_end_idx is not None:
            prompt_depth = prompt_depth.reshape(-1, num_views, *prompt_depth.shape[1:])
            prompt_depth = prompt_depth[:, view_start_idx:view_end_idx].reshape(-1, *prompt_depth.shape[2:])

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
            prompt_depth=prompt_depth if self.prompt_flag[3] else None,
        )
        path_3 = self.scratch.refinenet3(
            path_4,
            layer_3_rn,
            size=layer_2_rn.shape[2:],
            prompt_depth=prompt_depth if self.prompt_flag[2] else None,
        )
        path_2 = self.scratch.refinenet2(
            path_3,
            layer_2_rn,
            size=layer_1_rn.shape[2:],
            prompt_depth=prompt_depth if self.prompt_flag[1] else None,
        )
        path_1 = self.scratch.refinenet1(
            path_2,
            layer_1_rn,
            prompt_depth=prompt_depth if self.prompt_flag[0] else None,
        )

        out = self.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * self.patch_size), int(patch_w * self.patch_size)),
            mode="bilinear",
            align_corners=True,
        )

        if self.interp_refinenet_cfg is not None:
            out = self.interp_refinenet(out)

        if self.features_only:
            return out

        if self.depth_channel == 1:
            depth = self.output_conv2(out)
            return_dict = dict(depth=depth)
        elif self.depth_channel == 3:
            points = self.output_conv2(out)
            return_dict = dict(points=points)
        elif self.depth_channel == 0:
            return_dict = dict()
        else:
            raise NotImplementedError

        if self.return_features:
            return_dict["features"] = out

        if self.pred_confidence:
            confidence = self.confidence_conv(out)
            return_dict["confidence"] = confidence

        if self.pred_normal:
            normal = self.normal_conv(out)
            return_dict["normal"] = normal

        if self.pred_motion_mask:
            motion_mask = self.motion_mask_conv(out)
            return_dict["motion_mask"] = motion_mask

        if self.pred_invalid_mask:
            invalid_mask = self.invalid_mask_conv(out)
            return_dict["invalid_mask"] = invalid_mask

        if self.pred_ray:
            ray = self.ray_conv(out)
            return_dict["ray"] = ray
        
        if self.pred_ray_in_world:
            ray_in_world = self.ray_in_world_conv(out)
            return_dict["ray_in_world"] = ray_in_world

        return return_dict

    def forward(
        self,
        features,
        prompt_depth=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        if patch_h is None or patch_w is None:
            if "patch_h" in meta_data and "patch_w" in meta_data:
                patch_w = meta_data["patch_w"]
                patch_h = meta_data["patch_h"]
            else:
                patch_w = meta_data["input_width"][0] // self.patch_size
                patch_h = meta_data["input_height"][0] // self.patch_size
        meta_data["patch_w"] = patch_w
        meta_data["patch_h"] = patch_h

        if self.training or self.chunk_size == 0:
            return self._forward_impl(
                features=features,
                prompt_depth=prompt_depth,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_start_idx=patch_start_idx,
                view_start_idx=None,
                view_end_idx=None,
            )

        assert features[0].ndim == 4
        S = features[0].shape[1]

        total_datas = [] if self.features_only else {}
        for view_start_idx in range(0, S, self.chunk_size):
            view_end_idx = min(view_start_idx + self.chunk_size, S)

            outputs = self._forward_impl(
                features=features,
                prompt_depth=prompt_depth,
                num_views=S,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_start_idx=patch_start_idx,
                view_start_idx=view_start_idx,
                view_end_idx=view_end_idx,
            )

            if self.features_only:
                total_datas.append(outputs)
            else:
                for key, val in outputs.items():
                    if key not in total_datas:
                        total_datas[key] = []
                    total_datas[key].append(val)

        if isinstance(total_datas, list):
            total_datas = torch.stack(total_datas, dim=1)
        else:
            for key, val in total_datas.items():
                total_datas[key] = torch.stack(val, dim=1)

        return total_datas

    def onnx_forward(
        self,
        features,
        prompt_depth,
        patch_h,
        patch_w,
        **kwargs,
    ):
        return self._forward_impl(
                features=features,
                prompt_depth=prompt_depth,
                patch_h=patch_h,
                patch_w=patch_w,
                patch_start_idx=None,
                view_start_idx=None,
                view_end_idx=None,
            )
