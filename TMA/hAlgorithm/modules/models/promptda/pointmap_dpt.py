# Copyright (c) 2024, Depth Anything V2
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.moge.utils.geometry_torch import normalized_view_plane_uv
from hAlgorithm.utils import instantiate_from_config

from .blocks import Exp, _make_fusion_block, _make_scratch, conv_bn_relu


class PointMapDPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        output_act="sigmoid",
        with_uv=False,
        output_mask=False,
        prompt_inchannel=3,
        final_outchannel=3,
        prompt_enc_deg=None,
        prompt_fusion="add",
        prompt_cfg=None,
        prompt_pre_cfg=None,
        prompt_flag=None,
        fusion_out_cfg=None,
        **kwargs,
    ):
        super().__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.with_uv = with_uv
        self.prompt_fusion = prompt_fusion
        self.prompt_cfg = prompt_cfg
        self.prompt_pre_cfg = prompt_pre_cfg
        self.fusion_out_cfg = fusion_out_cfg
        self.prompt_flag = prompt_flag
        if self.prompt_flag is None:
            self.prompt_flag = [True, True, True, True]

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        if self.with_uv:
            features += 2

        self.scratch.refinenet1 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[0] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[1] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[2] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[3] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )

        head_features_1 = features
        head_features_2 = 32
        self.head_features_1 = head_features_1
        self.head_features_2 = head_features_2

        if output_act == "sigmoid":
            act_func = nn.Sigmoid()
        elif output_act == "relu":
            act_func = nn.ReLU(True)
        elif output_act == "exp":
            act_func = Exp()
        else:
            act_func = nn.Identity()

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, final_outchannel, kernel_size=1, stride=1, padding=0),
            act_func,
        )

        self.output_mask = output_mask
        if self.output_mask:
            self.output_mask_conv = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                # nn.Sigmoid(),  # do sigmoid outside
            )
            # NOTE: without activate

        if self.prompt_pre_cfg is not None:
            self.prompt_pre = instantiate_from_config(self.prompt_pre_cfg)

    def add_uv_pose(self, features):
        if self.with_uv:
            uv = normalized_view_plane_uv(
                width=features.shape[-1],
                height=features.shape[-2],
                dtype=features.dtype,
                device=features.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(features.shape[0], -1, -1, -1)
            features = torch.cat([features, uv], dim=1)
        return features

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None, return_dict=False):
        if self.prompt_pre_cfg is not None and prompt_depth is not None:
            prompt_depth = self.prompt_pre(prompt_depth)

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_1_rn = self.add_uv_pose(layer_1_rn)

        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_2_rn = self.add_uv_pose(layer_2_rn)

        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_3_rn = self.add_uv_pose(layer_3_rn)

        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_4_rn = self.add_uv_pose(layer_4_rn)

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
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        final_out = self.scratch.output_conv2(out)
        if self.output_mask:
            mask = self.output_mask_conv(out)
        else:
            mask = None
        if not return_dict:
            return final_out, mask
        else:
            return dict(
                depth=final_out,
                mask=mask,
                layers=(layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn),
                paths=(path_1, path_2, path_3, path_4),
            )


class MTPointMapDPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        output_act="sigmoid",
        with_uv=False,
        pred_confidence=False,
        pred_mask=False,
        pred_gradient=False,
        pred_prompt_confidence=False,
        prompt_inchannel=3,
        final_outchannel=3,
        prompt_enc_deg=None,
        prompt_fusion="add",
        prompt_cfg=None,
        prompt_pre_cfg=None,
        prompt_flag=None,
        fusion_out_cfg=None,
        interpolate_out_cfg=None,
        interpolate_out_size=None,
        return_features=False,
        upsample_scale=None,
        **kwargs,
    ):
        super().__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.with_uv = with_uv
        self.prompt_fusion = prompt_fusion
        self.prompt_cfg = prompt_cfg
        self.prompt_pre_cfg = prompt_pre_cfg
        self.fusion_out_cfg = fusion_out_cfg
        self.interpolate_out_cfg = interpolate_out_cfg
        self.interpolate_out_size = interpolate_out_size

        self.prompt_flag = prompt_flag
        if self.prompt_flag is None:
            self.prompt_flag = [True, True, True, True]
        self.return_features = return_features

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        if self.with_uv:
            features += 2

        self.scratch.refinenet1 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[0] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[1] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[2] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            features,
            use_bn,
            prompt_inchannel=prompt_inchannel if self.prompt_flag[3] else 0,
            sin_enc=prompt_enc_deg,
            mode=self.prompt_fusion,
            depth_cfg=self.prompt_cfg,
            out_conv_cfg=self.fusion_out_cfg,
        )

        head_features_1 = features
        head_features_2 = 32
        self.head_features_1 = head_features_1
        self.head_features_2 = head_features_2

        if output_act == "sigmoid":
            act_func = nn.Sigmoid()
        elif output_act == "relu":
            act_func = nn.ReLU(True)
        elif output_act == "exp":
            act_func = Exp()
        else:
            act_func = nn.Identity()

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, final_outchannel, kernel_size=1, stride=1, padding=0),
            act_func,
        )

        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.confidence_conv = self.build_output_conv(head_features_1, head_features_2, 1)

        self.pred_mask = pred_mask
        if self.pred_mask:
            self.mask_conv = self.build_output_conv(head_features_1, head_features_2, 1)

        self.pred_gradient = pred_gradient
        if self.pred_gradient:
            self.gradient_conv = self.build_output_conv(head_features_1, head_features_2, 2)

        self.pred_prompt_confidence = pred_prompt_confidence
        if self.pred_prompt_confidence:
            self.prompt_confidence_conv = self.build_output_conv(
                head_features_1, head_features_2, 1
            )

        if self.prompt_pre_cfg is not None:
            self.prompt_pre = instantiate_from_config(self.prompt_pre_cfg)

        if self.interpolate_out_cfg is not None:
            self.interpolate_out_cfg["input_channel"] = head_features_1 // 2
            self.interpolate_out_conv = instantiate_from_config(self.interpolate_out_cfg)

        self.upsample_scale = upsample_scale

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

    def add_uv_pose(self, features):
        if self.with_uv:
            uv = normalized_view_plane_uv(
                width=features.shape[-1],
                height=features.shape[-2],
                dtype=features.dtype,
                device=features.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(features.shape[0], -1, -1, -1)
            features = torch.cat([features, uv], dim=1)
        return features

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None, return_dict=False):
        if self.prompt_pre_cfg is not None and prompt_depth is not None:
            prompt_depth = self.prompt_pre(prompt_depth)

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_1_rn = self.add_uv_pose(layer_1_rn)

        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_2_rn = self.add_uv_pose(layer_2_rn)

        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_3_rn = self.add_uv_pose(layer_3_rn)

        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_4_rn = self.add_uv_pose(layer_4_rn)

        if self.upsample_scale is not None:
            layer_1_rn = F.interpolate(
                layer_1_rn,
                (
                    int(layer_1_rn.shape[-2] * self.upsample_scale),
                    int(layer_1_rn.shape[-1] * self.upsample_scale),
                ),
                mode="bilinear",
                align_corners=True,
            )
            layer_2_rn = F.interpolate(
                layer_2_rn,
                (
                    int(layer_2_rn.shape[-2] * self.upsample_scale),
                    int(layer_2_rn.shape[-1] * self.upsample_scale),
                ),
                mode="bilinear",
                align_corners=True,
            )
            layer_3_rn = F.interpolate(
                layer_3_rn,
                (
                    int(layer_3_rn.shape[-2] * self.upsample_scale),
                    int(layer_3_rn.shape[-1] * self.upsample_scale),
                ),
                mode="bilinear",
                align_corners=True,
            )
            layer_4_rn = F.interpolate(
                layer_1_rn,
                (
                    int(layer_4_rn.shape[-2] * self.upsample_scale),
                    int(layer_4_rn.shape[-1] * self.upsample_scale),
                ),
                mode="bilinear",
                align_corners=True,
            )

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
        out = self.scratch.output_conv1(path_1)

        if self.interpolate_out_size is not None:
            out = F.interpolate(
                out,
                self.interpolate_out_size,
                mode="bilinear",
                align_corners=True,
            )
        elif self.upsample_scale is None:
            out = F.interpolate(
                out,
                (int(patch_h * 14), int(patch_w * 14)),
                mode="bilinear",
                align_corners=True,
            )
        else:
            out = F.interpolate(
                out,
                (int(patch_h * 14 * self.upsample_scale), int(patch_w * 14 * self.upsample_scale)),
                mode="bilinear",
                align_corners=True,
            )

        if self.interpolate_out_cfg is not None:
            out = self.interpolate_out_conv(out)

        depth = self.scratch.output_conv2(out)
        return_dict = dict(depth=depth)

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
            return_dict["layers"] = (layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn)
            return_dict["paths"] = (path_1, path_2, path_3, path_4)

        return return_dict


class SRMTPointMapDPTHead(MTPointMapDPTHead):
    """Super-Resolution"""

    def __init__(self, sr_interpolate_out_size=None, sr_interpolate_out_cfg=None, **kwargs):
        super().__init__(**kwargs)

        self.sr_interpolate_out_size = sr_interpolate_out_size
        self.sr_interpolate_out_cfg = sr_interpolate_out_cfg

        if self.sr_interpolate_out_cfg is not None:
            self.sr_interpolate_out_cfg["input_channel"] = self.head_features_1 // 2
            self.sr_interpolate_out_conv = instantiate_from_config(self.sr_interpolate_out_cfg)

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None, return_dict=False):
        if self.prompt_pre_cfg is not None and prompt_depth is not None:
            prompt_depth = self.prompt_pre(prompt_depth)

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_1_rn = self.add_uv_pose(layer_1_rn)

        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_2_rn = self.add_uv_pose(layer_2_rn)

        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_3_rn = self.add_uv_pose(layer_3_rn)

        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_4_rn = self.add_uv_pose(layer_4_rn)

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
        out = self.scratch.output_conv1(path_1)

        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )

        if self.interpolate_out_cfg is not None:
            out = self.interpolate_out_conv(out)

        out = F.interpolate(
            out,
            self.sr_interpolate_out_size,
            mode="bilinear",
            align_corners=True,
        )

        if self.sr_interpolate_out_cfg is not None:
            out = self.sr_interpolate_out_conv(out)

        depth = self.scratch.output_conv2(out)
        return_dict = dict(depth=depth)

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
            return_dict["layers"] = (layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn)
            return_dict["paths"] = (path_1, path_2, path_3, path_4)

        return return_dict
