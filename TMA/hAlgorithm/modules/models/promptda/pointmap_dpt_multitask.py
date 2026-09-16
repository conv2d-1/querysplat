# Copyright (c) 2024, Depth Anything V2
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.moge.utils.geometry_torch import normalized_view_plane_uv
from hAlgorithm.modules.models.promptda.pointmap_dpt import PointMapDPTHead

from .blocks import _make_fusion_block, _make_scratch, conv_bn_relu


class MTPointMapDPTHead(PointMapDPTHead):
    def __init__(
        self,
        pred_mask=False,
        pred_confidence=False,
        pred_segms=0,
        pred_grad=False,
        pred_inputconf=False,
        prompt_inputconf=None,
        **kwargs,
    ):
        kwargs["output_mask"] = False

        if prompt_inputconf is not None:
            if isinstance(prompt_inputconf, bool):
                self.prompt_inputconf = "concat" if prompt_inputconf else None
            else:
                self.prompt_inputconf = prompt_inputconf
        else:
            self.prompt_inputconf = None

        super().__init__(**kwargs)

        if self.prompt_inputconf is not None and self.prompt_inputconf in [
            "concat",
            "cat",
        ]:
            output_act = kwargs.get("output_act", None)
            if output_act == "sigmoid":
                act_func = nn.Sigmoid()
            elif output_act == "relu":
                act_func = nn.ReLU(True)
            else:
                act_func = nn.Identity()

            out_feat1 = self.head_features_1 + 1
            out_feat2 = self.head_features_2 + 1
            self.scratch.output_conv1 = nn.Conv2d(
                out_feat1,
                out_feat1 // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(
                    out_feat1 // 2,
                    out_feat2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(True),
                nn.Conv2d(out_feat2, 3, kernel_size=1, stride=1, padding=0),
                act_func,
            )

        self.pred_inputconf = pred_inputconf
        self.pred_mask = pred_mask
        self.pred_confidence = pred_confidence
        self.pred_segms = pred_segms
        self.pred_grad = pred_grad
        self.pred_inputconf = pred_inputconf

        if self.pred_inputconf:
            self.inputconf_dec1 = conv_bn_relu(
                self.head_features_1,
                self.head_features_1 // 2,
                kernel=3,
                stride=1,
                padding=1,
            )
            self.inputconf_dec0 = nn.Conv2d(
                self.head_features_1 // 2,
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

        if self.pred_mask:
            self.mask_dec1 = conv_bn_relu(
                self.head_features_1,
                self.head_features_1 // 2,
                kernel=3,
                stride=1,
                padding=1,
            )
            self.mask_dec0 = nn.Conv2d(
                self.head_features_1 // 2,
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

        if self.pred_confidence:
            self.conf_dec1 = conv_bn_relu(
                self.head_features_1,
                self.head_features_1 // 2,
                kernel=3,
                stride=1,
                padding=1,
            )
            self.conf_dec0 = nn.Conv2d(
                self.head_features_1 // 2,
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

        if self.pred_segms > 0:
            self.segms_dec1 = conv_bn_relu(
                self.head_features_1,
                self.head_features_1 // 2,
                kernel=3,
                stride=1,
                padding=1,
            )
            self.segms_dec0 = nn.Conv2d(
                self.head_features_1 // 2,
                self.pred_segms,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

        if self.pred_grad:
            self.grad_dec1 = conv_bn_relu(
                self.head_features_1,
                self.head_features_1 // 2,
                kernel=3,
                stride=1,
                padding=1,
            )
            # 2 channels for gradx, grady
            self.grad_dec0 = nn.Conv2d(
                self.head_features_1 // 2,
                2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

    def interpolate(self, out, patch_h, patch_w):
        if out.shape[-2] != int(patch_h * 14) or out.shape[-1] != int(patch_w * 14):
            out = F.interpolate(
                out,
                (int(patch_h * 14), int(patch_w * 14)),
                mode="bilinear",
                align_corners=True,
            )
        return out

    def forward(self, out_features, patch_h, patch_w, prompt_depth=None, **kwargs):
        predict = super().forward(out_features, patch_h, patch_w, prompt_depth, return_dict=True)
        path_1, path_2, path_3, path_4 = predict["paths"]
        layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = predict["layers"]

        # predict input confidence
        if self.pred_inputconf:
            inputConf = self.inputconf_dec1(path_1)
            inputConf = self.inputconf_dec0(inputConf)

        if self.prompt_inputconf is not None:
            if self.prompt_inputconf in ["concat", "cat"]:
                depth_path = torch.concat([path_1, inputConf], dim=1)
            elif self.prompt_inputconf in ["mul", "multiply"]:
                depth_path = torch.mul(path_1, F.sigmoid(inputConf))
            else:
                raise NotImplementedError
            depth_out = self.scratch.output_conv1(depth_path)
            depth_out = self.interpolate(depth_out, patch_h, patch_w)
            depth_out = self.scratch.output_conv2(depth_out)
        else:
            depth_out = predict["depth"]

        out = dict(depth=depth_out)

        if self.pred_inputconf:
            inputConf = self.interpolate(inputConf, patch_h, patch_w)
            out["input_confidence"] = inputConf

        if self.pred_mask:
            mask_out = self.mask_dec1(path_1)
            mask_out = self.interpolate(mask_out, patch_h, patch_w)
            mask_out = self.mask_dec0(mask_out)
            out["mask"] = mask_out
        else:
            out["mask"] = predict.get("mask", None)

        if self.pred_confidence:
            conf_out = self.conf_dec1(path_1)
            conf_out = self.interpolate(conf_out, patch_h, patch_w)
            conf_out = self.conf_dec0(conf_out)
            out["confidence"] = conf_out

        if self.pred_segms > 0:
            segms_out = self.segms_dec1(path_1)
            segms_out = self.interpolate(segms_out, patch_h, patch_w)
            segms_out = self.segms_dec0(segms_out)
            out["segms"] = segms_out

        if self.pred_grad:
            grad_out = self.grad_dec1(path_1)
            grad_out = self.interpolate(grad_out, patch_h, patch_w)
            grad_out = self.grad_dec0(grad_out)
            out["grad"] = grad_out

        return out
