import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config

from .blocks import Exp, InverseLog, PointmapExp


class PointMapHead(nn.Module):
    def __init__(
        self,
        patch_size=14,
        features=256,
        features2=32,
        output_act="sigmoid",
        pred_confidence=False,
        pred_mask=False,
        mask_cls_num=1,
        pred_seg=False,
        seg_cls_num=1,
        pred_gradient=False,
        pred_normal=False,
        pred_prompt_confidence=False,
        final_outchannel=3,
        interpolate_out_cfg=None,
        interpolate_out_size=None,
        return_features=False,
        **kwargs,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.head_features_1 = features
        self.head_features_2 = features2
        self.final_outchannel = final_outchannel
        self.interpolate_out_cfg = interpolate_out_cfg
        self.interpolate_out_size = interpolate_out_size
        self.return_features = return_features
        self.mask_cls_num = mask_cls_num
        self.seg_cls_num = seg_cls_num

        if output_act == "sigmoid":
            act_func = nn.Sigmoid()
        elif output_act == "relu":
            act_func = nn.ReLU(True)
        elif output_act == "exp":
            act_func = Exp()
        elif output_act == "inverse_log":
            act_func = InverseLog()
        elif output_act == "point_exp":
            act_func = PointmapExp()
        else:
            act_func = nn.Identity()

        self.output_conv1 = nn.Conv2d(
            self.head_features_1,
            self.head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.output_conv2 = nn.Sequential(
            nn.Conv2d(
                self.head_features_1 // 2,
                self.head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                self.head_features_2, self.final_outchannel, kernel_size=1, stride=1, padding=0
            ),
            act_func,
        )

        if self.interpolate_out_cfg is not None:
            self.interpolate_out_cfg["input_channel"] = self.head_features_1 // 2
            self.interpolate_out_conv = instantiate_from_config(self.interpolate_out_cfg)

        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.confidence_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, 1
            )

        self.pred_mask = pred_mask
        if self.pred_mask:
            self.mask_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, self.mask_cls_num
            )

        self.pred_seg = pred_seg
        if self.pred_seg:
            self.seg_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, self.seg_cls_num
            )

        self.pred_gradient = pred_gradient
        if self.pred_gradient:
            self.gradient_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, 2
            )

        self.pred_normal = pred_normal
        if self.pred_normal:
            self.normal_conv = self.build_output_conv(self.head_features_1, self.head_features_2, 3)

        self.pred_prompt_confidence = pred_prompt_confidence
        if self.pred_prompt_confidence:
            self.prompt_confidence_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, 1
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

    def forward(self, x, patch_h=None, patch_w=None, return_dict=False, meta_data=None, **kwargs):
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        out = self.output_conv1(x)

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

        if self.pred_seg:
            seg = self.seg_conv(out)
            return_dict["seg"] = seg

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


class RelDepthHead(nn.Module):
    def __init__(
        self,
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
        final_outchannel=1,
        interpolate_out_cfg=None,
        interpolate_out_size=None,
        return_features=False,
        **kwargs,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.head_features_1 = features
        self.head_features_2 = features2
        self.final_outchannel = final_outchannel
        self.interpolate_out_cfg = interpolate_out_cfg
        self.interpolate_out_size = interpolate_out_size
        self.return_features = return_features
        self.mask_cls_num = mask_cls_num

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
            self.head_features_1,
            self.head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.output_conv2 = nn.Sequential(
            nn.Conv2d(
                self.head_features_1 // 2,
                self.head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                self.head_features_2, self.final_outchannel, kernel_size=1, stride=1, padding=0
            ),
            act_func,
        )

        if self.interpolate_out_cfg is not None:
            self.interpolate_out_cfg["input_channel"] = self.head_features_1 // 2
            self.interpolate_out_conv = instantiate_from_config(self.interpolate_out_cfg)

        self.pred_confidence = pred_confidence
        if self.pred_confidence:
            self.confidence_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, 1
            )

        self.pred_mask = pred_mask
        if self.pred_mask:
            self.mask_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, self.mask_cls_num
            )

        self.pred_gradient = pred_gradient
        if self.pred_gradient:
            self.gradient_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, 2
            )

        self.pred_normal = pred_normal
        if self.pred_normal:
            self.normal_conv = self.build_output_conv(self.head_features_1, self.head_features_2, 3)

        self.pred_prompt_confidence = pred_prompt_confidence
        if self.pred_prompt_confidence:
            self.prompt_confidence_conv = self.build_output_conv(
                self.head_features_1, self.head_features_2, 1
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

    def forward(self, x, patch_h=None, patch_w=None, return_dict=False, meta_data=None):
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        out = self.output_conv1(x)

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

        rel_depth = self.output_conv2(out)
        return_dict = dict(rel_depth=rel_depth)

        if self.pred_confidence:
            confidence = self.confidence_conv(out)
            return_dict["rgb_confidence"] = confidence

        if self.pred_mask:
            mask = self.mask_conv(out)
            return_dict["rgb_mask"] = mask

        if self.pred_gradient:
            gradient = self.gradient_conv(out)
            return_dict["rgb_gradient"] = gradient

        if self.pred_normal:
            normal = self.normal_conv(out)
            return_dict["rgb_normal"] = normal

        if self.pred_prompt_confidence:
            prompt_confidence = self.prompt_confidence_conv(out)
            return_dict["rgb_prompt_confidence"] = prompt_confidence

        if self.return_features:
            return_dict["rgb_features"] = out

        return return_dict
