import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.moge.utils.geometry_torch import normalized_view_plane_uv
from hAlgorithm.utils import instantiate_from_config

from .blocks import Exp, _make_fusion_block, _make_scratch, conv_bn_relu


class MTPointMapMultiFrameDPTHead(nn.Module):
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
        upsample_scale=None,
        return_features=False,
        dino_fusion_cfg=None,
        path_fusion_cfg=None,
        only_last_frame=True,
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
        self.dino_fusion_cfg = dino_fusion_cfg
        self.path_fusion_cfg = path_fusion_cfg
        self.only_last_frame = only_last_frame

        if self.prompt_flag is None:
            self.prompt_flag = [True, True, True, True]
        self.return_features = return_features

        # dino fusion
        dino_fusion_cfg_default = dict(
            type="hAlgorithm.modules.models.fusion.mf_transformer.DinoFusionBlockTransformer",
            use_attn=False,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        if self.dino_fusion_cfg is not None:
            dino_fusion_cfg_default.update(self.dino_fusion_cfg)
        self.dino_fuser = instantiate_from_config(dino_fusion_cfg_default)

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

        # out path fusion
        path_fusion_cfg_default = dict(
            type="hAlgorithm.modules.models.fusion.mf_transformer.FeatureFusionBlockTransformer",
            head_features_1=head_features_1,
            use_attn=False,
        )
        if self.path_fusion_cfg is not None:
            path_fusion_cfg_default.update(self.path_fusion_cfg)
        self.path_fuser = instantiate_from_config(path_fusion_cfg_default)

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

    @torch.no_grad()
    def clear_hidden(self):
        print("Clearing all hidden states due to new clip.")
        self.dino_fuser.hidden = []
        self.path_fuser.hidden = []

    @torch.no_grad()
    def infer_flow(self, out_features, patch_h, patch_w, prompt_depth=None):
        """
        out_features: [[B,C,H,W]*4] * 1
        prompt_depth: B,1,C,H,W
        """
        assert len(out_features) == 1
        out_list = self.dino_fuser.infer_flow(out_features, patch_h, patch_w)
        path_list = [
            self.get_fusion_path_single_batch(out_list[-1], prompt_depth=prompt_depth[:, -1, ...])
        ]
        path_list = self.path_fuser.infer_flow(path_list)
        depth_out = [self.get_output_single_batch(path_list[-1], patch_h, patch_w)]
        return depth_out

    def forward(
        self,
        out_features,
        patch_h,
        patch_w,
        prompt_depth=None,
        return_dict=False,
        flow_forward=False,
        flow_clear=False,
    ):
        """
        out_features: [[B,C,H,W]*4] * T
        prompt_depth: B,T,C,H,W
        """
        if flow_clear:
            self.clear_hidden()
        if flow_forward:
            return self.infer_flow(out_features, patch_h, patch_w, prompt_depth)

        input_frames = len(out_features)

        out_list = self.dino_fuser(out_features, patch_h, patch_w)
        path_list = []
        if self.dino_fuser.only_last_frame:
            path = self.get_fusion_path_single_batch(
                out_list[-1], prompt_depth=prompt_depth[:, -1, ...]
            )
            path_list.append(path)
        else:
            for i, out in enumerate(out_list):
                path = self.get_fusion_path_single_batch(out, prompt_depth=prompt_depth[:, i, ...])
                path_list.append(path)

        path_list = self.path_fuser(path_list)

        if self.only_last_frame:
            path_list = [path_list[-1]]

        output_frames = len(path_list)
        depth_out = []
        # depth_out: [B,C,H,W] * T
        # fill in None
        for i in range(input_frames - output_frames):
            depth_out.append(None)

        for path in path_list:
            depth = self.get_output_single_batch(path, patch_h, patch_w)
            depth_out.append(depth)

        return depth_out

    def get_fusion_path_single_batch(self, out_features, prompt_depth=None):
        """
        out_features: [B,C,H,W] * 4
        """

        if self.prompt_pre_cfg is not None and prompt_depth is not None:
            prompt_depth = self.prompt_pre(prompt_depth)

        layer_1, layer_2, layer_3, layer_4 = out_features

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
        return path_1

    def get_output_single_batch(self, path_1, patch_h, patch_w):
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

        return return_dict
