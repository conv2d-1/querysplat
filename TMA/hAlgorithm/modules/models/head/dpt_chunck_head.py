import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.decoder.blocks import _make_fusion_block, _make_scratch
from hAlgorithm.modules.models.vggt.heads.utils import create_uv_grid, position_grid_to_embed
from hAlgorithm.utils import instantiate_from_config

from .blocks import Exp, InverseLog, MogeActivation
from .dpt_head import DPTHead as BaseDPTHead


class DPTHead(BaseDPTHead):
    def __init__(self, chunk_size=4, **kwargs):
        super(DPTHead, self).__init__(**kwargs)

        self.chunk_size = chunk_size
       
    def _forward_impl(
        self,
        features,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        return_dict=False,
        meta_data=None,
        patch_start_idx=None,
        frames_start_idx=None,
        frames_end_idx=None,
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

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

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

        pointmap = self.output_conv2(out)
        return_dict = dict(pointmap=pointmap)

        if self.pred_confidence:
            confidence = self.confidence_conv(out)
            return pointmap, confidence
        else:
            return pointmap, None
    
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

        aggregated_tokens_list = features

        if aggregated_tokens_list[0].ndim == 4:
            B, S = aggregated_tokens_list[0].shape[0:2]
        H, W = meta_data["input_height"][0], meta_data["input_width"][0]

        if not self.training:
            preds_list = []
            conf_list = []
            return_features_list = []
            for frames_start_idx in range(0, S, self.chunk_size):
                frames_end_idx = min(frames_start_idx + self.chunk_size, S)

                pointmap, confidence = self._forward_impl(
                    aggregated_tokens_list,
                    patch_h,
                    patch_w,
                    return_dict=return_dict,
                    meta_data=meta_data,
                    frames_start_idx=frames_start_idx,
                    frames_end_idx=frames_end_idx,
                    patch_start_idx=patch_start_idx,
                )

                preds_list.append(pointmap)
                conf_list.append(confidence)

            return_dict = dict(
                pointmap=torch.cat(preds_list, dim=0),
                confidence=torch.cat(conf_list, dim=0),
            )
            return return_dict


