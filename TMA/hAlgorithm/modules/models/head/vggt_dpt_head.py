# import torch.nn as nn
# import torch.nn.functional as F
import logging

import torch

# from .blocks import Exp, InverseLog
from hAlgorithm.modules.models.vggt.heads.dpt_head import DPTHead, activate_head, custom_interpolate

# from hAlgorithm.modules.models.decoder.blocks import _make_fusion_block, _make_scratch
# from hAlgorithm.modules.models.vggt.heads.utils import create_uv_grid, position_grid_to_embed
# from hAlgorithm.utils import instantiate_from_config


class VGGTDPTHead(DPTHead):
    def __init__(
        self,
        patch_size=14,
        pretrain=None,
        return_features=False,
        return_refine_features=False,
        **kwargs,
    ):
        super(VGGTDPTHead, self).__init__(**kwargs)

        self.patch_size = patch_size
        self.pretrain = pretrain
        self.return_features = return_features
        self.return_refine_features = return_refine_features

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"VGGTDPTHead, load pretrain {self.pretrain}")

    def scratch_forward_return_refine_features(self, features):
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        refine_features = []
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        refine_features.append(out)

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        refine_features.append(out)

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        refine_features.append(out)

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        refine_features.append(out)

        out = self.scratch.output_conv1(out)
        refine_features.append(out)

        return out, refine_features

    def forward(
        self,
        features,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        patch_start_idx=None,
        **kwargs,
    ):
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if patch_h is None or patch_w is None:
            patch_w = meta_data["input_width"][0] // self.patch_size
            patch_h = meta_data["input_height"][0] // self.patch_size

        aggregated_tokens_list = features

        if aggregated_tokens_list[0].ndim == 4:
            B, S = aggregated_tokens_list[0].shape[0:2]
            BS = B * S
        else:
            BS = aggregated_tokens_list[0].shape[0]
        H, W = meta_data["input_height"][0], meta_data["input_width"][0]

        # B, S, _, H, W = images.shape
        # patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx]

            if patch_start_idx is not None:
                if x.ndim == 4:
                    x = x[:, :, patch_start_idx:]
                elif x.ndim == 3:
                    x = x[:, patch_start_idx:]

            # x = x.view(BS, -1, x.shape[-1])
            if x.ndim == 4:
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)

            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        if self.return_refine_features:
            out, refine_features = self.scratch_forward_return_refine_features(out)
        else:
            out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (
                int(patch_h * self.patch_size / self.down_ratio),
                int(patch_w * self.patch_size / self.down_ratio),
            ),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(BS, *out.shape[1:])

        if self.return_features:
            return_features = out
        else:
            return_features = None

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(
            out, activation=self.activation, conf_activation=self.conf_activation
        )

        preds = preds.view(BS, *preds.shape[1:])
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1).unsqueeze(1)
        else:
            preds = preds.permute(0, 3, 1, 2)

        conf = conf.view(BS, *conf.shape[1:]).unsqueeze(1)

        return_dict = dict(pointmap=preds, confidence=conf, features=return_features)

        if self.return_refine_features:
            return_dict["refine_features"] = refine_features

        return return_dict
