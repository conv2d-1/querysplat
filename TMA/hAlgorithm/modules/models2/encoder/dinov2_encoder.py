import logging

import torch
import torch.nn as nn

from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa

from .dinov2_config import model_configs


class Dinov2Encoder(nn.Module):
    def __init__(
        self,
        patch_size=14,
        name="vitl",
        use_clstoken=False,
        dinov2_attention_with_sdpa=True,
        pretrain=None,
        normalize=True,
        with_register=False,
        dinov2_custom_cfg=dict(),
        layer_idxs=None,
        debug=False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.use_clstoken = use_clstoken
        self.dinov2_custom_cfg = dinov2_custom_cfg

        self.name = name
        self.model_configs = model_configs[self.name]

        self.dinov2_attention_with_sdpa = dinov2_attention_with_sdpa
        self.pretrain = pretrain
        self.with_register = with_register
        self.layer_idxs = layer_idxs
        self.debug = debug

        self.build_dino()

        # mean and std of the pretrained dinov2 model
        self.normalize = normalize
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def enable_pytorch_native_sdpa(self):
        """
        Enables PyTorch's native scaled dot product attention (SDPA) for the backbone's attention layers.
        """
        for block in self.dinov2.blocks:
            block.attn = wrap_dinov2_attention_with_sdpa(block.attn)

    def build_dino(self):
        model_name = "dinov2_{:}14".format(self.name) if not self.with_register else "dinov2_{:}14_reg".format(self.name)
        self.dinov2 = torch.hub.load(
            "hAlgorithm/modules/models/facebookresearch_dinov2_main",
            model_name,
            source="local",
            pretrained=False,
            **self.dinov2_custom_cfg,
        )

        if self.dinov2_attention_with_sdpa:
            self.enable_pytorch_native_sdpa()

        if self.pretrain is not None:
            self.dinov2.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"Dinov2Encoder, load pretrain {self.pretrain}")

    def get_out_channels(self):
        return self.dinov2.blocks[0].attn.qkv.in_features

    def forward(self, x, meta_data=None, **kwargs):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        meta_data["patch_h"] = patch_h
        meta_data["patch_w"] = patch_w

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        if self.layer_idxs is not None:
            features = self.dinov2.get_intermediate_layers(x, self.layer_idxs, return_class_token=self.use_clstoken)
        else:
            features = self.dinov2.get_intermediate_layers(x, self.model_configs["layer_idxs"], return_class_token=self.use_clstoken)

        return features

    def onnx_forward(self, x, **kwargs):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        #  N * [[B, patch_h*patch_w, C]]
        if self.layer_idxs is not None:
            features = self.dinov2.get_intermediate_layers(x, self.layer_idxs, return_class_token=self.use_clstoken)
        else:
            features = self.dinov2.get_intermediate_layers(x, self.model_configs["layer_idxs"], return_class_token=self.use_clstoken)

        return features