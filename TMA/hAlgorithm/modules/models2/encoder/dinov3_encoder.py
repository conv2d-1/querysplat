import logging

import torch
import torch.nn as nn

from .dinov2_config import model_configs

class Dinov3Encoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        name="vitl",
        use_clstoken=False,
        pretrain=None,
        normalize=True,
        dinov3_custom_cfg=dict(),
        use_plus_version=False,
        input_patch_size=16,
        debug=False,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.use_clstoken = use_clstoken
        self.dinov3_custom_cfg = dinov3_custom_cfg
        self.use_plus_version = use_plus_version
        self.input_patch_size = input_patch_size

        self.name = name
        self.model_configs = model_configs[self.name]

        self.pretrain = pretrain
        self.debug = debug

        self.build_dino()

        # mean and std of the pretrained dinov3 model
        self.normalize = normalize
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def build_dino(self):
        model_name = "dinov3_{:}16".format(self.name) if not self.use_plus_version else "dinov3_{:}16plus".format(self.name)
        self.dinov3 = torch.hub.load(
            "hAlgorithm/modules/models2/external/romav2/models/dinov3",
            model=model_name,
            source="local",
            pretrained=False,
            **self.dinov3_custom_cfg,
        )

        if self.pretrain is not None:
            self.dinov3.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"Dinov3Encoder, load pretrain {self.pretrain}")

    def get_out_channels(self):
        return self.dinov3.blocks[0].attn.qkv.in_features

    def forward(self, x, meta_data=None, **kwargs):
        h, w = x.shape[-2:]
        if self.input_patch_size != self.patch_size:
            patch_h, patch_w = h // self.input_patch_size, w // self.input_patch_size
            tgt_h, tgt_w = int(patch_h*self.patch_size), int(patch_w*self.patch_size)
            x = torch.nn.functional.interpolate(x, size=(tgt_h, tgt_w), mode='bilinear', align_corners=True)

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        features = self.dinov3.get_intermediate_layers(x, n=self.model_configs["layer_idxs"], return_class_token=self.use_clstoken)
        return features

    def onnx_forward(self, x, **kwargs):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        #  N * [[B, patch_h*patch_w, C]]
        features = self.dinov3.get_intermediate_layers(x, n=self.model_configs["layer_idxs"], return_class_token=self.use_clstoken)
        return features
