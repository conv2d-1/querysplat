import torch
import torch.nn as nn

from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa

from .config import model_configs


class PromptDinov2Encoder(nn.Module):
    def __init__(
        self,
        patch_size=14,
        name="vitl",
        use_clstoken=False,
        out_channels=None,
        dinov2_attention_with_sdpa=True,
        pretrain=None,
        normalize=True,
        dinov2_custom_cfg=None,
        layer_idxs=None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.use_clstoken = use_clstoken

        self.name = name
        self.model_configs = model_configs[self.name]
        self.out_channels = (
            out_channels if out_channels is not None else self.model_configs["out_channels"]
        )

        self.dinov2_attention_with_sdpa = dinov2_attention_with_sdpa
        self.pretrain = pretrain
        self.dinov2_custom_cfg = dinov2_custom_cfg
        self.layer_idxs = layer_idxs

        self.build_dino()
        self.build_act_postprocess()

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
        if self.name == "vit":
            self.dinov2 = torch.hub.load(
                "hAlgorithm/modules/models/facebookresearch_dinov2_main",
                "dinov2_{:}14".format(self.name),
                source="local",
                pretrained=False,
                **self.dinov2_custom_cfg,
            )
        else:
            self.dinov2 = torch.hub.load(
                "hAlgorithm/modules/models/facebookresearch_dinov2_main",
                "dinov2_{:}14".format(self.name),
                source="local",
                pretrained=False,
            )

        if self.dinov2_attention_with_sdpa:
            self.enable_pytorch_native_sdpa()

        if self.pretrain is not None:
            self.dinov2.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
        self.dino_output_dim = self.dinov2.blocks[0].attn.qkv.in_features

    def build_act_postprocess(self):
        in_channels = self.dino_output_dim
        out_channels = self.out_channels

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

        if self.use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

    def get_out_channels(self):
        return self.out_channels

    def forward(self, x, meta_data):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        meta_data["patch_h"] = patch_h
        meta_data["patch_w"] = patch_w

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        #  N * [[B, patch_h*patch_w, C]]
        if self.layer_idxs is None:
            features = self.dinov2.get_intermediate_layers(
                x, self.model_configs["layer_idxs"], return_class_token=self.use_clstoken
            )
        else:
            features = self.dinov2.get_intermediate_layers(
                x, self.layer_idxs, return_class_token=self.use_clstoken
            )

        # act_postprocess
        out = []
        for i, x in enumerate(features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        return out

    def onnx_forward(self, x, **kwargs):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        if self.normalize:
            x = ((x + 1) * 0.5 - self._mean) / self._std

        #  N * [[B, patch_h*patch_w, C]]
        if self.layer_idxs is None:
            features = self.dinov2.get_intermediate_layers(
                x, self.model_configs["layer_idxs"], return_class_token=self.use_clstoken
            )
        else:
            features = self.dinov2.get_intermediate_layers(
                x, self.layer_idxs, return_class_token=self.use_clstoken
            )

        # act_postprocess
        out = []
        for i, x in enumerate(features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        return out
