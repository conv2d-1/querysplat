import torch
import torch.nn as nn

from hAlgorithm.utils import instantiate_from_config


class MdePromptEncoder(nn.Module):
    def __init__(
        self,
        prompt_encoder_cfg,
        fuse_type="cat",
        mde_encoder_cfg=None,
        fused_encoder_cfg=None,
        post_encoder_cfg=None,
    ):
        super().__init__()
        self.fuse_type = fuse_type
        self.mde_encoder = instantiate_from_config(mde_encoder_cfg)
        self.prompt_encoder = instantiate_from_config(prompt_encoder_cfg)
        self.fused_encoder = instantiate_from_config(fused_encoder_cfg)
        self.post_encoder = instantiate_from_config(post_encoder_cfg)

        assert self.prompt_encoder is not None

    def forward(self, x, **kwargs):
        prompt = self.prompt_encoder(x)

        if self.mde_encoder:
            mde = self.mde_encoder(x)
            if self.fuse_type in ["cat", "concat"]:
                output = torch.concat([prompt, mde], dim=1)
            elif self.fuse_type in ["add"]:
                output = prompt + mde
            else:
                raise NotImplementedError
        else:
            output = prompt

        if self.fused_encoder:
            output = self.fused_encoder(output)

        if self.post_encoder:
            output = self.post_encoder(output)

        return output
