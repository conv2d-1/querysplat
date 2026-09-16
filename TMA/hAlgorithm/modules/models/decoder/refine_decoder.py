import torch
import torch.nn as nn

from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers import PatchEmbed


class Mix(nn.Module):
    def __init__(
        self,
        embed_dim,
        prompt_in_chans,
        prompt_patch_size=None,
        prompt_embed_dim=None,
        fuse_embed_dim=None,
        fuse="add",
    ):
        super().__init__()

        self.fuse = fuse
        self.embed_dim = embed_dim

        self.prompt_patch_size = prompt_patch_size
        self.prompt_in_chans = prompt_in_chans
        self.prompt_embed_dim = prompt_embed_dim

        self.fuse_embed_dim = fuse_embed_dim

        if self.fuse == "add":
            self.prompt_embed_dim = self.embed_dim

        if self.prompt_in_chans > 0:
            self.prompt_patch_embed = PatchEmbed(
                img_size=224,
                patch_size=self.prompt_patch_size,
                in_chans=self.prompt_in_chans,
                embed_dim=self.prompt_embed_dim,
            )

    def forward(self, patch_tokens, prompt_features, meta_data, **kwargs):
        if self.prompt_in_chans > 0 and prompt_features is not None:
            prompt_tokens = self.prompt_patch_embed(prompt_features)

            if isinstance(patch_tokens, (list, tuple)):
                patch_tokens = list(patch_tokens)
                for idx in range(len(patch_tokens)):
                    if self.fuse == "add":
                        patch_tokens[idx] = patch_tokens[idx] + prompt_tokens
                    else:
                        patch_tokens[idx] = torch.cat([patch_tokens[idx], prompt_tokens], dim=-1)
            else:
                if self.fuse == "add":
                    patch_tokens = patch_tokens + prompt_tokens
                else:
                    patch_tokens = torch.cat([patch_tokens, prompt_tokens], dim=-1)

        return patch_tokens