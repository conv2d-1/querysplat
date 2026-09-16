import torch
import torch.nn as nn
import torch.nn.functional as F


class FuseEncoder(nn.Module):
    def __init__(
        self,
        enc_embed_dim,
        use_cls_token=False,
        with_layer_norm=False,
        fuse_type="add",
        fuse_depth=False,
        fuse_ray=False,
        fuse_extra=False,
        prompt_depth_in_channel=0,
        prompt_depth_patch_size=7,
        prompt_ray_in_channel=0,
        prompt_ray_patch_size=14,
    ):
        super().__init__()

        self.fuse_type = fuse_type
        assert self.fuse_type in ["add"], f"Supported fuse type is [add], but got {self.fuse_type}"
        self.fuse_depth = fuse_depth
        self.fuse_ray = fuse_ray
        self.fuse_extra = fuse_extra

        self.use_cls_token = use_cls_token

        self.with_layer_norm = with_layer_norm
        if self.with_layer_norm:
            self.layer_norm = nn.LayerNorm(enc_embed_dim, eps=1e-6)

        # prompt depth patch embed
        self.prompt_depth_in_channel = prompt_depth_in_channel
        self.prompt_depth_patch_size = prompt_depth_patch_size
        if self.prompt_depth_in_channel > 0:
            self.prompt_depth_proj = nn.Conv2d(
                self.prompt_depth_in_channel,
                enc_embed_dim,
                kernel_size=self.prompt_depth_patch_size,
                stride=self.prompt_depth_patch_size,
            )

        # prompt ray patch embed
        self.prompt_ray_in_channel = prompt_ray_in_channel
        self.prompt_ray_patch_size = prompt_ray_patch_size
        if self.prompt_ray_in_channel > 0:
            self.prompt_ray_proj = nn.Conv2d(
                self.prompt_ray_in_channel,
                enc_embed_dim,
                kernel_size=self.prompt_ray_patch_size,
                stride=self.prompt_ray_patch_size,
            )

    def forward(
        self,
        patch_features,
        prompt_depth=None,
        prompt_ray=None,
        prompt_extra=None,
        prompt_scale=None,
        intrinsics=None,
        w2c=None,
        meta_data=None,
        **kwargs,
    ):
        if not isinstance(patch_features, (list, tuple)):
            patch_features = [patch_features]
        num_features = len(patch_features)
        cls_features = []
        patch_features_out = []
        for i, x in enumerate(patch_features):
            if self.use_cls_token:
                patch_features_out.append(x[0])
                cls_features.append(x[1])
            elif isinstance(x, (list, tuple)):
                patch_features_out.append(x[0])
            else:
                patch_features_out.append(x)
        if self.use_cls_token:
            cls_features = torch.cat(cls_features, dim=0).unsqueeze(-2)  # B*N,1,C

        patch_features = torch.cat(patch_features_out, dim=0)  # B*N,L,C
        patch_h, patch_w = meta_data["patch_h"], meta_data["patch_w"]

        fused_tokens = []
        if prompt_depth is not None and self.fuse_depth:
            # h,w  to ph,pw, patch embed
            tgt_h = int(patch_h * self.prompt_depth_patch_size)
            tgt_w = int(patch_w * self.prompt_depth_patch_size)
            prompt_depth = F.interpolate(
                prompt_depth, size=(tgt_h, tgt_w), mode="bilinear", align_corners=True
            )
            prompt_depth = self.prompt_depth_proj(prompt_depth)
            prompt_depth = prompt_depth.flatten(2).permute(0, 2, 1)  # B,C,H,W -> B,L,C
            assert prompt_depth.shape[-2:] == patch_features.shape[-2:]
            fused_tokens.append(prompt_depth)

        if prompt_ray is not None and self.fuse_ray:
            if self.prompt_ray_in_channel > 0:
                # h,w  to ph,pw, patch embed
                tgt_h = int(patch_h * self.prompt_ray_patch_size)
                tgt_w = int(patch_w * self.prompt_ray_patch_size)
                prompt_ray = F.interpolate(
                    prompt_ray, size=(tgt_h, tgt_w), mode="bilinear", align_corners=True
                )
                prompt_ray = self.prompt_ray_proj(prompt_ray)  # B,C,H,W -> B,L,C
                prompt_ray_proj = prompt_ray_proj.flatten(2).permute(0, 2, 1)
                assert prompt_ray.shape[-2:] == patch_features.shape[-2:]
            else:
                prompt_ray = prompt_ray.unsqueeze(-2)  # B,C -> B,1,C
            fused_tokens.append(prompt_ray)

        if prompt_extra is not None and self.fuse_extra:  # None Pixel-Align Input (eg: pose, scale)
            prompt_extra = prompt_extra.unsqueeze(-2)  # B,C -> B,1,C
            fused_tokens.append(prompt_extra)

        if self.fuse_type == "add":
            for token in fused_tokens:
                patch_features = patch_features + token.repeat(
                    num_features, 1, 1
                )  # B,L,C -> B*N,L,C
                if self.use_cls_token and token.shape[1] == 1:
                    cls_features = cls_features + token.repeat(num_features, 1, 1)
        else:
            raise NotImplementedError(f"Supported fuse type is [add], but got {self.fuse_type}")

        if self.with_layer_norm:
            patch_features = self.layer_norm(patch_features)
            if self.use_cls_token:
                cls_features = self.layer_norm(cls_features)

        patch_features = patch_features.chunk(num_features, dim=0)
        if self.use_cls_token:
            cls_features = cls_features.squeeze(-2).chunk(num_features, dim=0)
            patch_features = list(zip(patch_features, cls_features))
        return patch_features
