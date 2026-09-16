import torch
from einops import rearrange
from torch import nn

from hAlgorithm.modules.models.gaussiansv2.gaussian_adapter import GaussianAdapter
from hAlgorithm.modules.models.gaussiansv2.gaussian_adapter_voxelize import GaussianAdapterVoxelize
from hAlgorithm.modules.models.pi3.models.layers.pos_embed import RoPE2D
from hAlgorithm.modules.models.pi3.models.layers.transformer_head import (
    LinearPts3d,
    TransformerDecoder,
)
from hAlgorithm.modules.utils.gaussians.projection import sample_image_grid


class GaussianHead(nn.Module):
    def __init__(
        self,
        in_channels,
        depth=5,
        patch_size=14,
        embed_dim=1024,
        num_heads=16,
        pos_type="rope100",
        cfg=None,
        voxelization=False,
        rgb_features_channels=None,
        rgb_features_index=None,
        head_mlp_ratio=None,
    ):
        super(GaussianHead, self).__init__()

        self.patch_size = patch_size
        self.rgb_features_channels = rgb_features_channels
        self.rgb_features_index = rgb_features_index
        self.head_mlp_ratio = head_mlp_ratio

        if voxelization:
            self.gaussian_adapter = GaussianAdapterVoxelize(cfg)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg)

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        num_gaussian_parameters += 1  # predict mask
        if self.gaussian_adapter.offset:
            num_gaussian_parameters += 3  # prpedict xyz offset
        self.output_dim = num_gaussian_parameters

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):  # eg rope100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError

        # ----------------------
        #  Points Decoder
        # ----------------------
        self.decoder = TransformerDecoder(
            in_dim=in_channels,
            depth=depth,
            dec_embed_dim=embed_dim,
            dec_num_heads=num_heads,
            out_dim=embed_dim,
            rope=self.rope,
        )

        if self.rgb_features_channels is not None and self.rgb_features_channels > 0:
            embed_dim = embed_dim + self.rgb_features_channels

        self.head = LinearPts3d(
            patch_size=patch_size,
            dec_embed_dim=embed_dim,
            output_dim=self.output_dim,
            permute=False,
            mlp_ratio=self.head_mlp_ratio,
        )

    def forward(
        self,
        features,
        pos,
        patch_start_idx,
        depth,
        extrinsics,
        intrinsics,
        meta_data,
        rgb_features=None,
        **kwargs
    ):
        """
        images: [B, F*V, C, H, W]
        features: [B, F*V, C, H, W]
        depth: [B, F*V, 1, H, W]
        extrinsics: [B, F*V, 4, 4]
        intrinsics: [B, F*V, 3, 3]
        """
        patch_h = meta_data["patch_h"]
        patch_w = meta_data["patch_w"]

        if isinstance(features, (list, tuple)):
            features = features[-1]

        B, N, L, C = features.shape
        features = features.reshape(B * N, L, C)
        pos = pos.reshape(B * N, L, 2)

        hidden = self.decoder(features, xpos=pos)

        W = patch_w * self.patch_size
        H = patch_h * self.patch_size

        if self.rgb_features_channels is not None and self.rgb_features_channels > 0:
            if isinstance(rgb_features, (list, tuple)):
                if self.rgb_features_index is not None:
                    rgb_features = rgb_features[self.rgb_features_index]
                else:
                    rgb_features = rgb_features[-1]

            hidden = torch.cat([hidden[:, patch_start_idx:], rgb_features], dim=-1)

            gaussians = self.head([hidden], (H, W)).reshape(B, N, self.output_dim, H, W)
        else:
            gaussians = self.head([hidden[:, patch_start_idx:]], (H, W)).reshape(
                B, N, self.output_dim, H, W
            )

        # [B, V, C, H, W]
        # gaussians = rearrange(gaussians, "b v c h w -> b v c h w", b=B, v=N)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")  # [B, V, H*W, 86]
        opacities = raw_gaussians[..., :1].unsqueeze(-1)  # [B, V, H*W, 1, 1]
        raw_gaussians = raw_gaussians[..., 1:]

        xy_ray, _ = sample_image_grid((H, W), hidden.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")  # [H*W, 1, 2]
        # [B*num_depths, V, H*W, 1, 84]
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=1,  # num_surfaces
        )
        xy_ray = xy_ray[None, None].repeat(B, N, 1, 1, 1)  # [B, V, H*W, 1, 2]

        gaussians = self.gaussian_adapter(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            rearrange(depth, "b v 1 h w -> b v (h w) () ()", b=B, v=N),
            opacities,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (H, W),
            input_images=None,
        )

        return gaussians
