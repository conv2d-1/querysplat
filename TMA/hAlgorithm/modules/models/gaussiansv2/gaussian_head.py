import torch
from einops import rearrange
from torch import nn

from hAlgorithm.modules.models.gaussiansv2.gaussian_adapter import GaussianAdapter
from hAlgorithm.modules.models.gaussiansv2.gaussian_adapter_voxelize import GaussianAdapterVoxelize
from hAlgorithm.modules.utils.gaussians.projection import sample_image_grid

from hAlgorithm.utils import instantiate_from_config


class GaussianHead(nn.Module):
    def __init__(self, feat_dim, downsample=False, cfg=None, img_encoder=None):
        super().__init__()

        self.gaussian_adapter = GaussianAdapter(cfg)

        self.img_encoder = instantiate_from_config(img_encoder)
        if self.img_encoder is None:
            img_channels = 3
        else:
            img_channels = self.img_encoder.get_output_channel()

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        num_gaussian_parameters += 1  # predict mask
        in_channels = img_channels + feat_dim  # concat(img, features)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(
                num_gaussian_parameters * 2,
                num_gaussian_parameters,
                3,
                1,
                1,
                padding_mode="replicate",
            ),
        )
        self.downsample = downsample

    def forward(self, images, features, depth, extrinsics, intrinsics, **kwargs):
        """
        images: [B, F*V, C, H, W]
        features: [B, F*V, C, H, W]
        depth: [B, F*V, 1, H, W]
        extrinsics: [B, F*V, 4, 4]
        intrinsics: [B, F*V, 3, 3]
        """
        if self.downsample:
            images = images[..., ::2, ::2]
            features = features[..., ::2, ::2]
            depth = depth[..., ::2, ::2]

        b, v, _, h, w = images.shape

        features = features.view(b * v, -1, h, w)
        images = images.view(b * v, -1, h, w)

        if self.img_encoder is not None:
            images = self.img_encoder(images)

        features = torch.cat([features, images], dim=1)
        gaussians = self.head(features)  # [B*V, C, H, W]

        # [B, V, C, H, W]
        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")  # [B, V, H*W, 86]
        opacities = raw_gaussians[..., :1].unsqueeze(-1)  # [B, V, H*W, 1, 1]
        raw_gaussians = raw_gaussians[..., 1:]

        xy_ray, _ = sample_image_grid((h, w), images.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")  # [H*W, 1, 2]
        # [B*num_depths, V, H*W, 1, 84]
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=1,  # num_surfaces
        )
        xy_ray = xy_ray[None, None].repeat(b, v, 1, 1, 1)  # [B, V, H*W, 1, 2]

        gaussians = self.gaussian_adapter(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            rearrange(depth, "b v 1 h w -> b v (h w) () ()", b=b, v=v),
            opacities,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
            input_images=None,
        )

        return gaussians


class GaussianHeadNoRGB(nn.Module):
    def __init__(self, feat_dim, downsample=False, cfg=None):
        super().__init__()

        self.gaussian_adapter = GaussianAdapter(cfg)

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        num_gaussian_parameters += 1  # predict mask
        num_gaussian_parameters += 3  # prpedict xyz offset
        in_channels = feat_dim  # no rgb input
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(num_gaussian_parameters * 2, num_gaussian_parameters, 1, 1, 0, bias=False),
        )
        self.downsample = downsample

    def forward(self, images, features, depth, extrinsics, intrinsics, **kwargs):
        """
        images: [B, F*V, C, H, W]
        features: [B, F*V, C, H, W]
        depth: [B, F*V, 1, H, W]
        extrinsics: [B, F*V, 4, 4]
        intrinsics: [B, F*V, 3, 3]
        """
        if self.downsample:
            images = images[..., ::2, ::2]
            features = features[..., ::2, ::2]
            depth = depth[..., ::2, ::2]

        b, v, _, h, w = images.shape

        features = features.view(b * v, -1, h, w)
        gaussians = self.head(features)  # [B*V, C, H, W]

        # [B, V, C, H, W]
        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")  # [B, V, H*W, 86]
        opacities = raw_gaussians[..., :1].unsqueeze(-1)  # [B, V, H*W, 1, 1]
        raw_gaussians = raw_gaussians[..., 1:]

        xy_ray, _ = sample_image_grid((h, w), images.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        # [B*num_depths, V, H*W, 1, 84]
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=1,  # num_surfaces
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=images.device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        gaussians = self.gaussian_adapter(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            rearrange(depth, "b v c h w -> b v (h w) () c"),
            opacities,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
            input_images=None,
        )

        return gaussians


class GaussianHeadRGBConvK1(GaussianHead):
    def __init__(self, feat_dim, downsample=False, cfg=None, voxelization=False, img_encoder=None):
        super(GaussianHeadRGBConvK1, self).__init__(feat_dim, downsample, cfg, img_encoder=img_encoder)
        if voxelization:
            self.gaussian_adapter = GaussianAdapterVoxelize(cfg)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg)
        
        self.img_encoder = instantiate_from_config(img_encoder)
        if self.img_encoder is None:
            img_channels = 3
        else:
            img_channels = self.img_encoder.get_output_channel()

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        num_gaussian_parameters += 1  # predict mask
        if self.gaussian_adapter.offset:
            num_gaussian_parameters += 3  # prpedict xyz offset
        in_channels = feat_dim + img_channels  # rgb input
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(num_gaussian_parameters * 2, num_gaussian_parameters, 1, 1, 0, bias=True),
        )
        self.downsample = downsample
