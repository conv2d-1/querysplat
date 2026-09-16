import torch
from einops import einsum, rearrange
from jaxtyping import Float
from pytorch3d import transforms
from torch import Tensor, nn

from hAlgorithm.modules.utils.gaussians.projection import (
    get_world_rays,
    homogenize_points,
    sample_image_grid,
)
from hAlgorithm.modules.utils.gaussians.sh_rotation import rotate_sh
from hAlgorithm.modules.utils.gaussians.types import GaussiansV2 as Gaussians
from hAlgorithm.utils import instantiate_from_config


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class GaussianAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sh_degree = cfg["sh_degree"]
        self.normalize = cfg.get("normalize", False)
        self.offset = cfg.get("offset", False)
        self.fix_scale = cfg.get("fix_scale", None)
        self.use_prompt_scale = cfg.get("use_prompt_scale", False)
        self.use_glb_depth = cfg.get("use_glb_depth", False)

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "b v 1 1 1 4 4"],
        intrinsics: Float[Tensor, "b v 1 1 1 3 3"] | None,
        coordinates: Float[Tensor, "b v n 1 1 2"],
        depths: Float[Tensor, "b v n 1 c"] | None,
        opacities: Float[Tensor, "b v n 1 1"],
        raw_gaussians: Float[Tensor, "b v n 1 1 c"],
        input_images: Tensor | None = None,
        prompt_scale: Tensor | None = None,
        eps: float = 1e-8,
        **kwargs,
    ) -> Gaussians:
        # TODO: compute scales and rotations from pointmap, predict their residuals
        if self.offset:
            offset, mask, scales, rotations, sh = raw_gaussians.split(
                (3, 1, 3, 4, 3 * self.d_sh), dim=-1
            )
        else:
            offset = None
            mask, scales, rotations, sh = raw_gaussians.split((1, 3, 4, 3 * self.d_sh), dim=-1)
        # scales will be activated with exp, so subtract 5 to make it smaller
        scales = scales - 4
        # 看一下这个scale的分布情况。load模型
        # scales[..., -1] = scales[..., -1] * 0 - 6  # make the last dimension small
        mask = mask.sigmoid()[..., 0] > 0.5  # [b, v, n, 1, 1]

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)
        # https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2127778441

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # transform rotations to world-space
        # TODO: double check if this is correct
        c2w_rotations = extrinsics[..., :3, :3]
        if not torch.allclose(torch.det(c2w_rotations), c2w_rotations.new_tensor(1.0)):
            cw2_quat = transforms.matrix_to_quaternion(c2w_rotations)
            c2w_rotations = transforms.quaternion_to_matrix(cw2_quat)

        rotations = c2w_rotations @ transforms.quaternion_to_matrix(rotations)
        rotations = transforms.matrix_to_quaternion(rotations)

        # Compute Gaussian means.
        if self.use_glb_depth:
            means = depths
        elif depths.shape[-1] == 3:
            means = (extrinsics @ homogenize_points(depths).unsqueeze(-1))[..., :3, 0]
        else:
            origins, directions = get_world_rays(
                coordinates, extrinsics, intrinsics, with_normalize=False
            )
            means = origins + directions * depths

        norm_scales = None
        B = means.shape[0]
        if self.normalize:
            if self.use_prompt_scale:
                norm_scales = prompt_scale[:, 0]
            else:
                norm_scales = torch.quantile(
                    torch.linalg.norm(means.reshape(B, -1, 3), dim=-1), 0.9, dim=1
                )

            if self.fix_scale is not None:
                norm_scales = norm_scales / self.fix_scale

        gaussians = Gaussians(
            means=rearrange(
                means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            harmonics=rearrange(
                rotate_sh(sh, c2w_rotations[..., None, :, :]),
                "b v r srf spp c d_sh -> b (v r srf spp) d_sh c",
            ),
            opacities=rearrange(
                opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            scales=rearrange(
                scales,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rotations=rearrange(
                rotations.broadcast_to((*scales.shape[:-1], 4)),
                "b v r srf spp xyzq -> b (v r srf spp) xyzq",
            ),
            mask=rearrange(
                mask,
                "b v r srf spp -> b (v r srf spp)",
            ),
            norm_scales=norm_scales,
        )

        return gaussians

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        if self.offset:
            return 11 + 3 * self.d_sh
        else:
            return 8 + 3 * self.d_sh


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.conv1(x))))


class GaussianHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        cfg=None,
        img_encoder=None,
        pre_encoder=None,
        downsample=False,
        down_ratio=None,
        conv_down=None,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.downsample = downsample
        self.down_ratio = down_ratio
        self.conv_down = conv_down

        self.gaussian_adapter = GaussianAdapter(cfg)

        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        num_gaussian_parameters += 1  # predict opacity
        in_channels = 3 + feat_dim  # concat(img, features)

        self.img_encoder = instantiate_from_config(img_encoder)
        self.pre_encoder = instantiate_from_config(pre_encoder)

        if self.img_encoder is not None:
            in_channels = self.img_encoder.get_output_channel() + feat_dim  # concat(img, features)
        else:
            in_channels = 3 + feat_dim

        if self.pre_encoder is not None:
            in_channels = self.pre_encoder.get_output_channel()

        if self.conv_down is not None and self.conv_down > 0:
            self.donw = nn.Sequential(
                *[DownSampleBlock(feat_dim, feat_dim) for i in range(int(self.conv_down))]
            )

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

        self.xy_ray_cache = dict()
        self.pixel_size_cache = dict()

    def forward(self, images, features, depth, extrinsics, intrinsics, prompt_scale):
        """
        images: [B, F*V, C, H, W]
        features: [B, F*V, C, H, W]
        depth: [B, F*V, C, H, W]
        extrinsics: [B, F*V, 4, 4]
        intrinsics: [B, F*V, 3, 3]
        """

        b, v, _, h, w = images.shape
        images = images.view(b * v, -1, h, w)
        depth = depth.view(b * v, -1, h, w)

        if self.img_encoder is not None:
            images = self.img_encoder(images)

        if features.shape[-2:] != images.shape[-2:]:
            f_h, f_w = features.shape[-2:]
            features = features.view(b * v, -1, f_h, f_w)

            if (self.down_ratio is None and self.down_ratio != 1) and (
                self.conv_down is None or self.conv_down == 0
            ):
                images = nn.functional.interpolate(
                    images,
                    (f_h, f_w),
                    mode="bilinear",
                    align_corners=True,
                )
                depth = nn.functional.interpolate(
                    depth,
                    (f_h, f_w),
                    mode="bilinear",
                    align_corners=True,
                )
        else:
            features = features.view(b * v, -1, h, w)

        if self.downsample:
            images = images[..., ::2, ::2]
            features = features[..., ::2, ::2]
            depth = depth[..., ::2, ::2]

        elif self.down_ratio is not None and self.down_ratio != 1:
            new_h = int(h / self.down_ratio)
            new_w = int(w / self.down_ratio)

            images = nn.functional.interpolate(
                images,
                (new_h, new_w),
                mode="bilinear",
                align_corners=True,
            )
            features = nn.functional.interpolate(
                features,
                (new_h, new_w),
                mode="bilinear",
                align_corners=True,
            )
            depth = nn.functional.interpolate(
                depth,
                (new_h, new_w),
                mode="bilinear",
                align_corners=True,
            )

        elif self.conv_down is not None and self.conv_down > 0:
            features = self.donw(features)
            new_h, new_w = features.shape[-2:]

            images = nn.functional.interpolate(
                images,
                (new_h, new_w),
                mode="bilinear",
                align_corners=True,
            )
            depth = nn.functional.interpolate(
                depth,
                (new_h, new_w),
                mode="bilinear",
                align_corners=True,
            )

        features = torch.cat([features, images], dim=1)

        if self.pre_encoder is not None:
            features = self.pre_encoder(features)

        gaussians = self.head(features)  # [B*V, C, H, W]
        h, w = gaussians.shape[-2:]

        assert gaussians.shape[-2:] == depth.shape[-2:], f"{gaussians.shape}, {depth.shape}"

        # [B, V, C, H, W]
        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(gaussians, "b v c h w -> b v (h w) c")  # [B, V, H*W, 86]
        opacities = raw_gaussians[..., :1].unsqueeze(-1)  # [B, V, H*W, 1, 1]
        raw_gaussians = raw_gaussians[..., 1:]

        cache_key = f"xy_ray_{h}_{w}"
        if cache_key in self.xy_ray_cache and self.xy_ray_cache[cache_key].device == images.device:
            xy_ray = self.xy_ray_cache[cache_key]
        else:
            xy_ray, _ = sample_image_grid((h, w), images.device)
            self.xy_ray_cache[cache_key] = xy_ray

        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        # [B*num_depths, V, H*W, 1, 84]
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=1,  # num_surfaces
        )
        offset_xy = gaussians[..., :2].sigmoid()

        cache_key = f"pixel_size_{h}_{w}"
        if (
            cache_key in self.pixel_size_cache
            and self.pixel_size_cache[cache_key].device == images.device
        ):
            pixel_size = self.pixel_size_cache[cache_key]
        else:
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=images.device)
            self.pixel_size_cache[cache_key] = pixel_size
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        gaussians = self.gaussian_adapter(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            rearrange(depth, "(b v) c h w -> b v (h w) () () c", b=b, v=v),
            opacities,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            input_images=None,
            prompt_scale=prompt_scale[:, :, 0, 0, 0] if prompt_scale is not None else None,
        )

        return gaussians


class MSGaussianHead(nn.Module):
    def __init__(self, feat_dim, cfg, down_ratio=None, conv_down=None, hooks=None):
        super().__init__()

        self.cfg = cfg
        self.hooks = hooks

        assert down_ratio is not None or conv_down is not None

        if down_ratio is not None:
            self.heads = nn.ModuleList(
                [
                    GaussianHead(feat_dim=feat_dim, cfg=cfg, down_ratio=down_ratio[i])
                    for i in range(len(down_ratio))
                ]
            )
        else:
            self.heads = nn.ModuleList(
                [
                    GaussianHead(feat_dim=feat_dim, cfg=cfg, conv_down=conv_down[i])
                    for i in range(len(conv_down))
                ]
            )

        self.normalize = self.heads[0].gaussian_adapter.normalize
        self.fix_scale = self.heads[0].gaussian_adapter.fix_scale
        self.use_prompt_scale = self.heads[0].gaussian_adapter.use_prompt_scale

    def forward(self, images, features, depth, extrinsics, intrinsics, prompt_scale):
        """
        images: [B, F*V, C, H, W]
        features: [B, F*V, C, H, W]
        depth: [B, F*V, C, H, W]
        extrinsics: [B, F*V, 4, 4]
        intrinsics: [B, F*V, 3, 3]
        """

        bs = images.shape[0]

        ms_gaussians = []
        ms_means = []
        ms_scales = []
        ms_rotations = []
        ms_harmonics = []
        ms_opacities = []

        for head in self.heads:
            gaussians = head(
                images=images,
                features=features,
                depth=depth,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                prompt_scale=prompt_scale,
            )
            ms_gaussians.append(gaussians)

            if gaussians.norm_scales is not None:
                ms_means.append(
                    [gaussians.means[bi] * gaussians.norm_scales[bi] for bi in range(bs)]
                )
            else:
                ms_means.append(gaussians.means)

            ms_scales.append(gaussians.scales)
            ms_rotations.append(gaussians.rotations)
            ms_harmonics.append(gaussians.harmonics)
            ms_opacities.append(gaussians.opacities)

        ms_means = torch.stack(
            [
                torch.cat([ms_means[hi][bi] for hi in range(len(ms_means))], dim=0)
                for bi in range(bs)
            ],
            dim=0,
        )
        ms_scales = torch.stack(
            [
                torch.cat([ms_scales[hi][bi] for hi in range(len(ms_scales))], dim=0)
                for bi in range(bs)
            ],
            dim=0,
        )
        ms_rotations = torch.stack(
            [
                torch.cat([ms_rotations[hi][bi] for hi in range(len(ms_rotations))], dim=0)
                for bi in range(bs)
            ],
            dim=0,
        )
        ms_harmonics = torch.stack(
            [
                torch.cat([ms_harmonics[hi][bi] for hi in range(len(ms_harmonics))], dim=0)
                for bi in range(bs)
            ],
            dim=0,
        )
        ms_opacities = torch.stack(
            [
                torch.cat([ms_opacities[hi][bi] for hi in range(len(ms_opacities))], dim=0)
                for bi in range(bs)
            ],
            dim=0,
        )

        if self.normalize:
            if self.use_prompt_scale:
                norm_scales = prompt_scale[:, 0]
            else:
                norm_scales = torch.quantile(
                    torch.linalg.norm(ms_means.reshape(bs, -1, 3), dim=-1), 0.9, dim=1
                )

            if self.fix_scale is not None:
                norm_scales = norm_scales / self.fix_scale

        gaussians = Gaussians(
            means=ms_means,
            harmonics=ms_harmonics,
            opacities=ms_opacities,
            scales=ms_scales,
            rotations=ms_rotations,
            mask=None,
            norm_scales=norm_scales,
        )
        ms_gaussians.append(gaussians)

        if self.hooks is not None:
            return [ms_gaussians[i] for i in self.hooks]
        else:
            return ms_gaussians
