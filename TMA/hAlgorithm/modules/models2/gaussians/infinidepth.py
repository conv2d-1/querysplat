"""Pixel-aligned Gaussian splatting head.

Adapted from InfiniDepth's GSPixelAlignPredictor. Takes dense backbone features,
a depth map, and an RGB image to predict per-pixel 3D Gaussian parameters for
Gaussian splatting reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _homogenize_points(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def _homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def _transform_cam2world(homogeneous: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    # extrinsics: [B, 4, 4], homogeneous: [B, N, 4]
    # Need extrinsics [B, 1, 4, 4] to broadcast with homogeneous [B, N, 4, 1]
    return torch.matmul(extrinsics.unsqueeze(1), homogeneous.unsqueeze(-1)).squeeze(-1)


def _unproject(coordinates_xy: torch.Tensor, z: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    coordinates_h = _homogenize_points(coordinates_xy)
    intr_inv = torch.linalg.inv(intrinsics)
    rays = torch.matmul(intr_inv.unsqueeze(1), coordinates_h.unsqueeze(-1)).squeeze(-1)
    return rays * z.unsqueeze(-1)


def _get_world_rays(coordinates_xy, extrinsics, intrinsics):
    ones = torch.ones_like(coordinates_xy[..., 0])
    directions_cam = _unproject(coordinates_xy, ones, intrinsics)
    directions_cam = directions_cam / torch.clamp(directions_cam[..., 2:], min=1e-6)
    directions_world = _transform_cam2world(_homogenize_vectors(directions_cam), extrinsics)[..., :3]
    origins_world = extrinsics[:, None, :3, 3].expand_as(directions_world)
    return origins_world, directions_world


def _sample_image_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


class GaussianAdapter(nn.Module):
    """Converts raw per-pixel network outputs into world-space Gaussian parameters.

    Applies activation functions (softplus for scales, normalize for quaternions),
    degree-wise SH masking, DC anchoring to image color, and camera-based unprojection
    to produce valid 3D Gaussians.
    """

    def __init__(self, gaussian_scale_min=1e-10, gaussian_scale_max=5.0, sh_degree=2):
        super().__init__()
        self.gaussian_scale_min = gaussian_scale_min
        self.gaussian_scale_max = gaussian_scale_max
        self.sh_degree = sh_degree

        self.register_buffer(
            "sh_mask",
            torch.ones(((sh_degree + 1) ** 2,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, sh_degree + 1):
            self.sh_mask[degree ** 2: (degree + 1) ** 2] = 0.1 * (0.25 ** degree)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh

    def forward(self, image, extrinsics, intrinsics, coordinates_xy, depths, opacities, raw_gaussians):
        """
        Args:
            image: [B, 3, H, W] RGB in [0, 1].
            extrinsics: [B, 4, 4] camera-to-world.
            intrinsics: [B, 3, 3] pixel intrinsics.
            coordinates_xy: [B, N, 2] pixel-space (x, y).
            depths: [B, N] per-point depth.
            opacities: [B, N] per-point opacity.
            raw_gaussians: [B, N, 7 + 3*d_sh] raw scale/rotation/SH.
        Returns:
            Dict with means, harmonics, opacities, scales, rotations.
        """
        b, _, h, w = image.shape
        scales_raw, rotations_raw, sh_raw = torch.split(
            raw_gaussians, [3, 4, 3 * self.d_sh], dim=-1
        )

        scales = torch.clamp(
            F.softplus(scales_raw - 4.0),
            min=self.gaussian_scale_min,
            max=self.gaussian_scale_max,
        )
        rotations = rotations_raw / (torch.norm(rotations_raw, dim=-1, keepdim=True) + 1e-8)

        harmonics = sh_raw.view(b, -1, 3, self.d_sh) * self.sh_mask.view(1, 1, 1, -1)

        image_flat = image.permute(0, 2, 3, 1).reshape(b, h * w, 3)
        harmonics[..., 0] = harmonics[..., 0] + _rgb_to_sh(image_flat)
        origins, directions = _get_world_rays(coordinates_xy, extrinsics, intrinsics)
        means = origins + directions * depths.unsqueeze(-1)

        return dict(
            means=means,
            harmonics=harmonics,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )


class GaussianHead(nn.Module):
    """Pixel-aligned Gaussian head for 3D Gaussian splatting.

    Fuses dense backbone features (ViT patch tokens), depth map, and RGB image
    through lightweight CNN encoders to regress per-pixel Gaussian parameters,
    then lifts them to world space via camera geometry.

    Compatible with InfiniDepth's GSPixelAlignPredictor checkpoint format.
    """

    def __init__(
        self,
        backbone_feature_dim=1024,
        patch_size=14,
        rgb_feature_dim=64,
        depth_feature_dim=32,
        backbone_reduced_dim=128,
        regressor_channels=64,
        gaussian_scale_min=1e-10,
        gaussian_scale_max=5.0,
        sh_degree=2,
        feature_layer_idx=-1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.feature_layer_idx = feature_layer_idx

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, rgb_feature_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, depth_feature_dim, 3, 1, 1),
            nn.GELU(),
        )
        self.dino_projector = nn.Sequential(
            nn.Conv2d(backbone_feature_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, backbone_reduced_dim, 1),
        )

        reg_in = rgb_feature_dim + depth_feature_dim + backbone_reduced_dim
        self.gaussian_regressor = nn.Sequential(
            nn.Conv2d(reg_in, regressor_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(regressor_channels, regressor_channels, 3, 1, 1),
        )

        self.gaussian_adapter = GaussianAdapter(
            gaussian_scale_min=gaussian_scale_min,
            gaussian_scale_max=gaussian_scale_max,
            sh_degree=sh_degree,
        )

        num_gaussian_params = self.gaussian_adapter.d_in + 2 + 1
        head_in = regressor_channels + rgb_feature_dim + backbone_reduced_dim
        self.gaussian_head = nn.Sequential(
            nn.Conv2d(head_in, num_gaussian_params, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(num_gaussian_params, num_gaussian_params, 3, 1, 1, padding_mode="replicate"),
        )

    @torch.no_grad()
    def load_from_infinidepth_checkpoint(self, checkpoint_path: str) -> None:
        """Load weights from an InfiniDepth GS checkpoint (encoder.* prefix)."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        own_sd = self.state_dict()
        load_sd = {}
        for k in own_sd:
            prefixed = f"encoder.{k}"
            if prefixed in state_dict and state_dict[prefixed].shape == own_sd[k].shape:
                load_sd[k] = state_dict[prefixed]
        self.load_state_dict(load_sd, strict=False)

    def _prepare_inputs(self, rgb, depth, depth_height, depth_width, intrinsics, w2c, c2w, scale):
        """Preprocess raw pipeline data into the format needed by the CNN backbone.

        Handles: depth reshape from query format, scale denormalization,
        RGB range conversion, camera param flattening and w2c->c2w inversion.

        Args:
            depth: [B, N, Q, D] query decoder depth (D>=1, first channel used).
            rgb: [B, N, C, H, W] or [B, C, H, W] input image in [-1, 1].
            depth_height: int, height of the query grid.
            depth_width: int, width of the query grid.
            intrinsics: [B, N, 3, 3] or [B*N, 3, 3] or None.
            w2c: [B, N, 4, 4] or [B*N, 4, 4] or None — world-to-camera.
            c2w: [B, N, 4, 4] or [B*N, 4, 4] or None — camera-to-world.
            scale: Depth scale factor (pipeline normalization) or None.

        Returns:
            depth_map [BN, 1, H, W], image_01 [BN, 3, H, W],
            gs_intrinsics [BN, 3, 3] or None, gs_c2w [BN, 4, 4] or None, bn int.
        """
        if rgb.ndim == 5:
            b, n, c, h, w = rgb.shape
            image_01 = (rgb.view(b * n, c, h, w) + 1.0) * 0.5
        else:
            b, c, h, w = rgb.shape
            n = 1
            image_01 = (rgb + 1.0) * 0.5
        bn = b * n

        # pred_depth [B, N, Q, D] -> [BN, 1, qH, qW]
        depth_map = depth.reshape(bn, depth_height, depth_width, -1)
        depth_map = depth_map[..., :1].permute(0, 3, 1, 2).contiguous()
        if depth_map.shape[-2:] != (h, w):
            depth_map = F.interpolate(depth_map, size=(h, w), mode="bilinear", align_corners=False)

        if scale is not None:
            depth_map = depth_map * scale.reshape(bn, *([1] * (depth_map.ndim - 1)))

        # Camera intrinsics
        gs_intrinsics = None
        if intrinsics is not None:
            gs_intrinsics = intrinsics.reshape(bn, 3, 3) if intrinsics.numel() > 0 else None

        # Camera-to-world extrinsics
        gs_c2w = None
        if c2w is not None:
            gs_c2w = c2w.reshape(bn, 4, 4)
        elif w2c is not None:
            gs_c2w = torch.linalg.inv(w2c.reshape(bn, 4, 4))

        return depth_map, image_01, gs_intrinsics, gs_c2w, bn

    def _extract_backbone_features(self, patch_features, patch_start_idx, bn, h_img, w_img):
        """Extract and reshape spatial backbone tokens to a feature map.

        Returns:
            backbone_feat: [BN, backbone_reduced_dim, H, W] upsampled feature map.
        """
        if isinstance(patch_features, (list, tuple)):
            backbone_tokens = patch_features[self.feature_layer_idx]
        else:
            backbone_tokens = patch_features

        if backbone_tokens.ndim == 4:
            spatial_tokens = backbone_tokens[:, :, patch_start_idx:, :]
            bs, s, n_spatial, c = spatial_tokens.shape
            spatial_tokens = spatial_tokens.reshape(bs * s, n_spatial, c)
        elif backbone_tokens.ndim == 3:
            c = backbone_tokens.shape[-1]
            spatial_tokens = backbone_tokens[:, patch_start_idx:]
        else:
            raise ValueError(f"Unexpected backbone_tokens shape: {backbone_tokens.shape}")

        patch_h = h_img // self.patch_size
        patch_w = w_img // self.patch_size

        backbone_map = spatial_tokens.reshape(bn, patch_h, patch_w, c).permute(0, 3, 1, 2)
        backbone_feat = self.dino_projector(backbone_map)
        return F.interpolate(backbone_feat, size=(h_img, w_img), mode="bilinear", align_corners=False)

    def forward(
        self,
        patch_features,
        patch_start_idx,
        rgb,
        depth,
        depth_width,
        depth_height,
        intrinsics=None,
        w2c=None,
        c2w=None,
        scale=None,
        meta_data=None,
    ):
        depth_map, image_01, gs_intrinsics, gs_c2w, bn = self._prepare_inputs(
            rgb=rgb,
            depth=depth.detach(),
            depth_height=depth_height,
            depth_width=depth_width,
            intrinsics=intrinsics,
            w2c=w2c,
            c2w=c2w,
            scale=scale,
        )
        h_img, w_img = image_01.shape[2], image_01.shape[3]

        backbone_feat = self._extract_backbone_features(
            patch_features, patch_start_idx, bn, h_img, w_img,
        )

        rgb_feat = self.rgb_encoder(image_01)
        depth_feat = self.depth_encoder(depth_map)

        reg_input = torch.cat([rgb_feat, depth_feat, backbone_feat], dim=1)
        reg_feat = self.gaussian_regressor(reg_input)
        head_input = torch.cat([reg_feat, rgb_feat, backbone_feat], dim=1)
        raw = self.gaussian_head(head_input)

        raw = raw.permute(0, 2, 3, 1).reshape(bn, h_img * w_img, -1)
        opacities = torch.sigmoid(raw[..., :1]).squeeze(-1)
        gaussian_core = raw[..., 1:]

        offset_xy = torch.sigmoid(gaussian_core[..., :2])
        raw_gaussians = gaussian_core[..., 2:]

        base = _sample_image_grid(h_img, w_img, image_01.device).unsqueeze(0).expand(bn, -1, -1)
        coords = base + (offset_xy - 0.5)

        depths = depth_map[:, 0].reshape(bn, -1)

        if gs_c2w is None:
            gs_c2w = torch.eye(4, device=image_01.device, dtype=image_01.dtype).unsqueeze(0).expand(bn, -1, -1)

        gaussians = self.gaussian_adapter(
            image=image_01,
            extrinsics=gs_c2w,
            intrinsics=gs_intrinsics,
            coordinates_xy=coords,
            depths=depths,
            opacities=opacities,
            raw_gaussians=raw_gaussians,
        )

        return {"gaussians": gaussians}
