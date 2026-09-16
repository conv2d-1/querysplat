import torch
import torch.nn as nn

import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_camera_frame_fisheye, debug_visualize_rays



class RayEncoder(nn.Module):
    def __init__(self, model, debug=False, **kwargs):
        super(RayEncoder, self).__init__()
        self.model = instantiate_from_config(model)
        self.debug = debug

    def forward(self, intrinsics, ray_directions=None, w2c=None, rgb_mask=None, meta_data=None, **kwargs):
        if ray_directions is None:
            w = meta_data["input_width"][0].item()
            h = meta_data["input_height"][0].item()

            camera_type = meta_data.get("camera_type", ["PINHOLE"])[0]
            if camera_type == "PINHOLE":
                ray_directions = get_rays_in_camera_frame(
                    intrinsics=intrinsics,
                    height=h, width=w,
                    normalize_to_unit_sphere=True,
                )
            elif camera_type == "FISHEYE_EQUIDISTANT":
                ray_directions = get_rays_in_camera_frame_fisheye(
                    intrinsics=intrinsics,
                    height=h, width=w,
                    fisheye_model="equidistant",
                )
            elif camera_type == "FISHEYE_OPENCV":
                # ray_directions = get_rays_in_camera_frame_fisheye(
                #     intrinsics=intrinsics,
                #     height=h, width=w,
                #     fisheye_model="opencv_fisheye",
                #     distortion_coeffs=distortion_coeffs,
                # )
                # TODO: add opencv fisheye ray directions
                pass
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")
        
        if rgb_mask is not None:
            ray_directions = ray_directions * rgb_mask.float()

        if self.debug:
            debug_visualize_rays(ray_directions, rgb_mask=rgb_mask, save_path=f"{camera_type}_ray_directions.png")
            breakpoint()

        ray_features = self.model(ray_directions)
        return ray_features


class PluckerRayEncoder(nn.Module):
    """
    Encodes camera intrinsics and extrinsics into pixel-aligned Plücker embeddings.
    MoVieS Strategy 1: "Plücker embedding provides a dense and spatially aligned encoding"

    Output: [B*N, embed_dim, H_feat, W_feat] matching the patch_feature shape.
    """
    def __init__(self, embed_dim=1024, patch_size=14, use_harmonic_embed=True, n_harmonic_freqs=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_harmonic_embed = use_harmonic_embed

        # Raw Plucker coordinates are 6D: 3D direction (d) + 3D moment (m)
        input_dim = 6
        
        if use_harmonic_embed:
            # [sin(x), cos(x)] for each freq * input_dim + original input
            self.mlp_in_dim = input_dim + (2 * n_harmonic_freqs * input_dim)
            self.freqs = nn.Parameter(2.0 ** torch.arange(n_harmonic_freqs), requires_grad=False)
        else:
            self.mlp_in_dim = input_dim

        # MLP to project geometric features to semantic channel dimension
        self.proj = nn.Sequential(
            nn.Linear(self.mlp_in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Initialize to ensure initial contribution is small/stable
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.xavier_uniform_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def _get_rays(self, intrinsics, w2c, H, W):
        """
        Generate rays in World Coordinate System.
        Args:
            intrinsics: [B, 3, 3] Scaled intrinsics for feature map resolution
            w2c: [B, 4, 4] World-to-Camera matrix
            H, W: Feature map height and width
        Returns:
            rays_o: [B, H, W, 3] Ray Origins
            rays_d: [B, H, W, 3] Ray Directions (Normalized)
        """
        B, device = intrinsics.shape[0], intrinsics.device
        
        # 1. Invert w2c to get c2w (Camera-to-World)
        # w2c mapping: P_cam = R * P_world + t
        # c2w mapping: P_world = R^T * (P_cam - t) = R^T * P_cam - R^T * t
        # Standard inversion:
        c2w = torch.inverse(w2c) 

        # 2. Generate Grid
        # i: Width (x), j: Height (y)
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W, device=device),
            torch.linspace(0, H - 1, H, device=device),
            indexing='xy'
        )
        # [B, H, W]
        i = i.unsqueeze(0).expand(B, -1, -1)
        j = j.unsqueeze(0).expand(B, -1, -1)

        # 3. Unproject to Camera Coordinates
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
        cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

        # Expand to [B, H, W]
        fx, fy = fx[:, None, None], fy[:, None, None]
        cx, cy = cx[:, None, None], cy[:, None, None]

        # OpenGL/NeRF coordinate convention: -z looks forward.
        # Directions: [(u-cx)/fx, -(v-cy)/fy, -1]
        # OpenCV convention: z looks forward.
        # Directions: [(u-cx)/fx, (v-cy)/fy, 1]
        # MoVieS/VGGT usually follows the dataset convention. Assuming OpenCV here:
        dirs = torch.stack([
            (i - cx) / fx,
            (j - cy) / fy,
            torch.ones_like(i)
        ], -1) # [B, H, W, 3]

        # 4. Rotate to World Coordinates
        # ray_d_world = dirs @ R_c2w.T
        # c2w rotation part: c2w[:, :3, :3]
        rot = c2w[:, :3, :3] 
        rays_d = torch.sum(dirs[..., None, :] * rot[:, None, None, :3, :3], -1)
        
        # Normalize directions
        rays_d = F.normalize(rays_d, dim=-1)

        # 5. Ray Origin (Camera Center in World)
        # c2w translation part: c2w[:, :3, 3]
        rays_o = c2w[:, :3, 3].view(B, 1, 1, 3).expand(B, H, W, 3)

        return rays_o, rays_d

    def forward(self, ray_directions, intrinsics, w2c, meta_data, **kwargs):
        """
        Args:
            ray_directions: Unused here, calculated internally from K/w2c.
            intrinsics: [B*N, 3, 3]
            w2c: [B*N, 4, 4]
            meta_data: Dict with 'input_height', 'input_width'
        
        Returns:
            embedding: [B*N, embed_dim, H_feat, W_feat]
        """
        H_feat = meta_data["input_height"][0] // self.patch_size
        W_feat = meta_data["input_width"][0] // self.patch_size

        K_scaled = intrinsics.clone()
        K_scaled[:, 0, 0] /= self.patch_size
        K_scaled[:, 1, 1] /= self.patch_size
        K_scaled[:, 0, 2] /= self.patch_size
        K_scaled[:, 1, 2] /= self.patch_size

        rays_o, rays_d = self._get_rays(K_scaled, w2c, H_feat, W_feat)
        rays_m = torch.cross(rays_o, rays_d, dim=-1)
        plucker = torch.cat([rays_d, rays_m], dim=-1)

        if self.use_harmonic_embed:
            embed = plucker.unsqueeze(-1) * self.freqs
            embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=-1)
            embed = embed.flatten(-2)
            x = torch.cat([plucker, embed], dim=-1)
        else:
            x = plucker

        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)

        return x
