import logging

import math
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch_scatter import scatter_add, scatter_max

from hAlgorithm.modules.models2.external.anysplat.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from hAlgorithm.modules.models2.external.anysplat.model.encoder.common.gaussian_adapter import GaussianAdapterCfg, UnifiedGaussianAdapter
from hAlgorithm.modules.models2.external.anysplat.model.encoder.vggt.utils.geometry import batchify_unproject_depth_map_to_point_map
from hAlgorithm.modules.models2.external.anysplat.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from hAlgorithm.utils import instantiate_from_config


class FFGS(nn.Module):
    def __init__(
        self,
        param_head_in_channels,
        adapter,
        decoder,
        opacity_mapping,
        voxel_decoder,
        image_encoder=None,
        depth_head=None,
        voxel_size=None,
        near=0.01,
        far=100.0,
        pretrain=None,
        mean_vfe=True,
    ):
        super(FFGS, self).__init__()

        self.image_encoder = instantiate_from_config(image_encoder)
        self.depth_head = instantiate_from_config(depth_head)

        self.adapter = UnifiedGaussianAdapter(GaussianAdapterCfg(**adapter))
        gs_dim = self.adapter.d_in + 3 + 1 # xyz + opacity

        # voxel_decoder["output_channels"] = gs_dim
        self.voxel_decoder = instantiate_from_config(voxel_decoder)
        
        self.param_head = nn.Sequential(
            nn.Linear(param_head_in_channels, gs_dim * 2),
            nn.GELU(),
            nn.Linear(gs_dim * 2, gs_dim),
        )

        self.decoder = DecoderSplattingCUDA(DecoderSplattingCUDACfg(**decoder))

        self.opacity_mapping = opacity_mapping

        self.near = near
        self.far = far
        self.voxel_size = voxel_size
        self.pretrain = pretrain
        self.mean_vfe = mean_vfe

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"FFGS, load pretrain {self.pretrain}")

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

        if self.mean_vfe:
            # Flatten confidence scores and features
            anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

            # Aggregate per voxel
            voxel_feats = scatter_add(anchor_feats_flat.squeeze(1), inverse_indices, dim=0)  # [num_unique_voxels, feat_dim]
            voxel_feats = voxel_feats / counts.unsqueeze(-1)

        else:
            # Flatten confidence scores and features
            conf_flat = conf.flatten()  # [B*V*N]
            anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

            # Compute softmax weights per voxel
            conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
            conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
            voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
            weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [B*V*N, 1]

            # Compute weighted average of positions and features
            # weighted_pts = pts3d_flatten * weights
            weighted_feats = anchor_feats_flat.squeeze(1) * weights

            # Aggregate per voxel
            # voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)  # [num_unique_voxels, 3]
            voxel_feats = scatter_add(weighted_feats, inverse_indices, dim=0)  # [num_unique_voxels, feat_dim]

        return unique_voxels, voxel_feats

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def render(self, gaussians, extrinsics, intrinsics, image_shape, depth_mode=None, cam_rot_delta=None, cam_trans_delta=None):
        batch_size, view_num = extrinsics.shape[:2]
        device = extrinsics.device
        output = self.decoder.forward(
            gaussians=gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            near=torch.ones(batch_size, view_num, device=device) * self.near,
            far=torch.ones(batch_size, view_num, device=device) * self.far,
            image_shape=image_shape,
            depth_mode=depth_mode,
            cam_rot_delta=cam_rot_delta,
            cam_trans_delta=cam_trans_delta,
        )
        return dict(render_rgb=output.color, render_depth=output.depth)

    def forward(
        self,
        image=None,
        patch_features=None,
        patch_start_idx=None,
        pose_enc=None,
        intrinsics=None,
        w2c=None,
        meta_data=None,
        **kwargs,
    ):
        device = image.device
        b, v, _, h, w = image.shape
        global_step = meta_data.get("total_iter", 0)

        results = self.depth_head(
            patch_features,
            patch_start_idx=patch_start_idx,
            meta_data=meta_data,
        )
        depth_map = results["depth"].view(b, v, 1, h, w)
        depth_conf = results["confidence"].view(b, v, 1, h, w)
        features = results["features"]

        if self.image_encoder is not None:
            image_features = self.image_encoder(image.view(b * v, -1, h, w))
            features = torch.cat([features, image_features], dim=1)

        anchor_feats, conf = features.view(b, v, -1, h, w), depth_conf.squeeze(2)

        if self.training:
            # NOTE: 训练时使用 GT
            R, T = w2c[..., :3, :3], w2c[..., :3, 3]
            extrinsics = torch.cat([R, T[..., None]], dim=-1)
        else:
            with torch.amp.autocast("cuda", enabled=False):
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc.float(), image.shape[-2:])  # only for debug
                # NOTE: anysplat extrinsics:[3,4]([R, T])
        pts_all = batchify_unproject_depth_map_to_point_map(depth_map.squeeze(2), extrinsics, intrinsics)

        torch.cuda.empty_cache()

        voxel_coords_list, voxel_feats_list = [], []
        for b_i in range(b):
            voxel_coords, voxel_feats = self.voxelizaton_with_fusion(
                anchor_feats[b_i],
                pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                self.voxel_size,
                conf=conf[b_i],
            )
            voxel_coords = torch.cat([voxel_coords.new_ones(voxel_coords.shape[0], 1) * b_i, voxel_coords], dim=-1)
            voxel_coords_list.append(voxel_coords)
            voxel_feats_list.append(voxel_feats)

        voxel_coords = torch.cat(voxel_coords_list, dim=0)[:, [0, 3, 2, 1]]  # bxyz->bzyx
        voxel_feats = torch.cat(voxel_feats_list, dim=0)

        # normalize voxel_coords
        min_z, min_y, min_x = voxel_coords[:, 1:].min(dim=0)[0]
        min_zyx = torch.cat([min_z[None, None], min_y[None, None], min_x[None, None]], dim=-1)
        voxel_coords[:, 1:4] = voxel_coords[:, 1:4] - min_zyx

        max_z, max_y, max_x = voxel_coords[:, 1:].max(dim=0)[0]

        # sparse_shape = [z_dim, y_dim, x_dim]
        sparse_shape = [int(max_z.item()) + 1, int(max_y.item()) + 1, int(max_x.item()) + 1]

        refine_voxel_feats = self.voxel_decoder(batch_size=b, sparse_shape=sparse_shape, voxel_features=voxel_feats, voxel_coords=voxel_coords)
        voxel_feats = voxel_feats + refine_voxel_feats
        voxel_feats = self.param_head(voxel_feats)

        with torch.amp.autocast("cuda", enabled=False):
            voxel_feats = voxel_feats.float()
            extrinsics = extrinsics.float()
            intrinsics = intrinsics.float()

            batch_voxel_feats = [voxel_feats[voxel_coords[:, 0] == i] for i in range(b)]
            max_voxels = max(f.shape[0] for f in batch_voxel_feats)
            batch_voxel_feats = self.pad_tensor_list(batch_voxel_feats, (max_voxels,), value=-1e10)

            voxel_centers = (voxel_coords[:, 1:].float() + min_zyx + 0.5) * self.voxel_size
            voxel_centers = voxel_centers[:, [2, 1, 0]] # zyx->xyz
            batch_voxel_centers = [voxel_centers[voxel_coords[:, 0] == i] for i in range(b)]
            batch_voxel_centers = self.pad_tensor_list(batch_voxel_centers, (max_voxels,), value=-1e10)

            t = 3 * self.voxel_size
            means = t * (batch_voxel_feats[:, :, :3].sigmoid() * 2 - 1) + batch_voxel_centers
            opacity = batch_voxel_feats[:, :, 3].sigmoid()

            gaussians = self.adapter.forward(
                means,
                means[:, :, -1],
                opacity,
                batch_voxel_feats[:, :, 4:],
            )

            extrinsic_padding = torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsics.dtype).view(1, 1, 1, 4).repeat(b, v, 1, 1)
            intrinsics = intrinsics.clone()  # Create a new tensor
            intrinsics = torch.stack([intrinsics[:, :, 0] / w, intrinsics[:, :, 1] / h, intrinsics[:, :, 2]], dim=2)

            pred_context_pose = dict(
                extrinsics=torch.cat([extrinsics, extrinsic_padding], dim=2).inverse(),
                intrinsics=intrinsics,
            )
            output = self.decoder.forward(
                gaussians,
                pred_context_pose["extrinsics"],
                pred_context_pose["intrinsics"],
                torch.ones(b, v, device=device) * self.near,
                torch.ones(b, v, device=device) * self.far,
                (h, w),
                "depth",
            )

        results = dict(gaussians=gaussians, render_rgb=output.color, render_depth=output.depth)

        if self.depth_head is not None:
            results["depth"] = depth_map
            results["confidence"] = depth_conf

        return results
