import logging

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
        image_encoder=None,
        depth_head=None,
        voxel_size=None,
        near=0.01,
        far=100.0,
        pretrain=None,
    ):
        super(FFGS, self).__init__()

        self.image_encoder = instantiate_from_config(image_encoder)
        self.depth_head = instantiate_from_config(depth_head)

        self.adapter = UnifiedGaussianAdapter(GaussianAdapterCfg(**adapter))
        self.raw_gs_dim = 1 + self.adapter.d_in  # 1 for opacity

        gs_dim = self.raw_gs_dim + 1
        self.param_head = nn.Sequential(
            nn.Conv2d(param_head_in_channels, gs_dim * 2, 3, 1, 1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(gs_dim * 2, gs_dim, 3, 1, 1, padding_mode="replicate"),
        )

        self.decoder = DecoderSplattingCUDA(DecoderSplattingCUDACfg(**decoder))

        self.opacity_mapping = opacity_mapping

        self.near = near
        self.far = far
        self.voxel_size = voxel_size
        self.pretrain = pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"FFGS, load pretrain {self.pretrain}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.opacity_mapping
        x = cfg["initial"] + min(global_step / cfg["warm_up"], 1) * (cfg["final"] - cfg["initial"])
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(weighted_feats, inverse_indices, dim=0)  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats

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
        with torch.amp.autocast("cuda", enabled=False):
            if image is not None:
                image = image.float()
            if patch_features is not None:
                patch_features = [feats.float() for feats in patch_features]
            if pose_enc is not None:
                pose_enc = pose_enc.float()
            if intrinsics is not None:
                intrinsics = intrinsics.float()
            if w2c is not None:
                w2c = w2c.float()

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

            out = self.param_head(features)
            out = out.view(b, v, -1, h, w)
            anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

            if self.training:
                # NOTE: 训练时使用 GT
                R, T = w2c[..., :3, :3], w2c[..., :3, 3]
                extrinsics = torch.cat([R, T[..., None]], dim=-1)
            else:
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image.shape[-2:])  # only for debug
                # NOTE: anysplat extrinsics:[3,4]([R, T])
            pts_all = batchify_unproject_depth_map_to_point_map(depth_map.squeeze(2), extrinsics, intrinsics)

            torch.cuda.empty_cache()

            neural_feats_list, neural_pts_list = [], []
            if self.voxel_size is not None and self.voxel_size > 0:
                for b_i in range(b):
                    neural_pts, neural_feats = self.voxelizaton_with_fusion(
                        anchor_feats[b_i],
                        pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                        self.voxel_size,
                        conf=conf[b_i],
                    )
                    neural_feats_list.append(neural_feats)
                    neural_pts_list.append(neural_pts)
            else:
                for b_i in range(b):
                    neural_feats_list.append(anchor_feats[b_i].permute(0, 2, 3, 1))
                    neural_pts_list.append(pts_all[b_i])

            max_voxels = max(f.shape[0] for f in neural_feats_list)
            neural_feats = self.pad_tensor_list(neural_feats_list, (max_voxels,), value=-1e10)

            neural_pts = self.pad_tensor_list(neural_pts_list, (max_voxels,), -1e4)  # -1 == invalid voxel

            depths = neural_pts[..., -1].unsqueeze(-1)
            densities = neural_feats[..., 0].sigmoid()

            assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
            assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

            opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)

            gaussians = self.adapter.forward(
                neural_pts,
                depths,
                opacity,
                neural_feats[..., 1:].squeeze(2),
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

