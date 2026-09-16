import logging

import torch
import torch.nn as nn


class LocalL1Loss(nn.Module):
    def __init__(self, radius=3, loss_weight=1, with_anchor_pred=False, debug=False, **kwargs):
        super().__init__()

        self.radius = radius
        self.with_anchor_pred = with_anchor_pred

        self.loss_weight = loss_weight
        self.debug = debug

        if isinstance(self.loss_weight, dict):
            assert "default" in self.loss_weight

    def get_loss_weight(self, name=None):
        if name is not None and isinstance(self.loss_weight, dict) and name in self.loss_weight:
            loss_weight = self.loss_weight[name]
        elif isinstance(self.loss_weight, dict) and name not in self.loss_weight:
            loss_weight = self.loss_weight["default"]
        else:
            loss_weight = self.loss_weight
        return loss_weight

    def forward(self, prediction, target, mask, sift_point_mask, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)
        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        if sift_point_mask.sum() < 1:
            return 0 * torch.sum(prediction)

        radius = self.radius
        diameter = 2 * radius + 1
        range_index = prediction.new_tensor(
            [[i, j] for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]
        )

        target = target * mask
        prediction = prediction * mask

        if sift_point_mask.ndim == 4:
            sift_point_mask = sift_point_mask * mask[:, :, 0]

            bs, ss, ys, xs = torch.where(sift_point_mask.bool())
            B, S, H, W = sift_point_mask.shape
            N = sift_point_mask.sum()

            # NOTE: range_index [y,x]
            range_index = range_index[None].repeat(N, 1, 1)
            range_index[:, :, 0] = torch.clip(range_index[:, :, 0] + ys[:, None], min=0, max=H - 1)
            range_index[:, :, 1] = torch.clip(range_index[:, :, 1] + xs[:, None], min=0, max=W - 1)

            bs = bs[:, None, None].repeat(1, diameter * diameter, 1)
            ss = ss[:, None, None].repeat(1, diameter * diameter, 1)
            range_index = torch.cat([bs, ss, range_index], dim=-1).long()

            anchor_gt = target[:, :, -1][bs[:, 0, 0], ss[:, 0, 0], ys, xs][:, None]
            if self.with_anchor_pred:
                anchor_pred = prediction[:, :, -1][bs[:, 0, 0], ss[:, 0, 0], ys, xs][:, None]

            range_gt = target[:, :, -1][
                range_index[..., 0], range_index[..., 1], range_index[..., 2], range_index[..., 3]
            ]
            range_pred = prediction[:, :, -1][
                range_index[..., 0], range_index[..., 1], range_index[..., 2], range_index[..., 3]
            ]

        else:
            sift_point_mask = sift_point_mask * mask[:, 0]

            bs, ys, xs = torch.where(sift_point_mask.bool())
            B, H, W = sift_point_mask.shape
            N = sift_point_mask.sum()

            range_index = range_index[None].repeat(N, 1, 1)
            range_index[:, :, 0] = torch.clip(range_index[:, :, 0] + ys[:, None], min=0, max=H - 1)
            range_index[:, :, 1] = torch.clip(range_index[:, :, 1] + xs[:, None], min=0, max=W - 1)

            bs = bs[:, None, None].repeat(1, diameter * diameter, 1)
            range_index = torch.cat([bs, range_index], dim=-1).long()

            anchor_gt = target[:, -1][bs[:, 0, 0], ys, xs][:, None]
            if self.with_anchor_pred:
                anchor_pred = prediction[:, -1][bs[:, 0, 0], ys, xs][:, None]

            range_gt = target[:, -1][range_index[..., 0], range_index[..., 1], range_index[..., 2]]
            range_pred = prediction[:, -1][
                range_index[..., 0], range_index[..., 1], range_index[..., 2]
            ]

        range_gt = range_gt - anchor_gt
        if self.with_anchor_pred:
            range_pred = range_pred - anchor_pred
        else:
            range_pred = range_pred - anchor_gt

        dis_mask = range_gt.abs() <= (anchor_gt * 0.1)

        if self.debug:
            if sift_point_mask.ndim == 4:
                self.debug_anchor_and_range(
                    target, bs, ss, ys, xs, range_index, dis_mask, name="gt"
                )
                self.debug_anchor_and_range(
                    prediction, bs, ss, ys, xs, range_index, dis_mask, name="pred"
                )
            else:
                self.debug_anchor_and_range(
                    target, bs, None, ys, xs, range_index, dis_mask, name="gt"
                )
                self.debug_anchor_and_range(
                    prediction, bs, None, ys, xs, range_index, dis_mask, name="pred"
                )
            breakpoint()

        range_gt = range_gt * dis_mask
        range_pred = range_pred * dis_mask

        scale_gt = range_gt.abs().max(dim=-1)[0] + 1e-6
        range_gt = range_gt / scale_gt[:, None]

        scale_pred = range_pred.abs().max(dim=-1)[0] + 1e-6
        range_pred = range_pred / scale_pred[:, None]

        loss = (range_pred - range_gt).abs().mean(dim=-1).mean()

        if torch.isinf(loss).item() or torch.isnan(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(prediction))
            logging.warning(f"Data {name}, LocalL1Loss NAN error, {loss}")

        return loss * loss_weight

    def debug_anchor_and_range(self, points, bs, ss, ys, xs, range_index, dis_mask, name, **kwargs):
        import os

        import open3d as o3d

        bs = bs[:, 0, 0]
        N = bs.shape[0]

        if points.ndim == 5:
            ss = ss[:, 0, 0]
            B, S, C, H, W = points.shape
            anchor = points[bs, ss, :, ys, xs]
            anchor_range = points[
                range_index[..., 0],
                range_index[..., 1],
                :,
                range_index[..., 2],
                range_index[..., 3],
            ]
        else:
            S = 0
            B, C, H, W = points.shape
            anchor = points[bs, :, ys, xs]
            anchor_range = points[range_index[..., 0], :, range_index[..., 1], range_index[..., 2]]

        os.makedirs(f"./debug/anchor_and_range/{name}/", exist_ok=True)
        for bi in range(B):
            mask = bs == bi

            if S > 0:
                for si in range(S):
                    smask = ss == si
                    cur_points = points[bi, si].permute(1, 2, 0).reshape(-1, 3)
                    cur_anchor = anchor[mask & smask].reshape(-1, 3)
                    cur_anchor_range = anchor_range[mask & smask]
                    cur_dis_mask = dis_mask[mask & smask]
                    cur_anchor_range = cur_anchor_range[cur_dis_mask]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_points.detach().cpu().numpy())
                    save_path = f"./debug/anchor_and_range/{name}/b{bi:02d}_s{si:02d}_points.ply"
                    o3d.io.write_point_cloud(save_path, pcd)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_anchor.detach().cpu().numpy())
                    save_path = f"./debug/anchor_and_range/{name}/b{bi:02d}_s{si:02d}_anchor.ply"
                    o3d.io.write_point_cloud(save_path, pcd)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cur_anchor_range.detach().cpu().numpy())
                    save_path = f"./debug/anchor_and_range/{name}/b{bi:02d}_s{si:02d}_range.ply"
                    o3d.io.write_point_cloud(save_path, pcd)
            else:
                cur_points = points[bi].permute(1, 2, 0).reshape(-1, 3)
                cur_anchor = anchor[mask].reshape(-1, 3)
                cur_anchor_range = anchor_range[mask]
                cur_dis_mask = dis_mask[mask]
                cur_anchor_range = cur_anchor_range[cur_dis_mask]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_points.detach().cpu().numpy())
                save_path = f"./debug/anchor_and_range/{name}/b{bi:02d}_points.ply"
                o3d.io.write_point_cloud(save_path, pcd)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_anchor.detach().cpu().numpy())
                save_path = f"./debug/anchor_and_range/{name}/b{bi:02d}_anchor.ply"
                o3d.io.write_point_cloud(save_path, pcd)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cur_anchor_range.detach().cpu().numpy())
                save_path = f"./debug/anchor_and_range/{name}/b{bi:02d}_range.ply"
                o3d.io.write_point_cloud(save_path, pcd)


class GlobalLocalL1Loss(nn.Module):
    def __init__(
        self,
        radius=3,
        loss_weight=1,
        with_vis_mask=True,
        with_anchor_pred=False,
        debug=False,
        **kwargs,
    ):
        super().__init__()

        self.radius = radius
        self.with_vis_mask = with_vis_mask
        self.with_anchor_pred = with_anchor_pred

        self.loss_weight = loss_weight
        self.debug = debug

        if isinstance(self.loss_weight, dict):
            assert "default" in self.loss_weight

    def get_loss_weight(self, name=None):
        if name is not None and isinstance(self.loss_weight, dict) and name in self.loss_weight:
            loss_weight = self.loss_weight[name]
        elif isinstance(self.loss_weight, dict) and name not in self.loss_weight:
            loss_weight = self.loss_weight["default"]
        else:
            loss_weight = self.loss_weight
        return loss_weight

    def forward(
        self,
        prediction,
        target,
        mask,
        sift_point_mask,
        sift_track_points,
        sift_track_points_cam,
        sift_track_points_uv,
        sift_track_mask,
        sift_track_vis,
        name=None,
        **kwargs,
    ):

        loss_weight = self.get_loss_weight(name)
        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        if sift_track_vis.sum() < 1:
            return 0 * torch.sum(prediction)

        radius = self.radius
        diameter = 2 * radius + 1
        area = diameter * diameter
        range_index = prediction.new_tensor(
            [[i, j] for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]
        )

        target = target * mask
        prediction = prediction * mask

        B, S, C, H, W = prediction.shape

        sift_uv_int = (sift_track_points_uv + 0.5).long()
        uv_int_mask = (
            (sift_uv_int[..., 0] >= 0)
            & (sift_uv_int[..., 0] < W)
            & (sift_uv_int[..., 1] >= 0)
            & (sift_uv_int[..., 1] < H)
        )
        sift_uv_int = sift_uv_int * uv_int_mask.unsqueeze(-1)
        # NOTE: range_index [x,y]
        range_index = range_index[None, None, None] + sift_uv_int.unsqueeze(-2).expand(
            -1, -1, -1, area, -1
        )
        range_index[..., 0] = torch.clip(range_index[..., 0], min=0, max=W - 1)
        range_index[..., 1] = torch.clip(range_index[..., 1], min=0, max=H - 1)
        range_index = range_index.long()

        batch_indices = (
            torch.arange(B).view(B, 1, 1).expand(-1, S, sift_uv_int.shape[2]).to(prediction.device)
        )
        frame_indices = (
            torch.arange(S).view(1, S, 1).expand(B, -1, sift_uv_int.shape[2]).to(prediction.device)
        )

        # 仅第一帧作为 anchor
        anchor_gt = (
            target[
                batch_indices[:, 0],
                frame_indices[:, 0],
                :,
                sift_uv_int[:, 0, :, 1],
                sift_uv_int[:, 0, :, 0],
            ]
            .unsqueeze(1)
            .unsqueeze(-2)
        )  # [B,1,N,1,3]

        if self.with_anchor_pred:
            anchor_pred = (
                prediction[
                    batch_indices[:, 0],
                    frame_indices[:, 0],
                    :,
                    sift_uv_int[:, 0, :, 1],
                    sift_uv_int[:, 0, :, 0],
                ]
                .unsqueeze(1)
                .unsqueeze(-2)
            )  # [B,1,N,1,3]

        range_gt = target[
            batch_indices.unsqueeze(-1).expand(-1, -1, -1, area),
            frame_indices.unsqueeze(-1).expand(-1, -1, -1, area),
            :,
            range_index[..., 1],
            range_index[..., 0],
        ]  # [B, S, N, D*D, 3]
        range_pred = prediction[
            batch_indices.unsqueeze(-1).expand(-1, -1, -1, area),
            frame_indices.unsqueeze(-1).expand(-1, -1, -1, area),
            :,
            range_index[..., 1],
            range_index[..., 0],
        ]  # [B, S, N, D*D, 3]

        range_gt = range_gt - anchor_gt
        if self.with_anchor_pred:
            range_pred = range_pred - anchor_pred
        else:
            range_pred = range_pred - anchor_gt

        dis_mask = range_gt.norm(dim=-1) <= (anchor_gt.norm(dim=-1) * 0.1)
        dis_mask = dis_mask.unsqueeze(-1)

        if self.with_vis_mask:
            range_gt = (
                range_gt * dis_mask * uv_int_mask[..., None, None] * sift_track_vis[..., None, None]
            )
            range_pred = (
                range_pred
                * dis_mask
                * uv_int_mask[..., None, None]
                * sift_track_vis[..., None, None]
            )
        else:
            range_gt = (
                range_gt
                * dis_mask
                * uv_int_mask[..., None, None]
                * sift_track_mask[..., None, None]
            )
            range_pred = (
                range_pred
                * dis_mask
                * uv_int_mask[..., None, None]
                * sift_track_mask[..., None, None]
            )

        range_gt = range_gt.permute(0, 2, 1, 3, 4).reshape(B, -1, S * area, 3)
        range_pred = range_pred.permute(0, 2, 1, 3, 4).reshape(B, -1, S * area, 3)

        if self.debug:
            self.debug_anchor_and_range(
                target, anchor_gt, range_gt, sift_uv_int, sift_track_mask, sift_track_vis
            )

        scale_gt = range_gt.norm(dim=-1).max(dim=-1)[0] + 1e-6
        range_gt = range_gt / scale_gt[..., None, None]

        scale_pred = range_pred.norm(dim=-1).max(dim=-1)[0] + 1e-6
        range_pred = range_pred / scale_pred[..., None, None]

        loss = torch.nan_to_num(range_pred - range_gt).abs().mean()

        if torch.isinf(loss).item() or torch.isnan(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(prediction))
            logging.warning(f"Data {name}, GlobalLocalL1Loss NAN error, {loss}")

        return loss * loss_weight

    def debug_anchor_and_range(
        self, target, anchor_gt, range_gt, sift_uv_int, sift_track_mask, sift_track_vis
    ):
        import os

        import numpy as np
        import open3d as o3d

        from hAlgorithm.datasets_mv.track.vggt_track import get_track_colors_by_position

        B, S, C, H, W = target.shape
        N = anchor_gt.shape[2]
        A = range_gt.shape[-2] // S

        range_gt = range_gt + anchor_gt[:, :, :, 0].permute(0, 2, 1, 3).expand(-1, -1, S * A, -1)

        track_colors_rgb = get_track_colors_by_position(
            sift_uv_int[0],
            vis_mask_b=None,
            image_width=W,
            image_height=H,
            cmap_name="hsv",
        )

        os.makedirs("./debug/anchor_and_range_glb", exist_ok=True)

        for bi in range(B):
            cur_target = target[bi].detach().cpu().permute(0, 2, 3, 1).reshape(-1, 3).numpy()
            cur_anchor_gt = anchor_gt[bi, 0, :, 0, :].detach().cpu().numpy()
            cur_range_gt = range_gt[bi].detach().cpu().numpy().reshape(N * S, -1, 3)
            rgb = track_colors_rgb.copy()[:, None, :].repeat(S, axis=1).reshape(-1, 3)

            cur_valid_mask = sift_track_mask[bi].detach().cpu().permute(1, 0).numpy().reshape(-1)
            cur_vis_mask = sift_track_vis[bi].detach().cpu().permute(1, 0).numpy().reshape(-1)

            cur_range_gt = cur_range_gt[cur_valid_mask].reshape(-1, 3)
            rgb = rgb[cur_valid_mask]
            cur_vis_mask = cur_vis_mask[cur_valid_mask]
            rgb[~cur_vis_mask] = 0
            rgb = rgb[:, None, :].repeat(A, axis=1).reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cur_target)
            save_path = f"./debug/anchor_and_range_glb/{bi:02d}_target.ply"
            o3d.io.write_point_cloud(save_path, pcd)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cur_anchor_gt)
            colors = np.zeros([cur_anchor_gt.shape[0], 3])
            colors[:, 0] = 1.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            save_path = f"./debug/anchor_and_range_glb/{bi:02d}_anchor.ply"
            o3d.io.write_point_cloud(save_path, pcd)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cur_range_gt)
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
            save_path = f"./debug/anchor_and_range_glb/{bi:02d}_range.ply"
            o3d.io.write_point_cloud(save_path, pcd)

        breakpoint()
