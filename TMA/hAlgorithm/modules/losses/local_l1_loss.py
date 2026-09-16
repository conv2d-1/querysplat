import logging

import torch
import torch.nn as nn


class LocalL1Loss(nn.Module):

    def __init__(
        self,
        loss_weight=1,
        radius=3,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.radius = radius

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
        B, H, W, C = target.shape
        local_error = 0
        local_num = 0

        radius = self.radius
        for b in range(B):
            if sift_point_mask[b].sum() < 1:
                continue
            points = torch.argwhere(sift_point_mask[b, 0].bool())
            actual_target = target[b, ..., -1]
            actual_output = prediction[b, ..., -1]
            valid_mask = mask[b, 0]
            for y, x in points:
                y_start = max(0, y - radius)
                y_end = min(H, y + radius)
                x_start = max(0, x - radius)
                x_end = min(W, x + radius)
                pt_mask = valid_mask[y_start:y_end, x_start:x_end]
                if pt_mask.sum() < 1:
                    continue
                gt_area = actual_target[y_start:y_end, x_start:x_end][pt_mask]
                pred_area = actual_output[y_start:y_end, x_start:x_end][pt_mask]
                max_val = gt_area.max()
                min_val = gt_area.min()
                gt_area = (gt_area - min_val) / (max_val - min_val)
                pred_area = (pred_area - min_val) / (max_val - min_val)
                local_error += torch.abs(pred_area - gt_area).mean()
                local_num += 1

        if local_num > 0:
            return (local_error / local_num) * loss_weight

        return 0 * torch.sum(prediction)


class LocalL1LossV2(nn.Module):

    def __init__(
        self,
        loss_weight=1,
        radius=3,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.radius = radius

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
        self, prediction, target, mask, sift_point_mask, name=None, prompt_scale=None, **kwargs
    ):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(prediction)

        sift_point_mask = sift_point_mask.squeeze(1) & mask
        if sift_point_mask.sum() < 1:
            return 0 * torch.sum(prediction)

        radius = self.radius
        diameter = 2 * radius + 1
        range_index = prediction.new_tensor(
            [[i, j] for i in range(-radius, radius + 1) for j in range(-radius, radius + 1)]
        )
        mask = mask.unsqueeze(-1)
        target = target * mask
        prediction = prediction * mask

        bs, ys, xs = torch.where(sift_point_mask.bool())
        B, H, W = sift_point_mask.shape
        N = sift_point_mask.sum()

        range_index = range_index[None].repeat(N, 1, 1)
        range_index[:, :, 0] = torch.clip(range_index[:, :, 0] + ys[:, None], min=0, max=H - 1)
        range_index[:, :, 1] = torch.clip(range_index[:, :, 1] + xs[:, None], min=0, max=W - 1)
        bs = bs[:, None, None].repeat(1, diameter * diameter, 1)
        range_index = torch.cat([bs, range_index], dim=-1).long()

        anchor_gt = target[..., -1][bs[:, 0, 0], ys, xs][:, None]
        anchor_pred = prediction[..., -1][bs[:, 0, 0], ys, xs][:, None]
        range_gt = target[..., -1][range_index[..., 0], range_index[..., 1], range_index[..., 2]]
        range_pred = prediction[..., -1][
            range_index[..., 0], range_index[..., 1], range_index[..., 2]
        ]

        dis_mask = (range_gt - anchor_gt).abs() <= (anchor_gt * 0.1)
        val_mask = (range_gt > 0) * dis_mask
        N = val_mask.sum(dim=-1)

        _max = (((range_gt - anchor_gt) * val_mask).abs().max(dim=-1)[0] + 1e-6).unsqueeze(-1)
        if prompt_scale is not None:
            _max_scale = prompt_scale[:, 0, 0][bs[:, 0, 0]]
            _max_mask = (_max * _max_scale) > 0.01
        else:
            _max_mask = _max > 0
        val_mask = val_mask * _max_mask

        if val_mask.sum() < 1:
            return 0 * torch.sum(prediction)

        range_gt_norm = (range_gt - anchor_gt) * val_mask / _max
        range_pred_norm = (range_pred - anchor_gt) * val_mask / _max
        local_error = (range_pred_norm - range_gt_norm) * val_mask
        local_error = (local_error.abs().sum(dim=-1) / N)[_max_mask[:, 0]].mean()

        if not torch.isfinite(local_error):
            return 0 * torch.sum(prediction)

        return local_error * loss_weight


class LocalL1LossTS(nn.Module):
    def __init__(self, radius=3, loss_weight=1, debug=False, **kwargs):
        super().__init__()
        self.radius = radius
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
        prediction = prediction.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
        mask = mask.unsqueeze(1)
        sift_point_mask = sift_point_mask.squeeze(1)

        loss_weight = self.get_loss_weight(name)
        if loss_weight == 0:
            return 0 * torch.sum(prediction)
        if sift_point_mask.sum() < 1:
            logging.info("No Sift Point in Local L1 Loss.")
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
            range_index = range_index[None].repeat(N, 1, 1)
            range_index[:, :, 0] = torch.clip(range_index[:, :, 0] + ys[:, None], min=0, max=H - 1)
            range_index[:, :, 1] = torch.clip(range_index[:, :, 1] + xs[:, None], min=0, max=W - 1)
            bs = bs[:, None, None].repeat(1, diameter * diameter, 1)
            ss = ss[:, None, None].repeat(1, diameter * diameter, 1)
            range_index = torch.cat([bs, ss, range_index], dim=-1).long()
            anchor_gt = target[:, :, -1][bs[:, 0, 0], ss[:, 0, 0], ys, xs][:, None]
            # anchor_pred = prediction[:, :, -1][bs[:, 0, 0], ss[:, 0, 0], ys, xs][:, None]
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
            # anchor_pred = prediction[:, -1][bs[:, 0, 0], ys, xs][:, None]
            range_gt = target[:, -1][range_index[..., 0], range_index[..., 1], range_index[..., 2]]
            range_pred = prediction[:, -1][
                range_index[..., 0], range_index[..., 1], range_index[..., 2]
            ]
        range_gt = range_gt - anchor_gt
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
