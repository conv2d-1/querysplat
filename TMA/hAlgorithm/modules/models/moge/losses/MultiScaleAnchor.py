import logging

import numpy as np
import torch
import torch.nn as nn


class MultiScaleAnchorLoss(nn.Module):
    """
    Compute Global point map supervision, eq.(3) off MoGe
    """

    def __init__(
        self,
        loss_weight=1,
        data_type=["lidar", "denselidar", "stereo", "denselidar_syn", "sfm"],
        nums=50,
        debug=False,
        **kwargs,
    ):
        super(MultiScaleAnchorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.nums = nums
        self.debug = debug

    def sample_anchors(self, points, points2, mask, nums=1000):
        """
        sample anchor points from the point map with mask
        points: (B, h, w, 3)
        mask: (B, h, w)
        return: (B, nums, 3)
        """
        batch_size = points.shape[0]
        points = points.reshape(batch_size, -1, 3)
        points2 = points2.reshape(batch_size, -1, 3)
        mask = mask.reshape(batch_size, -1)
        anchors = []
        anchors2 = []
        for i in range(batch_size):
            points_i = points[i][mask[i]]
            points2_i = points2[i][mask[i]]

            # idx = np.random.choice(points_i.shape[0], nums, replace=True, p=(points_i[..., 2]/points_i[..., 2].sum()).detach().cpu().numpy())

            # NOTE: norm points z
            norm_points_i_z = points_i[..., 2] - points_i[..., 2].min()
            probabilities = (norm_points_i_z / norm_points_i_z.sum()).detach()
            idx = torch.multinomial(probabilities, nums, replacement=True)

            points_i = points_i[idx]
            points2_i = points2_i[idx]
            anchors.append(points_i)
            anchors2.append(points2_i)

        anchors = torch.stack(anchors)
        anchors2 = torch.stack(anchors2)
        return anchors, anchors2

    def find_neighbors(self, anchors, points, focal, alpha=[1 / 4]):
        """
        anchors: (B, nums, 3)
        points: (B, h, w, 3)
        focal: (B, 1)
        """
        dist = torch.norm(
            anchors[..., None, None, :] - points[:, None, ...], dim=-1
        )  # (B, nums, h, w)

        h, w = points.shape[1], points.shape[2]
        masks = []
        for i in range(len(alpha)):
            radius = alpha[i] * anchors[..., 2] * np.sqrt(h**2 + w**2) / (2 * focal)  # (B, nums)
            mask_ = dist < radius[:, :, None, None]  # (B, nums, h, w)
            #  maks_ = mask_.sum(1) > 0  # (B, h, w)
            masks.append(mask_)

        return masks

    def to_local_coord(self, points, anchors):
        """
        points: (B, h, w, 3)
        anchors: (B, nums, 3)
        return: (B, nums, h, w, 3)
        """
        return (
            points[:, None, ...] - anchors[..., None, None, :]
        )  # (b,1, h, w, 3)-(b, nums, 1, 1, 3)

    def forward(self, prediction, target, mask=None, **kwargs):

        valid = (mask.reshape(mask.shape[0], -1).sum(-1)) > 0
        target = target[valid]
        prediction = prediction[valid]
        mask = mask[valid]

        target_anchors, prediction_anchors = self.sample_anchors(
            target, prediction, mask, nums=self.nums
        )
        target_local_coord = self.to_local_coord(target, target_anchors)
        prediction_local_coord = self.to_local_coord(prediction, prediction_anchors)
        neig_masks = self.find_neighbors(
            target_anchors,
            target,
            focal=kwargs["intrinsics"][:, 0:1, 0],
            alpha=[1 / 8, 1 / 16, 1 / 32],
        )
        loss = 0
        for neig_mask in neig_masks:
            neig_mask = neig_mask & mask[:, None, :, :]

            diff = torch.sum(
                torch.abs(prediction_local_coord[neig_mask] - target_local_coord[neig_mask]),
                dim=-1,
            )
            # diff = torch.sum(torch.abs(prediction_local_coord - target_local_coord), dim=-1)*neig_mask# / (target[..., 2, None]+self.eps)
            loss += torch.mean(diff)

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            logging.warning(f"GlobalPointZLoss NAN error, {loss}")

        if self.debug:
            self.debug_tool(
                target,
                prediction,
                target_anchors,
                prediction_anchors,
                target_local_coord,
                prediction_local_coord,
                neig_masks,
            )

        return loss * self.loss_weight

    def debug_tool(
        self,
        target,
        prediction,
        target_anchors,
        prediction_anchors,
        target_local_coord,
        prediction_local_coord,
        neig_masks,
    ):
        import os

        import open3d as o3d

        os.makedirs("debug/MultiScaleAnchorLoss", exist_ok=True)

        points = target.cpu().numpy()[0].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        save_path = os.path.join("debug/MultiScaleAnchorLoss", f"target.ply")
        o3d.io.write_point_cloud(save_path, pcd)

        points = target_anchors.cpu().numpy()[0].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        save_path = os.path.join("debug/MultiScaleAnchorLoss", f"target_anchors.ply")
        o3d.io.write_point_cloud(save_path, pcd)

        points = target.cpu().numpy()[0].reshape(-1, 3)
        for mi, mask in enumerate(neig_masks):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            colors = np.ones_like(points)

            for i in range(mask.shape[1]):
                cur_mask = mask.cpu().numpy()[0, i].reshape(-1)
                colors[cur_mask, 1:3] = 0.0

            pcd.colors = o3d.utility.Vector3dVector(colors)
            save_path = os.path.join(
                "debug/MultiScaleAnchorLoss",
                f"target_local_coord_n{points.shape[0]}_m{mi}.ply",
            )
            o3d.io.write_point_cloud(save_path, pcd)


class MultiScaleAnchorLossFix(MultiScaleAnchorLoss):
    def find_neighbors(self, anchors, points, focal, alpha=[1 / 4]):
        """
        anchors: (B, nums, 3)
        points: (B, h, w, 3)
        focal: (B, 1)
        """
        dist = torch.norm(
            anchors[..., None, None, :] - points[:, None, ...], dim=-1
        )  # (B, nums, h, w)

        h, w = points.shape[1], points.shape[2]
        masks = []
        for i in range(len(alpha)):
            radius = (
                alpha[i] * anchors[..., 2].abs() * np.sqrt(h**2 + w**2) / (2 * focal)
            )  # (B, nums)
            mask_ = dist < radius[:, :, None, None]  # (B, nums, h, w)
            #  maks_ = mask_.sum(1) > 0  # (B, h, w)
            masks.append(mask_)

        return masks


class MultiScaleAnchorLossFix2(MultiScaleAnchorLossFix):
    @torch.no_grad()
    def sample_anchors(self, points, points2, mask, nums=1000):
        """
        sample anchor points from the point map with mask
        points: (B, h, w, 3)
        mask: (B, h, w)
        return: (B, nums, 3)
        """
        batch_size = points.shape[0]
        points = points.reshape(batch_size, -1, 3)
        points2 = points2.reshape(batch_size, -1, 3)
        mask = mask.reshape(batch_size, -1)
        anchors = []
        anchors2 = []
        for i in range(batch_size):
            points_i = points[i][mask[i]]
            points2_i = points2[i][mask[i]]

            # NOTE: norm points z
            norm_points_i_z = points_i[..., 2] - points_i[..., 2].min()
            probabilities = (norm_points_i_z / norm_points_i_z.sum()).detach()
            idx = torch.multinomial(probabilities, nums, replacement=True)

            points_i = points_i[idx]
            points2_i = points2_i[idx]

            anchors.append(points_i)
            anchors2.append(points2_i)

        anchors = torch.stack(anchors)
        anchors2 = torch.stack(anchors2)
        return anchors, anchors2
