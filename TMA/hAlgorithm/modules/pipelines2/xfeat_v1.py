import logging
import os

import cv2
import numpy as np
import torch

from hAlgorithm.modules.models2.external.xfeat.training.losses import alike_distill_loss, coordinate_classification_loss, dual_softmax_loss, keypoint_loss
from hAlgorithm.modules.pipelines2.base import Pipeline
from hAlgorithm.modules.pipelines2.utils.outputs import MatchingOutput
from hAlgorithm.modules.utils.ray import get_rays_in_camera_frame, get_rays_in_world_frame


class XFeatPipeline(Pipeline):
    """Pipeline for xfeat."""

    def __init__(self, intrinsics_name=None, extrinsics_name=None, downsample=8, inputs_normalize=True, with_ray_directions=False, with_ray_in_world=False, save_output_cfg=None, **kwargs):
        super(XFeatPipeline, self).__init__(**kwargs)

        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.downsample = int(downsample)

        self.inputs_normalize = inputs_normalize
        self.with_ray_directions = with_ray_directions
        self.with_ray_in_world = with_ray_in_world

        self.save_output_cfg = dict(
            save_everything=True,
            gt_out_dir="gt",
        )
        if save_output_cfg is not None:
            self.save_output_cfg.update(save_output_cfg)

    def get_inputs(self, batch):
        meta_data = batch["meta_data"]
        name = meta_data["name"][0]
        total_iter = batch.get("total_iter")

        if total_iter is not None:
            meta_data["total_iter"] = total_iter

        image = batch["image"].to(device=self.device, dtype=self.dtype)
        # NOTE: xfeat: ch3 -> ch1
        if self.inputs_normalize:
            image = image.mean(-3, keepdim=True)

        positives = intrinsics = extrinsics = image_show = None
        ray_directions = ray_world = None

        if "lut_mat21" in batch:
            lut_mat21 = batch["lut_mat21"].to(device=self.device, dtype=self.dtype)

            positives = []
            for bi in range(lut_mat21.shape[0]):
                for vi in range(lut_mat21.shape[1]):
                    mask_valid21 = torch.all(lut_mat21[bi, vi] >= 0, dim=-1)
                    points = lut_mat21[bi, vi][mask_valid21]
                    positives.append(points)

        if self.intrinsics_name is not None and self.intrinsics_name in batch:
            intrinsics = batch[self.intrinsics_name].to(device=self.device)

            if self.with_ray_directions:
                w = meta_data["input_width"][0].item()
                h = meta_data["input_height"][0].item()
                ray_directions = get_rays_in_camera_frame(
                    intrinsics=intrinsics.reshape(-1, 3, 3),
                    height=h,
                    width=w,
                    normalize_to_unit_sphere=True,
                )
                ray_directions = ray_directions.view(*intrinsics.shape[:2], 3, h, w)

        if self.extrinsics_name is not None and self.extrinsics_name in batch:
            extrinsics = batch[self.extrinsics_name].to(device=self.device)

            if self.with_ray_in_world:
                w = meta_data["input_width"][0].item()
                h = meta_data["input_height"][0].item()
                ray_origins_world, ray_directions_world = get_rays_in_world_frame(
                    intrinsics=intrinsics.reshape(-1, 3, 3),
                    height=h,
                    width=w,
                    normalize_to_unit_sphere=True,
                    camera_pose=extrinsics.reshape(-1, 4, 4).inverse(),  # NOTE: camera_pose is camera2word, extrinsics is word2camera
                )
                ray_origins_world = ray_origins_world.view(*intrinsics.shape[:2], 3, h, w)
                ray_directions_world = ray_directions_world.view(*intrinsics.shape[:2], 3, h, w)
                ray_world = torch.cat([ray_origins_world, ray_directions_world], dim=2)

        if not self.training:
            if "image_show" in batch:
                image_show = batch["image_show"]
                image_show = image_show.float().numpy()

        return (name, total_iter, meta_data, image, image_show, positives, intrinsics, extrinsics, ray_directions, ray_world)

    def train_step(self, batch):
        self.train()

        (
            name,
            total_iter,
            meta_data,
            image,
            image_show,
            positives,
            intrinsics,
            extrinsics,
            ray_directions,
            ray_world,
        ) = self.get_inputs(batch)

        results = self.model(
            image=image,
            positives=positives,
            ray_directions=ray_directions,
            ray_world=ray_world,
        )

        feats1, kpts1, hmap1 = results["feats1"], results["kpts1"], results["hmap1"]
        feats2, kpts2, hmap2 = results["feats2"], results["kpts2"], results["hmap2"]

        coords1 = results["coords1"]

        total_loss_dict = dict(loss_ds=[], loss_coords=[], loss_kp=[])

        # Compute losses
        for b in range(len(positives)):
            if len(positives[b]) == 0:
                continue
            # Get positive correspondencies
            pts1, pts2 = positives[b][:, :2], positives[b][:, 2:]

            # Grab features at corresponding idxs
            m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
            m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

            # grab heatmaps at corresponding idxs
            h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
            h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]

            loss_ds, conf = dual_softmax_loss(m1, m2)
            total_loss_dict["loss_ds"].append(loss_ds)

            if coords1[b] is not None:
                loss_coords, acc_coords = coordinate_classification_loss(coords1[b], pts1, pts2, conf)
                total_loss_dict["loss_coords"].append(loss_coords)

            # loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], image[b, 0])
            # loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], image[b, 1])
            # loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
            # acc_pos = (acc_pos1 + acc_pos2)/2
            # total_loss_dict["loss_kp_pos"].append(loss_kp_pos)

            loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)
            total_loss_dict["loss_kp"].append(loss_kp)

        total_loss_dict = {key: sum(val) / len(val) for key, val in total_loss_dict.items()}
        total_loss = sum(total_loss_dict.values())
        total_loss_dict["loss"] = total_loss.item()

        return total_loss, total_loss_dict

    @torch.no_grad()
    def infer(self, **batch):
        self.eval()

        (
            name,
            total_iter,
            meta_data,
            image,
            image_show,
            positives,
            intrinsics,
            extrinsics,
            ray_directions,
            ray_world,
        ) = self.get_inputs(batch)

        B, V, C, H, W = image.shape
        image0 = image[:, 0:1].expand(-1, V - 1, -1, -1, -1).reshape(-1, C, H, W)
        image1 = image[:, 1:].reshape(-1, C, H, W)

        ray_directions_image0 = ray_directions_image1 = ray_world_image0 = ray_world_image1 = None

        if ray_directions is not None:
            ray_directions_image0 = ray_directions[:, 0:1].expand(-1, V - 1, -1, -1, -1).reshape(-1, ray_directions.shape[-3], H, W)
            ray_directions_image1 = ray_directions[:, 1:].reshape(-1, ray_directions.shape[-3], H, W)

        if ray_world is not None:
            ray_world_image0 = ray_world[:, 0:1].expand(-1, V - 1, -1, -1, -1).reshape(-1, ray_world.shape[-3], H, W)
            ray_world_image1 = ray_world[:, 1:].reshape(-1, ray_world.shape[-3], H, W)

        if "use_amp" in batch and batch["use_amp"]:
            with torch.autocast("cuda", enabled=bool(batch["use_amp"]), dtype=batch["amp_dtype"]):
                results = self.model(
                    image0=image0,
                    image1=image1,
                    ray_directions_image0=ray_directions_image0,
                    ray_directions_image1=ray_directions_image1,
                    ray_world_image0=ray_world_image0,
                    ray_world_image1=ray_world_image1,
                )

        else:
            results = self.model(
                image0=image0,
                image1=image1,
                ray_directions_image0=ray_directions_image0,
                ray_directions_image1=ray_directions_image1,
                ray_world_image0=ray_world_image0,
                ray_world_image1=ray_world_image1,
            )

        B, V, C, H, W = image_show.shape
        image0 = np.repeat(image_show[:, :1], V - 1, axis=1)
        image0 = np.ascontiguousarray(image0.reshape(-1, C, H, W).transpose(0, 2, 3, 1).astype(np.uint8))
        image1 = np.ascontiguousarray(image_show[:, 1:].reshape(-1, C, H, W).transpose(0, 2, 3, 1).astype(np.uint8))

        matches = [match.cpu().numpy() for match in results["matches"]]
        if positives is not None:
            positives = [match.cpu().numpy() * self.downsample for match in positives]

        if extrinsics is not None:
            extrinsics_image0 = extrinsics[:, 0:1].expand(-1, V - 1, -1, -1).cpu().float().numpy().reshape(-1, 4, 4)
            extrinsics_image1 = extrinsics[:, 1:].cpu().float().numpy().reshape(-1, 4, 4)
        else:
            extrinsics_image0 = extrinsics_image1 = None

        if intrinsics is not None:
            intrinsics_image0 = intrinsics[:, 0:1].expand(-1, V - 1, -1, -1).cpu().float().numpy().reshape(-1, 3, 3)
            intrinsics_image1 = intrinsics[:, 1:].cpu().float().numpy().reshape(-1, 3, 3)
        else:
            intrinsics_image0 = intrinsics_image1 = None

        outputs_list = []
        for i in range(len(matches)):
            outputs = MatchingOutput(
                image0=image0[i],
                image1=image1[i],
                matches=matches[i],
                matches_gt=positives[i] if positives is not None else None,
                intrinsics_image0=intrinsics_image0[i] if intrinsics_image0 is not None else None,
                intrinsics_image1=intrinsics_image1[i] if intrinsics_image1 is not None else None,
                extrinsics_image0=extrinsics_image0[i] if extrinsics_image0 is not None else None,
                extrinsics_image1=extrinsics_image1[i] if extrinsics_image1 is not None else None,
            )
            outputs_list.append(outputs)

        return outputs_list

    def get_out_dir(self, out_dir, data_idx=None):
        if data_idx is not None:
            matching_out_dir = os.path.join(out_dir, f"matching/{data_idx:06d}")
        else:
            matching_out_dir = os.path.join(out_dir, "matching")
        return matching_out_dir

    def get_gt_out_dir(self, out_dir):
        if self.save_output_cfg["gt_out_dir"] is None:
            return None
        gt_out_dir = os.path.join(os.path.dirname(os.path.dirname(out_dir)), self.save_output_cfg["gt_out_dir"], os.path.basename(out_dir))
        return gt_out_dir

    def visualize(self, outputs_list, meta_data, out_dir):
        data_idx = meta_data["data_idx"][0]

        gt_out_dir = self.get_gt_out_dir(out_dir)
        out_dir = self.get_out_dir(out_dir, data_idx=data_idx)

        for idx in range(len(outputs_list)):
            outputs = outputs_list[idx]
            image0 = outputs.image0
            image1 = outputs.image1
            matches = outputs.matches
            matches_gt = outputs.matches_gt

            p1 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
            p2 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            
            # Images Visualization
            if gt_out_dir is not None and self.save_output_cfg["save_everything"]:
                image0_save_path = os.path.join(gt_out_dir, f"images/{data_idx:03d}/{idx:03d}/img0.jpg")
                image1_save_path = os.path.join(gt_out_dir, f"images/{data_idx:03d}/{idx:03d}/img1.jpg")

                if not os.path.exists(image0_save_path) or not os.path.exists(image1_save_path):
                    os.makedirs(os.path.dirname(image0_save_path), exist_ok=True)
                    cv2.imwrite(image0_save_path, p1)
                    cv2.imwrite(image1_save_path, p2)

            if matches_gt is not None:
                path = os.path.join(gt_out_dir, f"matching/{data_idx:03d}", f"{idx:06d}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)

                src_pts = matches_gt[:, :2]  # (N, 2)
                tgt_pts = matches_gt[:, 2:]  # (N, 2)

                colors = np.random.randint(0, 255, size=(len(src_pts), 3), dtype=np.uint8)
                for i, pt in enumerate(src_pts):
                    cv2.circle(p1, tuple(pt.astype(int)), radius=3, color=colors[i].tolist(), thickness=-1)
                for i, pt in enumerate(tgt_pts):
                    cv2.circle(p2, tuple(pt.astype(int)), radius=3, color=colors[i].tolist(), thickness=-1)
                combined = np.hstack((p1, p2))
                # cv2.imwrite(path, combined)

                W = p1.shape[1]
                combined_line = combined.copy()
                for i in range(len(src_pts)):
                    src = src_pts[i].astype(int).tolist()
                    tgt = tgt_pts[i].astype(int).tolist()
                    tgt[0] += W
                    cv2.line(combined_line, src, tgt, color=colors[i].tolist(), thickness=1)
                combined = np.vstack((combined, combined_line))
                cv2.imwrite(path, combined)
            
            # Prediction
            p1 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
            p2 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            src_pts = matches[:, :2]  # (N, 2)
            tgt_pts = matches[:, 2:]  # (N, 2)

            path = os.path.join(out_dir, f"{idx:06d}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)

            colors = np.random.randint(0, 255, size=(len(src_pts), 3), dtype=np.uint8)

            for i, pt in enumerate(src_pts):
                cv2.circle(p1, tuple(pt.astype(int)), radius=3, color=colors[i].tolist(), thickness=-1)
            for i, pt in enumerate(tgt_pts):
                cv2.circle(p2, tuple(pt.astype(int)), radius=3, color=colors[i].tolist(), thickness=-1)
            combined = np.hstack((p1, p2))
            # cv2.imwrite(path, combined)

            W = p1.shape[1]
            combined_line = combined.copy()
            for i in range(len(src_pts)):
                src = src_pts[i].astype(int).tolist()
                tgt = tgt_pts[i].astype(int).tolist()
                tgt[0] += W
                cv2.line(combined_line, src, tgt, color=colors[i].tolist(), thickness=1)
            combined = np.vstack((combined, combined_line))
            cv2.imwrite(path, combined)

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        pass
