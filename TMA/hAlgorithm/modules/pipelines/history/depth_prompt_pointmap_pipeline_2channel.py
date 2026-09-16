import json
import logging
import os
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa
from hAlgorithm.modules.pipelines.depth_prompt_pointmap_pipeline_v2 import (
    DepthPromptPointMapPipeline,
)
from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.utils import (
    colorize_depth_maps,
    cuda_timing_context,
    instantiate_from_config,
)

from .outputs import DepthOutput


def depth_to_point(depth, K, device):
    B, C, H, W = depth.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
    points = (
        torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
        .reshape(3, -1)
        .float()
        .to(device)
    )
    rays_d = K.inverse().to(device) @ points  # (B, 3, HW)
    pts = depth.flatten(2) * rays_d
    depth = pts.reshape(B, 3, H, W)
    return depth


class DepthPromptPointMap2ChannelPipeline(DepthPromptPointMapPipeline):

    def __init__(
        self,
        head_iters=1,
        **kwargs,
    ):
        super(DepthPromptPointMap2ChannelPipeline, self).__init__(**kwargs)

        self.head_iters = head_iters

    def share_step(
        self,
        x,
        prompt_depth,
        intrinsics,
        prompt_scale,
        prompt_center,
        prompt_clear_prob=0,
    ):
        """
        This method processes an input image `x` along with optional depth information `prompt_depth`.
        It normalizes the depth information if provided, prepares the image for the encoder,
        extracts features using a pretrained model, and then predicts depth using the depth head.

        Parameters:
        - x (Tensor): The input image tensor.
        - prompt_depth (Tensor or None): Optional tensor containing depth information to guide the prediction.
        - prompt_scale (float or Tensor): Maximum range value for normalizing the prompt depth.
        - prompt_center (float or Tensor): Center value for normalizing the prompt depth.

        Returns:
        - depth (Tensor): Predicted depth map from the depth head.
        - invalid_mask (Tensor or None): Predicted invalid depth mask from the depth head.
        """
        # Normalize the prompt depth if it is provided, using the maximum range and center values.
        if prompt_depth is not None:
            if prompt_clear_prob > 0 and random.random() < prompt_clear_prob:
                prompt_depth[...] = 0
            else:
                prompt_depth = self.normalize(prompt_depth, prompt_scale, prompt_center)

        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        x = ((x + 1) * 0.5 - self._mean) / self._std

        features = self.pretrained(x, self.model_config["layer_idxs"], return_class_token=True)
        for i in range(self.head_iters):
            depth, invalid_mask = self.depth_head(features, patch_h, patch_w, prompt_depth)

            if depth.shape[1] == 1:
                depth = depth_to_point(depth, K=intrinsics, device=self.device)

            prompt_depth = depth

        invalid_mask = nn.functional.sigmoid(invalid_mask)
        return depth, invalid_mask

    def get_loss(
        self,
        name,
        total_iter,
        predict,
        invalid_mask_pred,
        target,
        invalid_mask_target,
        intrinsics,
        valid_mask,
        edge_mask,
        prompt_mask,
        image,
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        predict = predict.float().permute(0, 2, 3, 1).contiguous()
        target = target.float().permute(0, 2, 3, 1).contiguous()
        valid_mask = valid_mask.squeeze(1)

        if prompt_mask is not None:
            prompt_mask = prompt_mask.squeeze(1)

        loss = 0
        loss_dict = dict()
        # Base l1 loss
        if self.l1_loss is not None:
            l1_Loss = self.l1_loss(
                predict,
                target,
                valid_mask,
                name=name,
                image=image,
                mask_pred=invalid_mask_pred,
            )
            loss_dict["l1_loss"] = l1_Loss
            loss += l1_Loss

        # Multi-Scale Anchor Loss
        if total_iter > self.warmup_iters and self.msa_loss is not None:
            msa_loss = self.msa_loss(
                predict,
                target,
                valid_mask,
                intrinsics=intrinsics,
                prompt_mask=prompt_mask,
                name=name,
            )
            loss_dict["msa_loss"] = msa_loss
            loss += msa_loss

        # Edge loss in warmup
        if (
            (total_iter <= self.warmup_iters)
            and self.edge_loss is not None
            and (edge_mask is not None)
        ):
            edge_mask = edge_mask.squeeze(1)
            edge_loss = self.edge_loss(predict, target, edge_mask, name=name)
            loss_dict["edge_loss"] = edge_loss
            loss += edge_loss

        # Edge loss
        if (
            total_iter > self.warmup_iters
            and self.edge_msa_loss is not None
            and edge_mask is not None
        ):
            edge_mask = edge_mask.squeeze(1)
            edge_msa_loss = self.edge_msa_loss(
                predict, target, edge_mask, intrinsics=intrinsics, name=name
            )
            loss_dict["edge_msa_loss"] = edge_msa_loss
            loss += edge_msa_loss

        # Normal loss
        if self.normal_loss is not None:
            if self.normal_loss.start_iter is None or (
                self.normal_loss.start_iter is not None and total_iter > self.normal_loss.start_iter
            ):
                normal_loss = self.normal_loss(predict, target, valid_mask, name=name)
                loss_dict["normal_loss"] = normal_loss
                loss += normal_loss

        # Prompt sparse depth loss
        if self.prompt_loss is not None and self.prompt_mask_name is not None:
            prompt_loss = self.prompt_loss(predict, target, prompt_mask, name=name)
            loss_dict["prompt_loss"] = prompt_loss
            loss += prompt_loss

        # Depth loss
        if self.depth_loss is not None:
            depth_loss = self.depth_loss(predict[..., -1], target[..., -1], valid_mask, name=name)
            loss_dict["depth_loss"] = depth_loss
            loss += depth_loss

        return loss, loss_dict

    def train_step(self, batch):
        """
        Executes a single training step using a batch of data.

        Parameters:
        - batch (dict): A dictionary containing the batch of data, including images, depth information, and masks.

        Returns:
        - loss (Tensor): The total loss for this training step.
        - loss_dict (dict): A dictionary containing individual loss components.
        """
        self.train()

        # Extract and move tensors to the appropriate device and dtype if necessary.
        name = batch["meta_data"]["name"][0]
        total_iter = batch["total_iter"]
        image = batch["image"].to(device=self.device, dtype=self.dtype)
        intrinsics = batch["intrinsics"].to(device=self.device)
        target = batch[self.target_name].to(device=self.device)
        valid_mask = batch[self.target_mask_name].to(self.device)
        invalid_mask_target = (
            batch[self.invalid_mask_target_name].to(device=self.device)
            if self.invalid_mask_target_name
            else None
        )

        if self.prompt_name is not None:
            prompt_depth = batch[self.prompt_name].to(self.device, self.dtype)
            prompt_mask = (
                batch[self.prompt_mask_name].to(self.device) if self.prompt_mask_name else None
            )
            prompt_scale = batch[self.prompt_scale_name].to(self.device, self.dtype)[
                :, None, None, None
            ]

            if self.prompt_center_name is not None:
                prompt_center = batch[self.prompt_center_name].to(self.device, self.dtype)[
                    :, :, None, None
                ]
            else:
                # Otherwise, create a tensor of zeros with the same shape as prompt_scale.
                prompt_center = None

            # Normalize the target depth using the provided max range and center values.
            target = self.normalize(target, prompt_scale, prompt_center)

            # Optionally clip the target depth to a specified range.
            if self.target_clip is not None:
                target = target.clip(self.target_clip[0], self.target_clip[1])

            # After normalize target
            if self.prompt_set_none:
                prompt_depth = prompt_scale = prompt_center = prompt_mask = None
        else:
            prompt_depth = prompt_scale = prompt_center = prompt_mask = None

        # Extract edge mask from the batch if it exists.
        edge_mask = batch.get(self.edge_mask_name, None)  # Use get() with a default value.
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=self.device)

        predict, invalid_mask_pred = self.share_step(
            image,
            prompt_depth,
            intrinsics,
            prompt_scale,
            prompt_center=prompt_center,
            prompt_clear_prob=self.prompt_clear_prob,
        )

        return self.get_loss(
            name,
            total_iter,
            predict,
            invalid_mask_pred,
            target,
            invalid_mask_target,
            intrinsics,
            valid_mask,
            edge_mask,
            prompt_mask,
            image,
        )

    @torch.no_grad()
    def infer(self, image: torch.Tensor, **kwargs):
        """
        Executes inference on a given input image and returns the predicted depth map and related outputs.

        Parameters:
        - image (Tensor): The input image tensor.
        - **kwargs: Additional keyword arguments containing depth information and other optional parameters.

        Returns:
        - DepthOutput: An object containing the predicted depth map and other related outputs.
        """

        # Set the model to evaluation mode.
        self.eval()

        image = image.to(self.device, self.dtype)
        intrinsics = kwargs.get("intrinsics", None)  # [n, 3, 3]

        if self.prompt_name is not None:
            prompt_depth = kwargs[self.prompt_name].to(self.device, self.dtype)
            prompt_scale = kwargs[self.prompt_scale_name].to(self.device, self.dtype)[
                :, None, None, None
            ]

            if self.prompt_center_name is not None:
                prompt_center = kwargs[self.prompt_center_name].to(self.device, self.dtype)[
                    :, :, None, None
                ]
            else:
                # Otherwise, create a tensor of zeros with the same shape as prompt_scale.
                prompt_center = None
        else:
            prompt_depth = prompt_scale = prompt_center = None

        # Use the shared step method to make predictions based on the input image and depth information.
        if self.prompt_set_none:
            pointmap_pred, invalid_mask_pred = self.share_step(
                image,
                prompt_depth=None,
                intrinsics=intrinsics,
                prompt_scale=None,
                prompt_center=None,
                prompt_clear_prob=1.0,
            )
        else:
            pointmap_pred, invalid_mask_pred = self.share_step(
                image,
                prompt_depth,
                intrinsics,
                prompt_scale,
                prompt_center=prompt_center,
                prompt_clear_prob=1.0 if self.test_prompt_clear else 0,
            )

            if not self.test_prompt_clear:
                # Denormalize the predicted pointmap using the provided max range and center values.
                pointmap_pred = self.denormalize(
                    pointmap_pred, scale=prompt_scale, center=prompt_center
                )

        # Convert the predicted pointmap to a NumPy array and extract the depth channel.
        pointmap_pred = (
            pointmap_pred.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        )
        depth = pointmap_pred[:, 2].reshape((image.shape[-2], image.shape[-1])).clip(1e-3)

        # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
        if self.match_input_res:
            depth_gt = kwargs[self.align_name].squeeze().numpy()
            h, w = depth_gt.shape
            depth = cv2.resize(
                depth,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )

            if self.post_align:
                depth_gt_valid_mask = kwargs[self.align_mask_name].squeeze().numpy()
                depth = align_depth_least_square(
                    gt_arr=depth_gt,
                    pred_arr=depth,
                    valid_mask_arr=depth_gt_valid_mask,
                    return_scale_shift=False,
                    max_resolution=None,
                )

        # Optionally convert the ground truth pointmap to a NumPy array.
        pointmap_gt = kwargs.get(self.target_name, None)
        if pointmap_gt is not None:
            pointmap_gt = (
                pointmap_gt.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
            )

        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        pointmap_color = image.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        pointmap_color = (pointmap_color + 1) * 0.5 * 255

        return DepthOutput(
            depth_align=depth,
            pointmap=pointmap_pred,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            intrinsics=intrinsics,
            invalid_mask=invalid_mask_pred,
        )

    def visualize(self, outputs, meta_data, out_dir):
        depth = outputs.depth_align.copy()
        max_val, min_val = depth.max(), depth.min()
        depth_norm = (depth - min_val) / (max_val - min_val)
        depth_pred_colored = colorize_depth_maps(
            depth_norm, 0, 1, cmap="turbo"
        )  # [3, H, W], value in (0, 1)
        save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.jpg")
        depth_pred_colored.save(save_path)

        logging.info(f"visualize save: {save_path}")

        import open3d as o3d

        pointmap = outputs.pointmap.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointmap)
        if outputs.pointmap_color is not None:
            pointmap_color = outputs.pointmap_color.copy()
            pcd.colors = o3d.utility.Vector3dVector(pointmap_color / 255.0)
        save_path = os.path.join(out_dir, f"point_{meta_data['data_idx'][0]:06d}.ply")
        o3d.io.write_point_cloud(save_path, pcd)

        logging.info(f"visualize save: {save_path}")

        if outputs.pointmap_gt is not None:
            pointmap_gt = outputs.pointmap_gt.copy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointmap_gt)
            if outputs.pointmap_color is not None:
                pointmap_color = outputs.pointmap_color.copy()
                pcd.colors = o3d.utility.Vector3dVector(pointmap_color / 255.0)
            save_path = os.path.join(out_dir, f"point_{meta_data['data_idx'][0]:06d}_gt.ply")
            o3d.io.write_point_cloud(save_path, pcd)

            logging.info(f"visualize save: {save_path}")

        # TODO add invalid_mask visualize func
        if outputs.invalid_mask is not None:
            invalid_mask_colored = colorize_depth_maps(
                outputs.invalid_mask.cpu().numpy(), 0, 1, cmap="turbo"
            )
            save_path = os.path.join(out_dir, f"mask_{meta_data['data_idx'][0]:06d}.jpg")
            invalid_mask_colored.save(save_path)
