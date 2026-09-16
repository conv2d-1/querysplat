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
    DepthPromptPointMapPipeline as BasePipeline,
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
    if C > 1:
        depth = depth[:, [-1], ...]
    pts = depth.flatten(2) * rays_d
    depth = pts.reshape(B, 3, H, W)
    return depth


class DepthPromptPointMapPipeline(BasePipeline):
    def __init__(
        self,
        prompt_diffmap_name=None,
        invalid_mask_target_name=None,
        conf_loss=None,
        inputconf_loss=None,
        grad_pred_loss=None,
        mask_loss=None,
        z_pointmap=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.prompt_diffmap_name = prompt_diffmap_name
        self.invalid_mask_target_name = invalid_mask_target_name
        self.z_pointmap = z_pointmap

        self.conf_loss = instantiate_from_config(conf_loss)
        self.grad_pred_loss = instantiate_from_config(grad_pred_loss)
        self.mask_loss = instantiate_from_config(mask_loss)
        self.inputconf_loss = instantiate_from_config(inputconf_loss)

    def share_step(
        self,
        x,
        prompt_depth,
        prompt_scale,
        prompt_center,
        prompt_clear_prob=0,
        intrinsics=None,
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
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)

        if isinstance(depth, dict):
            invalid_mask = depth.get("mask", None)
            confidence = depth.get("confidence", None)
            segms = depth.get("segms", None)
            grad = depth.get("grad", None)
            input_confidence = depth.get("input_confidence", None)
            depth = depth["depth"]
            if self.z_pointmap and intrinsics is not None:
                depth = depth_to_point(depth, K=intrinsics, device=depth.device)
            return depth, invalid_mask, confidence, segms, grad, input_confidence
        elif isinstance(depth, (list, tuple)):
            assert len(depth) == 2
            depth, invalid_mask = depth
            if self.z_pointmap and intrinsics is not None:
                depth = depth_to_point(depth, K=intrinsics, device=depth.device)
            return depth, invalid_mask, None, None, None, None
        else:
            if self.z_pointmap and intrinsics is not None:
                depth = depth_to_point(depth, K=intrinsics, device=depth.device)
            return depth, None, None, None, None, None

    def get_extend_loss(
        self,
        name,
        depth_pred,
        confidence_pred,
        intrinsics,
        grad_pred,
        segms_pred,
        invalid_mask_pred,
        input_confidence_pred,
        depth_target,
        valid_mask,
        invalid_mask_target,
        prompt_depth,
        prompt_diffmap,
        prompt_scale,
        prompt_center,
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        depth_pred = depth_pred.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        depth_target = depth_target.float().permute(0, 2, 3, 1)
        valid_mask = valid_mask.squeeze(1)
        if confidence_pred is not None:
            confidence_pred = confidence_pred.squeeze(1).unsqueeze(3)  # B,1,H,W -> B,H,W,1

        if prompt_depth is not None:
            prompt_depth = prompt_depth.float().permute(0, 2, 3, 1)

        if prompt_diffmap is not None:
            prompt_diffmap = prompt_diffmap.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C

        loss = 0
        loss_dict = {}

        def update_loss(loss, cur_loss, name):
            if cur_loss is None:
                return loss
            if isinstance(cur_loss, dict):
                loss += sum([val for key, val in cur_loss.items()])
                loss_dict.update(cur_loss)
            else:
                loss += cur_loss
                loss_dict[name] = cur_loss
            return loss

        if self.conf_loss is not None and confidence_pred is not None:
            conf_loss = self.conf_loss(
                confidence_pred,
                depth_pred,
                depth_target,
                prompt_scale,
                prompt_center,
                valid_mask,
                intrinsics=intrinsics,
                name=name,
            )
            loss = update_loss(loss, conf_loss, "conf_loss_extend")

        if self.grad_pred_loss is not None and grad_pred is not None:
            grad_pred_loss = self.grad_pred_loss(
                grad_pred, depth_target[..., -1], valid_mask, name=name
            )
            loss = update_loss(loss, grad_pred_loss, "grad_pred_loss_extend")

        if self.mask_loss is not None and invalid_mask_pred is not None:
            mask_loss = self.mask_loss(
                invalid_mask_pred.unsqueeze(1),
                invalid_mask_target.unsqueeze(1),
                valid_mask,
                name=name,
            )
            loss = update_loss(loss, mask_loss, "mask_loss_extend")

        if self.inputconf_loss is not None and input_confidence_pred is not None:
            inputconf_loss = self.inputconf_loss(
                input_confidence_pred,
                prompt_diffmap=prompt_diffmap,
                valid_mask=valid_mask,
                name=name,
            )
            loss = update_loss(loss, inputconf_loss, "inputconf_loss_extend")

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
            if self.invalid_mask_target_name is not None
            else None
        )
        prompt_diffmap = (
            batch[self.prompt_diffmap_name].to(self.device)
            if self.prompt_diffmap_name is not None
            else None
        )

        if self.prompt_name is not None:
            prompt_depth = batch[self.prompt_name].to(self.device, self.dtype)
            prompt_mask = (
                batch[self.prompt_mask_name].to(self.device)
                if self.prompt_mask_name is not None
                else None
            )
            if self.prompt_interpolate:
                import knn_interpolate

                prompt_depth_interpolate = torch.zeros_like(prompt_depth)
                knn_interpolate.k1_interpolate_batch(
                    prompt_depth.float().contiguous(),
                    prompt_mask.int().contiguous(),
                    prompt_depth_interpolate,
                )
                if self.debug:
                    for bi in range(prompt_depth.shape[0]):
                        prompt_depth_interpolate_norm = (
                            prompt_depth_interpolate[bi : bi + 1, 2].detach().cpu().numpy()
                        )
                        prompt_depth_interpolate_norm = (
                            prompt_depth_interpolate_norm - prompt_depth_interpolate_norm.min()
                        ) / (
                            prompt_depth_interpolate_norm.max()
                            - prompt_depth_interpolate_norm.min()
                            + 1e-6
                        )
                        prompt_depth_interpolate_norm = colorize_depth_maps(
                            prompt_depth_interpolate_norm,
                            0,
                            1,
                            cmap="turbo",
                            valid_mask=np.ones(prompt_depth_interpolate_norm.shape).astype(bool),
                        )
                        prompt_depth_interpolate_norm.save(f"{bi:04d}_dense_depth.jpg")

                        prompt_depth_norm = prompt_depth[bi : bi + 1, 2].detach().cpu().numpy()
                        prompt_depth_norm = (prompt_depth_norm - prompt_depth_norm.min()) / (
                            prompt_depth_norm.max() - prompt_depth_norm.min() + 1e-6
                        )
                        prompt_depth_norm = colorize_depth_maps(
                            prompt_depth_norm,
                            0,
                            1,
                            cmap="turbo",
                            valid_mask=np.ones(prompt_depth_norm.shape).astype(bool),
                        )
                        prompt_depth_norm.save(f"{bi:04d}_sparse_depth.jpg")

                prompt_depth = prompt_depth_interpolate

            prompt_scale = batch[self.prompt_scale_name].to(self.device, self.dtype)[
                :, None, None, None
            ]

            if self.prompt_center_name is not None:
                prompt_center = batch[self.prompt_center_name].to(self.device, self.dtype)[
                    :, :, None, None
                ]
            else:
                # Otherwise, create a tensor of zeros with the same shape as prompt_scale.
                # prompt_center = torch.zeros_like(prompt_scale)
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

        (
            depth_pred,
            invalid_mask_pred,
            confidence_pred,
            segms_pred,
            grad_pred,
            input_confidence_pred,
        ) = self.share_step(
            image,
            prompt_depth,
            prompt_scale,
            prompt_center=prompt_center,
            prompt_clear_prob=self.prompt_clear_prob,
            intrinsics=intrinsics,
        )

        loss, loss_dict = self.get_loss(
            name=name,
            total_iter=total_iter,
            predict=depth_pred,
            confidence_pred=confidence_pred,
            target=target,
            intrinsics=intrinsics,
            valid_mask=valid_mask,
            edge_mask=edge_mask,
            prompt_mask=prompt_mask,
            image=image,
        )

        loss_extend, loss_dict_extend = self.get_extend_loss(
            name=name,
            depth_pred=depth_pred,
            invalid_mask_pred=invalid_mask_pred,
            confidence_pred=confidence_pred,
            intrinsics=intrinsics,
            segms_pred=segms_pred,
            grad_pred=grad_pred,
            input_confidence_pred=input_confidence_pred,
            invalid_mask_target=invalid_mask_target,
            valid_mask=valid_mask,
            depth_target=target,
            prompt_depth=prompt_depth,
            prompt_diffmap=prompt_diffmap,
            prompt_scale=prompt_scale,
            prompt_center=(
                prompt_center if prompt_center is not None else torch.zeros_like(prompt_scale)
            ),
        )

        return loss + loss_extend, {**loss_dict, **loss_dict_extend}

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

            if self.prompt_interpolate:
                import knn_interpolate

                prompt_mask = (
                    kwargs[self.prompt_mask_name].to(self.device) if self.prompt_mask_name else None
                )
                prompt_depth_interpolate = torch.zeros_like(prompt_depth)
                knn_interpolate.k1_interpolate_batch(
                    prompt_depth.float().contiguous(),
                    prompt_mask.int().contiguous(),
                    prompt_depth_interpolate,
                )
                prompt_depth = prompt_depth_interpolate

            if self.prompt_center_name is not None:
                prompt_center = kwargs[self.prompt_center_name].to(self.device, self.dtype)[
                    :, :, None, None
                ]
            else:
                # Otherwise, create a tensor of zeros with the same shape as prompt_scale.
                # prompt_center = torch.zeros_like(prompt_scale)
                prompt_center = None
        else:
            prompt_depth = prompt_scale = prompt_center = None

        # Use the shared step method to make predictions based on the input image and depth information.
        if self.prompt_set_none:
            (
                pointmap_pred,
                invalid_mask_pred,
                confidence_pred,
                segms_pred,
                grad_pred,
                input_confidence_pred,
            ) = self.share_step(
                image,
                prompt_depth=None,
                prompt_scale=None,
                prompt_center=None,
                prompt_clear_prob=1.0,
                intrinsics=intrinsics,
            )
        else:
            (
                pointmap_pred,
                invalid_mask_pred,
                confidence_pred,
                segms_pred,
                grad_pred,
                input_confidence_pred,
            ) = self.share_step(
                image,
                prompt_depth,
                prompt_scale,
                prompt_center=prompt_center,
                prompt_clear_prob=1.0 if self.test_prompt_clear else 0,
                intrinsics=intrinsics,
            )

            if not self.test_prompt_clear:
                # Denormalize the predicted pointmap using the provided max range and center values.
                pointmap_pred = self.denormalize(
                    pointmap_pred, scale=prompt_scale, center=prompt_center
                )

        filtered_pointmap = None
        filtered_pointmap_color = None
        if confidence_pred is not None:
            filter_mask = confidence_pred.squeeze(0).squeeze(0) > 0  # H,W
            confidence_pred = confidence_pred.cpu()[0, 0].float().numpy()
            filtered_pointmap = (
                pointmap_pred.squeeze(0).permute(1, 2, 0)[filter_mask].cpu().float().numpy()
            )  # num,3

            filtered_pointmap_color = (
                image.squeeze(0).permute(1, 2, 0)[filter_mask].cpu().float().numpy()  # num,3
            )
            filtered_pointmap_color = (filtered_pointmap_color + 1) * 0.5 * 255

        inconf_filtered_pointmap = None
        inconf_filtered_pointmap_color = None
        if input_confidence_pred is not None:
            filter_mask = input_confidence_pred.squeeze(0).squeeze(0) > 0  # H,W
            input_confidence_pred = input_confidence_pred.cpu().float().squeeze(0).numpy()
            inconf_filtered_pointmap = (
                pointmap_pred.squeeze(0).permute(1, 2, 0)[filter_mask].cpu().float().numpy()
            )  # num,3

            inconf_filtered_pointmap_color = (
                image.squeeze(0).permute(1, 2, 0)[filter_mask].cpu().float().numpy()  # num,3
            )
            inconf_filtered_pointmap_color = (inconf_filtered_pointmap_color + 1) * 0.5 * 255

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
            if confidence_pred is not None:
                confidence_pred = cv2.resize(
                    confidence_pred,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
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
            pointmap_h=image.shape[-2],
            pointmap_w=image.shape[-1],
            confidence=confidence_pred,
            filtered_pointmap=filtered_pointmap,
            filtered_pointmap_color=filtered_pointmap_color,
            inconf_filtered_pointmap=inconf_filtered_pointmap,
            inconf_filtered_pointmap_color=inconf_filtered_pointmap_color,
            invalid_mask=(
                invalid_mask_pred.cpu().float().squeeze(0).numpy()
                if invalid_mask_pred is not None
                else None
            ),
            depth_grad=(
                grad_pred.cpu().float().squeeze(0).numpy() if grad_pred is not None else None
            ),
            input_confidence=input_confidence_pred,
        )

    def visualize(self, outputs, meta_data, out_dir):
        super().visualize(
            outputs=outputs,
            meta_data=meta_data,
            out_dir=out_dir,
        )

        import open3d as o3d

        if outputs.filtered_pointmap is not None:
            filtered_pointmap = outputs.filtered_pointmap.copy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_pointmap)
            if outputs.filtered_pointmap_color is not None:
                pointmap_color = outputs.filtered_pointmap_color.copy()
                pcd.colors = o3d.utility.Vector3dVector(pointmap_color / 255.0)
            save_path = os.path.join(out_dir, f"filtered_point_{meta_data['data_idx'][0]:06d}.ply")
            o3d.io.write_point_cloud(save_path, pcd)

        if outputs.inconf_filtered_pointmap is not None:
            filtered_pointmap = outputs.inconf_filtered_pointmap.copy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_pointmap)
            if outputs.inconf_filtered_pointmap_color is not None:
                pointmap_color = outputs.inconf_filtered_pointmap_color.copy()
                pcd.colors = o3d.utility.Vector3dVector(pointmap_color / 255.0)
            save_path = os.path.join(
                out_dir, f"inconf_filtered_point_{meta_data['data_idx'][0]:06d}.ply"
            )
            o3d.io.write_point_cloud(save_path, pcd)

        if outputs.input_confidence is not None:
            input_confidence = outputs.input_confidence.copy()
            input_confidence_color = colorize_depth_maps(
                input_confidence,
                input_confidence.min(),
                input_confidence.max(),
                cmap="turbo",
            )
            save_path = os.path.join(
                out_dir, f"input_confidence_{meta_data['data_idx'][0]:06d}.jpg"
            )
            input_confidence_color.save(save_path)

            input_confidence_mask = (input_confidence > 0).astype(float)
            input_confidence_mask_color = colorize_depth_maps(
                input_confidence_mask, 0, 1, cmap="turbo"
            )
            save_path = os.path.join(
                out_dir, f"input_confidence_mask_{meta_data['data_idx'][0]:06d}.jpg"
            )
            input_confidence_mask_color.save(save_path)

        # out conf
        if outputs.confidence is not None:
            confidence = outputs.confidence.copy()
            confidence_pred_colored = colorize_depth_maps(
                confidence, confidence.min(), confidence.max(), cmap="turbo"
            )
            print(f"Confidence min:{confidence.min()}; max:{confidence.max()}.")
            save_path = os.path.join(out_dir, f"confidence_{meta_data['data_idx'][0]:06d}.jpg")
            confidence_pred_colored.save(save_path)

            confidence_mask = (confidence > 0).astype(float)
            confidence_mask_color = colorize_depth_maps(confidence_mask, 0, 1, cmap="turbo")
            save_path = os.path.join(out_dir, f"confidence_mask_{meta_data['data_idx'][0]:06d}.jpg")
            confidence_mask_color.save(save_path)

        if outputs.depth_grad is not None:
            depth_grad = outputs.depth_grad.copy()
            grad_x = depth_grad[0, ...]
            grad_y = depth_grad[1, ...]
            grad_x_color = colorize_depth_maps(grad_x, grad_x.min(), grad_x.max(), cmap="turbo")
            save_path = os.path.join(out_dir, f"gradx_{meta_data['data_idx'][0]:06d}.jpg")
            grad_x_color.save(save_path)
            grad_y_color = colorize_depth_maps(grad_y, grad_y.min(), grad_y.max(), cmap="turbo")
            save_path = os.path.join(out_dir, f"grady_{meta_data['data_idx'][0]:06d}.jpg")
            grad_y_color.save(save_path)
