import json
import logging
import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn

from hAlgorithm.modules.models.ddi.ddiv2 import DDI
from hAlgorithm.modules.pipelines.visualize import (  # save_gradient,
    save_depth_map,
    save_error,
    save_point_cloud,
)
from hAlgorithm.modules.utils.alignment import align_depth_least_square, depth2disparity
from hAlgorithm.utils import (
    colorize_depth_maps,
    instantiate_from_config,
)

from .outputs import DepthOutput
from .prompt_pointmap_pipeline_v2 import PromptPointMapPipeline as BasePipeline


class PromptPointMapPipelineWithRel(BasePipeline):
    def __init__(
        self,
        post_align_rel=False,
        post_align_rel_disp=True,
        rel_l1_loss=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.post_align_rel = post_align_rel
        self.post_align_rel_disp = post_align_rel_disp

        # Rel Loss
        self.rel_l1_loss = instantiate_from_config(rel_l1_loss)
        self.use_rel_loss = any(
            [
                self.rel_l1_loss is not None,
            ]
        )

    def get_rel_loss(
        self,
        name,
        rel_depth_pred,
        intrinsics,
        depth_target,
        valid_mask,
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        rel_depth_pred = rel_depth_pred.float().squeeze(1)  # B,1,H,W -> B,H,W
        depth_target = depth_target.float()[:, -1, ...]  # B,C,H,W -> B,H,W
        valid_mask = valid_mask.squeeze(1)

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

        if self.rel_l1_loss is not None:
            rel_l1_loss = self.rel_l1_loss(
                rel_depth_pred,
                depth_target,
                valid_mask,
                name=name,
            )
            loss = update_loss(loss, rel_l1_loss, "l1_loss_rel")
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

        (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
            prompt_depth,
            prompt_depth_norm,
            prompt_scale,
            prompt_center,
            prompt_mask,
            prompt_diffmap,
            target,
            target_norm,
            valid_mask,
            edge_mask,
            image_show,
            sift_point_mask,
            sift_eval_mask,
        ) = self.get_inputs(batch)

        results = self.model(
            image, prompt_depth_norm, with_freeze=True, meta_data=batch["meta_data"]
        )
        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
        gradient_pred = results.get("gradient", None)
        prompt_confidence_pred = results.get("prompt_confidence", None)
        rel_depth = results.get("rel_depth", None)
        invalid_mask_pred = results.get("mask", None)

        # DepthMap trans to PointMap
        if pointmap_pred.shape[1] == 1:
            pointmap_pred = self.depth_to_point(pointmap_pred, K=intrinsics)

        loss, loss_dict = self.get_loss(
            name=name,
            total_iter=total_iter,
            pointmap_pred=pointmap_pred,
            confidence_pred=confidence_pred,
            target=target_norm,
            intrinsics=intrinsics,
            valid_mask=valid_mask,
            edge_mask=edge_mask,
            prompt_mask=prompt_mask,
            image=image,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            sift_mask=sift_point_mask,
        )

        if self.use_extend_loss:
            loss_extend, loss_dict_extend = self.get_extend_loss(
                name=name,
                pointmap_pred=pointmap_pred,
                invalid_mask_pred=invalid_mask_pred,
                confidence_pred=confidence_pred,
                intrinsics=intrinsics,
                gradient_pred=gradient_pred,
                prompt_confidence_pred=prompt_confidence_pred,
                invalid_mask_target=None,
                valid_mask=valid_mask,
                depth_target=target_norm,
                prompt_diffmap=prompt_diffmap,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
            )
            loss = loss + loss_extend
            loss_dict.update(loss_dict_extend)

        if self.use_rel_loss:
            loss_rel, loss_dict_rel = self.get_rel_loss(
                name=name,
                rel_depth_pred=rel_depth,
                intrinsics=intrinsics,
                depth_target=target_norm,
                valid_mask=valid_mask,
            )
            loss = loss + loss_rel
            loss_dict.update(loss_dict_rel)

        return loss, loss_dict

    def postprocess(
        self,
        intrinsics,
        extrinsics,
        image,
        target,
        prompt_depth,
        prompt_scale,
        prompt_center,
        image_show,
        pointmap_pred,
        confidence_pred,
        invalid_mask_pred=None,
        gradient_pred=None,
        prompt_confidence_pred=None,
        align_gt=None,
        align_mask=None,
        rel_depth_pred=None,
    ):
        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if image_show is not None and pointmap_pred is not None:
            pointmap_color = image_show
        else:
            pointmap_color = image.cpu().float().squeeze(0).numpy().transpose(1, 2, 0)
            pointmap_color = (pointmap_color + 1) * 0.5 * 255

        if pointmap_pred is not None:
            # Denormalize the predicted pointmap using the provided max range and center values.
            pointmap_pred = self.denormalize(
                pointmap_pred, scale=prompt_scale, center=prompt_center
            )

            if self.output_depth2point and pointmap_pred.shape[1] == 3:
                pointmap_pred = pointmap_pred[:, 2:3, :, :]

            # DepthMap trans to PointMap
            if pointmap_pred.shape[1] == 1:
                pointmap_pred = self.depth_to_point(pointmap_pred, K=intrinsics)

            # Adjust pointmap_color shape
            if (
                pointmap_pred.shape[2] != pointmap_color.shape[0]
                or pointmap_pred.shape[3] != pointmap_color.shape[1]
            ):
                pointmap_color = cv2.resize(
                    pointmap_color,
                    dsize=(pointmap_pred.shape[3], pointmap_pred.shape[2]),
                    interpolation=cv2.INTER_LINEAR,
                )
            pointmap_color = pointmap_color.reshape(-1, 3)

            # Convert the predicted pointmap to a NumPy array and extract the depth channel.
            depth = pointmap_pred[0, 2].cpu().float().numpy().clip(1e-3)
            pointmap_h, pointmap_w = depth.shape[:2]
            pointmap_pred = (
                pointmap_pred.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
            )
        else:
            depth = None
            pointmap_h, pointmap_w = pointmap_color.shape[:2]
            pointmap_color = pointmap_color.reshape(-1, 3)

        # Filtered noise pointmap base on confidence_pred
        if confidence_pred is not None:
            confidence_pred = confidence_pred.cpu()[0, 0].float().numpy()
            filtered_pointmap = pointmap_pred[confidence_pred.reshape(-1) > self.output_conf_thresh]
            filtered_pointmap_color = pointmap_color[
                confidence_pred.reshape(-1) > self.output_conf_thresh
            ]
        else:
            filtered_pointmap = filtered_pointmap_color = None

        if invalid_mask_pred is not None:
            invalid_mask_pred = invalid_mask_pred.detach().cpu()[0, 0].float().numpy()

        if rel_depth_pred is not None:
            rel_depth = rel_depth_pred.cpu().float().squeeze().numpy()
        else:
            rel_depth = None

        # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
        pointmap_pred_align = pointmap_color_align = None
        filtered_pointmap_pred_align = filtered_pointmap_color_align = None
        if self.match_input_res and pointmap_pred is not None:
            align_gt = align_gt.squeeze().numpy()
            h, w = align_gt.shape
            depth = cv2.resize(
                depth,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            if rel_depth is not None:
                rel_depth = cv2.resize(
                    rel_depth,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            if confidence_pred is not None:
                confidence_pred = cv2.resize(
                    confidence_pred,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.pointmap_match_input_res:
                ratio_h, ratio_w = 1.0 * h / pointmap_h, 1.0 * w / pointmap_w
                intrinsics_align = intrinsics.clone()
                intrinsics_align[:, 0, 0] = intrinsics_align[:, 0, 0] * ratio_w
                intrinsics_align[:, 1, 1] = intrinsics_align[:, 1, 1] * ratio_h
                intrinsics_align[:, 0, 2] = intrinsics_align[:, 0, 2] * ratio_w
                intrinsics_align[:, 1, 2] = intrinsics_align[:, 1, 2] * ratio_h

                pointmap_pred_align = self.depth_to_point(
                    torch.from_numpy(depth)[None, None],
                    K=intrinsics_align.cpu(),
                    device="cpu",
                    cache=False,
                )
                pointmap_pred_align = (
                    pointmap_pred_align.squeeze(0).permute(1, 2, 0).numpy().reshape(-1, 3)
                )

                pointmap_color_align = cv2.resize(
                    pointmap_color.reshape(pointmap_h, pointmap_w, 3),
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                ).reshape(-1, 3)

                if confidence_pred is not None:
                    filtered_pointmap_pred_align = pointmap_pred_align[
                        confidence_pred.reshape(-1) > self.output_conf_thresh
                    ]
                    filtered_pointmap_color_align = pointmap_color_align[
                        confidence_pred.reshape(-1) > self.output_conf_thresh
                    ]

            if self.post_align:
                align_mask_np = align_mask.squeeze().numpy()
                depth = align_depth_least_square(
                    gt_arr=align_gt,
                    pred_arr=depth,
                    valid_mask_arr=align_mask_np,
                    return_scale_shift=False,
                    max_resolution=None,
                )

            if self.post_align_rel:
                align_mask_np = align_mask.squeeze().numpy()
                if self.post_align_rel_disp:
                    align_gt_rel = depth2disparity(align_gt)
                else:
                    align_gt_rel = align_gt
                rel_depth = align_depth_least_square(
                    gt_arr=align_gt_rel,
                    pred_arr=rel_depth,
                    valid_mask_arr=align_mask_np,
                    return_scale_shift=False,
                    max_resolution=None,
                )

        # Optionally convert the ground truth pointmap to a NumPy array.
        if target is not None:
            pointmap_gt = target.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)

        if intrinsics is not None:
            intrinsics = intrinsics.cpu().squeeze(0).numpy()

        if extrinsics is not None:
            extrinsics = extrinsics.cpu().squeeze(0).numpy()

        # Global predicted point cloud
        pointmap_gt_global = pointmap_pred_global = filtered_pointmap_global = None
        if self.output_global_pointmap and extrinsics is not None:
            extrinsics_inv = np.linalg.inv(extrinsics)
            R = extrinsics_inv[:3, :3]
            T = extrinsics_inv[:3, 3]
            if pointmap_gt is not None:
                pointmap_gt_global = np.dot(R, pointmap_gt.T).T + T
            if pointmap_pred is not None:
                pointmap_pred_global = np.dot(R, pointmap_pred.T).T + T
            if confidence_pred is not None:
                filtered_pointmap_global = np.dot(R, filtered_pointmap.T).T + T

        if gradient_pred is not None:
            gradient_pred = gradient_pred.cpu().float().squeeze(0).numpy().transpose(1, 2, 0)

        if prompt_confidence_pred is not None:
            prompt_confidence_pred = prompt_confidence_pred.cpu().float()[0, 0].numpy()

        if prompt_depth is not None:
            _, _, prompt_h, prompt_w = prompt_depth.shape
            prompt_pointmap = (
                prompt_depth.squeeze(0)[:3].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
            )
        else:
            prompt_h = prompt_w = None

        return self.output_type(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_align=depth,
            pointmap=pointmap_pred,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            pointmap_h=pointmap_h,
            pointmap_w=pointmap_w,
            confidence=confidence_pred,
            filtered_pointmap=filtered_pointmap,
            filtered_pointmap_color=filtered_pointmap_color,
            filtered_pointmap_global=filtered_pointmap_global,
            pointmap_gt_global=pointmap_gt_global,
            pointmap_global=pointmap_pred_global,
            depth_grad=gradient_pred,
            input_confidence=prompt_confidence_pred,
            prompt_pointmap=prompt_pointmap,
            prompt_h=prompt_h,
            prompt_w=prompt_w,
            pointmap_align=pointmap_pred_align,
            pointmap_color_align=pointmap_color_align,
            filtered_pointmap_align=filtered_pointmap_pred_align,
            filtered_pointmap_color_align=filtered_pointmap_color_align,
            rel_depth=rel_depth,
            invalid_mask=invalid_mask_pred,
        )

    @torch.no_grad()
    def infer(self, **batch):
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

        (
            name,
            total_iter,
            image,
            intrinsics,
            extrinsics,
            prompt_depth,
            prompt_depth_norm,
            prompt_scale,
            prompt_center,
            prompt_mask,
            prompt_diffmap,
            target,
            target_norm,
            valid_mask,
            edge_mask,
            image_show,
            sift_point_mask,
            sift_eval_mask,
        ) = self.get_inputs(batch)

        # Use the shared step method to make predictions based on the input image and depth information.
        results = self.model(image, prompt_depth_norm, meta_data=batch["meta_data"])
        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
        gradient_pred = results.get("gradient", None)
        prompt_confidence_pred = results.get("prompt_confidence", None)
        rel_depth_pred = results.get("rel_depth", None)
        invalid_mask_pred = results.get("mask", None)

        if self.output_ddi:
            tmp_target = batch[self.target_name].to(device=self.device)

            tem_pointmap_pred_ = self.denormalize(
                pointmap_pred, scale=prompt_scale, center=prompt_center
            )
            from hAlgorithm.utils import cuda_timing_context

            with cuda_timing_context("ddi", True):
                depth_ddi = self.ddi(
                    tem_pointmap_pred_[:, -1],
                    prompt_depth[:, -1],
                    prompt_mask[:, 0],
                    prompt_scale,
                    tmp_target[:, -1],
                )
            pointmap_pred[:, -1, :, :] = depth_ddi / prompt_scale

        align_gt = align_mask = None
        if self.match_input_res:
            align_gt = batch[self.align_name]
        if self.match_input_res and (self.post_align or self.post_align_rel):
            align_mask = batch[self.align_mask_name]

        return self.postprocess(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image=image,
            target=target,
            prompt_depth=prompt_depth,
            prompt_scale=prompt_scale,
            prompt_center=prompt_center,
            image_show=image_show,
            pointmap_pred=pointmap_pred,
            confidence_pred=confidence_pred,
            invalid_mask_pred=invalid_mask_pred,
            gradient_pred=gradient_pred,
            prompt_confidence_pred=prompt_confidence_pred,
            align_gt=align_gt,
            align_mask=align_mask,
            rel_depth_pred=rel_depth_pred,
        )

    def visualize(self, outputs, meta_data, out_dir, prefix=""):
        data_idx = meta_data["data_idx"][0]
        if "rgb" in meta_data["data_info"]:
            rgb = meta_data["data_info"]["rgb"]
            logging.info(f"vis {data_idx}, out_dir:{out_dir}, rgb:{rgb}")
        else:
            logging.info(f"vis {data_idx}, out_dir:{out_dir}")

        # Depth Map Visualization
        if outputs.depth_align is not None:
            save_depth_map(
                outputs.depth_align.copy(), out_dir, f"{prefix}depth", data_idx, info=False
            )

        if outputs.rel_depth is not None:
            save_depth_map(
                outputs.rel_depth.copy(), out_dir, f"{prefix}rel_depth", data_idx, info=False
            )

        # Point Cloud Visualization
        if outputs.pointmap is not None:
            save_point_cloud(
                outputs.pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point",
                data_idx,
                info=False,
            )

        # Ground Truth Point Cloud Visualization
        if outputs.pointmap_gt is not None:
            save_point_cloud(
                outputs.pointmap_gt.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point_gt",
                data_idx,
            )

            # Error Map Visualization
            if outputs.pointmap is not None:
                pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
                pred_depthmap = outputs.pointmap.copy()[:, 2].reshape(pointmap_shape)
                gt_depthmap = outputs.pointmap_gt.copy()[:, 2].reshape(pointmap_shape)
                save_error(pred_depthmap, gt_depthmap, out_dir, f"{prefix}error", data_idx)

        # Confidence Visualization
        if outputs.confidence is not None:
            confidence = outputs.confidence.copy()
            save_depth_map(confidence, out_dir, f"{prefix}conf", data_idx, info=False)
            confidence_mask = (confidence > self.output_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, f"{prefix}conf_mask", data_idx)

        # invalid mask Visualization
        if outputs.invalid_mask is not None:
            invalid_mask = outputs.invalid_mask.copy()
            save_depth_map(invalid_mask, out_dir, f"{prefix}invalid_mask", data_idx, info=False)
            invalid_mask_binary = (invalid_mask > 0).astype(float)
            save_depth_map(invalid_mask_binary, out_dir, f"{prefix}invalid_mask_binary", data_idx)

        # Filtered Point Cloud Visualization
        if outputs.filtered_pointmap is not None:
            save_point_cloud(
                outputs.filtered_pointmap.copy(),
                (
                    outputs.filtered_pointmap_color.copy()
                    if outputs.filtered_pointmap_color is not None
                    else None
                ),
                out_dir,
                f"{prefix}filtered_point",
                data_idx,
            )

        # Global Point Cloud Visualization
        if outputs.pointmap_global is not None:
            save_point_cloud(
                outputs.pointmap_global.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}global_point",
                data_idx,
            )

        # Global Point Cloud Visualization
        if outputs.filtered_pointmap_global is not None:
            save_point_cloud(
                outputs.filtered_pointmap_global.copy(),
                (
                    outputs.filtered_pointmap_color.copy()
                    if outputs.filtered_pointmap_color is not None
                    else None
                ),
                out_dir,
                f"{prefix}global_filtered_point",
                data_idx,
            )

        if outputs.pointmap_gt_global is not None:
            save_point_cloud(
                outputs.pointmap_gt_global.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}global_point_gt",
                data_idx,
            )

        # Confidence Visualization
        if outputs.input_confidence is not None:
            confidence = outputs.input_confidence.copy()
            confidence_act = 1 / (1 + np.exp(-confidence))
            save_depth_map(
                confidence_act,
                out_dir,
                f"{prefix}prompt_conf",
                data_idx,
                min_val=0,
                max_val=1,
                info=False,
            )
            confidence_mask = (confidence > self.output_prompt_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, f"{prefix}prompt_conf_mask", data_idx)

        if outputs.prompt_pointmap is not None:
            if outputs.prompt_h != outputs.pointmap_h or outputs.prompt_w != outputs.pointmap_w:
                if outputs.pointmap_color is not None:
                    tmp_color = cv2.resize(
                        outputs.pointmap_color.copy().reshape(
                            outputs.pointmap_h, outputs.pointmap_w, -1
                        ),
                        dsize=(outputs.prompt_w, outputs.prompt_h),
                        interpolation=cv2.INTER_LINEAR,
                    ).reshape(outputs.prompt_w * outputs.prompt_h, -1)
                else:
                    tmp_color = None

                save_point_cloud(
                    outputs.prompt_pointmap.copy(),
                    tmp_color,
                    out_dir,
                    f"{prefix}prompt_point",
                    data_idx,
                )
            else:
                save_point_cloud(
                    outputs.prompt_pointmap.copy(),
                    (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                    out_dir,
                    f"{prefix}prompt_point",
                    data_idx,
                )

            save_depth_map(
                outputs.prompt_pointmap.copy().reshape(outputs.prompt_h, outputs.prompt_w, 3)[
                    :, :, 2
                ],
                out_dir,
                f"{prefix}prompt_point",
                data_idx,
                info=False,
            )

        if outputs.pointmap_align is not None:
            save_point_cloud(
                outputs.pointmap_align.copy(),
                (
                    outputs.pointmap_color_align.copy()
                    if outputs.pointmap_color_align is not None
                    else None
                ),
                out_dir,
                f"{prefix}point_align",
                data_idx,
            )

        if outputs.filtered_pointmap_align is not None:
            save_point_cloud(
                outputs.filtered_pointmap_align.copy(),
                (
                    outputs.filtered_pointmap_color_align.copy()
                    if outputs.filtered_pointmap_color_align is not None
                    else None
                ),
                out_dir,
                f"{prefix}filtered_point_align",
                data_idx,
            )
