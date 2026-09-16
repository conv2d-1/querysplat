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
from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.utils import (
    colorize_depth_maps,
    instantiate_from_config,
)

from .outputs import DepthOutput


class PromptPointMapPipeline(nn.Module):
    def __init__(
        self,
        model,
        target_name="pointmap",
        target_mask_name="depth_mask",
        intrinsics_name="intrinsics",
        extrinsics_name="extrinsics",
        invalid_mask_target_name=None,
        prompt_name=None,
        prompt_mask_name=None,
        prompt_center_name=None,
        prompt_scale_name=None,
        prompt_diffmap_name=None,
        align_name=None,
        align_mask_name=None,
        match_input_res=False,
        pointmap_match_input_res=False,
        post_align=False,
        l1_loss=None,
        local_loss=None,
        msa_loss=None,
        edge_mask_name=None,
        edge_loss=None,
        edge_msa_loss=None,
        normal_loss=None,
        prompt_loss=None,
        depth_loss=None,
        grad_loss=None,
        chamfer_loss=None,
        conf_loss=None,
        prompt_conf_loss=None,
        mask_loss=None,
        grad_pred_loss=None,
        normal_pred_loss=None,
        warmup_iters=-1,
        target_clip=None,
        dinov2_attention_with_sdpa=True,
        prompt_set_none=False,
        prompt_interpolate=False,
        output_depth2point=False,
        output_conf_thresh=0,
        output_conf_ratio=None,
        output_prompt_conf_thresh=0,
        output_global_pointmap=False,
        output_ddi=False,
        debug=False,
        debug_size=False,
        debug_scale=False,
        exclude_modules=[],
        save_everything=False,
        save_depth_np=True,
        save_pointmap=True,
        save_pointmap_filtered=False,
        save_confidence=False,
        # save_intrinsics=False,
        # save_extrinsics=False,
        vis_everything=False,
        vis_prompt=True,
        vis_confidence=False,
        vis_depthmap=True,
        vis_pointmap=True,
        vis_pointmap_filtered=False,
        vis_gt=True,
        **kwargs,
    ):
        super().__init__()

        self.model = instantiate_from_config(model)

        # Targe names
        self.warmup_iters = warmup_iters

        self.target_name = target_name
        self.target_mask_name = target_mask_name
        self.target_clip = target_clip
        self.intrinsics_name = intrinsics_name
        self.extrinsics_name = extrinsics_name
        self.edge_mask_name = edge_mask_name

        self.prompt_name = prompt_name
        self.prompt_mask_name = prompt_mask_name
        self.prompt_center_name = prompt_center_name
        self.prompt_scale_name = prompt_scale_name
        self.prompt_diffmap_name = prompt_diffmap_name
        self.prompt_set_none = prompt_set_none
        self.prompt_interpolate = prompt_interpolate

        self.align_name = align_name
        self.align_mask_name = align_mask_name
        self.post_align = post_align
        self.match_input_res = match_input_res
        self.pointmap_match_input_res = self.match_input_res and pointmap_match_input_res

        self.output_depth2point = output_depth2point
        self.output_conf_thresh = output_conf_thresh
        self.output_conf_ratio = output_conf_ratio
        self.output_prompt_conf_thresh = output_prompt_conf_thresh
        self.output_global_pointmap = output_global_pointmap
        self.output_type = DepthOutput

        self.output_ddi = output_ddi
        self.ddi = DDI()

        # Build Loss
        self.l1_loss = instantiate_from_config(l1_loss)
        self.local_loss = instantiate_from_config(local_loss)
        self.msa_loss = instantiate_from_config(msa_loss)
        self.edge_loss = instantiate_from_config(edge_loss)
        self.edge_msa_loss = instantiate_from_config(edge_msa_loss)
        self.normal_loss = instantiate_from_config(normal_loss)
        self.prompt_loss = instantiate_from_config(prompt_loss)
        self.depth_loss = instantiate_from_config(depth_loss)
        self.grad_loss = instantiate_from_config(grad_loss)
        self.chamfer_loss = instantiate_from_config(chamfer_loss)

        # Extend Loss
        self.invalid_mask_target_name = invalid_mask_target_name

        self.conf_loss = instantiate_from_config(conf_loss)
        self.mask_loss = instantiate_from_config(mask_loss)
        self.prompt_conf_loss = instantiate_from_config(prompt_conf_loss)
        self.grad_pred_loss = instantiate_from_config(grad_pred_loss)
        self.normal_pred_loss = instantiate_from_config(normal_pred_loss)
        self.use_extend_loss = any(
            [
                self.conf_loss is not None,
                self.mask_loss is not None,
                self.prompt_conf_loss is not None,
                self.grad_pred_loss is not None,
                self.normal_pred_loss is not None,
            ]
        )

        self.cache_dict = dict()
        self.debug = debug
        self.debug_size = debug_size
        self.debug_scale = debug_scale
        self.exclude_modules = exclude_modules

        #save
        self.save_everything = save_everything 
        self.save_depth_np = save_depth_np or self.save_everything
        self.save_pointmap = save_pointmap or self.save_everything
        self.save_pointmap_filtered = save_pointmap_filtered or self.save_everything
        self.save_confidence = save_confidence or self.save_everything
        # self.save_intrinsics = save_intrinsics or self.save_everything
        # self.save_extrinsics = save_extrinsics or self.save_everything
        
        self.vis_everything = vis_everything 
        self.vis_prompt = vis_prompt or self.vis_everything
        self.vis_confidence = vis_confidence or self.vis_everything
        self.vis_pointmap = vis_pointmap or self.vis_everything
        self.vis_depthmap = vis_depthmap or self.vis_everything
        self.vis_pointmap_filtered = vis_pointmap_filtered or self.vis_everything
        self.vis_gt = vis_gt or self.vis_everything

    def get_train_parameters(self):
        return self.parameters()

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        weight_dtype = torch.float32
        if accelerator.mixed_precision in ["fp16", "fp8"]:
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        logging.info(f"weight_dtype: {weight_dtype}")

        self.cuda(accelerator.device)
        self.device = accelerator.device
        self.dtype = weight_dtype

        if train_dataloader is None:
            self.model = accelerator.prepare(self.model)
            return None, None, None

        (
            self.model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
        return optimizer, train_dataloader, lr_scheduler

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_dict is not None:
            if len(self.exclude_modules) > 0:  # drop some modules which may have wrong shape
                new_state_dict = dict()
                drop_keys = []
                for key, val in state_dict.items():
                    ignore_val = False
                    for name in self.exclude_modules:
                        if name in key:
                            ignore_val = True
                            break
                    if ignore_val:
                        drop_keys.append(key)
                        continue
                    new_state_dict[key] = val
                res = self.model.load_state_dict(new_state_dict, strict=False)
                logging.info(f"drop keys: {drop_keys}")
            else:
                res = self.model.load_state_dict(state_dict, strict=False)

            logging.info(f"Model parameters are loaded from {ckpt_path}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def save_checkpoint(self, accelerator, ckpt_dir=None):
        model = accelerator.unwrap_model(self.model)
        state_dict = model.state_dict()

        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")
            try:
                torch.save(state_dict, ckpt_path)
                logging.info(f"Model is saved to: {ckpt_path}")
            except Exception:
                logging.warning(f"Model is saved error!!")

        return state_dict

    def depth_to_point(self, depth, K, device=None, cache=True):
        device = device or self.device
        B, C, H, W = depth.shape
        cache_key = f"depth_to_point_{H}_{W}"
        if cache_key in self.cache_dict:
            points = self.cache_dict[cache_key]
            if points.device != device:
                points.to(device)
        else:
            grid_x, grid_y = torch.meshgrid(
                torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy"
            )
            points = (
                torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
                .reshape(3, -1)
                .float()
                .to(device)
            )
            if cache:
                self.cache_dict[cache_key] = points
        rays_d = K.inverse() @ points  # (B, 3, HW)
        pts = depth.flatten(2) * rays_d
        depth = pts.reshape(B, 3, H, W)
        return depth

    def get_inputs(self, batch):
        # Extract and move tensors to the appropriate device and dtype if necessary.
        name = batch["meta_data"]["name"][0]
        total_iter = batch.get("total_iter", None)
        image = batch["image"].to(device=self.device, dtype=self.dtype)

        intrinsics = batch.get(self.intrinsics_name, None)
        extrinsics = batch.get(self.extrinsics_name, None)  # [n, 4, 4]

        if intrinsics is not None:
            intrinsics = intrinsics.to(device=self.device)
        if extrinsics is not None:
            extrinsics = extrinsics.to(device=self.device)

        image_show = batch.get("image_show", None)
        if image_show is not None:
            if image_show.ndim == 4:
                image_show = image_show.float().squeeze(0).numpy().transpose(1, 2, 0)
            else:
                image_show = image_show.float().squeeze(0).numpy().transpose(0, 2, 3, 1)

        prompt_scale = prompt_center = None
        if self.prompt_scale_name is not None and self.prompt_scale_name in batch:
            prompt_scale = batch[self.prompt_scale_name].to(self.device, self.dtype)[
                ..., None, None, None
            ]

        if self.prompt_center_name is not None and self.prompt_center_name in batch:
            prompt_center = batch[self.prompt_center_name].to(self.device, self.dtype)[
                ..., None, None, None
            ]

        if self.prompt_name is not None and self.prompt_name in batch:
            prompt_depth = batch[self.prompt_name].to(self.device, self.dtype)
            prompt_mask = (
                batch[self.prompt_mask_name].to(self.device) if self.prompt_mask_name else None
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
                    self.debug_prompt_interpolate(prompt_depth, prompt_depth_interpolate)

                prompt_depth = prompt_depth_interpolate
            else:
                prompt_diffmap = (
                    batch[self.prompt_diffmap_name].to(self.device)
                    if self.prompt_diffmap_name
                    else None
                )

            # Normalize the prompt depth if it is provided, using the maximum range and center values.
            prompt_depth_norm = self.normalize(prompt_depth, prompt_scale, prompt_center)

        else:
            prompt_depth = prompt_depth_norm = prompt_mask = prompt_diffmap = None
        
        if self.target_name is not None and self.target_name in batch:
            target = batch[self.target_name].to(device=self.device)
            valid_mask = batch[self.target_mask_name].to(self.device)

            if self.debug:
                logging.info(
                    f'{valid_mask.reshape(valid_mask.shape[0], -1).sum(-1),} {batch["meta_data"]["data_info"][0]["rgb"]}'
                )

            if self.prompt_interpolate:
                prompt_diffmap = torch.abs(target - prompt_depth)

            target_norm = self.normalize(target, prompt_scale, prompt_center)

            # Optionally clip the target depth to a specified range.
            if self.target_clip is not None:
                target_norm = target_norm.clip(self.target_clip[0], self.target_clip[1])

        else:
            target = target_norm = valid_mask = None

        # Extract edge mask from the batch if it exists.
        edge_mask = batch.get(self.edge_mask_name, None)
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=self.device)

        # After normalize
        if self.prompt_set_none:
            prompt_depth = prompt_depth_norm = prompt_scale = prompt_center = prompt_mask = (
                prompt_diffmap
            ) = None

        if self.debug_size:
            logging.info(
                f"image: {image.shape}, target: {target_norm.shape}, prompt: {prompt_depth.shape}"
            )
        if self.debug_scale:
            bs = image.shape[0]
            logging.info(
                f"prompt_scale: {prompt_scale.reshape(-1).cpu().numpy().tolist()}, target_norm:[{target_norm[:, 2].reshape(bs, -1).min(dim=1)[0].cpu().numpy().tolist()}, {target_norm[:, 2].reshape(bs, -1).max(dim=1)[0].cpu().numpy().tolist()}]"
            )

        return (
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
        )

    def get_loss(
        self,
        name,
        total_iter,
        pointmap_pred,
        confidence_pred,
        target,
        intrinsics,
        valid_mask,
        edge_mask,
        prompt_mask,
        image,
        sift_mask=None,
        **kwargs,
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        pointmap_pred = pointmap_pred.float().permute(0, 2, 3, 1).contiguous()
        target = target.float().permute(0, 2, 3, 1).contiguous()
        valid_mask = valid_mask.squeeze(1)

        if confidence_pred is not None:
            confidence_pred = confidence_pred.squeeze(1).unsqueeze(-1)

        if prompt_mask is not None:
            prompt_mask = prompt_mask.squeeze(1)

        if edge_mask is not None:
            edge_mask = edge_mask.squeeze(1)

        loss = 0
        loss_dict = dict()

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

        # Base l1 loss
        if self.l1_loss is not None:
            l1_Loss = self.l1_loss(
                pointmap_pred,
                target,
                valid_mask,
                name=name,
                image=image,
                confidence=confidence_pred,
            )
            loss = update_loss(loss, l1_Loss, "l1_loss")

        if self.local_loss is not None:
            local_loss = self.local_loss(
                pointmap_pred,
                target,
                valid_mask,
                name=name,
                sift_point_mask=sift_mask,
                intrinsics=intrinsics,
                prompt_scale=kwargs.get("prompt_scale", None),
                prompt_center=kwargs.get("prompt_center", None),
            )
            loss = update_loss(loss, local_loss, "local_loss")

        # Multi-Scale Anchor Loss
        if total_iter > self.warmup_iters and self.msa_loss is not None:
            msa_loss = self.msa_loss(
                pointmap_pred,
                target,
                valid_mask,
                intrinsics=intrinsics,
                prompt_mask=prompt_mask,
                name=name,
            )
            loss = update_loss(loss, msa_loss, "msa_loss")

        # Edge loss in warmup
        if (
            (total_iter <= self.warmup_iters)
            and self.edge_loss is not None
            and (edge_mask is not None)
        ):
            edge_loss = self.edge_loss(pointmap_pred, target, edge_mask, name=name)
            loss = update_loss(loss, edge_loss, "edge_loss")

        # Edge loss
        if (
            total_iter > self.warmup_iters
            and self.edge_msa_loss is not None
            and edge_mask is not None
        ):
            edge_msa_loss = self.edge_msa_loss(
                pointmap_pred, target, edge_mask, intrinsics=intrinsics, name=name
            )
            loss = update_loss(loss, edge_msa_loss, "edge_msa_loss")

        # Normal loss
        if self.normal_loss is not None and pointmap_pred.shape[-1] == 3:
            if self.normal_loss.start_iter is None or (
                self.normal_loss.start_iter is not None and total_iter > self.normal_loss.start_iter
            ):
                normal_loss = self.normal_loss(pointmap_pred, target, valid_mask, name=name)
                loss = update_loss(loss, normal_loss, "normal_loss")

        # Prompt sparse depth loss
        if self.prompt_loss is not None and self.prompt_mask_name is not None:
            prompt_loss = self.prompt_loss(pointmap_pred, target, prompt_mask, name=name)
            loss = update_loss(loss, prompt_loss, "prompt_loss")

        # Depth loss
        # TODO: for global pointmap, this should be adjusted
        if self.depth_loss is not None:
            if getattr(self.depth_loss, "start_iter", None) is None or (
                self.depth_loss.start_iter is not None and total_iter > self.depth_loss.start_iter
            ):
                depth_loss = self.depth_loss(
                    pointmap_pred[..., -1], target[..., -1], valid_mask, name=name
                )
                loss = update_loss(loss, depth_loss, "depth_loss")

        # Gradient loss
        # TODO: for global pointmap, this should be adjusted
        if self.grad_loss is not None:
            if getattr(self.grad_loss, "require_pointmap", False):
                grad_loss = self.grad_loss(pointmap_pred, target, valid_mask, name=name)
            else:
                grad_loss = self.grad_loss(
                    pointmap_pred[..., -1], target[..., -1], valid_mask, name=name
                )
            loss = update_loss(loss, grad_loss, "grad_loss")

        # Chamfer distance loss
        if self.chamfer_loss is not None:
            chamfer_loss = self.chamfer_loss(
                pointmap_pred,
                target,
                valid_mask,
                edge_mask=edge_mask,
                confidence=confidence_pred,
                name=name,
            )
            loss = update_loss(loss, chamfer_loss, "chamfer_loss")

        return loss, loss_dict

    def get_extend_loss(
        self,
        name,
        pointmap_pred,
        confidence_pred,
        intrinsics,
        gradient_pred,
        invalid_mask_pred,
        prompt_confidence_pred,
        depth_target,
        valid_mask,
        invalid_mask_target,
        prompt_diffmap,
        prompt_scale,
        prompt_center,
        **kwargs
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        pointmap_pred = pointmap_pred.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
        depth_target = depth_target.float().permute(0, 2, 3, 1)
        valid_mask = valid_mask.squeeze(1)

        if confidence_pred is not None:
            confidence_pred = confidence_pred.squeeze(1).unsqueeze(3)  # B,1,H,W -> B,H,W,1

        if prompt_diffmap is not None:
            prompt_diffmap = prompt_diffmap.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C

        if gradient_pred is not None:
            gradient_pred = gradient_pred.float().permute(0, 2, 3, 1)

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
                pointmap_pred,
                depth_target,
                prompt_scale,
                prompt_center,
                valid_mask,
                intrinsics=intrinsics,
                name=name,
            )
            loss = update_loss(loss, conf_loss, "conf_loss")

        if self.grad_pred_loss is not None and gradient_pred is not None:
            grad_pred_loss = self.grad_pred_loss(
                gradient_pred, depth_target[..., -1], valid_mask, name=name
            )
            loss = update_loss(loss, grad_pred_loss, "grad_pred_loss")

        if self.mask_loss is not None and invalid_mask_pred is not None and invalid_mask_target is not None:
            mask_loss = self.mask_loss(
                invalid_mask_pred.unsqueeze(1),
                invalid_mask_target.unsqueeze(1),
                valid_mask,
                name=name,
            )
            loss = update_loss(loss, mask_loss, "mask_loss")

        if self.prompt_conf_loss is not None and prompt_confidence_pred is not None:
            prompt_conf_loss = self.prompt_conf_loss(
                prompt_confidence_pred,
                prompt_diffmap=prompt_diffmap,
                valid_mask=valid_mask,
                name=name,
            )
            loss = update_loss(loss, prompt_conf_loss, "prompt_conf_loss")

        normal_pred = kwargs.get('normal_pred', None)
        if self.normal_pred_loss is not None and normal_pred is not None:
            normal_pred = depth_target.permute(0, 2, 3, 1)
            normal_pred_loss = self.normal_pred_loss(
                normal_pred, depth_target, valid_mask, name=name,
            )
            loss = update_loss(loss, normal_pred_loss, "normal_pred_loss")

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
        ) = self.get_inputs(batch)

        sift_eval_mask = batch.get("sift_eval_mask", None)
        sift_point_mask = batch.get("sift_mask", None)

        inf_mask = batch.get("inf_mask", None)
        results = self.model(
            image, prompt_depth_norm, with_freeze=True, meta_data=batch["meta_data"]
        )
        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
        gradient_pred = results.get("gradient", None)
        prompt_confidence_pred = results.get("prompt_confidence", None)
        invalid_mask_pred = results.get("mask", None)
        normal_pred = results.get('normal', None)

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
                invalid_mask_target=inf_mask,
                valid_mask=valid_mask,
                depth_target=target_norm,
                prompt_diffmap=prompt_diffmap,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
                normal_pred=normal_pred,
            )
            loss = loss + loss_extend
            loss_dict.update(loss_dict_extend)

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
            if self.output_conf_ratio is not None:
                conf_thresh = np.percentile(
                    confidence_pred.reshape(-1), self.output_conf_ratio * 100
                )
                filtered_pointmap = pointmap_pred[confidence_pred.reshape(-1) > conf_thresh]
                filtered_pointmap_color = pointmap_color[confidence_pred.reshape(-1) > conf_thresh]
            else:
                filtered_pointmap = pointmap_pred[
                    confidence_pred.reshape(-1) > self.output_conf_thresh
                ]
                filtered_pointmap_color = pointmap_color[
                    confidence_pred.reshape(-1) > self.output_conf_thresh
                ]
        else:
            filtered_pointmap = filtered_pointmap_color = None

        if invalid_mask_pred is not None:
            invalid_mask_pred = invalid_mask_pred.detach().cpu()[0, 0].float().numpy()
            valid_pointmap = pointmap_pred[
                invalid_mask_pred.reshape(-1) < 0
            ]
            valid_pointmap_color = pointmap_color[
                invalid_mask_pred.reshape(-1) < 0
            ]
        else:
            valid_pointmap = valid_pointmap_color = None

        # If matching input resolution is required, resize the predicted pointmap and color pointmap to match the ground truth depth.
        pointmap_pred_align = pointmap_color_align = None
        filtered_pointmap_pred_align = filtered_pointmap_color_align = None
        if self.match_input_res and depth is not None and align_gt is not None:
            align_gt = align_gt.squeeze().numpy()
            h, w = align_gt.shape[:2]
            depth = cv2.resize(
                depth,
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
                    if self.output_conf_ratio is not None:
                        conf_thresh = np.percentile(
                            confidence_pred.reshape(-1), self.output_conf_ratio * 100
                        )
                        filtered_pointmap_pred_align = pointmap_pred_align[
                            confidence_pred.reshape(-1) > conf_thresh
                        ]
                        filtered_pointmap_color_align = pointmap_color_align[
                            confidence_pred.reshape(-1) > conf_thresh
                        ]
                    else:
                        filtered_pointmap_pred_align = pointmap_pred_align[
                            confidence_pred.reshape(-1) > self.output_conf_thresh
                        ]
                        filtered_pointmap_color_align = pointmap_color_align[
                            confidence_pred.reshape(-1) > self.output_conf_thresh
                        ]

            if self.post_align and align_mask is not None:
                align_mask = align_mask.squeeze().numpy()
                depth = align_depth_least_square(
                    gt_arr=align_gt,
                    pred_arr=depth,
                    valid_mask_arr=align_mask,
                    return_scale_shift=False,
                    max_resolution=None,
                )

        # Optionally convert the ground truth pointmap to a NumPy array.
        if target is not None:
            pointmap_gt = target.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        else:
            pointmap_gt = None

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
            prompt_pointmap = prompt_h = prompt_w = None

        if prompt_scale is not None:
            prompt_scale = float(prompt_scale.cpu().squeeze())

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
            invalid_mask=invalid_mask_pred,
            prompt_scale=prompt_scale,
            valid_pointmap=valid_pointmap,
            valid_pointmap_color=valid_pointmap_color,
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
        ) = self.get_inputs(batch)

        sift_eval_mask = batch.get("sift_eval_mask", None)
        sift_point_mask = batch.get("sift_mask", None)
        # Use the shared step method to make predictions based on the input image and depth information.
        results = self.model(image, prompt_depth_norm, meta_data=batch["meta_data"])
        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]
        gradient_pred = results.get("gradient", None)
        prompt_confidence_pred = results.get("prompt_confidence", None)
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
        if self.match_input_res and self.align_name in batch:
            align_gt = batch[self.align_name]
        if self.match_input_res and self.post_align and self.align_mask_name in batch:
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
        )

    def normalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if center is not None:
            depth = depth - center
        if scale is not None:
            depth = depth / scale
        return depth

    def denormalize(
        self, depth: torch.Tensor, scale: torch.Tensor = None, center: torch.Tensor = None
    ):
        if scale is not None:
            depth = depth * scale
        if center is not None:
            depth = depth + center
        return depth

    def visualize(self, outputs, meta_data, out_dir, prefix=""):
        data_idx = meta_data["data_idx"][0]
        if "rgb" in meta_data["data_info"][0]:
            rgb = meta_data["data_info"][0]["rgb"]
            logging.info(f"vis {data_idx}, out_dir:{out_dir}, rgb:{rgb}")
        else:
            logging.info(f"vis {data_idx}, out_dir:{out_dir}")

        # Depth Map Visualization
        if outputs.depth_align is not None and self.vis_depthmap:
            save_depth_map(
                outputs.depth_align.copy(), out_dir, f"{prefix}depth", data_idx, info=False
            )

        # Point Cloud Visualization
        if outputs.pointmap is not None and self.vis_pointmap:
            save_point_cloud(
                outputs.pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                f"{prefix}point",
                data_idx,
                info=False,
            )

        # Ground Truth Point Cloud Visualization
        if outputs.pointmap_gt is not None and self.vis_gt:
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
        if outputs.confidence is not None and self.vis_confidence:
            confidence = outputs.confidence.copy()
            save_depth_map(confidence, out_dir, f"{prefix}conf", data_idx, info=False)
            if self.output_conf_ratio is not None:
                conf_thresh = np.percentile(confidence.reshape(-1), self.output_conf_ratio * 100)
                confidence_mask = (confidence > conf_thresh).astype(float)
            else:
                confidence_mask = (confidence > self.output_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, f"{prefix}conf_mask", data_idx)

        # invalid mask Visualization
        if outputs.invalid_mask is not None:
            invalid_mask = outputs.invalid_mask.copy()
            save_depth_map(invalid_mask, out_dir, f"{prefix}invalid_mask", data_idx, info=False)
            invalid_mask_binary = (invalid_mask > 0).astype(float)
            save_depth_map(invalid_mask_binary, out_dir, f"{prefix}invalid_mask_binary", data_idx)

        if outputs.valid_pointmap is not None and self.vis_pointmap_filtered:
            save_point_cloud(
                outputs.valid_pointmap.copy(),
                (
                    outputs.valid_pointmap_color.copy()
                    if outputs.valid_pointmap_color is not None
                    else None
                ),
                out_dir,
                f"{prefix}valid_point",
                data_idx,
            )

        # Filtered Point Cloud Visualization
        if outputs.filtered_pointmap is not None and self.vis_pointmap_filtered:
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

        if outputs.prompt_pointmap is not None and self.vis_prompt:
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

        if outputs.pointmap_align is not None and self.vis_prompt:
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

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        data_info = meta_data["data_info"][0]
        scene = data_info.get("scene", None)

        if scene is not None:
            scene = scene[0]
            frame_id = int(data_info["frame_id"])
            view_id = int(data_info["view_id"])
            cur_out_dir = os.path.join(out_dir, scene, f"{frame_id:06d}")
            os.makedirs(cur_out_dir, exist_ok=True)
            save_path = os.path.join(cur_out_dir, f"depth_{view_id:06d}.npy")
            points_save_dir = cur_out_dir
            points_save_index = view_id
        else:
            save_path = os.path.join(out_dir, f"depth_{meta_data['data_idx'][0]:06d}.npy")
            points_save_dir = out_dir
            points_save_index = meta_data["data_idx"][0]

        depth_scale = float(data_info["depth_scale"])
        if self.save_depth_np:
            np.save(save_path, outputs.depth_align * depth_scale)
            logging.info(f"output save: {save_path}")

        if outputs.pointmap is not None and self.save_pointmap:
            points_save_path = save_point_cloud(
                outputs.pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                points_save_dir,
                f"point",
                points_save_index,
                info=False,
            )
        else:
            points_save_path = None

        if outputs.filtered_pointmap is not None and self.save_pointmap_filtered:
            filtered_point_save_path = save_point_cloud(
                outputs.filtered_pointmap.copy(),
                (
                    outputs.filtered_pointmap_color.copy()
                    if outputs.filtered_pointmap_color is not None
                    else None
                ),
                points_save_dir,
                f"filtered_point",
                points_save_index,
            )
        else:
            filtered_point_save_path = None

        # intrinsics = outputs.intrinsics
        # if intrinsics is not None:
        #     intrinsics = intrinsics.tolist()
        # if intrinsics is not None and self.save_intrinsics:
        #     if scene is not None:
        #         intrinsics_save_path = os.path.join(
        #             os.path.dirname(save_path), f"intrinsics_{view_id:06d}.json"
        #         )
        #     else:
        #         intrinsics_save_path = os.path.join(
        #             os.path.dirname(save_path), f"intrinsics_{meta_data['data_idx'][0]:06d}.json"
        #         )
        #     with open(intrinsics_save_path, "w") as f:
        #         json.dump(intrinsics, f, indent=2)

        # if outputs.extrinsics is not None and self.save_extrinsics:
        #     extrinsics = outputs.extrinsics
        #     extrinsics = extrinsics.tolist()
        #     if scene is not None:
        #         extrinsics_save_path = os.path.join(
        #             os.path.dirname(save_path), f"extrinsics_{view_id:06d}.json"
        #         )
        #     else:
        #         extrinsics_save_path = os.path.join(
        #             out_dir, f"extrinsics_{meta_data['data_idx'][0]:06d}.json"
        #         )
        #     with open(extrinsics_save_path, "w") as f:
        #         json.dump(extrinsics, f, indent=2)
        # else:
        #     extrinsics = None

        confidence = outputs.confidence
        if confidence is not None and self.save_confidence:
            if scene is not None:
                confidence_save_path = os.path.join(
                    os.path.dirname(save_path), f"conf_{view_id:06d}.npy"
                )
            else:
                confidence_save_path = os.path.join(
                    out_dir, f"conf_{meta_data['data_idx'][0]:06d}.npy"
                )
            np.save(confidence_save_path, confidence)
        else:
            confidence_save_path = None

        if output_meta_dict is not None:
            info = dict(
                rgb=data_info["rgb"],
                depth_scale=depth_scale,
                pred_depth=save_path,
                pred_confidence=confidence_save_path,
                pred_points=points_save_path,
                pred_filtered_points=filtered_point_save_path,
                # save_intrinsics=intrinsics,
                # save_extrinsics=extrinsics,
            )
            if "depth" in data_info:
                info["depth"] = data_info["depth"]
            if "lidar_depth" in data_info:
                info["lidar_depth"] = data_info["lidar_depth"]
            if "cam_in" in data_info:
                info["cam_in"] = data_info["cam_in"]
            if "extrinsics" in data_info:
                info["extrinsics"] = data_info["extrinsics"]

            if scene is None:
                if "files" not in output_meta_dict:
                    output_meta_dict["files"] = list()
                output_meta_dict["files"].append(info)
            else:
                if "mf_files" not in output_meta_dict:
                    output_meta_dict["mf_files"] = dict()
                if scene not in output_meta_dict["mf_files"]:
                    output_meta_dict["mf_files"][scene] = []
                frame_ids = [d["frame_id"] for d in output_meta_dict["mf_files"][scene]]
                if len(frame_ids) == 0:
                    index = None
                else:
                    if frame_id in set(frame_ids):
                        index = frame_ids.index(frame_id)
                    else:
                        index = None

                info["view_id"] = view_id
                info = {k:v for k,v in info.items() if v is not None}
                if index is None:
                    output_meta_dict["mf_files"][scene].append(
                        dict(
                            frame_id=frame_id,
                            views=[info],
                        )
                    )
                else:
                    output_meta_dict["mf_files"][scene][index]["views"].append(info)

        else:
            raise NotImplementedError

    def debug_prompt_interpolate(self, prompt_depth, prompt_depth_interpolate):
        for bi in range(prompt_depth_interpolate.shape[0]):
            prompt_depth_interpolate_norm = (
                prompt_depth_interpolate[bi : bi + 1, 2].detach().cpu().numpy()
            )
            prompt_depth_interpolate_norm = (
                prompt_depth_interpolate_norm - prompt_depth_interpolate_norm.min()
            ) / (prompt_depth_interpolate_norm.max() - prompt_depth_interpolate_norm.min() + 1e-6)
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

    @torch.no_grad()
    def trans_onnx(
        self,
        image,
        prompt_depth=None,
        prompt_scale=None,
        prompt_mask=None,
    ):
        if not hasattr(self, "patch_h"):
            h, w = image.shape[-2:]
            patch_h, patch_w = (
                h // self.model.rgb_encoder.patch_size,
                w // self.model.rgb_encoder.patch_size,
            )
        else:
            patch_h, patch_w = self.patch_h, self.patch_w

        if prompt_depth is not None and prompt_scale is not None and not self.prompt_set_none:
            prompt_depth = self.normalize(prompt_depth, prompt_scale, center=None)

        if prompt_depth is not None and self.model.prompt_encoder is not None:
            prompt_features = self.model.prompt_encoder(prompt_depth)
        else:
            prompt_features = None

        self.model.rgb_encoder.normalize = True
        self.model.rgb_encoder.__class__.forward = self.model.rgb_encoder.onnx_forward
        rgb_features = self.model.rgb_encoder(image, condition=prompt_features)

        features = self.model.decoder(rgb_features, prompt_features)
        results = self.model.head(features, patch_h, patch_w, return_dict=True)

        pointmap_pred, confidence_pred = results["pointmap"], results["confidence"]

        if prompt_depth is not None and prompt_scale is not None and not self.prompt_set_none:
            pointmap_pred = self.denormalize(pointmap_pred, scale=prompt_scale, center=None)

        return pointmap_pred, confidence_pred
