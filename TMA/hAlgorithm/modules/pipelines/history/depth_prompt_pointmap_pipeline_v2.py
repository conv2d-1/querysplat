import json
import logging
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from hAlgorithm.modules.models.ddi.ddiv2 import DDI
from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa
from hAlgorithm.modules.pipelines.visualize import (
    save_depth_map,
    save_error,
    save_gradient,
    save_point_cloud,
)
from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.utils import (
    colorize_depth_maps,
    instantiate_from_config,
)

from .outputs import DepthOutput


class DepthPromptPointMapPipeline(nn.Module):
    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
            "layer_idxs": [2, 5, 8, 11],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
            "layer_idxs": [2, 5, 8, 11],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
            "layer_idxs": [4, 11, 17, 23],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
            "layer_idxs": [9, 19, 29, 39],
        },
    }

    def __init__(
        self,
        head,
        encoder="vitl",
        patch_size=14,
        encoder_pretrain=None,
        head_pretrain=None,
        target_name="pointmap",
        target_mask_name="depth_mask",
        invalid_mask_target_name=None,
        prompt_name=None,
        prompt_mask_name=None,
        prompt_center_name=None,
        prompt_scale_name=None,
        prompt_diffmap_name=None,
        align_name=None,
        align_mask_name=None,
        match_input_res=False,
        post_align=False,
        l1_loss=None,
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
        warmup_iters=-1,
        target_clip=None,
        dinov2_attention_with_sdpa=True,
        prompt_clear_prob=0,
        prompt_set_none=False,
        prompt_interpolate=False,
        prompt_cat_image=False,
        prompt_cat_mask=False,
        test_prompt_clear=False,
        output_depth2point=False,
        output_conf_thresh=0,
        output_prompt_conf_thresh=0,
        output_global_pointmap=False,
        output_ddi=False,
        head_return_dict=False,
        debug=False,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.model_config = self.model_configs[self.encoder]
        self.patch_size = patch_size
        self.encoder_pretrain = encoder_pretrain
        self.depth_head_pretrain = head_pretrain

        # Build modules
        self.pretrained = torch.hub.load(
            "hAlgorithm/modules/models/facebookresearch_dinov2_main",
            "dinov2_{:}14".format(encoder),
            source="local",
            pretrained=False,
        )
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        if dinov2_attention_with_sdpa:
            self.enable_pytorch_native_sdpa()
        if self.encoder_pretrain is not None:
            self.pretrained.load_state_dict(
                torch.load(self.encoder_pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )

        head["in_channels"] = dim
        head["return_dict"] = head_return_dict
        head["features"] = self.model_config["features"]
        head["out_channels"] = self.model_config["out_channels"]
        self.head_cfg = head
        self.depth_head = instantiate_from_config(head)
        if self.depth_head_pretrain is not None:
            self.depth_head.load_state_dict(
                torch.load(self.head_pretrain, map_location="cpu", weights_only=False)
            )

        # mean and std of the pretrained dinov2 model
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Targe names
        self.target_name = target_name
        self.target_mask_name = target_mask_name
        self.prompt_name = prompt_name
        self.prompt_mask_name = prompt_mask_name
        self.prompt_center_name = prompt_center_name
        self.prompt_scale_name = prompt_scale_name
        self.prompt_diffmap_name = prompt_diffmap_name

        self.align_name = align_name
        self.align_mask_name = align_mask_name
        self.edge_mask_name = edge_mask_name

        self.target_clip = target_clip
        self.prompt_center_name = prompt_center_name
        self.warmup_iters = warmup_iters

        self.post_align = post_align
        self.match_input_res = match_input_res
        self.prompt_clear_prob = prompt_clear_prob
        self.test_prompt_clear = test_prompt_clear
        self.prompt_set_none = prompt_set_none
        self.prompt_interpolate = prompt_interpolate
        self.prompt_cat_image = prompt_cat_image
        self.prompt_cat_mask = prompt_cat_mask

        self.output_depth2point = output_depth2point
        self.output_conf_thresh = output_conf_thresh
        self.output_prompt_conf_thresh = output_prompt_conf_thresh
        self.output_global_pointmap = output_global_pointmap

        # Build Loss
        self.l1_loss = instantiate_from_config(l1_loss)
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
        self.use_extend_loss = any(
            [
                self.conf_loss is not None,
                self.mask_loss is not None,
                self.prompt_conf_loss is not None,
                self.grad_pred_loss is not None,
            ]
        )
        self.head_return_dict = head_return_dict

        # DDI
        self.output_ddi = output_ddi
        self.ddi = DDI()

        self.debug = debug

    def enable_pytorch_native_sdpa(self):
        """
        Enables PyTorch's native scaled dot product attention (SDPA) for the backbone's attention layers.
        """
        for block in self.pretrained.blocks:
            block.attn = wrap_dinov2_attention_with_sdpa(block.attn)

    def get_train_parameters(self):
        return self.parameters()

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        logging.info(f"weight_dtype: {weight_dtype}")

        self.cuda(accelerator.device)
        self.device = accelerator.device
        self.dtype = weight_dtype

        if train_dataloader is None:
            (
                self.pretrained,
                self.depth_head,
            ) = accelerator.prepare(
                self.pretrained,
                self.depth_head,
            )
            return None, None, None

        if accelerator.deepspeed_plugin is not None:
            self, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self, optimizer, train_dataloader, lr_scheduler
            )
        else:
            (
                self.pretrained,
                self.depth_head,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                self.pretrained,
                self.depth_head,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
        return optimizer, train_dataloader, lr_scheduler

    def load_checkpoint(self, ckpt_path=None, state_dict=None):
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state_dict is not None:
            res = self.load_state_dict(state_dict, strict=False)
            logging.info(f"Model parameters are loaded from {ckpt_path}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    def save_checkpoint(self, accelerator, ckpt_dir=None):
        pretrained = accelerator.unwrap_model(self.pretrained)
        depth_head = accelerator.unwrap_model(self.depth_head)

        if ckpt_dir is not None:
            pretrained_state = {
                f"pretrained.{key}": val for key, val in pretrained.state_dict().items()
            }
            depth_head_state = {
                f"depth_head.{key}": val for key, val in depth_head.state_dict().items()
            }
            state_dict = {**pretrained_state, **depth_head_state}

            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")

            torch.save(state_dict, ckpt_path)
            logging.info(f"Model is saved to: {ckpt_path}")
        else:
            pretrained_state = {
                f"pretrained.module.{key}": val for key, val in pretrained.state_dict().items()
            }
            depth_head_state = {
                f"depth_head.module.{key}": val for key, val in depth_head.state_dict().items()
            }
            state_dict = {**pretrained_state, **depth_head_state}

        return state_dict

    def depth_to_point(self, depth, K):
        B, C, H, W = depth.shape
        grid_x, grid_y = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
        points = (
            torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
            .reshape(3, -1)
            .float()
            .to(self.device)
        )
        rays_d = K.inverse().to(self.device) @ points  # (B, 3, HW)
        pts = depth.flatten(2) * rays_d
        depth = pts.reshape(B, 3, H, W)
        return depth

    def share_step(
        self,
        x,
        prompt_depth,
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
        - confidence (Tensor or None): Predicted confidence from the depth head.
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
        if self.head_return_dict:
            results = self.depth_head(features, patch_h, patch_w, prompt_depth, return_dict=True)
            return (
                results["depth"],
                results.get("confidence", None),
                results.get("gradient", None),
                results.get("prompt_confidence", None),
            )
        else:
            depth, confidence = self.depth_head(features, patch_h, patch_w, prompt_depth)
            return depth, confidence, None, None

    def get_loss(
        self,
        name,
        total_iter,
        predict,
        confidence_pred,
        target,
        intrinsics,
        valid_mask,
        edge_mask,
        prompt_mask,
        image,
        **kwargs,
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        predict = predict.float().permute(0, 2, 3, 1).contiguous()
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
                predict,
                target,
                valid_mask,
                name=name,
                image=image,
                confidence=confidence_pred,
            )
            loss = update_loss(loss, l1_Loss, "l1_loss")

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
            loss = update_loss(loss, msa_loss, "msa_loss")

        # Edge loss in warmup
        if (
            (total_iter <= self.warmup_iters)
            and self.edge_loss is not None
            and (edge_mask is not None)
        ):
            edge_loss = self.edge_loss(predict, target, edge_mask, name=name)
            loss = update_loss(loss, edge_loss, "edge_loss")

        # Edge loss
        if (
            total_iter > self.warmup_iters
            and self.edge_msa_loss is not None
            and edge_mask is not None
        ):
            edge_msa_loss = self.edge_msa_loss(
                predict, target, edge_mask, intrinsics=intrinsics, name=name
            )
            loss = update_loss(loss, edge_msa_loss, "edge_msa_loss")

        # Normal loss
        if self.normal_loss is not None:
            if self.normal_loss.start_iter is None or (
                self.normal_loss.start_iter is not None and total_iter > self.normal_loss.start_iter
            ):
                normal_loss = self.normal_loss(predict, target, valid_mask, name=name)
                loss = update_loss(loss, normal_loss, "normal_loss")

        # Prompt sparse depth loss
        if self.prompt_loss is not None and self.prompt_mask_name is not None:
            prompt_loss = self.prompt_loss(predict, target, prompt_mask, name=name)
            loss = update_loss(loss, prompt_loss, "prompt_loss")

        # Depth loss
        if self.depth_loss is not None:
            if getattr(self.depth_loss, "start_iter", None) is None or (
                self.depth_loss.start_iter is not None and total_iter > self.depth_loss.start_iter
            ):
                depth_loss = self.depth_loss(
                    predict[..., -1], target[..., -1], valid_mask, name=name
                )
                loss = update_loss(loss, depth_loss, "depth_loss")

        # Gradient loss
        if self.grad_loss is not None:
            grad_loss = self.grad_loss(predict[..., -1], target[..., -1], valid_mask, name=name)
            loss = update_loss(loss, grad_loss, "grad_loss")

        # Chamfer distance loss
        if self.chamfer_loss is not None:
            chamfer_loss = self.chamfer_loss(
                predict,
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
        depth_pred,
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
    ):
        # Convert the prediction and target tensors to float and permute dimensions for loss calculation.
        depth_pred = depth_pred.float().permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
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
                depth_pred,
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

        if self.mask_loss is not None and invalid_mask_pred is not None:
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

        if self.debug:
            logging.info(
                f'{valid_mask.reshape(valid_mask.shape[0], -1).sum(-1),} {batch["meta_data"]["data_info"]["rgb"]}'
            )

        if self.prompt_name is not None:
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
                    self.debug_prompt_interpolate(prompt_depth_interpolate)

                prompt_depth = prompt_depth_interpolate
                prompt_diffmap = torch.abs(target - prompt_depth)
            else:
                prompt_diffmap = (
                    batch[self.prompt_diffmap_name].to(self.device)
                    if self.prompt_diffmap_name
                    else None
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

            if prompt_mask is not None and self.prompt_cat_mask:
                prompt_depth = torch.cat([prompt_depth, prompt_mask], dim=1)
            if self.prompt_cat_image:
                prompt_depth = torch.cat([prompt_depth, image], dim=1)

            # After normalize target
            if self.prompt_set_none:
                prompt_depth = prompt_scale = prompt_center = prompt_mask = prompt_diffmap = None
        else:
            prompt_depth = prompt_scale = prompt_center = prompt_mask = prompt_diffmap = None

        # Extract edge mask from the batch if it exists.
        edge_mask = batch.get(self.edge_mask_name, None)
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=self.device)

        depth_pred, confidence_pred, gradient_pred, prompt_confidence_pred = self.share_step(
            image,
            prompt_depth,
            prompt_scale,
            prompt_center=prompt_center,
            prompt_clear_prob=self.prompt_clear_prob,
        )
        # DepthMap trans to PointMap
        if depth_pred.shape[1] == 1:
            depth_pred = self.depth_to_point(depth_pred, K=intrinsics)

        if not self.use_extend_loss:
            return self.get_loss(
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
        else:
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
                invalid_mask_pred=None,
                confidence_pred=confidence_pred,
                intrinsics=intrinsics,
                gradient_pred=gradient_pred,
                prompt_confidence_pred=prompt_confidence_pred,
                invalid_mask_target=None,
                valid_mask=valid_mask,
                depth_target=target,
                prompt_diffmap=prompt_diffmap,
                prompt_scale=prompt_scale,
                prompt_center=prompt_center,
            )
            loss = loss + loss_extend
            loss_dict.update(loss_dict_extend)

            return loss, loss_dict

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

        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if "image_show" in kwargs:
            pointmap_color = kwargs["image_show"].float().squeeze(0).numpy().transpose(1, 2, 0)
        else:
            pointmap_color = image.float().squeeze(0).numpy().transpose(1, 2, 0)
            pointmap_color = (pointmap_color + 1) * 0.5 * 255

        image = image.to(self.device, self.dtype)
        intrinsics = kwargs.get("intrinsics", None)  # [1, 3, 3]
        extrinsics = kwargs.get("extrinsics", None)  # [1, 4, 4]

        # Prepare prompt
        if self.prompt_name is not None:
            prompt_depth = kwargs[self.prompt_name].to(self.device, self.dtype)
            prompt_scale = kwargs[self.prompt_scale_name].to(self.device, self.dtype)[
                :, None, None, None
            ]
            prompt_mask = (
                kwargs[self.prompt_mask_name].to(self.device) if self.prompt_mask_name else None
            )
            if self.prompt_interpolate:
                import knn_interpolate

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
                prompt_center = None

            if prompt_mask is not None and self.prompt_cat_mask:
                prompt_depth = torch.cat([prompt_depth, prompt_mask], dim=1)
            if self.prompt_cat_image:
                prompt_depth = torch.cat([prompt_depth, image], dim=1)

        else:
            prompt_depth = prompt_scale = prompt_center = None

        # Use the shared step method to make predictions based on the input image and depth information.
        if self.prompt_set_none:
            pointmap_pred, confidence_pred, gradient_pred, prompt_confidence_pred = self.share_step(
                image,
                prompt_depth=None,
                prompt_scale=None,
                prompt_center=None,
                prompt_clear_prob=1.0,
            )
        else:
            pointmap_pred, confidence_pred, gradient_pred, prompt_confidence_pred = self.share_step(
                image,
                prompt_depth,
                prompt_scale,
                prompt_center=prompt_center,
                prompt_clear_prob=1.0 if self.test_prompt_clear else 0,
            )

            if self.output_ddi:
                tmp_target = kwargs[self.target_name].to(device=self.device)

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

            if not self.test_prompt_clear:
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
        pointmap_pred = (
            pointmap_pred.cpu().float().squeeze(0).numpy().reshape(3, -1).transpose(1, 0)
        )
        depth = pointmap_pred[:, 2].reshape((image.shape[-2], image.shape[-1])).clip(1e-3)

        # Filtered noise pointmap base on confidence_pred
        if confidence_pred is not None:
            confidence_pred = confidence_pred.cpu()[0, 0].float().numpy()
            filtered_pointmap = pointmap_pred[confidence_pred.reshape(-1) > self.output_conf_thresh]
            filtered_pointmap_color = pointmap_color[
                confidence_pred.reshape(-1) > self.output_conf_thresh
            ]
        else:
            filtered_pointmap = filtered_pointmap_color = None

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

        if intrinsics is not None:
            intrinsics = intrinsics.cpu().squeeze(0).numpy()

        if extrinsics is not None:
            extrinsics = extrinsics.cpu().squeeze(0).numpy()

        # Global predicted point cloud
        pointmap_gt_global = pointmap_pred_global = None
        if self.output_global_pointmap and extrinsics is not None:
            extrinsics_inv = np.linalg.inv(extrinsics)
            R = extrinsics_inv[:3, :3]
            T = extrinsics_inv[:3, 3]
            if pointmap_gt is not None:
                pointmap_gt_global = np.dot(R, pointmap_gt.T).T + T
            pointmap_pred_global = np.dot(R, pointmap_pred.T).T + T

        if gradient_pred is not None:
            # gradient_pred = self.denormalize(
            #     gradient_pred, scale=prompt_scale, center=None
            # )
            gradient_pred = gradient_pred.cpu().float().squeeze(0).numpy().transpose(1, 2, 0)

        if prompt_confidence_pred is not None:
            prompt_confidence_pred = prompt_confidence_pred.cpu().float()[0, 0].numpy()

        if prompt_depth is not None:
            prompt_pointmap = (
                prompt_depth.squeeze(0)[:3].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
            )

        return DepthOutput(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_align=depth,
            pointmap=pointmap_pred,
            pointmap_gt=pointmap_gt,
            pointmap_color=pointmap_color,
            pointmap_h=image.shape[-2],
            pointmap_w=image.shape[-1],
            confidence=confidence_pred,
            filtered_pointmap=filtered_pointmap,
            filtered_pointmap_color=filtered_pointmap_color,
            pointmap_gt_global=pointmap_gt_global,
            pointmap_global=pointmap_pred_global,
            depth_grad=gradient_pred,
            input_confidence=prompt_confidence_pred,
            prompt_pointmap=prompt_pointmap,
        )

    def normalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if center is not None:
            return (depth - center) / scale
        else:
            return depth / scale

    def denormalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if center is not None:
            return depth * scale + center
        else:
            return depth * scale

    def visualize(self, outputs, meta_data, out_dir):
        data_idx = meta_data["data_idx"][0]

        # Depth Map Visualization
        if outputs.depth_align is not None:
            save_depth_map(outputs.depth_align.copy(), out_dir, "depth", data_idx, info=True)

        # Point Cloud Visualization
        if outputs.pointmap is not None:
            save_point_cloud(
                outputs.pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                "point",
                data_idx,
                info=True,
            )

        # Ground Truth Point Cloud Visualization
        if outputs.pointmap_gt is not None:
            save_point_cloud(
                outputs.pointmap_gt.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                "point_gt",
                data_idx,
            )

            # Error Map Visualization
            pointmap_shape = (outputs.pointmap_h, outputs.pointmap_w)
            pred_depthmap = outputs.pointmap.copy()[:, 2].reshape(pointmap_shape)
            gt_depthmap = outputs.pointmap_gt.copy()[:, 2].reshape(pointmap_shape)
            save_error(pred_depthmap, gt_depthmap, out_dir, "error", data_idx)

        # Confidence Visualization
        if outputs.confidence is not None:
            confidence = outputs.confidence.copy()
            save_depth_map(confidence, out_dir, "conf", data_idx, info=True)
            confidence_mask = (confidence > self.output_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, "conf_mask", data_idx)

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
                "filtered_point",
                data_idx,
            )

        # Global Point Cloud Visualization
        if outputs.pointmap_global is not None:
            save_point_cloud(
                outputs.pointmap_global.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                "global_point",
                data_idx,
            )

        if outputs.pointmap_gt_global is not None:
            save_point_cloud(
                outputs.pointmap_gt_global.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                "global_point_gt",
                data_idx,
            )

        # if outputs.depth_grad is not None:
        #     save_gradient(outputs.depth_grad.copy(), out_dir, data_idx, info=False)

        # Confidence Visualization
        if outputs.input_confidence is not None:
            confidence = outputs.input_confidence.copy()
            confidence_act = 1 / (1 + np.exp(-confidence))
            save_depth_map(
                confidence_act, out_dir, "prompt_conf", data_idx, min_val=0, max_val=1, info=True
            )
            confidence_mask = (confidence > self.output_prompt_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, "prompt_conf_mask", data_idx)

        if outputs.prompt_pointmap is not None:
            save_point_cloud(
                outputs.prompt_pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                "prompt_point",
                data_idx,
            )

    def save_output(self, outputs, meta_data, out_dir):
        save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.npy")
        np.save(save_path, outputs.depth_align)

        data_path = meta_data["data_path"][0]
        new_data_path = os.path.join(out_dir, "data_info_with_depth.json")
        assert new_data_path != data_path
        if not os.path.exists(new_data_path):
            shutil.copy(data_path, new_data_path)

        with open(new_data_path, "r") as f:
            data_info = json.load(f)
            assert (
                data_info["files"][meta_data["data_idx"]]["rgb"] == meta_data["data_info"]["rgb"][0]
            )
            data_info["files"][meta_data["data_idx"]]["pred_depth"] = save_path
            data_info["files"][meta_data["data_idx"]]["depth_scale"] = meta_data["depth_scale"][
                0
            ].item()

        with open(new_data_path, "w") as f:
            json.dump(data_info, f, indent=2)

        logging.info(f"output save: {save_path}")

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
    ):
        features = self.pretrained(image, self.model_config["layer_idxs"], return_class_token=True)

        if prompt_depth is not None and prompt_scale is not None and not self.prompt_set_none:
            prompt_depth = self.normalize(prompt_depth, prompt_scale, center=None)

        if not hasattr(self, "patch_h"):
            h, w = image.shape[-2:]
            self.patch_h, self.patch_w = h // self.patch_size, w // self.patch_size

        if self.head_return_dict:
            results = self.depth_head(features, self.patch_h, self.patch_w, prompt_depth)
            pointmap_pred, confidence_pred, gradient_pred, prompt_confidence_pred = (
                results["depth"],
                results.get("confidence", None),
                results.get("gradient", None),
                results.get("prompt_confidence", None),
            )
        else:
            pointmap_pred, confidence_pred = self.depth_head(
                features, self.patch_h, self.patch_w, prompt_depth
            )

        if prompt_depth is not None and prompt_scale is not None and not self.prompt_set_none:
            pointmap_pred = self.denormalize(pointmap_pred, scale=prompt_scale, center=None)

        if confidence_pred is None:
            return pointmap_pred
        else:
            return pointmap_pred, confidence_pred
