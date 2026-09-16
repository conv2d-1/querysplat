import json
import logging
import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn

from hAlgorithm.modules.models.ddi.ddiv2 import DDI
from hAlgorithm.modules.models.facebookresearch_dinov2_main.config import model_configs
from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa
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
        pointmap_match_input_res=False,
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
        prompt_set_none=False,
        prompt_interpolate=False,
        output_depth2point=False,
        output_conf_thresh=0,
        output_prompt_conf_thresh=0,
        output_global_pointmap=False,
        output_ddi=False,
        freeze_modules=[],
        debug=False,
        custom_model_config=None,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.model_config = model_configs[self.encoder]
        self.patch_size = patch_size
        self.encoder_pretrain = encoder_pretrain
        self.dinov2_attention_with_sdpa = dinov2_attention_with_sdpa

        self.head_cfg = head
        self.depth_head_pretrain = head_pretrain

        # Store model modification config
        self.custom_model_config = custom_model_config or {}

        self.module_names = []
        self.build_encoder()
        self.build_decoder()

        # mean and std of the pretrained dinov2 model
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Targe names
        self.warmup_iters = warmup_iters

        self.target_name = target_name
        self.target_mask_name = target_mask_name
        self.target_clip = target_clip
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
        self.output_prompt_conf_thresh = output_prompt_conf_thresh
        self.output_global_pointmap = output_global_pointmap

        self.output_ddi = output_ddi
        self.ddi = DDI()

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

        self.freeze_modules = freeze_modules

        self.cache_dict = dict()
        self.debug = debug

    def enable_pytorch_native_sdpa(self):
        """
        Enables PyTorch's native scaled dot product attention (SDPA) for the backbone's attention layers.
        """
        for block in self.pretrained.blocks:
            block.attn = wrap_dinov2_attention_with_sdpa(block.attn)

    def build_encoder(self):
        self.pretrained = torch.hub.load(
            "hAlgorithm/modules/models/facebookresearch_dinov2_main",
            "dinov2_{:}14".format(self.encoder),
            source="local",
            pretrained=False,
        )
        if self.dinov2_attention_with_sdpa:
            self.enable_pytorch_native_sdpa()
        if self.encoder_pretrain is not None:
            self.pretrained.load_state_dict(
                torch.load(self.encoder_pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )

        self.encoder_output_dim = self.pretrained.blocks[0].attn.qkv.in_features

        # Add debug logging
        if self.custom_model_config:
            logging.info(f"Model modification config: {self.custom_model_config}")

            # Apply model modifications if config is provided
            from hAlgorithm.modules.models.transformer.custom_vit import CustomViT

            self.pretrained = CustomViT(self.pretrained, **self.custom_model_config)

            # Log specific configurations if present
            if "norm_type" in self.custom_model_config:
                logging.info(f"Using normalization type: {self.custom_model_config['norm_type']}")
            if "attention_type" in self.custom_model_config:
                logging.info(f'Using attention type: {self.custom_model_config["attention_type"]}')
            if (
                "lora_config" in self.custom_model_config
                and self.custom_model_config["lora_config"] is not None
            ):
                logging.info(
                    f"Applying LoRA with config: {self.custom_model_config['lora_config']}"
                )

                # Note: Parameter counting is already done inside CustomViT when LoRA is applied
                # No need to count parameters here as it's already logged in CustomViT

        else:
            logging.info("No model modifications applied, using standard model")

        self.module_names.append("pretrained")

    def build_decoder(self):
        self.head_cfg["in_channels"] = self.encoder_output_dim
        if "features" not in self.head_cfg:
            self.head_cfg["features"] = self.model_config["features"]
        if "out_channels" not in self.head_cfg:
            self.head_cfg["out_channels"] = self.model_config["out_channels"]

        self.depth_head = instantiate_from_config(self.head_cfg)

        if self.depth_head_pretrain is not None:
            self.depth_head.load_state_dict(
                torch.load(self.head_pretrain, map_location="cpu", weights_only=False)
            )
        self.module_names.append("depth_head")

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
        state_dict = dict()
        if ckpt_dir is not None:
            for module_name in self.module_names:
                state = {
                    f"{module_name}.{key}": val
                    for key, val in accelerator.unwrap_model(getattr(self, module_name))
                    .state_dict()
                    .items()
                }
                state_dict.update(state)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")

            torch.save(state_dict, ckpt_path)
            logging.info(f"Model is saved to: {ckpt_path}")
        else:
            for module_name in self.module_names:
                state = {
                    f"{module_name}.module.{key}": val
                    for key, val in accelerator.unwrap_model(getattr(self, module_name))
                    .state_dict()
                    .items()
                }
                state_dict.update(state)

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

        intrinsics = batch.get("intrinsics", None)
        extrinsics = batch.get("extrinsics", None)  # [n, 4, 4]

        if intrinsics is not None:
            intrinsics = intrinsics.to(device=self.device)
        if extrinsics is not None:
            extrinsics = extrinsics.to(device=self.device)

        image_show = batch.get("image_show", None)
        if image_show is not None:
            image_show = image_show.float().squeeze(0).numpy().transpose(1, 2, 0)

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
                    self.debug_prompt_interpolate(prompt_depth, prompt_depth_interpolate)

                prompt_depth = prompt_depth_interpolate
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

            # Normalize the prompt depth if it is provided, using the maximum range and center values.
            prompt_depth_norm = self.normalize(prompt_depth, prompt_scale, prompt_center)

        else:
            prompt_depth = prompt_depth_norm = prompt_scale = prompt_center = prompt_mask = (
                prompt_diffmap
            ) = None

        if self.target_name is not None and self.target_name in batch:
            target = batch[self.target_name].to(device=self.device)
            valid_mask = batch[self.target_mask_name].to(self.device)

            if self.debug:
                logging.info(
                    f'{valid_mask.reshape(valid_mask.shape[0], -1).sum(-1),} {batch["meta_data"]["data_info"]["rgb"]}'
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

    def share_step(self, x, prompt_depth, prompt_mask=None):
        """
        This method processes an input image `x` along with optional depth information `prompt_depth`.
        It normalizes the depth information if provided, prepares the image for the encoder,
        extracts features using a pretrained model, and then predicts depth using the depth head.

        Parameters:
        - x (Tensor): The input image tensor.
        - prompt_depth (Tensor or None): Optional tensor containing depth information to guide the prediction.

        Returns:
        - depth (Tensor): Predicted depth map from the depth head.
        - confidence (Tensor or None): Predicted confidence from the depth head.
        """
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        # Normalize the image. Assuming `x` is in [-1, 1], this converts it to [0, 1] and then normalizes using mean and std.
        x = ((x + 1) * 0.5 - self._mean) / self._std

        features = self.pretrained(x, self.model_config["layer_idxs"], return_class_token=True)
        results = self.depth_head(features, patch_h, patch_w, prompt_depth, return_dict=True)

        return (
            results["depth"],
            results.get("confidence", None),
            results.get("gradient", None),
            results.get("prompt_confidence", None),
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
        if self.normal_loss is not None:
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
        if self.depth_loss is not None:
            if getattr(self.depth_loss, "start_iter", None) is None or (
                self.depth_loss.start_iter is not None and total_iter > self.depth_loss.start_iter
            ):
                depth_loss = self.depth_loss(
                    pointmap_pred[..., -1], target[..., -1], valid_mask, name=name
                )
                loss = update_loss(loss, depth_loss, "depth_loss")

        # Gradient loss
        if self.grad_loss is not None:
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

    def freeze(self):
        for module_name in self.freeze_modules:
            if module_name in self.module_names:
                module = getattr(self, module_name)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

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
        self.freeze()

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

        pointmap_pred, confidence_pred, gradient_pred, prompt_confidence_pred = self.share_step(
            image,
            prompt_depth_norm,
            prompt_mask=prompt_mask,
        )
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
        )

        if self.use_extend_loss:
            loss_extend, loss_dict_extend = self.get_extend_loss(
                name=name,
                pointmap_pred=pointmap_pred,
                invalid_mask_pred=None,
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

        return loss, loss_dict

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

        # Convert the color pointmap to a NumPy array and normalize it to the [0, 255] range.
        if image_show is not None:
            pointmap_color = image_show
        else:
            pointmap_color = image.cpu().float().squeeze(0).numpy().transpose(1, 2, 0)
            pointmap_color = (pointmap_color + 1) * 0.5 * 255

        # Use the shared step method to make predictions based on the input image and depth information.
        pointmap_pred, confidence_pred, gradient_pred, prompt_confidence_pred = self.share_step(
            image,
            prompt_depth_norm,
            prompt_mask=prompt_mask,
        )

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

        # Denormalize the predicted pointmap using the provided max range and center values.
        pointmap_pred = self.denormalize(pointmap_pred, scale=prompt_scale, center=prompt_center)

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
        pointmap_pred_align = pointmap_color_align = None
        filtered_pointmap_pred_align = filtered_pointmap_color_align = None
        if self.match_input_res:
            depth_gt = batch[self.align_name].squeeze().numpy()
            h, w = depth_gt.shape
            depth = cv2.resize(
                depth,
                dsize=(w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            ratio_h, ratio_w = 1.0 * h / pointmap_h, 1.0 * w / pointmap_w
            intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio_w
            intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio_h
            intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio_w
            intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio_h

            if confidence_pred is not None:
                confidence_pred = cv2.resize(
                    confidence_pred,
                    dsize=(w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.pointmap_match_input_res:
                pointmap_pred_align = self.depth_to_point(
                    torch.from_numpy(depth)[None, None],
                    K=intrinsics.cpu(),
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
                depth_gt_valid_mask = batch[self.align_mask_name].squeeze().numpy()
                depth = align_depth_least_square(
                    gt_arr=depth_gt,
                    pred_arr=depth,
                    valid_mask_arr=depth_gt_valid_mask,
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
            _, _, prompt_h, prompt_w = prompt_depth.shape
            prompt_pointmap = (
                prompt_depth.squeeze(0)[:3].cpu().float().numpy().transpose(1, 2, 0).reshape(-1, 3)
            )
        else:
            prompt_h = prompt_w = None

        return DepthOutput(
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
        )

    def normalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if center is not None:
            depth = depth - center
        if scale is not None:
            depth = depth / scale
        return depth

    def denormalize(self, depth: torch.Tensor, scale: torch.Tensor, center: torch.Tensor):
        if scale is not None:
            depth = depth * scale
        if center is not None:
            depth = depth + center
        return depth

    def visualize(self, outputs, meta_data, out_dir):
        data_idx = meta_data["data_idx"][0]
        if "rgb" in meta_data["data_info"]:
            rgb = meta_data["data_info"]["rgb"][0]
            logging.info(f"vis {data_idx}, out_dir:{out_dir}, rgb:{rgb}")
        else:
            logging.info(f"vis {data_idx}, out_dir:{out_dir}")

        # Depth Map Visualization
        if outputs.depth_align is not None:
            save_depth_map(outputs.depth_align.copy(), out_dir, "depth", data_idx, info=False)

        # Point Cloud Visualization
        if outputs.pointmap is not None:
            save_point_cloud(
                outputs.pointmap.copy(),
                (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                out_dir,
                "point",
                data_idx,
                info=False,
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
            save_depth_map(confidence, out_dir, "conf", data_idx, info=False)
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
                confidence_act, out_dir, "prompt_conf", data_idx, min_val=0, max_val=1, info=False
            )
            confidence_mask = (confidence > self.output_prompt_conf_thresh).astype(float)
            save_depth_map(confidence_mask, out_dir, "prompt_conf_mask", data_idx)

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
                    "prompt_point",
                    data_idx,
                )
            else:
                save_point_cloud(
                    outputs.prompt_pointmap.copy(),
                    (outputs.pointmap_color.copy() if outputs.pointmap_color is not None else None),
                    out_dir,
                    "prompt_point",
                    data_idx,
                )

            save_depth_map(
                outputs.prompt_pointmap.copy().reshape(outputs.prompt_h, outputs.prompt_w, 3)[
                    :, :, 2
                ],
                out_dir,
                "prompt_point",
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
                "point_align",
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
                "filtered_point_align",
                data_idx,
            )

    def save_output(self, outputs, meta_data, out_dir, output_meta_dict=None):
        scene = meta_data["data_info"].get("scene", None)

        if scene is not None:
            scene = scene[0]
            frame_id = int(meta_data["data_info"]["frame_id"][0])
            view_id = int(meta_data["data_info"]["view_id"][0])
            cur_out_dir = os.path.join(out_dir, scene, f"{frame_id:06d}")
            os.makedirs(cur_out_dir, exist_ok=True)
            save_path = os.path.join(cur_out_dir, f"detph_{view_id:06d}.npy")
        else:
            save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.npy")

        np.save(save_path, outputs.depth_align)
        logging.info(f"output save: {save_path}")

        intrinsics = outputs.intrinsics
        extrinsics = outputs.extrinsics
        if intrinsics is not None:
            intrinsics = intrinsics.tolist()
            if scene is not None:
                intrinsics_save_path = os.path.join(
                    os.path.dirname(save_path), f"intrinsics_{view_id:06d}.json"
                )
            else:
                intrinsics_save_path = os.path.join(
                    os.path.dirname(save_path), f"intrinsics_{meta_data['data_idx'][0]:06d}.json"
                )
            with open(intrinsics_save_path, "w") as f:
                json.dump(intrinsics, f, indent=2)

        if extrinsics is not None:
            extrinsics = extrinsics.tolist()
            if scene is not None:
                extrinsics_save_path = os.path.join(
                    os.path.dirname(save_path), f"extrinsics_{view_id:06d}.json"
                )
            else:
                extrinsics_save_path = os.path.join(
                    out_dir, f"extrinsics_{meta_data['data_idx'][0]:06d}.json"
                )
            with open(extrinsics_save_path, "w") as f:
                json.dump(extrinsics, f, indent=2)

        confidence = outputs.confidence
        if confidence is not None:
            if scene is not None:
                confidence_save_path = os.path.join(
                    os.path.dirname(save_path), f"conf_{view_id:06d}.npy"
                )
            else:
                confidence_save_path = os.path.join(
                    out_dir, f"conf_{meta_data['data_idx'][0]:06d}.npy"
                )
            np.save(confidence_save_path, confidence)

        if output_meta_dict is not None:
            info = dict(
                rgb=meta_data["data_info"]["rgb"][0],
                depth_scale=meta_data["data_info"]["depth_scale"][0].item(),
                pred_depth=save_path,
                save_intrinsics=intrinsics,
                save_extrinsics=extrinsics,
                cam_in=[d[0].item() for d in meta_data["data_info"]["cam_in"]],
            )
            if "depth" in meta_data["data_info"]:
                info["depth"] = meta_data["data_info"]["depth"][0]
            if "lidar_depth" in meta_data["data_info"]:
                info["lidar_depth"] = meta_data["data_info"]["lidar_depth"][0]

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
                info["extrinsics"] = [
                    [dd.item() for dd in d] for d in meta_data["data_info"]["extrinsics"]
                ]

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
            data_path = meta_data["data_path"][0]
            new_data_path = os.path.join(out_dir, "data_info_with_depth.json")
            assert new_data_path != data_path
            if not os.path.exists(new_data_path):
                shutil.copy(data_path, new_data_path)

            with open(new_data_path, "r") as f:
                data_info = json.load(f)
                assert (
                    data_info["files"][meta_data["data_idx"]]["rgb"]
                    == meta_data["data_info"]["rgb"][0]
                )
                data_info["files"][meta_data["data_idx"]]["pred_depth"] = save_path
                data_info["files"][meta_data["data_idx"]]["depth_scale"] = meta_data["depth_scale"][
                    0
                ].item()

            with open(new_data_path, "w") as f:
                json.dump(data_info, f, indent=2)

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
        features = self.pretrained(image, self.model_config["layer_idxs"], return_class_token=True)

        if prompt_depth is not None and prompt_scale is not None and not self.prompt_set_none:
            prompt_depth = self.normalize(prompt_depth, prompt_scale, center=None)

        if not hasattr(self, "patch_h"):
            h, w = image.shape[-2:]
            self.patch_h, self.patch_w = h // self.patch_size, w // self.patch_size

        results = self.depth_head(features, self.patch_h, self.patch_w, prompt_depth)
        pointmap_pred, confidence_pred, gradient_pred, prompt_confidence_pred = (
            results["depth"],
            results.get("confidence", None),
            results.get("gradient", None),
            results.get("prompt_confidence", None),
        )

        if prompt_depth is not None and prompt_scale is not None and not self.prompt_set_none:
            pointmap_pred = self.denormalize(pointmap_pred, scale=prompt_scale, center=None)

        if confidence_pred is None:
            return pointmap_pred
        else:
            return pointmap_pred, confidence_pred

    def merge_lora_weights(self):
        """Merge LoRA weights into the model if LoRA is applied."""
        if hasattr(self, "pretrained") and isinstance(self.pretrained, CustomViT):
            self.pretrained.merge_lora_weights()
            logging.info("Successfully merged LoRA weights in the pipeline model")
        return self
