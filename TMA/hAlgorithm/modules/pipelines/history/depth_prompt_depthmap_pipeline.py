import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa
from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.utils import colorize_depth_maps, instantiate_from_config

from .outputs import DepthOutput


class DepthPromptDepthMapPipeline(nn.Module):
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
        target_depth_name="depth",
        target_depth_mask_name="depth_mask",
        prompt_depth_name="sparse_depth",
        prompt_depth_max=None,
        prompt_depth_min=None,
        post_align=False,
        align_depth_name="depth_raw",
        align_depth_mask_name="depth_raw",
        edge_mask_name=None,
        match_input_res=False,
        warmup_stage=False,
        denormalize_before_loss=True,
        loss_weight=1.0,
        edge_loss_weight=1.0,
        dinov2_attention_with_sdpa=True,
        debug=False,
        **kwargs,
    ):
        super().__init__()

        self.warmup_stage = warmup_stage

        self.encoder = encoder
        self.model_config = self.model_configs[self.encoder]
        self.patch_size = patch_size
        self.encoder_pretrain = encoder_pretrain
        self.depth_head_pretrain = head_pretrain

        self.target_depth_name = target_depth_name
        self.target_depth_mask_name = target_depth_mask_name
        self.prompt_depth_name = prompt_depth_name
        self.prompt_depth_max = prompt_depth_max
        self.prompt_depth_min = prompt_depth_min

        self.align_depth_name = align_depth_name
        self.align_depth_mask_name = align_depth_mask_name

        self.edge_mask_name = edge_mask_name

        self.post_align = post_align
        self.match_input_res = match_input_res

        self.denormalize_before_loss = denormalize_before_loss
        self.loss_weight = loss_weight
        self.edge_loss_weight = edge_loss_weight

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
        head["features"] = self.model_config["features"]
        head["out_channels"] = self.model_config["out_channels"]
        self.depth_head = instantiate_from_config(head)
        if self.depth_head_pretrain is not None:
            self.depth_head.load_state_dict(
                torch.load(self.head_pretrain, map_location="cpu", weights_only=False)
            )

        # mean and std of the pretrained dinov2 model
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.debug = debug

    def load_checkpoint(self, ckpt_path):
        res = self.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=False), strict=False
        )
        logging.info(f"Model parameters are loaded from {ckpt_path}")
        logging.info(f"unexpected_keys: {res.unexpected_keys}")
        logging.info(f"missing_keys: {res.missing_keys}")

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

    def save_checkpoint(self, accelerator, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")

        pretrained = accelerator.unwrap_model(self.pretrained)
        depth_head = accelerator.unwrap_model(self.depth_head)

        pretrained_state = {
            f"pretrained.{key}": val for key, val in pretrained.state_dict().items()
        }
        depth_head_state = {
            f"depth_head.{key}": val for key, val in depth_head.state_dict().items()
        }
        state_dict = {**pretrained_state, **depth_head_state}

        torch.save(state_dict, ckpt_path)
        logging.info(f"Model is saved to: {ckpt_path}")

    def share_step(self, x, prompt_depth, prompt_depth_min, prompt_depth_max):
        if prompt_depth is not None:
            prompt_depth = self.normalize(prompt_depth, prompt_depth_min, prompt_depth_max)

        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        # Normalize the image,
        x = ((x + 1) * 0.5 - self._mean) / self._std
        features = self.pretrained(x, self.model_config["layer_idxs"], return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)
        return depth

    def train_step(self, batch):
        self.train()

        image = batch["image"].to(self.device, self.dtype)
        prompt_depth = (
            batch[self.prompt_depth_name].to(self.device, self.dtype)
            if not self.warmup_stage
            else None
        )
        prompt_depth_min = batch[self.prompt_depth_min].to(self.device, self.dtype)[
            :, None, None, None
        ]
        prompt_depth_max = batch[self.prompt_depth_max].to(self.device, self.dtype)[
            :, None, None, None
        ]

        target = batch[self.target_depth_name].to(self.device)
        valid_mask = batch[self.target_depth_mask_name].to(self.device)
        edge_mask = (
            batch[self.edge_mask_name].to(self.device, self.dtype) if self.edge_mask_name else None
        )

        model_pred = self.share_step(image, prompt_depth, prompt_depth_min, prompt_depth_max)
        if self.denormalize_before_loss:
            model_pred = self.denormalize(model_pred, prompt_depth_min, prompt_depth_max)
        else:
            target = self.normalize(target, prompt_depth_min, prompt_depth_max)
            target = target.clip(0, 1)

        if self.debug:
            for i in range(target.shape[0]):
                data_idx = batch["meta_data"]["data_idx"][i]
                depth = target[i].cpu().squeeze().numpy()
                max_val, min_val = depth.max(), depth.min()
                depth_norm = (depth - min_val) / (max_val - min_val)
                depth_pred_colored = colorize_depth_maps(
                    depth_norm, 0, 1, cmap="turbo"
                )  # [3, H, W], value in (0, 1)
                save_path = os.path.join(
                    "debug/DepthPromptDaPipeline", f"target_{data_idx:06d}.jpg"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                depth_pred_colored.save(save_path)

                logging.info(f"visualize save: {save_path}")

                depth = model_pred[i].detach().cpu().squeeze().numpy()
                max_val, min_val = depth.max(), depth.min()
                depth_norm = (depth - min_val) / (max_val - min_val)
                depth_pred_colored = colorize_depth_maps(
                    depth_norm, 0, 1, cmap="turbo"
                )  # [3, H, W], value in (0, 1)
                save_path = os.path.join(
                    "debug/DepthPromptDaPipeline", f"model_pred_{data_idx:06d}.jpg"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                depth_pred_colored.save(save_path)

                logging.info(f"visualize save: {save_path}")

        # Losses
        loss = (model_pred.float() - target.float()).abs()

        # Masked loss
        batch_size = image.shape[0]
        valid_mask = valid_mask.float()
        batch_valid_nums = valid_mask.reshape(batch_size, -1).sum(-1)
        batch_nan_mask = batch_valid_nums == 0
        batch_valid_nums = batch_valid_nums.clip(1)

        mask_loss = (
            (loss * valid_mask).reshape(batch_size, -1) / batch_valid_nums.unsqueeze(-1)
        ).sum(-1)
        mask_loss = mask_loss * (~batch_nan_mask).float()
        mask_loss = mask_loss.mean() * self.loss_weight

        # Edge loss
        if edge_mask is not None:
            edge_mask = edge_mask.float()
            batch_edge_nums = edge_mask.reshape(batch_size, -1).sum(-1)
            batch_nan_edge_mask = batch_edge_nums == 0
            batch_edge_nums = batch_edge_nums.clip(1)

            edge_loss = (
                (loss * edge_mask).reshape(batch_size, -1) / batch_edge_nums.unsqueeze(-1)
            ).sum(-1)
            edge_loss = edge_loss * (~batch_nan_edge_mask).float()
            edge_loss = edge_loss.mean() * self.edge_loss_weight

            loss = mask_loss + edge_loss
            loss_dict = {
                "mask_loss": mask_loss,
                "edge_loss": edge_loss,
            }
            return loss, loss_dict

        else:
            return mask_loss, None

    @torch.no_grad()
    def infer(self, image: torch.Tensor, **kwargs):
        self.eval()

        image = image.to(self.device, self.dtype)
        prompt_depth = (
            kwargs[self.prompt_depth_name].to(self.device, self.dtype)
            if not self.warmup_stage
            else None
        )
        prompt_depth_min = kwargs[self.prompt_depth_min].to(self.device, self.dtype)[
            :, None, None, None
        ]
        prompt_depth_max = kwargs[self.prompt_depth_max].to(self.device, self.dtype)[
            :, None, None, None
        ]

        depth_pred = self.share_step(image, prompt_depth, prompt_depth_min, prompt_depth_max)
        depth_pred = self.denormalize(depth_pred, prompt_depth_min, prompt_depth_max)

        if self.match_input_res or self.post_align:
            depth_gt = kwargs[self.align_depth_name]
            depth_gt = kwargs[self.align_depth_name].squeeze().numpy()
            depth_gt_valid_mask = kwargs[self.align_depth_mask_name].squeeze().numpy()

        if self.match_input_res:
            depth_pred = resize(
                depth_pred,
                depth_gt.shape,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        if self.post_align:
            depth_pred = align_depth_least_square(
                gt_arr=depth_gt,
                pred_arr=depth_pred,
                valid_mask_arr=depth_gt_valid_mask,
                return_scale_shift=False,
                max_resolution=None,
            )

        depth_pred = depth_pred.cpu().squeeze().float().numpy()

        return DepthOutput(depth_align=depth_pred)

    def normalize(self, prompt_depth: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return (prompt_depth - min_val) / (max_val - min_val)

    def denormalize(self, depth: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return depth * (max_val - min_val) + min_val

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

    def save_output(self, outputs, meta_data, out_dir):
        save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.npy")
        np.save(save_path, outputs.depth_align)

        data_path = meta_data["data_path"][0]
        new_data_path = os.path.join(out_dir, f"data_info_with_depth.json")
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

    @torch.no_grad()
    def trans_onnx(self, image, prompt_depth, prompt_depth_min, prompt_depth_max, patch_h, patch_w):
        features = self.pretrained(image, self.model_config["layer_idxs"], return_class_token=True)
        prompt_depth = self.normalize(prompt_depth, prompt_depth_min, prompt_depth_max)
        depth = self.depth_head(features, patch_h, patch_w, prompt_depth)
        depth = self.denormalize(depth, prompt_depth_min, prompt_depth_max)
        return depth
