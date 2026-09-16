from typing import *
from numbers import Number
import importlib
import itertools
import functools
import sys

import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.lingbot_depth.mdm.model.dinov2_rgbd.models.vision_transformer import DinoVisionTransformer
from hAlgorithm.modules.models2.external.lingbot_depth.mdm.model.utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing


class DINOv2_RGBD_Encoder(nn.Module):
    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor
    dim_features: int

    def __init__(
        self,
        backbone: str,
        intermediate_layers: Union[int, List[int]],
        dim_out: int,
        ignore_layers: Union[str, List[str]] = [],
        in_chans: int = 3,
        pretrain: str = None,
        strict: bool = True,
        img_depth_fuse_mode="",
        depth_emb_mode="",
        depth_mask_ratio=0.6,
        img_mask_ratio=0.0,
        patch_size=14,
        normalize=True,
        depth_use_dims=None,
        enable_depth_mask=False,
        **deprecated_kwargs
    ):
        super(DINOv2_RGBD_Encoder, self).__init__()
        """
        'backbone': 'dinov2_vitl14', 
        'intermediate_layers': 1, 
        'dim_out': 1024, 
        'strict': False, 
        'depth_emb_mode': 'conv_1c', 
        'img_depth_fuse_mode': 'cat_token'
        """

        self.intermediate_layers = intermediate_layers
        self.strict = strict
        self.ignore_layers = ignore_layers
        self.img_mask_ratio = img_mask_ratio
        # Load the backbone
        self.hub_loader = getattr(importlib.import_module("hAlgorithm.modules.models2.external.lingbot_depth.mdm.model.dinov2_rgbd.hub.backbones", __package__), backbone)
        self.backbone_name = backbone
        self.backbone = self.hub_loader(
            pretrained=False, in_chans=in_chans, img_depth_fuse_mode=img_depth_fuse_mode, depth_emb_mode=depth_emb_mode, depth_mask_ratio=depth_mask_ratio, img_mask_ratio=img_mask_ratio
        )

        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)

        if img_mask_ratio > 0:
            self.mask_token_mae = nn.Parameter(torch.zeros(1, 1, self.dim_features))
            torch.nn.init.normal_(self.mask_token_mae, std=0.02)

        if dim_out is None:
            self.output_projections = None
        else:
            self.output_projections = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.dim_features,
                        out_channels=dim_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for _ in range(self.num_features)
                ]
            )

        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.patch_size = patch_size
        self.normalize = normalize
        self.depth_use_dims = depth_use_dims
        self.enable_depth_mask = enable_depth_mask

        self.pretrain = pretrain

        if self.pretrain is not None:
            state_dict = torch.load(self.pretrain, map_location="cpu", weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]
                state_dict = {key.replace("encoder.backbone.", ""):val for key, val in state_dict.items() if key.startswith("encoder.backbone.")}
            res = self.backbone.load_state_dict(
                state_dict,
                strict=self.strict,
            )
            logging.info(f"DINOv2_RGBD_Encoder, load pretrain {self.pretrain}")
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    # def init_weights(self):
    #     pretrained_backbone_state_dict = self.hub_loader(pretrained=True).state_dict()
    #     ignore_layers = []
    #     if isinstance(self.ignore_layers, str):
    #         ignore_layers = [self.ignore_layers]
    #     else:
    #         ignore_layers = self.ignore_layers

    #     if len(ignore_layers) == 0:
    #         self.backbone.load_state_dict(pretrained_backbone_state_dict, strict=self.strict)
    #     else:
    #         state_dict = {}
    #         for k, v in pretrained_backbone_state_dict.items():
    #             is_ignore = False
    #             for ig_k in ignore_layers:
    #                 if ig_k in k:
    #                     is_ignore = True
    #                     break
    #             if not is_ignore:
    #                 state_dict[k] = v
    #         self.backbone.load_state_dict(state_dict, strict=self.strict)


    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(
        self,
        x: torch.Tensor,
        prompt_depth: torch.Tensor = None,
        # token_rows: Union[int, torch.LongTensor],
        # token_cols: Union[int, torch.LongTensor],
        # return_class_token: bool = False,
        remap_depth_in: str = "linear",
        meta_data=None,
        **kwargs
    ):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        meta_data["patch_h"] = patch_h
        meta_data["patch_w"] = patch_w

        # image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14), mode="bilinear", align_corners=False, antialias=not self.onnx_compatible_mode)
        # depth_14 = F.interpolate(depth, (token_rows * 14, token_cols * 14), mode="nearest")

        image_14 = x
        if self.depth_use_dims is None:
            depth_14 = prompt_depth
        else:
            depth_14 = prompt_depth[:, self.depth_use_dims]

        if self.normalize:
            image_14 = (image_14 + 1) * 0.5
        image_14 = (image_14 - self.image_mean) / self.image_std

        # set invalid depth value to zero
        depth_14[torch.isinf(depth_14)] = 0.0
        depth_14[torch.isnan(depth_14)] = 0.0
        # dmask_14 = (depth_14 > 0.01).detach()
        # depth_14 = depth_14 * dmask_14.float()

        if remap_depth_in == "linear":
            pass  # do nothing
        elif remap_depth_in == "log":
            depth_14 = torch.log(depth_14)
            # depth_14[~dmask_14] = 0.0
            depth_14 = torch.nan_to_num(depth_14, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise NotImplementedError

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers_mae(x_img=x, x_depth=depth_14, n=self.intermediate_layers, return_class_token=True, enable_depth_mask=self.enable_depth_mask, **kwargs)

        assert self.img_mask_ratio == 0, "img_mask_ratio is not supported in this encoder"

        if isinstance(features[0][0], list):
            num_valid_tokens = patch_h * patch_w
            features = tuple((torch.cat([feat[:, :num_valid_tokens].contiguous() for feat in feats], dim=0), torch.cat(cls_tokens, dim=0)) for feats, cls_tokens in features)

        # Project features to the desired dimensionality
        # x = torch.stack([
        #     proj(feat.permute(0, 2, 1)[:, :, :patch_h * patch_w].unflatten(2, (patch_h, patch_w)).contiguous())
        #         for proj, (feat, clstoken) in zip(self.output_projections, features)
        # ], dim=1).sum(dim=1)
        # cls_token = features[-1][1]

        # if return_class_token:
        #     return x, cls_token, None, None
        # else:
        #     return x, None, None

        if self.output_projections is not None:
            x = [proj(feat.permute(0, 2, 1)[:, :, : patch_h * patch_w].unflatten(2, (patch_h, patch_w)).contiguous()) for proj, (feat, clstoken) in zip(self.output_projections, features)]
            x = [xi.view(*xi.shape[:2], patch_h * patch_w).permute(0, 2, 1).contiguous() for xi in x]
        else:
            x = [feats[0] for feats in features]
        
        return x
