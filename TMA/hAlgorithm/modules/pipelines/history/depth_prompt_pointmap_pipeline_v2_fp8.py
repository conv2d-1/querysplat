import logging
import random

import torch
import torch.nn as nn
import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te
from accelerate.utils import FP8RecipeKwargs, set_seed
from accelerate.utils.transformer_engine import convert_model
from transformer_engine.common.recipe import DelayedScaling

from hAlgorithm.modules.models.moge.model.utils import (
    wrap_dinov2_attention_with_flash_attention3,
    wrap_dinov2_attention_with_sdpa,
)

from .depth_prompt_pointmap_pipeline_v2 import (
    DepthPromptPointMapPipeline as DepthPromptPointMapPipelineV2,
)


class CombinedModel(nn.Module):
    """
    Defines a complete model that includes both the pretrained model and the depth head,
    so that only a single model is passed to accelerate.prepare.
    """

    def __init__(self, pretrained, depth_head, fp8_enabled=True):
        super().__init__()
        self.pretrained = pretrained
        self.depth_head = depth_head
        self.fp8_enabled = fp8_enabled
        FP8_RECIPE_KWARGS = {
            "fp8_format": te_recipe.Format.HYBRID,
            "amax_history_len": 32,
            "amax_compute_algo": "max",
        }
        self.fp8_recipe = DelayedScaling(**FP8_RECIPE_KWARGS)

    def forward(
        self,
        x,
        layer_idxs,
        return_class_token,
        patch_h,
        patch_w,
        prompt_depth,
        head_return_dict=False,
    ):
        # Perform forward pass using FP8.
        with te.fp8_autocast(enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe):
            features = self.pretrained(x, layer_idxs, return_class_token=return_class_token)
            if head_return_dict:
                results = self.depth_head(
                    features, patch_h, patch_w, prompt_depth, return_dict=True
                )
                return (
                    results["depth"],
                    results.get("confidence", None),
                    results.get("gradient", None),
                )
            else:
                depth, confidence = self.depth_head(features, patch_h, patch_w, prompt_depth)
                return depth, confidence, None


class DepthPromptPointMapPipeline(DepthPromptPointMapPipelineV2):
    """
    This class prepares the model for acceleration by configuring the appropriate data types
    and converting model layers when using FP8 precision.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        weight_dtype = torch.float32
        # For operators that do not support FP8, default to FP16.
        if accelerator.mixed_precision in ["fp16", "fp8"]:
            weight_dtype = torch.float16
        elif accelerator.mixed_precision in ["bf16"]:
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
            self.combined_model = CombinedModel(
                self.pretrained, self.depth_head, accelerator.mixed_precision == "fp8"
            )
            (
                self.combined_model,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                self.combined_model,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
            if accelerator.mixed_precision == "fp8":
                # Recursively converts the linear and layer normalization layers of the model
                # to their transformers_engine counterparts.
                # Reference: https://huggingface.co/docs/accelerate/en/package_reference/fp8
                with torch.no_grad():
                    convert_model(self.combined_model)

        return optimizer, train_dataloader, lr_scheduler

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

        return self.combined_model(
            x,
            self.model_config["layer_idxs"],
            True,
            patch_h,
            patch_w,
            prompt_depth,
            head_return_dict=self.head_return_dict,
        )

    def enable_pytorch_native_sdpa(self):
        """
        Enables PyTorch's native scaled dot product attention (SDPA) for the backbone's attention layers.
        """
        try:
            # check if flash attn 3 is installed
            from flash_attn_interface import flash_attn_func

            for block in self.pretrained.blocks:
                block.attn = wrap_dinov2_attention_with_flash_attention3(block.attn)
        except:
            # use pytorch sdpa instead
            for block in self.pretrained.blocks:
                block.attn = wrap_dinov2_attention_with_sdpa(block.attn)
