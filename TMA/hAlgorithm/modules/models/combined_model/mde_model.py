
import torch
import torch.nn.functional as F
from hAlgorithm.utils import instantiate_from_config
from hAlgorithm.modules.utils.alignment import (
    align_least_square_batch, 
    align_points_scale_z_shift,
)

import logging
from .base import CombinedModel


class MDEModel(CombinedModel):
    def __init__(
        self,
        mde_model,
        conf_head=None,
        freeze_modules=[],
        load_modules=None,
        align_prompt=None,
    ):
        self.mde_model_cfg = mde_model
        self.conf_head_cfg = conf_head
        self.align_prompt = align_prompt

        super().__init__(
            rgb_encoder=None,
            prompt_encoder=None,
            decoder=None,
            head=None,
            freeze_modules=freeze_modules,
        )
        if load_modules is not None:
            self.load_modules_pretrain(load_modules)

    def load_modules_pretrain(self, load_modules):
        assert isinstance(load_modules, dict)
        for module_name, module_ckpt in load_modules.items():
            if module_name in self.module_names:
                module = getattr(self, module_name)
                state_dict = torch.load(module_ckpt, map_location="cpu", weights_only=False)
                res = module.load_state_dict(state_dict, strict=False)
                logging.info(f"{module_name} parameters are loaded from {module_ckpt}")
                logging.info(f"unexpected_keys: {res.unexpected_keys}")
                logging.info(f"missing_keys: {res.missing_keys}")

    def build_modules(self):
        super().build_modules()
        self.mde_model = instantiate_from_config(self.mde_model_cfg)
        if self.mde_model is not None:
            self.module_names.append("mde_model")
        self.conf_head = instantiate_from_config(self.conf_head_cfg)
        if self.conf_head is not None:
            self.module_names.append("conf_head")

    def forward(self, x, prompt_depth, with_freeze=False, meta_data=None, **kwargs):
        """
        Processes a single frame of input data.

        Parameters:
        - x (Tensor): The input image tensor.
        - prompt_depth (Tensor or None): Optional depth information to guide the prediction.

        Returns:
        - results (Dict): A dictionary containing the output predictions.
        """
        if with_freeze:
            self.freeze()

        h, w = x.shape[-2:]
        if not self.fp8_enabled:
            results = self.mde_model(x, return_dict=True)
            rgb_features = results.pop('features', None)
            
            if self.conf_head is not None:
                confidence = self.conf_head(rgb_features)[-1]
                confidence = F.interpolate(confidence, (h, w), mode='bilinear', align_corners=False, antialias=False)
                results["confidence"] = confidence
            else:
                results["confidence"] = None

            if self.align_prompt is not None:
                if self.align_prompt in ['align_points_scale_z_shift']:
                    from hAlgorithm.modules.models.mde_model.moge.utils.geometry_torch import mask_aware_nearest_resize
                    prompt_mask = prompt_depth[:, -1] > 0
                    (pred_points_lr, prompt_points_lr), lr_mask = mask_aware_nearest_resize((results['pointmap'].permute(0,2,3,1), prompt_depth.permute(0,2,3,1)), mask=prompt_mask, size=(64, 64))
                    scale, shift = align_points_scale_z_shift(
                        pred_points_lr.flatten(-3, -2),
                        prompt_points_lr.flatten(-3, -2),
                        lr_mask.flatten(-2, -1)
                    )
                    scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    shift = shift.unsqueeze(-1).unsqueeze(-1)
                    breakpoint()
                    results['pointmap'] = results['pointmap'] * scale + shift
                elif self.align_prompt in [True, 'align_depths_scale']:
                    depthmap = results['pointmap'][:, -1,:, :]
                    prompt_depthmap = prompt_depth[:, -1, :, :]
                    _, scale, shift = align_least_square_batch(
                        depthmap, prompt_depthmap, prompt_depthmap > 0, return_scale_shift=True
                    )
                    results['pointmap'] = results['pointmap'] * scale.unsqueeze(1)

        return results
