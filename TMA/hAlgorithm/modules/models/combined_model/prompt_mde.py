"""
Inspired by https://arxiv.org/pdf/2504.01596, combining MDE model with a dToF sensor.
"""

import logging

import torch
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config

from .base import CombinedModel


class MDECombinedModel(CombinedModel):
    def __init__(
        self,
        rgb_encoder=None,
        prompt_encoder=None,
        mde_model=None,
        decoder=None,
        head=None,
        freeze_modules=[],
        load_modules=None,
    ):
        self.mde_model_cfg = mde_model

        super(MDECombinedModel, self).__init__(
            rgb_encoder=rgb_encoder,
            prompt_encoder=prompt_encoder,
            decoder=decoder,
            head=head,
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
        super(MDECombinedModel, self).build_modules()
        self.mde_model = instantiate_from_config(self.mde_model_cfg)
        if self.mde_model is not None:
            self.module_names.append("mde_model")

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
        patch_h, patch_w = h // self.rgb_encoder.patch_size, w // self.rgb_encoder.patch_size

        if not self.fp8_enabled:
            rgb_features = self.rgb_encoder(x, meta_data=meta_data)
            if self.prompt_encoder is not None and prompt_depth is not None:
                if self.mde_model is not None:
                    prompt_depth = self.mde_model(x, prompt_depth)
                prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)
            else:
                prompt_features = None
            features = self.decoder(rgb_features, prompt_features, meta_data=meta_data)
            results = self.head(features, patch_h, patch_w, return_dict=True, meta_data=meta_data)
            results["prompt_depth"] = prompt_depth
        return results


class ConditionedMDECombinedModel(MDECombinedModel):
    def __init__(self, **kwargs):
        super(ConditionedMDECombinedModel, self).__init__(**kwargs)

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
        patch_h, patch_w = h // self.rgb_encoder.patch_size, w // self.rgb_encoder.patch_size

        prompt_features = None
        if not self.fp8_enabled:
            if prompt_depth is not None:
                if self.mde_model is not None:
                    prompt_depth = self.mde_model(x, prompt_depth)

                if self.prompt_encoder is not None:
                    prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)
                else:
                    prompt_features = prompt_depth

            rgb_features = self.rgb_encoder(x, condition=prompt_features, meta_data=meta_data)
            features = self.decoder(rgb_features, prompt_features, meta_data=meta_data)
            results = self.head(features, patch_h, patch_w, return_dict=True, meta_data=meta_data)
            results["prompt_depth"] = prompt_depth
        return results


class MDECombinedModelShare(CombinedModel):
    def __init__(
        self,
        rgb_encoder=None,
        prompt_encoder=None,
        mde_head=None,
        decoder=None,
        head=None,
        freeze_modules=[],
        load_modules=None,
        prompt_mde=False,
        mde_act_norm="sigmoid",
        renormalize=False,
    ):
        self.mde_head_cfg = mde_head
        self.prompt_mde = prompt_mde
        self.renormalize = renormalize
        self.mde_act_norm = mde_act_norm

        super(MDECombinedModelShare, self).__init__(
            rgb_encoder=rgb_encoder,
            prompt_encoder=prompt_encoder,
            decoder=decoder,
            head=head,
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
        super(MDECombinedModelShare, self).build_modules()
        self.mde_head = instantiate_from_config(self.mde_head_cfg)
        if self.mde_head is not None:
            self.module_names.append("mde_head")

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
        patch_h, patch_w = h // self.rgb_encoder.patch_size, w // self.rgb_encoder.patch_size

        if not self.fp8_enabled:
            if self.renormalize:
                x = self.prepare_image(x)
            rgb_features = self.rgb_encoder(x, meta_data=meta_data)
            mde_results = self.mde_head(rgb_features, meta_data=meta_data)
            rel_depth = mde_results.pop("pointmap")
            if self.prompt_encoder is not None and prompt_depth is not None:
                if self.prompt_mde:
                    h, w = prompt_depth.shape[-2:]
                    mde_pred = self.act_normalize_rel(rel_depth)
                    mde_pred = F.interpolate(mde_pred, (h, w), mode="bilinear", align_corners=True)
                    prompt_depth = torch.concat([prompt_depth, mde_pred], dim=1)
                prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)
            else:
                prompt_features = None
            features = self.decoder(rgb_features, prompt_features, meta_data=meta_data)
            results = self.head(features, patch_h, patch_w, return_dict=True, meta_data=meta_data)
            results["rel_depth"] = rel_depth
            results.update(mde_results)
        return results

    def act_normalize_rel(self, rel_depth):
        if self.mde_act_norm in ["sigmoid"]:
            rel_depth_normalized = F.sigmoid(rel_depth)
        elif self.mde_act_norm in ["relu"]:
            B = rel_depth.shape[0]
            rel_depth = F.relu(rel_depth)
            # normalize to [0,1]
            _min = torch.quantile(rel_depth.reshape(B, -1).float(), 0.02, dim=1).reshape(B, 1, 1, 1)
            _max = torch.quantile(rel_depth.reshape(B, -1).float(), 0.98, dim=1).reshape(B, 1, 1, 1)
            rel_depth_normalized = 1.0 * (rel_depth - _min) / (_max - _min)
        return rel_depth_normalized

    def prepare_image(self, image):
        image = (image + 1) / 2.0
        return image


class MDECombinedModelShareV2(CombinedModel):
    def __init__(
        self,
        rgb_encoder=None,
        prompt_encoder=None,
        mde_head=None,
        decoder=None,
        head=None,
        freeze_modules=[],
        load_modules=None,
        mde_act_norm="sigmoid",
        renormalize=False,
    ):
        self.mde_head_cfg = mde_head
        self.renormalize = renormalize
        self.mde_act_norm = mde_act_norm

        super(MDECombinedModelShareV2, self).__init__(
            rgb_encoder=rgb_encoder,
            prompt_encoder=prompt_encoder,
            decoder=decoder,
            head=head,
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
        super(MDECombinedModelShareV2, self).build_modules()
        self.mde_head = instantiate_from_config(self.mde_head_cfg)
        if self.mde_head is not None:
            self.module_names.append("mde_head")

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
        patch_h, patch_w = h // self.rgb_encoder.patch_size, w // self.rgb_encoder.patch_size

        if not self.fp8_enabled:
            rgb_features = self.rgb_encoder(x, meta_data=meta_data)
            rgb_path, layer_rns = self.decoder(
                rgb_features, prompt_features=None, meta_data=meta_data, return_rns=True
            )
            mde_results = self.mde_head(
                rgb_path, patch_h, patch_w, return_dict=True, meta_data=meta_data
            )
            rel_depth = mde_results["rel_depth"]

            if self.prompt_encoder is not None and prompt_depth is not None:
                h, w = prompt_depth.shape[-2:]
                mde_pred = self.act_normalize_rel(rel_depth)
                mde_pred = F.interpolate(mde_pred, (h, w), mode="bilinear", align_corners=True)
                prompt_depth = torch.concat([prompt_depth, mde_pred], dim=1)
                prompt_features = self.prompt_encoder(prompt_depth, meta_data=meta_data)
            else:
                prompt_features = None

            features = self.decoder.get_fused_path(layer_rns, prompt_features)
            results = self.head(features, patch_h, patch_w, return_dict=True, meta_data=meta_data)
            results.update(mde_results)
        return results

    def act_normalize_rel(self, rel_depth):
        if self.mde_act_norm in ["sigmoid"]:
            rel_depth_normalized = F.sigmoid(rel_depth)
        elif self.mde_act_norm in ["relu"]:
            B = rel_depth.shape[0]
            rel_depth = F.relu(rel_depth)
            # normalize to [0,1]
            _min = torch.quantile(rel_depth.reshape(B, -1).float(), 0.02, dim=1).reshape(B, 1, 1, 1)
            _max = torch.quantile(rel_depth.reshape(B, -1).float(), 0.98, dim=1).reshape(B, 1, 1, 1)
            rel_depth_normalized = 1.0 * (rel_depth - _min) / (_max - _min)
        return rel_depth_normalized
