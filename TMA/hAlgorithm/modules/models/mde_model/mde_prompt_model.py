import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.utils.alignment import align_least_square_batch
from hAlgorithm.utils import instantiate_from_config


class MDEPromptModel(nn.Module):
    def __init__(self, mde_config, pretrain=None, sample_rel_depth=False, **kwargs) -> None:
        super().__init__()
        self.mde_model = instantiate_from_config(mde_config)
        self.sample_rel_depth = sample_rel_depth

        if self.mde_model is not None and pretrain is not None:
            self.mde_model.load_state_dict(torch.load(pretrain))

    def forward(self, x, prompt_depth, **kwargs):
        """
        Processes input image and depth prompt to generate enhanced depth prompt

        Args:
            x (Tensor): Input image tensor (BxCxHxW)
            prompt_depth (Tensor): Initial depth prompt tensor (Bx1xHxW)

        Returns:
            Tensor: Concatenated prompt tensor (Bx3xHxW) containing:
                - Original prompt_depth
                - Relative depth from MDE model
                - Inverse of relative depth
        """
        rel_depth_normalized = self.get_rel_depth(x, prompt_depth)
        if self.sample_rel_depth:
            rel_depth_normalized[~(prompt_depth[:, [-1], ...] > 1e-3)] = 0
        prompt_depth = torch.concat([prompt_depth, rel_depth_normalized], dim=1)
        return prompt_depth

    def get_rel_depth(self, x, prompt_depth):
        x = self.prepare_image(x)

        rel_depth = self.mde_model(x)  # Get relative depth prediction
        rel_depth = F.relu(rel_depth)
        h, w = prompt_depth.shape[-2:]
        B = prompt_depth.shape[0]

        rel_depth = F.interpolate(rel_depth, (h, w), mode="bilinear", align_corners=True)
        # normalize to [0,1]
        _min = torch.quantile(rel_depth.reshape(B, -1), 0.02, dim=1).reshape(B, 1, 1, 1)
        _max = torch.quantile(rel_depth.reshape(B, -1), 0.98, dim=1).reshape(B, 1, 1, 1)

        rel_depth_normalized = 1.0 * (rel_depth - _min) / (_max - _min)
        return rel_depth_normalized

    def prepare_image(self, image):
        image = (image + 1) / 2.0
        return image

    def debug_rel_depth(self, depth):
        import cv2
        import numpy as np

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        cv2.imwrite("./debug/rel_depth_debug.png", depth)


class MDEPromptModelSimple(MDEPromptModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, prompt_depth, **kwargs):
        return self.get_rel_depth(x, prompt_depth)


class MDEPromptModelDepthor(MDEPromptModel):
    def __init__(self, fit_prompt=False, normalize_inv=False, **kwargs):
        super().__init__(**kwargs)
        self.fit_prompt = fit_prompt
        self.normalize_inv = normalize_inv

    def forward(self, x, prompt_depth, **kwargs):
        """
        Processes input image and depth prompt to generate enhanced depth prompt

        Args:
            x (Tensor): Input image tensor (BxCxHxW)
            prompt_depth (Tensor): Initial depth prompt tensor (Bx1xHxW)

        Returns:
            Tensor: Concatenated prompt tensor (Bx3xHxW) containing:
                - Original prompt_depth
                - Relative depth from MDE model
                - Inverse of relative depth
        """
        rel_depth_normalized = self.get_rel_depth(x, prompt_depth)
        inv_rel_depth = 1 / (1 + rel_depth_normalized)

        if self.normalize_inv:
            B = rel_depth_normalized.shape[0]
            # normalize to [0,1]
            _min = torch.quantile(inv_rel_depth.reshape(B, -1), 0.02, dim=1).reshape(B, 1, 1, 1)
            _max = torch.quantile(inv_rel_depth.reshape(B, -1), 0.98, dim=1).reshape(B, 1, 1, 1)
            inv_rel_depth = 1.0 * (inv_rel_depth - _min) / (_max - _min)

        if self.fit_prompt:
            fit_target = prompt_depth[:, -1, ...]
            inv_rel_depth_fit = align_least_square_batch(
                inv_rel_depth.squeeze(1), fit_target, fit_target > 1e-3, debug=False
            )
            inv_rel_depth = torch.clamp(inv_rel_depth_fit, 0, 1).unsqueeze(1)
            rel_depth = 2 / (1 + inv_rel_depth) - 1
        else:
            rel_depth = rel_depth_normalized

        prompt_depth = torch.concat(
            [prompt_depth, rel_depth, inv_rel_depth], dim=1
        )  # Concatenate along channel dimension

        return prompt_depth
