import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config


class MDETeacherModel():
    def __init__(self, mde_config, pretrain=None, **kwargs) -> None:
        super().__init__()
        self.mde_model = instantiate_from_config(mde_config)

        if self.mde_model is not None and pretrain is not None:
            self.mde_model.load_state_dict(torch.load(pretrain))

    def forward(self, x, student_output=None, **kwargs):
        """
        Processes input image and depth prompt to generate enhanced depth prompt

        Args:
            x (Tensor): Input image tensor (BxCxHxW)

        Returns:
            Tensor: Concatenated prompt tensor (Bx3xHxW) containing:
                - Original prompt_depth
                - Relative depth from MDE model
                - Inverse of relative depth
        """
        x = self.prepare_image(x)  # B,C,H,W
        rel_depth = self.mde_model(x, **kwargs)  # Get relative depth prediction
        if student_output is not None:
            h, w = student_output.shape[-2:]
            rel_depth = F.interpolate(rel_depth, (h, w), mode="bilinear", align_corners=True)
        return rel_depth  # B,1,H,W

    def prepare_image(self, image):
        image = (image + 1) / 2.0
        return image

    def __call__(self, x, student_output=None, **kwargs):
        return self.forward(x, student_output, **kwargs)
