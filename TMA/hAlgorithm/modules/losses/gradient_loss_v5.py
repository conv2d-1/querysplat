import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import check_and_fix_inf_nan


class MultiScaleGradientLoss(nn.Module):
    def __init__(self, laplace=True, scharr=False, sobel=False, M=6, debug=False):
        """
        Initialize the multi-scale gradient loss.

        Args:
            M (int): The number of scales.
            p (int): The error norm (1 for L1, 2 for L2).
        """
        super(MultiScaleGradientLoss, self).__init__()

        self.laplace = laplace
        self.scharr = scharr
        self.sobel = sobel
        self.M = M
        self.debug = debug

        self.mask_kernel = nn.Parameter(
            torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False,
        )

        if self.laplace:
            self.laplace_xy = nn.Parameter(
                torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(
                    1, 1, 3, 3
                ),
                requires_grad=False,
            )

        if self.scharr:
            self.scharr_x = nn.Parameter(
                torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(
                    1, 1, 3, 3
                ),
                requires_grad=False,
            )
            self.scharr_y = nn.Parameter(
                torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32).view(
                    1, 1, 3, 3
                ),
                requires_grad=False,
            )

        if self.sobel:
            self.sobel_x = nn.Parameter(
                torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(
                    1, 1, 3, 3
                ),
                requires_grad=False,
            )
            self.sobel_y = nn.Parameter(
                torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(
                    1, 1, 3, 3
                ),
                requires_grad=False,
            )

    def apply_derivative_operator(self, image, operator="scharr"):
        """
        Apply a derivative operator to an image.

        Args:
            image (torch.Tensor): A 4D tensor of shape [batch_size, channels, height, width].
            operator (str): The type of derivative operator ('scharr' or 'laplace').

        Returns:
            torch.Tensor: The result of applying the derivative operator.
        """
        if operator == "scharr_xy":
            grad_x = F.conv2d(image, self.scharr_x.to(image.device, image.dtype), padding=1)
            grad_y = F.conv2d(image, self.scharr_y.to(image.device, image.dtype), padding=1)
            return grad_x, grad_y
        elif operator == "scharr":
            grad_x = F.conv2d(image, self.scharr_x.to(image.device, image.dtype), padding=1)
            grad_y = F.conv2d(image, self.scharr_y.to(image.device, image.dtype), padding=1)
            return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        elif operator == "sobel_xy":
            grad_x = F.conv2d(image, self.sobel_x.to(image.device, image.dtype), padding=1)
            grad_y = F.conv2d(image, self.sobel_y.to(image.device, image.dtype), padding=1)
            return grad_x, grad_y
        elif operator == "sobel":
            grad_x = F.conv2d(image, self.sobel_x.to(image.device, image.dtype), padding=1)
            grad_y = F.conv2d(image, self.sobel_y.to(image.device, image.dtype), padding=1)
            return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        elif operator == "laplace":
            return F.conv2d(image, self.laplace_xy.to(image.device, image.dtype), padding=1)
        else:
            raise ValueError("Unsupported derivative operator")

    def downsample_image(self, image):
        """
        Downsample an image by a factor of 2 using bilinear interpolation.

        Args:
            image (torch.Tensor): A 4D tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: The downsampled image.
        """
        return F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False)

    def forward(self, C_hat, C, mask, p=1, operator="scharr"):
        """
        Compute the multi-scale derivative loss between two inverse depth maps.

        Args:
            C_hat (torch.Tensor): A 4D tensor of shape [batch_size, 1, height, width] for the predicted inverse depth map.
            C (torch.Tensor): A 4D tensor of shape [batch_size, 1, height, width] for the ground truth inverse depth map.
            operator (str): The type of derivative operator ('scharr' or 'laplace').

        Returns:
            float: The computed multi-scale derivative loss.
        """
        total_loss = 0.0
        for j in range(self.M):
            r = 2**j
            C_this_res = F.avg_pool2d(C, r)
            C_hat_this_res = F.avg_pool2d(C_hat, r)
            mask_this_res = F.avg_pool2d(mask, r)
            mask_this_res = (mask_this_res == 1.0).float()

            mask_9 = F.pad(mask_this_res, (1, 1, 1, 1), mode="constant", value=1)
            mask_9 = F.conv2d(
                mask_9,
                self.mask_kernel.to(mask_this_res.device, mask_this_res.dtype),
                padding=0,
            )
            mask_9 = (mask_9 > 8.5).float()

            # Apply derivative operator
            grad_C = self.apply_derivative_operator(C_this_res, operator)
            grad_C_hat = self.apply_derivative_operator(C_hat_this_res, operator)

            if isinstance(grad_C, (list, tuple)):
                if self.debug:
                    self.debug_func(grad_C, grad_C_hat, m=j, p=p, operator=operator)

                # Compute the loss at this scale
                diff_x = (torch.abs(grad_C[0] - grad_C_hat[0]) ** p) * mask_9
                diff_x = check_and_fix_inf_nan(diff_x, "diff_x")
                diff_x = diff_x.sum() / mask_9.sum().clip(1)

                diff_y = (torch.abs(grad_C[1] - grad_C_hat[1]) ** p) * mask_9
                diff_y = check_and_fix_inf_nan(diff_y, "diff_y")
                diff_y = diff_y.sum() / mask_9.sum().clip(1)

                total_loss += diff_x + diff_y

            else:
                if self.debug:
                    self.debug_func(grad_C, grad_C_hat, m=j, p=p, operator=operator)

                # Compute the loss at this scale
                diff = ((torch.abs(grad_C - grad_C_hat) ** p) * mask_9)
                diff = check_and_fix_inf_nan(diff, "diff")
                diff = diff.sum() / mask_9.sum().clip(1)
                total_loss += diff

        return total_loss / self.M


class GradientLoss(nn.Module):
    def __init__(
        self,
        laplace=0.0,
        scharr=0.0,
        scharr_xy=0.0,
        sobel=0.0,
        sobel_xy=0.0,
        scharr_p=1,
        scharr_xy_p=1,
        sobel_p=1,
        sobel_xy_p=1,
        loss_weight=1.0,
        M=6,
        start_iter=None,
        debug=False,
        **kwargs,
    ):
        super().__init__()
        # Create an instance of the loss function
        self.loss_fn = MultiScaleGradientLoss(
            laplace=laplace > 0,
            scharr=(scharr + scharr_xy) > 0,
            sobel=(sobel + sobel_xy) > 0,
            M=M,
            debug=debug,
        )
        self.laplace = laplace
        self.scharr = scharr
        self.scharr_xy = scharr_xy
        self.sobel = sobel
        self.sobel_xy = sobel_xy

        assert (self.laplace + self.scharr + self.scharr_xy + self.sobel + self.sobel_xy) > 0

        self.scharr_p = scharr_p
        self.scharr_xy_p = scharr_xy_p
        self.sobel_p = sobel_p
        self.sobel_xy_p = sobel_xy_p

        self.loss_weight = loss_weight
        self.start_iter = start_iter
        self.debug = debug

        if isinstance(self.loss_weight, dict):
            assert "default" in self.loss_weight

    def get_loss_weight(self, name=None):
        if name is not None and isinstance(self.loss_weight, dict) and name in self.loss_weight:
            loss_weight = self.loss_weight[name]
        elif isinstance(self.loss_weight, dict) and name not in self.loss_weight:
            loss_weight = self.loss_weight["default"]
        else:
            loss_weight = self.loss_weight
        return loss_weight

    def forward(self, prediction, target, mask, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(prediction))

        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
            target = target.unsqueeze(1)

        mask = mask.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        assert prediction.shape[1] == 1 or prediction.shape[3] == 1
        if prediction.shape[3] == 1:
            prediction = prediction.squeeze(-1).unsqueeze(1)
            target = target.squeeze(-1).unsqueeze(1)

        loss = 0

        if self.laplace > 0:
            laplace_loss = self.loss_fn(prediction, target, mask=mask, operator="laplace")
            loss += laplace_loss * self.laplace

        if self.scharr > 0:
            scharr_loss = self.loss_fn(
                prediction, target, mask=mask, p=self.scharr_p, operator="scharr"
            )
            loss += scharr_loss * self.scharr

        if self.scharr_xy > 0:
            scharr_xy_loss = self.loss_fn(
                prediction, target, mask=mask, p=self.scharr_xy_p, operator="scharr_xy"
            )
            loss += scharr_xy_loss * self.scharr_xy

        if self.sobel > 0:
            sobel_loss = self.loss_fn(
                prediction, target, mask=mask, p=self.sobel_p, operator="sobel"
            )
            loss += sobel_loss * self.sobel

        if self.sobel_xy > 0:
            sobel_xy_loss = self.loss_fn(
                prediction, target, mask=mask, p=self.sobel_xy_p, operator="sobel_xy"
            )
            loss += sobel_xy_loss * self.sobel_xy

        return loss * loss_weight
