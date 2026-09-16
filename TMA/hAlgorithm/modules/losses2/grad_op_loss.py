import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Loss


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
                torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
                requires_grad=False,
            )

        if self.scharr:
            self.scharr_x = nn.Parameter(
                torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3),
                requires_grad=False,
            )
            self.scharr_y = nn.Parameter(
                torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32).view(1, 1, 3, 3),
                requires_grad=False,
            )

        if self.sobel:
            self.sobel_x = nn.Parameter(
                torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
                requires_grad=False,
            )
            self.sobel_y = nn.Parameter(
                torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3),
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

    def forward(self, C_hat, C, valid_mask, p=1, operator="scharr"):
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
            mask_this_res = F.avg_pool2d(valid_mask, r)
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
                diff_x = ((torch.abs(grad_C[0] - grad_C_hat[0]) ** p) * mask_9).sum() / mask_9.sum().clip(1)
                diff_y = ((torch.abs(grad_C[1] - grad_C_hat[1]) ** p) * mask_9).sum() / mask_9.sum().clip(1)
                total_loss += diff_x + diff_y

            else:
                if self.debug:
                    self.debug_func(grad_C, grad_C_hat, m=j, p=p, operator=operator)

                # Compute the loss at this scale
                diff = ((torch.abs(grad_C - grad_C_hat) ** p) * mask_9).sum() / mask_9.sum().clip(1)
                total_loss += diff

        return total_loss / self.M

    def debug_func(self, grad_C, grad_C_hat, m, p, operator):
        import os

        import matplotlib.pyplot as plt

        save_dir = f"debug/MultiScaleGradientLoss/{operator}"
        os.makedirs(save_dir, exist_ok=True)

        for batch_idx in range(grad_C.shape[0]):
            # Select the first batch item for plotting
            grad_C_i = grad_C[batch_idx].squeeze().detach().cpu().numpy()
            grad_C_hat_i = grad_C_hat[batch_idx].squeeze().detach().cpu().numpy()

            # Plot ground truth gradient
            fig, ax = plt.subplots(figsize=(5, 5))

            im = ax.imshow(grad_C_i, cmap="gray")
            ax.set_title(f"Scale {m}, Operator {operator} - Ground Truth")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            filename = os.path.join(save_dir, f"gt_scale_{m}_operator_{operator}_p{p}_b{batch_idx}.png")
            plt.savefig(filename)
            plt.close(fig)

            # Plot predicted gradient
            fig, ax = plt.subplots(figsize=(5, 5))

            im = ax.imshow(grad_C_hat_i, cmap="gray")
            ax.set_title(f"Scale {m}, Operator {operator} - Ground Truth")
            ax.axis("off")
            fig.colorbar(im, ax=ax)

            filename = os.path.join(save_dir, f"pred_scale_{m}_operator_{operator}_b{batch_idx}.png")
            plt.savefig(filename)
            plt.close(fig)


class GradientLoss(Loss):
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
        super().__init__(loss_weight=loss_weight)
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

        self.start_iter = start_iter
        self.debug = debug

    def forward(self, pred_depth, target_depth, valid_mask, name=None, **kwargs):

        loss_weight = self.get_loss_weight(name)

        if loss_weight == 0:
            return 0 * torch.sum(torch.nan_to_num(pred_depth))

        if pred_depth.ndim == 3:
            pred_depth = pred_depth.unsqueeze(1)
            target_depth = target_depth.unsqueeze(1)

        valid_mask = valid_mask.float()
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)

        assert pred_depth.shape[1] == 1 or pred_depth.shape[3] == 1
        if pred_depth.shape[3] == 1:
            pred_depth = pred_depth.squeeze(-1).unsqueeze(1)
            target_depth = target_depth.squeeze(-1).unsqueeze(1)

        loss = 0

        if self.laplace > 0:
            laplace_loss = self.loss_fn(pred_depth, target_depth, valid_mask=valid_mask, operator="laplace")
            loss += laplace_loss * self.laplace

        if self.scharr > 0:
            scharr_loss = self.loss_fn(pred_depth, target_depth, valid_mask=valid_mask, p=self.scharr_p, operator="scharr")
            loss += scharr_loss * self.scharr

        if self.scharr_xy > 0:
            scharr_xy_loss = self.loss_fn(pred_depth, target_depth, valid_mask=valid_mask, p=self.scharr_xy_p, operator="scharr_xy")
            loss += scharr_xy_loss * self.scharr_xy

        if self.sobel > 0:
            sobel_loss = self.loss_fn(pred_depth, target_depth, valid_mask=valid_mask, p=self.sobel_p, operator="sobel")
            loss += sobel_loss * self.sobel

        if self.sobel_xy > 0:
            sobel_xy_loss = self.loss_fn(pred_depth, target_depth, valid_mask=valid_mask, p=self.sobel_xy_p, operator="sobel_xy")
            loss += sobel_xy_loss * self.sobel_xy

        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(torch.nan_to_num(pred_depth))
            logging.warning(f"Data {name}, GradientLoss NAN error, {loss}")

        return loss * loss_weight
