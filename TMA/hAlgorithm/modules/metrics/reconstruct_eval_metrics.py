import json
import os

import cv2
import torch

from hAlgorithm.modules.losses.lpips import LPIPS, convert_to_buffer
from hAlgorithm.modules.losses.pgsr import ssim
from hAlgorithm.modules.utils.image_utils import psnr


def rgb_psnr(output, target, **kwargs):
    if kwargs.get("mask", None) is not None:
        mask = kwargs["mask"]
        if mask.ndim == 2:
            output = output[:, mask]
            target = target[:, mask]
        elif mask.ndim == 3:
            output = output[:, mask[0]]
            target = target[:, mask[0]]
        else:
            raise NotImplementedError
        return psnr(output, target).mean()
    else:
        return psnr(output, target).mean()


def rgb_ssim(output, target, **kwargs):
    if len(output.shape) == 3:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
    if kwargs.get("mask", None) is not None:
        return ssim(output, target, mask=kwargs["mask"]).mean()
    else:
        return ssim(output, target).mean()


def rgb_lpips(output, target, lpips, **kwargs):
    if len(output.shape) == 3:
        output = output.unsqueeze(0)
        target = target.unsqueeze(0)
    if kwargs.get("mask", None) is not None:
        mask = kwargs["mask"][None, None]
        return lpips.forward(output * mask, target * mask, normalize=False).mean()
    else:
        return lpips.forward(output, target, normalize=False).mean()


class ReconstructEvalMetrics:
    def __init__(
        self,
        metrics,
        target_name,
    ):
        self.metrics = metrics
        self.target_name = target_name

        if "rgb_psnr" in self.metrics:
            self.lpips = LPIPS(net="vgg").to("cuda")
            convert_to_buffer(self.lpips, persistent=False)
        else:
            self.lpips = None

    def eval_single_data(self, inputs, output, eval_idx):
        predict = getattr(output, "render_rgb", None)
        if predict is None:
            return dict()

        predict = torch.from_numpy(predict).permute(2, 0, 1)
        if isinstance(inputs, (list, tuple)):
            target = inputs[eval_idx][self.target_name][:, 0].squeeze().clone().cuda()
        else:
            target = inputs[self.target_name][:, eval_idx].squeeze().clone()
        target = (target + 1) * 0.5

        results_dict = dict()
        for metric in self.metrics:
            results = eval(metric)(predict, target, lpips=self.lpips)
            if results is not None:
                results_dict[metric] = results
        return results_dict

    def eval_mf_data(self, inputs, outputs):
        results_dict = dict()
        valid_result = 0
        for i, output in enumerate(outputs):
            if output is None:
                continue
            result = self.eval_single_data(inputs, output, eval_idx=i)
            for k, v in result.items():
                results_dict[k] = results_dict.get(k, 0) + v
            valid_result += 1
        if valid_result == 0:
            raise ValueError("Valid Result is zero!")
        results_dict = {k: v / valid_result for k, v in results_dict.items()}
        return results_dict

    def __call__(self, inputs, outputs):
        if isinstance(outputs, list):
            return self.eval_mf_data(inputs, outputs)
        else:
            return self.eval_single_data(inputs, outputs, eval_idx=None)


class ReconstructEvalMetricsWithNovelView(ReconstructEvalMetrics):
    def __init__(
        self,
        metrics,
        target_name,
        target_normalize=True,
    ):
        super().__init__(metrics, target_name)
        novel_metrics = ["novel_" + metric for metric in self.metrics]
        self.metrics = self.metrics + novel_metrics
        self.target_normalize = target_normalize

    def eval_single_data(self, inputs, output, eval_idx):
        predict = getattr(output, "render_rgb", None)
        if predict is None:
            return dict()

        predict = torch.from_numpy(predict).permute(2, 0, 1).cuda()
        if isinstance(inputs, (list, tuple)):
            target = inputs[eval_idx][self.target_name][:, 0].squeeze().clone().cuda()
        else:
            target = inputs[self.target_name][:, eval_idx].squeeze().clone().cuda()
        if self.target_normalize:
            target = (target + 1) * 0.5

        results_dict = dict()
        for metric in self.metrics:
            if "novel" in metric:
                continue
            if self.lpips is not None:
                results = eval(metric)(predict, target, lpips=self.lpips.cuda())
            if results is not None:
                results_dict[metric] = results
        return results_dict

    def eval_novel_view(self, inputs, outputs):
        novel_views = getattr(outputs[0], "novel_render_rgb", None)
        novel_mask = getattr(outputs[0], "novel_mask", None)
        if novel_views is None:
            return dict()
        novel_views = outputs[0].novel_render_rgb
        novel_views = torch.from_numpy(novel_views).permute(0, 3, 1, 2).cuda()
        novel_target = inputs[self.target_name].squeeze().clone().cuda()
        novel_target = (novel_target + 1) * 0.5

        novel_views[novel_mask == 0] = 0
        novel_target[novel_mask == 0] = 0

        results_dict = dict()
        for metric in self.metrics:
            if "novel" in metric:
                continue
            if self.lpips is not None:
                results = eval(metric)(novel_views, novel_target, lpips=self.lpips.cuda())
            if results is not None:
                results_dict["novel_" + metric] = results
        return results_dict

    def eval_mf_data(self, inputs, outputs):
        results_dict = dict()
        valid_result = 0
        for i, output in enumerate(outputs):
            if output is None:
                continue
            result = self.eval_single_data(inputs, output, eval_idx=i)
            for k, v in result.items():
                results_dict[k] = results_dict.get(k, 0) + v
            valid_result += 1
        if valid_result == 0:
            raise ValueError("Valid Result is zero!")
        results_dict = {k: v / valid_result for k, v in results_dict.items()}

        if "novel" in inputs:
            results_dict.update(self.eval_novel_view(inputs["novel"], outputs))

        return results_dict


class ReconstructEvalMetricsWithFinetuneGS(ReconstructEvalMetrics):
    def __init__(
        self,
        metrics,
        target_name,
        with_mask=False,
    ):
        super().__init__(metrics, target_name)

        self.with_mask = with_mask

        if self.with_mask :
            self.metrics = ["mask_" + metric for metric in self.metrics]

    def eval_single_data(self, inputs, output, eval_idx):
        predict = getattr(output, "render_rgb", None)
        target = getattr(output, "rgb", None)

        if predict is None or target is None:
            return dict()

        if output.object_mask is not None:
            mask = torch.from_numpy(output.object_mask).bool().cuda()
        else:
            mask = None

        predict = torch.from_numpy(predict).permute(2, 0, 1).cuda()
        target = torch.from_numpy(target).permute(2, 0, 1).cuda()

        results_dict = dict()
        for metric in self.metrics:
            if mask is not None and "mask_" in metric:
                results = eval(metric[5:])(predict, target, lpips=self.lpips, mask=mask)
            else:
                results = eval(metric)(predict, target, lpips=self.lpips)
            if results is not None:
                results_dict[metric] = results
        return results_dict

    def eval_novel_view(self, inputs, outputs):
        novel_views = getattr(outputs[0], "novel_render_rgb", None)
        novel_mask = getattr(outputs[0], "novel_mask", None)
        if novel_views is None:
            return dict()
        novel_views = outputs[0].novel_render_rgb
        novel_views = torch.from_numpy(novel_views).permute(0, 3, 1, 2).cuda()
        novel_target = inputs[self.target_name].squeeze().clone().cuda()
        # novel_target = (novel_target + 1) * 0.5 # NOTE

        novel_views[novel_mask == 0] = 0
        novel_target[novel_mask == 0] = 0

        results_dict = dict()
        for metric in self.metrics:
            if "novel" in metric:
                continue
            results = eval(metric)(novel_views, novel_target, lpips=self.lpips)
            if results is not None:
                results_dict["novel_" + metric] = results
        return results_dict

    def eval_mf_data(self, inputs, outputs):
        results_dict = dict()
        valid_result = 0
        for i, output in enumerate(outputs):
            if output is None:
                continue
            result = self.eval_single_data(inputs, output, eval_idx=i)
            for k, v in result.items():
                results_dict[k] = results_dict.get(k, 0) + v
            valid_result += 1
        if valid_result == 0:
            raise ValueError("Valid Result is zero!")
        results_dict = {k: v / valid_result for k, v in results_dict.items()}

        if "novel" in inputs:
            results_dict.update(self.eval_novel_view(inputs.get("novel", None), outputs))

        return results_dict


# 注意：_extract_hf_single_channel函数已被弃用，现在使用PyTorch FFT实现


def extract_high_frequency_component(image_tensor, radius=30):
    """
    提取图像的高频分量 - 使用PyTorch原生FFT避免OpenCV兼容性问题

    Args:
        image_tensor: torch.Tensor，形状为 [C, H, W] 或 [B, C, H, W]
        radius: int，低频区域的半径

    Returns:
        torch.Tensor: 高频分量图像，形状与输入相同
    """
    original_shape = image_tensor.shape
    device = image_tensor.device

    # 统一处理为 [B, C, H, W] 格式
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    batch_size, channels, height, width = image_tensor.shape
    result = torch.zeros_like(image_tensor)

    for b in range(batch_size):
        for c in range(channels):
            img = image_tensor[b, c]  # [H, W]

            # 使用PyTorch FFT
            fft = torch.fft.fft2(img)
            fft_shift = torch.fft.fftshift(fft)

            # 创建高通滤波器
            rows, cols = height, width
            crow, ccol = rows // 2, cols // 2

            # 创建圆形掩膜
            y, x = torch.meshgrid(
                torch.arange(rows, device=device), torch.arange(cols, device=device), indexing="ij"
            )
            center_dist = torch.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
            mask = (center_dist >= radius).float()

            # 应用高通滤波器
            fft_filtered = fft_shift * mask

            # 逆FFT
            fft_ishift = torch.fft.ifftshift(fft_filtered)
            img_back = torch.fft.ifft2(fft_ishift)
            img_back = torch.abs(img_back)

            result[b, c] = img_back

    # 恢复原始形状
    if len(original_shape) == 3:
        result = result.squeeze(0)  # [1, C, H, W] -> [C, H, W]

    return result


def create_high_frequency_mask(high_freq_component, percentile=90):
    """
    基于高频分量创建掩码，取前10%高频区域

    Args:
        high_freq_component: torch.Tensor，高频分量图像
        percentile: float，百分位阈值，默认90（前10%）

    Returns:
        torch.Tensor: 二值掩码，形状与输入相同
    """
    # 如果是多通道，转换为单通道
    if len(high_freq_component.shape) == 4:
        # [B, C, H, W] -> [B, H, W]
        hf_magnitude = torch.mean(high_freq_component, dim=1)
    elif len(high_freq_component.shape) == 3:
        # [C, H, W] -> [H, W]
        hf_magnitude = torch.mean(high_freq_component, dim=0)
    else:
        # [H, W]
        hf_magnitude = high_freq_component

    # 计算阈值（第percentile百分位数）
    threshold_value = torch.quantile(hf_magnitude.flatten(), percentile / 100.0)

    # 创建二值掩码
    high_freq_mask = (hf_magnitude >= threshold_value).float()

    # 扩展掩码维度以匹配原始图像
    if len(high_freq_component.shape) == 4:
        # [B, H, W] -> [B, C, H, W]
        batch_size, channels = high_freq_component.shape[0], high_freq_component.shape[1]
        high_freq_mask = high_freq_mask.unsqueeze(1).expand(batch_size, channels, -1, -1)
    elif len(high_freq_component.shape) == 3:
        # [H, W] -> [C, H, W]
        channels = high_freq_component.shape[0]
        high_freq_mask = high_freq_mask.unsqueeze(0).expand(channels, -1, -1)

    return high_freq_mask


def _masked_ssim(output, target, mask):
    """
    计算掩码区域的SSIM，使用pgsr中的ssim函数

    Args:
        output: 预测图像 [C,H,W]
        target: 目标图像 [C,H,W]
        mask: 二值掩码 [H,W]

    Returns:
        torch.Tensor: 掩码区域的SSIM值
    """
    # 确保输入格式正确 [B,C,H,W]
    if len(output.shape) == 3:
        output = output.unsqueeze(0)  # [1,C,H,W]
    if len(target.shape) == 3:
        target = target.unsqueeze(0)  # [1,C,H,W]

    # 使用pgsr中已有的支持mask的ssim函数
    ssim_val = ssim(output, target, mask=mask)

    return ssim_val


def rgb_psnr_hf(output, target, radius=30, percentile=90, **kwargs):
    """
    计算高频分量的PSNR
    修正逻辑：基于GT图片的高频区域作为掩码（更合理）

    Args:
        output: 预测图像 (render)
        target: 目标图像 (GT)
        radius: 高通滤波器半径
        percentile: 高频区域的百分位阈值，默认90（前10%）

    Returns:
        torch.Tensor: 高频PSNR值
    """
    # 1. 从GT图片提取高频分量（更合理，因为GT是真实的高频参考）
    target_hf = extract_high_frequency_component(target, radius)

    # 2. 基于GT的高频分量创建掩码（前10%高频区域）
    high_freq_mask = create_high_frequency_mask(target_hf, percentile)

    # 3. 只计算掩码为True的像素
    mask_bool = high_freq_mask > 0.5  # 确保是布尔掩码

    # 提取掩码区域的像素值
    output_masked_pixels = output[mask_bool]
    target_masked_pixels = target[mask_bool]

    # 4. 计算掩码区域的PSNR
    valid_pixels = output_masked_pixels.numel()
    if valid_pixels == 0:
        return torch.tensor(float("inf"))

    mse = torch.mean((output_masked_pixels - target_masked_pixels) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))

    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_val


def rgb_ssim_hf(output, target, radius=30, percentile=90, **kwargs):
    """
    计算高频分量的SSIM
    修正逻辑：基于GT图片的高频区域作为掩码（更合理）

    Args:
        output: 预测图像 (render)
        target: 目标图像 (GT)
        radius: 高通滤波器半径
        percentile: 高频区域的百分位阈值，默认90（前10%）

    Returns:
        torch.Tensor: 高频SSIM值
    """
    # 1. 从GT图片提取高频分量（更合理，因为GT是真实的高频参考）
    target_hf = extract_high_frequency_component(target, radius)

    # 2. 基于GT的高频分量创建掩码（前10%高频区域）
    high_freq_mask = create_high_frequency_mask(target_hf, percentile)

    # 3. 使用掩码式SSIM计算，避免0值区域影响
    return _masked_ssim(output, target, high_freq_mask)


def rgb_psnr_hf_with_mask(output, target, mask=None, radius=30, percentile=90, **kwargs):
    """
    计算带mask的高频分量PSNR
    修正逻辑：基于GT图片的高频区域作为掩码，同时支持外部mask

    Args:
        output: 预测图像 (render)
        target: 目标图像 (GT)
        mask: 外部二值掩膜，None表示不使用外部mask
        radius: 高通滤波器半径
        percentile: 高频区域的百分位阈值，默认90（前10%）

    Returns:
        torch.Tensor: 高频PSNR值
    """
    # 1. 从GT图片提取高频分量并创建高频掩码
    target_hf = extract_high_frequency_component(target, radius)
    high_freq_mask = create_high_frequency_mask(target_hf, percentile)

    # 2. 如果有外部mask，与高频mask结合
    if mask is not None:
        # 确保外部mask维度匹配
        if len(mask.shape) == 2:
            if len(high_freq_mask.shape) == 3:
                mask = mask.unsqueeze(0).expand_as(high_freq_mask)
            elif len(high_freq_mask.shape) == 4:
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(high_freq_mask)
        elif len(mask.shape) == 3 and len(high_freq_mask.shape) == 4:
            mask = mask.unsqueeze(0).expand_as(high_freq_mask)

        # 组合两个mask（都必须为True）
        combined_mask = high_freq_mask * mask
    else:
        combined_mask = high_freq_mask

    # 3. 只计算掩码为True的像素
    mask_bool = combined_mask > 0.5

    # 提取掩码区域的像素值
    output_masked_pixels = output[mask_bool]
    target_masked_pixels = target[mask_bool]

    # 4. 计算掩码区域的PSNR
    valid_pixels = output_masked_pixels.numel()
    if valid_pixels == 0:
        return torch.tensor(float("inf"))

    mse = torch.mean((output_masked_pixels - target_masked_pixels) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))

    psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_val


def rgb_ssim_hf_with_mask(output, target, mask=None, radius=30, percentile=90, **kwargs):
    """
    计算带mask的高频分量SSIM
    修正逻辑：基于GT图片的高频区域作为掩码，同时支持外部mask

    Args:
        output: 预测图像 (render)
        target: 目标图像 (GT)
        mask: 外部二值掩膜，None表示不使用外部mask
        radius: 高通滤波器半径
        percentile: 高频区域的百分位阈值，默认90（前10%）

    Returns:
        torch.Tensor: 高频SSIM值
    """
    # 1. 从GT图片提取高频分量并创建高频掩码
    target_hf = extract_high_frequency_component(target, radius)
    high_freq_mask = create_high_frequency_mask(target_hf, percentile)

    # 2. 如果有外部mask，与高频mask结合
    if mask is not None:
        # 确保外部mask维度匹配
        if len(mask.shape) == 2:
            if len(high_freq_mask.shape) == 3:
                mask = mask.unsqueeze(0).expand_as(high_freq_mask)
            elif len(high_freq_mask.shape) == 4:
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(high_freq_mask)
        elif len(mask.shape) == 3 and len(high_freq_mask.shape) == 4:
            mask = mask.unsqueeze(0).expand_as(high_freq_mask)

        # 组合两个mask（都必须为True）
        combined_mask = high_freq_mask * mask
    else:
        combined_mask = high_freq_mask

    # 3. 使用掩码式SSIM计算，避免0值区域影响
    return _masked_ssim(output, target, combined_mask)


def rgb_sharpness(output, target=None, **kwargs):
    """
    计算图像清晰度指标，使用更完善的多指标方法
    基于test_image_metrics_custom中的实现

    Args:
        output: 预测图像
        target: 目标图像（可选，如果提供则计算两者的清晰度差异）

    Returns:
        torch.Tensor: 清晰度值或差异值
    """
    import torch.nn.functional as F

    def calculate_comprehensive_sharpness(img):
        """计算综合清晰度指标"""
        # 确保输入是3D张量 [C, H, W]
        if len(img.shape) == 4:
            img = img.squeeze(0)  # 去掉batch维度

        # 1. 梯度幅值 (Gradient Magnitude)
        def gradient_magnitude(image_tensor):
            # 转为灰度
            if image_tensor.shape[0] == 3:
                gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            else:
                gray = image_tensor.mean(dim=0)

            # Sobel算子
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32,
                device=image_tensor.device,
            )
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32,
                device=image_tensor.device,
            )

            sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
            sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

            gray = gray.unsqueeze(0).unsqueeze(0)

            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)

            magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            return magnitude.mean()

        # 2. 高频能量分析 (FFT)
        def high_frequency_energy(image_tensor):
            # 转为灰度
            if image_tensor.shape[0] == 3:
                gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            else:
                gray = image_tensor.mean(dim=0)

            # FFT
            fft = torch.fft.fft2(gray)
            fft_shift = torch.fft.fftshift(fft)
            magnitude = torch.abs(fft_shift)

            # 计算高频区域的能量
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            # 创建高频mask (距离中心的环形区域)
            y, x = torch.meshgrid(
                torch.arange(h, device=image_tensor.device),
                torch.arange(w, device=image_tensor.device),
                indexing="ij",
            )
            dist = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

            # 高频区域：距离中心20%-40%范围
            radius_min = min(h, w) * 0.2
            radius_max = min(h, w) * 0.4
            high_freq_mask = (dist >= radius_min) & (dist <= radius_max)

            total_energy = magnitude.sum()
            high_freq_energy = magnitude[high_freq_mask].sum()

            return high_freq_energy / (total_energy + 1e-8)

        # 3. 拉普拉斯方差 (Laplacian Variance)
        def laplacian_variance(image_tensor):
            if image_tensor.shape[0] == 3:
                gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            else:
                gray = image_tensor.mean(dim=0)

            # 拉普拉斯核
            laplacian_kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image_tensor.device
            )
            laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)

            gray = gray.unsqueeze(0).unsqueeze(0)
            laplacian = F.conv2d(gray, laplacian_kernel, padding=1)

            return laplacian.var()

        # 4. 边缘密度 (Edge Density)
        def edge_density(image_tensor):
            if image_tensor.shape[0] == 3:
                gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            else:
                gray = image_tensor.mean(dim=0)

            # 使用高斯滤波 + 梯度检测
            gaussian_kernel = (
                torch.tensor(
                    [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                    dtype=torch.float32,
                    device=image_tensor.device,
                )
                / 16.0
            )
            gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)

            gray = gray.unsqueeze(0).unsqueeze(0)
            smoothed = F.conv2d(gray, gaussian_kernel, padding=1)

            # 梯度计算
            sobel_x = (
                torch.tensor(
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    dtype=torch.float32,
                    device=image_tensor.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sobel_y = (
                torch.tensor(
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                    dtype=torch.float32,
                    device=image_tensor.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            grad_x = F.conv2d(smoothed, sobel_x, padding=1)
            grad_y = F.conv2d(smoothed, sobel_y, padding=1)

            magnitude = torch.sqrt(grad_x**2 + grad_y**2)

            # 边缘阈值
            threshold = magnitude.mean() + magnitude.std()
            edges = (magnitude > threshold).float()

            edge_density = edges.mean()
            return edge_density

        # 计算各项指标
        grad_mag = gradient_magnitude(img)
        hf_energy = high_frequency_energy(img)
        lap_var = laplacian_variance(img)
        edge_dens = edge_density(img)

        # 综合清晰度分数（加权平均）
        # 权重可根据实际需要调整
        comprehensive_sharpness = (
            grad_mag * 0.3  # 梯度幅值权重30%
            + hf_energy * 0.25  # 高频能量权重25%
            + lap_var * 0.25  # 拉普拉斯方差权重25%
            + edge_dens * 0.2  # 边缘密度权重20%
        )

        return comprehensive_sharpness

    # 计算输出图像的清晰度
    output_sharpness = calculate_comprehensive_sharpness(output)

    if target is not None:
        # 如果提供了目标图像，计算清晰度差异
        target_sharpness = calculate_comprehensive_sharpness(target)
        # 返回清晰度差异的绝对值（越小越好）
        return torch.abs(output_sharpness - target_sharpness)
    else:
        # 只返回输出图像的清晰度值（越大越好）
        return output_sharpness
