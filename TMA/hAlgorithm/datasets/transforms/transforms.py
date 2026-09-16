import collections.abc as collections
import math
import numbers
import random
from copy import deepcopy

import cv2
import numpy as np
import torch

from hAlgorithm.utils import instantiate_from_config


class Compose(object):
    # Composes transforms: transforms.Compose([transforms.RandScale([0.5, 2.0]), transforms.ToTensor()])
    def __init__(self, transforms):
        self.transforms = [
            (instantiate_from_config(transform) if isinstance(transform, dict) else transform)
            for transform in transforms
        ]

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        for t in self.transforms:
            (image, intrinsics, depth, depth_mask, normal, other_labels, transform_info) = t(
                image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
            )
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __init__(self, **kwargs):
        return

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):

        if image is not None:
            assert isinstance(image, np.ndarray)
            assert len(image.shape) == 3 or len(image.shape) == 2
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()

        if intrinsics is not None:
            assert isinstance(intrinsics, list)
            assert len(intrinsics) == 4
            intrinsics = torch.tensor(intrinsics, dtype=torch.float)

        if depth is not None:
            assert isinstance(depth, np.ndarray)
            assert len(depth.shape) == 2
            depth = np.expand_dims(depth, axis=0)
            depth = torch.from_numpy(depth).float()

        if depth_mask is not None:
            assert isinstance(depth_mask, np.ndarray)
            assert len(depth_mask.shape) == 2
            depth_mask = np.expand_dims(depth_mask, axis=0)
            depth_mask = torch.from_numpy(depth_mask).long()

        if normal is not None:
            normal = torch.from_numpy(normal.transpose((2, 0, 1))).float()

        if other_labels is not None:
            for i, label in enumerate(other_labels):
                if label is None:
                    continue
                if len(label.shape) == 2:
                    other_labels[i] = torch.from_numpy(label).unsqueeze(0).float()
                else:
                    other_labels[i] = torch.from_numpy(label.transpose((2, 0, 1))).float()
        
        if "image_backup" in transform_info:
            transform_info["image_backup"] = torch.from_numpy(transform_info["image_backup"].transpose((2, 0, 1))).float()

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], **kwargs):
        assert len(mean) == len(std)
        self.mean = torch.tensor(mean).float()[:, None, None]
        self.std = torch.tensor(std).float()[:, None, None]

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        transform_info["image_show"] = image
        image = torch.div((image - self.mean), self.std)
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h) + 0.5).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w) + 0.5).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth


class Resize(object):
    def __init__(self, width, height, is_lidar=False, backup=False):
        self.width = width
        self.height = height
        self.is_lidar = is_lidar
        self.backup = backup

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):

        if image is not None:
            assert len(image.shape) == 3
            ori_h, ori_w, _ = image.shape
            image = cv2.resize(
                image,
                dsize=(self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
            cur_h, cur_w, _ = image.shape
            ratio_h, ratio_w = 1.0 * cur_h / ori_h, 1.0 * cur_w / ori_w

            transform_info["resize_ratio_h"] = ratio_h
            transform_info["resize_ratio_w"] = ratio_w

        if intrinsics is not None:  # NOTE: maybe, it is error
            intrinsics[0] = intrinsics[0] * ratio_w
            intrinsics[1] = intrinsics[1] * ratio_h
            intrinsics[2] = intrinsics[2] * ratio_w
            intrinsics[3] = intrinsics[3] * ratio_h

        if depth is not None:
            assert len(depth.shape) == 2
            if self.is_lidar:
                depth = resize_depth_preserve(depth, (self.height, self.width))
            else:
                depth = cv2.resize(
                    depth,
                    dsize=(self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if depth_mask is not None:
            if self.is_lidar:
                depth_mask = resize_depth_preserve(depth_mask, (self.height, self.width))
            else:
                depth_mask = cv2.resize(
                    depth_mask.astype(int),
                    dsize=(self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if normal is not None:
            normal = cv2.resize(
                normal,
                dsize=(self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )

        for i, other_label in enumerate(other_labels):
            if other_label is not None:
                label_name = transform_info["other_labels"][i]
                other_labels[i] = cv2.resize(
                    other_label,
                    dsize=(self.width, self.height),
                    interpolation=cv2.INTER_NEAREST if label_name not in ["invalid_mask"] else cv2.INTER_LINEAR,
                )
        
        if self.backup:
            transform_info["image_backup"] = image.copy()

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class ResizePatch(object):
    def __init__(self, width, height, low_resolution=True, is_lidar=False, patch_size=None):
        self.width = width
        self.height = height
        self.is_lidar = is_lidar
        self.patch_size = patch_size
        self.low_resolution = low_resolution

    def update_max_size(self, max_size):
        cur_max = max(self.width, self.height)
        scale = max_size / cur_max
        self.width = int(self.width * scale)
        self.height = int(self.height * scale)
        if self.patch_size is not None:
            self.width = min(int(self.width / self.patch_size + 0.5) * self.patch_size, max_size)
            self.height = min(int(self.height / self.patch_size + 0.5) * self.patch_size, max_size)

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        new_height = self.height
        new_width = self.width

        if image is not None:
            assert len(image.shape) == 3
            ori_h, ori_w, _ = image.shape

            if self.patch_size is not None:
                if self.low_resolution and (ori_h < self.height or ori_w < self.width):
                    new_height = int(ori_h / self.patch_size) * self.patch_size
                    new_width = int(ori_w / self.patch_size) * self.patch_size

            image = cv2.resize(
                image,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            cur_h, cur_w, _ = image.shape
            ratio_h, ratio_w = 1.0 * cur_h / ori_h, 1.0 * cur_w / ori_w

            transform_info["resize_ratio_h"] = ratio_h
            transform_info["resize_ratio_w"] = ratio_w

        if intrinsics is not None:  # NOTE: maybe, it is error
            intrinsics[0] = intrinsics[0] * ratio_w
            intrinsics[1] = intrinsics[1] * ratio_h
            intrinsics[2] = intrinsics[2] * ratio_w
            intrinsics[3] = intrinsics[3] * ratio_h

        if depth is not None:
            assert len(depth.shape) == 2

            if image is None and self.patch_size is not None:
                ori_h, ori_w = depth.shape
                if ori_h < self.height or ori_w < self.width:
                    new_height = int(ori_h / self.patch_size) * self.patch_size
                    new_width = int(ori_w / self.patch_size) * self.patch_size

            if self.is_lidar:
                depth = resize_depth_preserve(depth, (new_height, new_width))
            else:
                depth = cv2.resize(
                    depth,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if depth_mask is not None:
            if self.is_lidar:
                depth_mask = resize_depth_preserve(depth_mask, (new_height, new_width))
            else:
                depth_mask = cv2.resize(
                    depth_mask.astype(int),
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if normal is not None:
            normal = cv2.resize(
                normal,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )

        for i, other_label in enumerate(other_labels):
            if other_label is not None:
                label_name = transform_info["other_labels"][i]
                other_labels[i] = cv2.resize(
                    other_label,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST if label_name not in ["invalid_mask"] else cv2.INTER_LINEAR,
                )

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class ResizeSR(object):
    def __init__(self, width, height, depth_width=None, depth_height=None, is_lidar=False):
        self.width = width
        self.height = height
        self.is_lidar = is_lidar

        self.depth_width = depth_width
        self.depth_height = depth_height

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):

        assert len(image.shape) == 3
        ori_h, ori_w, _ = image.shape
        image = cv2.resize(
            image,
            dsize=(self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

        if intrinsics is not None:
            prompt_intrinsics = deepcopy(intrinsics)
            ratio_h, ratio_w = 1.0 * self.height / ori_h, 1.0 * self.width / ori_w
            prompt_intrinsics[0] = prompt_intrinsics[0] * ratio_w
            prompt_intrinsics[1] = prompt_intrinsics[1] * ratio_h
            prompt_intrinsics[2] = prompt_intrinsics[2] * ratio_w
            prompt_intrinsics[3] = prompt_intrinsics[3] * ratio_h
            transform_info["prompt_intrinsics"] = prompt_intrinsics

        if self.depth_width is not None and self.depth_height is not None:
            depth_width = self.depth_width
            depth_height = self.depth_height
        else:
            depth_width = self.width
            depth_height = self.height

        ratio_h, ratio_w = 1.0 * depth_height / ori_h, 1.0 * depth_width / ori_w
        # transform_info["resize_ratio_h"] = ratio_h
        # transform_info["resize_ratio_w"] = ratio_w

        if intrinsics is not None:  # NOTE: maybe, it is error
            intrinsics[0] = intrinsics[0] * ratio_w
            intrinsics[1] = intrinsics[1] * ratio_h
            intrinsics[2] = intrinsics[2] * ratio_w
            intrinsics[3] = intrinsics[3] * ratio_h

        if depth is not None:
            assert len(depth.shape) == 2
            if self.is_lidar:
                depth = resize_depth_preserve(depth, (depth_height, depth_width))
            else:
                depth = cv2.resize(
                    depth,
                    dsize=(depth_width, depth_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if depth_mask is not None:
            if self.is_lidar:
                depth_mask = resize_depth_preserve(depth_mask, (depth_height, depth_width))
            else:
                depth_mask = cv2.resize(
                    depth_mask.astype(int),
                    dsize=(depth_width, depth_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if normal is not None:
            normal = cv2.resize(
                normal,
                dsize=(depth_width, depth_height),
                interpolation=cv2.INTER_LINEAR,
            )

        for i, other_label in enumerate(other_labels):
            if other_label is not None:
                label_name = transform_info["other_labels"][i]
                other_labels[i] = cv2.resize(
                    other_label,
                    dsize=(depth_width, depth_height),
                    interpolation=cv2.INTER_NEAREST if label_name not in ["invalid_mask"] else cv2.INTER_LINEAR,
                )

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class ResizeKeepRatio(object):
    def __init__(
        self, max_size, patch_size=14, is_lidar=False, low_resolution=False, aspect_ratio=None, backup=False, names=None
    ):
        self.max_size = max_size
        self.patch_size = patch_size
        self.is_lidar = is_lidar
        self.low_resolution = low_resolution
        self.aspect_ratio = aspect_ratio
        self.backup = backup
        self.names = names

    def update_aspect_ratio(self, aspect_ratio=None):
        self.aspect_ratio = aspect_ratio

    def update_max_size(self, max_size):
        self.max_size = max_size

    def get_shape(self, image):
        height, width = image.shape[:2]

        if self.aspect_ratio is not None:
            if width >= height:
                if self.low_resolution and width < self.max_size:
                    new_width = int(width / self.patch_size) * self.patch_size
                else:
                    new_width = self.max_size
                new_height = int(new_width / self.aspect_ratio / self.patch_size) * self.patch_size
            else:
                if self.low_resolution and height < self.max_size:
                    new_height = int(height / self.patch_size) * self.patch_size
                else:
                    new_height = self.max_size
                new_width = int(new_height / self.aspect_ratio / self.patch_size) * self.patch_size

        elif self.low_resolution and width < self.max_size and height < self.max_size:
            new_width = int(width / self.patch_size) * self.patch_size
            new_height = int(height / self.patch_size) * self.patch_size
        elif width >= height:
            new_width = self.max_size
            new_height = round(height * (new_width / width) / self.patch_size) * self.patch_size
        else:
            new_height = self.max_size
            new_width = round(width * (new_height / height) / self.patch_size) * self.patch_size

        return new_width, new_height

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):

        if self.names is not None and transform_info.get("name", None) not in self.names:
            return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        if image is not None:
            assert len(image.shape) == 3
            ori_h, ori_w, _ = image.shape
            new_width, new_height = self.get_shape(image)
            image = cv2.resize(
                image,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
            cur_h, cur_w, _ = image.shape
            ratio_h, ratio_w = 1.0 * cur_h / ori_h, 1.0 * cur_w / ori_w

            transform_info["resize_ratio_h"] = ratio_h
            transform_info["resize_ratio_w"] = ratio_w
        else:
            new_width = new_height = None

        if intrinsics is not None:  # NOTE: maybe, it is error
            intrinsics[0] = intrinsics[0] * ratio_w
            intrinsics[1] = intrinsics[1] * ratio_h
            intrinsics[2] = intrinsics[2] * ratio_w
            intrinsics[3] = intrinsics[3] * ratio_h

        if depth is not None:
            assert len(depth.shape) == 2
            if new_width is None or new_height is None:
                new_width, new_height = self.get_shape(depth)

            if self.is_lidar:
                depth = resize_depth_preserve(depth, (new_height, new_width))
            else:
                depth = cv2.resize(
                    depth,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if depth_mask is not None:
            if self.is_lidar:
                depth_mask = resize_depth_preserve(depth_mask, (new_height, new_width))
            else:
                depth_mask = cv2.resize(
                    depth_mask.astype(int),
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if normal is not None:
            normal = cv2.resize(
                normal,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )

        for i, other_label in enumerate(other_labels):
            if other_label is not None:
                label_name = transform_info["other_labels"][i]
                other_labels[i] = cv2.resize(
                    other_label,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST if label_name not in ["invalid_mask"] else cv2.INTER_LINEAR,
                )
        
        if self.backup:
            transform_info["image_backup"] = image.copy()

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5, **kwargs):
        self.p = prob

    def __call__(
        self,
        image,
        intrinsics,
        depth,
        depth_mask,
        normal,
        other_labels,
        transform_info,
    ):
        if "horizontal_flip" in transform_info:
            if not transform_info["horizontal_flip"]:
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
        else:
            if random.random() > self.p:
                transform_info["horizontal_flip"] = False
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        transform_info["horizontal_flip"] = True

        if image is not None:
            image = cv2.flip(image, 1)

        if intrinsics is not None:
            h, w, _ = image.shape
            intrinsics[2] = w - 1 - intrinsics[2]
            # intrinsics[3] = h - intrinsics[3]

        if depth is not None:
            depth = cv2.flip(depth, 1)

        if depth_mask is not None:
            depth_mask = cv2.flip(depth_mask, 1)

        if normal is not None:
            normal = cv2.flip(normal, 1)
            normal[:, :, 0] = -normal[
                :, :, 0
            ]  # NOTE: check the direction of normal coordinates axis, this is used in https://github.com/baegwangbin/surface_normal_uncertainty

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                other_labels[i] = cv2.flip(other_lab, 1)
        
        if "image_backup" in transform_info:
            transform_info["image_backup"] = cv2.flip(transform_info["image_backup"], 1)

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5, **kwargs):
        self.p = prob

    def __call__(
        self,
        image,
        intrinsics,
        depth,
        depth_mask,
        normal,
        other_labels,
        transform_info,
    ):
        if "vertical_flip" in transform_info:
            if not transform_info["vertical_flip"]:
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
        else:
            if random.random() > self.p:
                transform_info["vertical_flip"] = False
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        transform_info["vertical_flip"] = True

        if image is not None:
            image = cv2.flip(image, 0)  # 0 表示垂直翻转

        if intrinsics is not None:
            h, w = image.shape[:2]
            # 只修改 cy：主点 y 坐标关于图像中心对称
            intrinsics[3] = h - 1 - intrinsics[3]  # intrinsics = [fx, fy, cx, cy]

        if depth is not None:
            depth = cv2.flip(depth, 0)

        if depth_mask is not None:
            depth_mask = cv2.flip(depth_mask, 0)

        if normal is not None:
            normal = cv2.flip(normal, 0)
            normal[:, :, 1] = -normal[:, :, 1]  # 垂直翻转：y 分量取反

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                other_labels[i] = cv2.flip(other_lab, 0)
        
        if "image_backup" in transform_info:
            transform_info["image_backup"] = cv2.flip(transform_info["image_backup"], 0)

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class KittiBenchmarkCrop(object):
    def __init__(self):
        pass

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        assert image is not None and len(image.shape) == 3

        KB_CROP_HEIGHT = 352
        KB_CROP_WIDTH = 1216

        height, width, _ = image.shape
        top_margin = int(height - KB_CROP_HEIGHT)
        left_margin = int((width - KB_CROP_WIDTH) / 2)

        image = image[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]

        if intrinsics is not None:
            intrinsics[2] = image.shape[1] / 2.0
            intrinsics[3] = image.shape[0] / 2.0

        if depth is not None:
            assert len(depth.shape) == 2
            depth = depth[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]

        if depth_mask is not None:
            depth_mask = depth_mask[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]

        if normal is not None:
            normal = normal[
                top_margin : top_margin + KB_CROP_HEIGHT,
                left_margin : left_margin + KB_CROP_WIDTH,
            ]

        for i, other_label in enumerate(other_labels):
            if other_label is not None:
                label_name = transform_info["other_labels"][i]
                other_labels[i] = other_label[
                    top_margin : top_margin + KB_CROP_HEIGHT,
                    left_margin : left_margin + KB_CROP_WIDTH,
                ]

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class RandomCrop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, crop_size, crop_type="center", padding=[0, 0, 0], **kwargs):
        if isinstance(crop_size, int):
            self.crop_h = crop_size
            self.crop_w = crop_size
        elif (
            isinstance(crop_size, collections.Iterable)
            and len(crop_size) == 2
            and isinstance(crop_size[0], int)
            and isinstance(crop_size[1], int)
            and crop_size[0] > 0
            and crop_size[1] > 0
        ):
            self.crop_h = crop_size[0]
            self.crop_w = crop_size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == "center" or crop_type == "rand" or crop_type == "rand_in_field":
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center | rand_in_field \n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

    def cal_padding_paras(self, h, w):
        # padding if current size is not satisfied
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        return pad_h, pad_w, pad_h_half, pad_w_half

    def cal_cropping_paras(self, h, w, intrinsics):
        u0 = intrinsics[2]
        v0 = intrinsics[3]
        if self.crop_type == "rand":
            h_min = 0
            h_max = h - self.crop_h
            w_min = 0
            w_max = w - self.crop_w
        elif self.crop_type == "center":
            h_min = (h - self.crop_h) / 2
            h_max = (h - self.crop_h) / 2
            w_min = (w - self.crop_w) / 2
            w_max = (w - self.crop_w) / 2
        else:  # rand in field
            h_min = min(max(0, v0 - 0.75 * self.crop_h), h - self.crop_h)
            h_max = min(max(v0 - 0.25 * self.crop_h, 0), h - self.crop_h)
            w_min = min(max(0, u0 - 0.75 * self.crop_w), w - self.crop_w)
            w_max = min(max(u0 - 0.25 * self.crop_w, 0), w - self.crop_w)

        h_off = random.randint(int(h_min), int(h_max))
        w_off = random.randint(int(w_min), int(w_max))
        return h_off, w_off

    def main_data_transform(
        self,
        image,
        label,
        mask,
        intrinsics,
        pad_h,
        pad_w,
        pad_h_half,
        pad_w_half,
        h_off,
        w_off,
    ):

        # padding if current size is not satisfied
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (
                    RuntimeError(
                        "depthtransform.Crop() need padding while padding argument is None\n"
                    )
                )
            image = cv2.copyMakeBorder(
                image,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=self.padding,
            )
            if label is not None:
                label = cv2.copyMakeBorder(
                    label,
                    pad_h_half,
                    pad_h - pad_h_half,
                    pad_w_half,
                    pad_w - pad_w_half,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            if mask is not None:
                mask = cv2.copyMakeBorder(
                    mask,
                    pad_h_half,
                    pad_h - pad_h_half,
                    pad_w_half,
                    pad_w - pad_w_half,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )

        # cropping
        image = image[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        if label is not None:
            label = label[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        if mask is not None:
            mask = mask[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]

        if intrinsics is not None:
            intrinsics[2] = intrinsics[2] + pad_w_half - w_off
            intrinsics[3] = intrinsics[3] + pad_h_half - h_off
        return image, label, mask, intrinsics

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        target_h, target_w, _ = image.shape
        pad_h, pad_w, pad_h_half, pad_w_half = self.cal_padding_paras(target_h, target_w)
        h_off, w_off = self.cal_cropping_paras(target_h + pad_h, target_w + pad_w, intrinsics)

        image, depth, depth_mask, intrinsics = self.main_data_transform(
            image,
            depth,
            depth_mask,
            intrinsics,
            pad_h,
            pad_w,
            pad_h_half,
            pad_w_half,
            h_off,
            w_off,
        )

        pad = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        if normal is not None:
            # padding if current size is not satisfied
            normal = cv2.copyMakeBorder(
                normal,
                pad_h_half,
                pad_h - pad_h_half,
                pad_w_half,
                pad_w - pad_w_half,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            normal = normal[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                # padding if current size is not satisfied
                other_lab = cv2.copyMakeBorder(
                    other_lab,
                    pad_h_half,
                    pad_h - pad_h_half,
                    pad_w_half,
                    pad_w - pad_w_half,
                    cv2.BORDER_CONSTANT,
                    value=-1,
                )
                other_labels[i] = other_lab[
                    h_off : h_off + self.crop_h, w_off : w_off + self.crop_w
                ]

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class FOVCrop(object):
    def __init__(self, fx=1122.73, fy=959.75, fov_x=60, fov_y=46.9, **kwargs):

        self.crop_width = int(math.tan(fov_x / 180 * math.pi / 2) * fx * 2)
        self.crop_height = int(math.tan(fov_y / 180 * math.pi / 2) * fy * 2)

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        height, width, _ = image.shape

        if "fov_crop_width" in transform_info:
            CROP_WIDTH = transform_info["fov_crop_width"]
            CROP_HEIGHT = transform_info["fov_crop_height"]
        else:
            CROP_HEIGHT = min(height, self.crop_height)
            CROP_WIDTH = min(width, self.crop_width)

            transform_info["fov_crop_width"] = CROP_WIDTH
            transform_info["fov_crop_height"] = CROP_HEIGHT

        top_margin = int((height - CROP_HEIGHT) / 2)
        left_margin = int((width - CROP_WIDTH) / 2)

        image = image[
            top_margin : top_margin + CROP_HEIGHT,
            left_margin : left_margin + CROP_WIDTH,
        ]

        if intrinsics is not None:
            intrinsics[2] -= left_margin
            intrinsics[3] -= top_margin

        if depth is not None:
            assert len(depth.shape) == 2
            depth = depth[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if depth_mask is not None:
            assert len(depth_mask.shape) == 2
            depth_mask = depth_mask[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if normal is not None:
            normal = normal[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                if other_lab is not None:
                    # padding if current size is not satisfied
                    assert len(other_lab.shape) == 2
                    other_labels[i] = other_lab[
                        top_margin : top_margin + CROP_HEIGHT,
                        left_margin : left_margin + CROP_WIDTH,
                    ]

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class FOVRandomCrop(object):
    def __init__(self, fov_x, fov_y, prob=0.5, names=None, **kwargs):

        self.fov_x = fov_x
        self.fov_y = fov_y
        self.prob = prob
        self.names = names

    def get_fov(self, transform_info):
        if "fov_x" in transform_info:
            fov_x = transform_info["fov_x"]
        elif isinstance(self.fov_x, (list, tuple)):
            fov_x = random.uniform(min(self.fov_x), max(self.fov_x))
        else:
            fov_x = self.fov_x

        if "fov_y" in transform_info:
            fov_y = transform_info["fov_y"]
        elif isinstance(self.fov_y, (list, tuple)):
            fov_y = random.uniform(min(self.fov_y), max(self.fov_y))
        else:
            fov_y = self.fov_y

        return fov_x, fov_y

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        if self.names is not None and transform_info.get("name", None) not in self.names:
            return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        if "FOVRandomCrop" in transform_info and not transform_info["FOVRandomCrop"]:
            return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        if "FOVRandomCrop" not in transform_info:
            prob = random.random()
            if prob > self.prob:
                transform_info["FOVRandomCrop"] = False
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
            else:
                transform_info["FOVRandomCrop"] = True

        height, width, _ = image.shape

        if "fov_crop_width" in transform_info:
            CROP_WIDTH = transform_info["fov_crop_width"]
            CROP_HEIGHT = transform_info["fov_crop_height"]
        else:
            fov_x, fov_y = self.get_fov(transform_info)
            fx, fy = intrinsics[0], intrinsics[1]

            crop_width = int(math.tan(fov_x / 180 * math.pi / 2) * fx * 2)
            crop_height = int(math.tan(fov_y / 180 * math.pi / 2) * fy * 2)

            CROP_WIDTH = min(width, crop_width)
            CROP_HEIGHT = min(height, crop_height)

            transform_info["fov_crop_width"] = CROP_WIDTH
            transform_info["fov_crop_height"] = CROP_HEIGHT

        top_margin = int((height - CROP_HEIGHT) / 2)
        left_margin = int((width - CROP_WIDTH) / 2)

        image = image[
            top_margin : top_margin + CROP_HEIGHT,
            left_margin : left_margin + CROP_WIDTH,
        ]

        if intrinsics is not None:
            intrinsics[2] -= left_margin
            intrinsics[3] -= top_margin

        if depth is not None:
            assert len(depth.shape) == 2
            depth = depth[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if depth_mask is not None:
            assert len(depth_mask.shape) == 2
            depth_mask = depth_mask[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if normal is not None:
            # assert len(normal.shape) == 2
            normal = normal[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                if other_lab is not None:
                    # padding if current size is not satisfied
                    assert len(other_lab.shape) == 2
                    other_labels[i] = other_lab[
                        top_margin : top_margin + CROP_HEIGHT,
                        left_margin : left_margin + CROP_WIDTH,
                    ]

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class KeepRatioRandomCrop(object):
    def __init__(self, max_size, crop_size=None, patch_size=14, crop_type="center", prob=0.5):
        self.max_size = max_size
        self.crop_size = crop_size
        self.patch_size = patch_size
        assert self.max_size % self.patch_size == 0
        if self.crop_size is not None:
            for csize in self.crop_size:
                assert csize % self.patch_size == 0
        self.crop_type = crop_type
        self.prob = prob

    def cal_cropping_paras(self, h, w, crop_h, crop_w, intrinsics):
        if self.crop_type == "rand":
            h_min = 0
            h_max = h - crop_h
            w_min = 0
            w_max = w - crop_w
        elif self.crop_type == "center":
            h_min = (h - crop_h) / 2
            h_max = (h - crop_h) / 2
            w_min = (w - crop_w) / 2
            w_max = (w - crop_w) / 2
        else:  # rand in field
            u0 = intrinsics[2]
            v0 = intrinsics[3]
            h_min = min(max(0, v0 - 0.75 * crop_h), h - crop_h)
            h_max = min(max(v0 - 0.25 * crop_h, 0), h - crop_h)
            w_min = min(max(0, u0 - 0.75 * crop_w), w - crop_w)
            w_max = min(max(u0 - 0.25 * crop_w, 0), w - crop_w)

        h_off = random.randint(int(h_min), int(h_max))
        w_off = random.randint(int(w_min), int(w_max))
        return h_off, w_off

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        if "KeepRatioRandomCropFlag" in transform_info:
            if not transform_info["KeepRatioRandomCropFlag"]:
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
        else:
            if random.random() > self.prob:
                transform_info["KeepRatioRandomCropFlag"] = False
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
            transform_info["KeepRatioRandomCropFlag"] = True

        if "KeepRatioRandomCrop" in transform_info:
            top, bottom, left, right = transform_info["KeepRatioRandomCrop"]
        else:
            height, width, _ = image.shape
            # assert width >= height

            if self.crop_size is not None:
                new_width, new_height = self.crop_size
                new_height = min(new_height, height)
                new_width = min(new_width, width)

            elif max([height, width]) < self.max_size:
                new_height = (height // self.patch_size) * self.patch_size
                new_width = (width // self.patch_size) * self.patch_size

            elif width >= height:
                new_width = self.max_size
                new_height = int(height * (new_width / width) / self.patch_size) * self.patch_size

            else:
                new_height = self.max_size
                new_width = int(width * (new_height / height) / self.patch_size) * self.patch_size

            top, left = self.cal_cropping_paras(
                h=height, w=width, crop_h=new_height, crop_w=new_width, intrinsics=intrinsics
            )

            right = left + new_width
            bottom = top + new_height

            transform_info["KeepRatioRandomCrop"] = [top, bottom, left, right]

        image = image[top:bottom, left:right]

        if intrinsics is not None:
            intrinsics[2] -= left
            intrinsics[3] -= top

        if depth is not None:
            assert len(depth.shape) == 2
            depth = depth[top:bottom, left:right]

        if depth_mask is not None:
            assert len(depth_mask.shape) == 2
            depth_mask = depth_mask[top:bottom, left:right]

        if normal is not None:
            normal = normal[top:bottom, left:right]

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                if other_lab is not None:
                    # padding if current size is not satisfied
                    assert len(other_lab.shape) == 2
                    other_labels[i] = other_lab[top:bottom, left:right]
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

class RandomResizeKeepRatioCropSize(object):
    def __init__(
        self, max_size, patch_size=14, is_lidar=False, low_resolution=False, aspect_ratio=None,
        crop_size=None, crop_type="center", crop_prob=0.5,
    ):
        self.resizer = ResizeKeepRatio(max_size, patch_size, is_lidar, low_resolution, aspect_ratio)
        self.cropper = KeepRatioRandomCrop(max_size, crop_size, patch_size, crop_type, prob=crop_prob)
    
    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        (
            image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
        ) = self.cropper(image, intrinsics, depth, depth_mask, normal, other_labels, transform_info)

        if not transform_info.get("KeepRatioRandomCropFlag", False):
            (
                image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
            ) = self.resizer(image, intrinsics, depth, depth_mask, normal, other_labels, transform_info)

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

class RandomPatchChannelPermutation(object):
    def __init__(self, patch_range=(0.05, 0.25), patch_nums=[1, 3], prob=0.5, invert_prob=0.5):
        self.patch_range = patch_range if isinstance(patch_range, (list, tuple)) else [patch_range, patch_range]
        self.prob = prob
        self.invert_prob = invert_prob
        self.patch_nums = patch_nums if isinstance(patch_nums, (list, tuple)) else [patch_nums, patch_nums]
        self.patch_nums[1] = self.patch_nums[1] + 1

    def patch_permute(self, image, patch_h, patch_w):
        h, w, c = image.shape
        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)
        new_channel = np.random.permutation(c)
        image[y:y+patch_h, x:x+patch_w, :] = image[y:y+patch_h, x:x+patch_w, new_channel]
        if random.random() > self.invert_prob:
            image[y:y+patch_h, x:x+patch_w, :] = 255 - image[y:y+patch_h, x:x+patch_w, :]
        return image

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        if "RandomPatchChannelPermutation" in transform_info:
            if not transform_info["RandomPatchChannelPermutation"]:
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
        else:
            if random.random() > self.prob:
                transform_info["RandomPatchChannelPermutation"] = False
                return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
            transform_info["RandomPatchChannelPermutation"] = True
        h, w = image.shape[:2]
        patch_w = int(np.random.uniform(*self.patch_range) * w)
        patch_h = int(np.random.uniform(*self.patch_range) * h)

        patch_num = np.random.randint(*self.patch_nums)
        for i in range(patch_num):
            image = self.patch_permute(image, patch_h, patch_w)
        
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class MapAnyThingResizeCrop(object):
    def __init__(self, resolution_set=518, is_lidar=False):
        self.is_lidar = is_lidar
        self.resolution_set = resolution_set

        self.RESOLUTION_MAPPINGS = {
            518: {
                1.000: (518, 518),  # 1:1
                1.321: (518, 392),  # 4:3
                1.542: (518, 336),  # 3:2
                1.762: (518, 294),  # 16:9
                2.056: (518, 252),  # 2:1
                3.083: (518, 168),  # 3.2:1
                0.757: (392, 518),  # 3:4
                0.649: (336, 518),  # 2:3
                0.567: (294, 518),  # 9:16
                0.486: (252, 518),  # 1:2
            },
            512: {
                1.000: (512, 512),  # 1:1
                1.333: (512, 384),  # 4:3
                1.524: (512, 336),  # 3:2
                1.778: (512, 288),  # 16:9
                2.000: (512, 256),  # 2:1
                3.200: (512, 160),  # 3.2:1
                0.750: (384, 512),  # 3:4
                0.656: (336, 512),  # 2:3
                0.562: (288, 512),  # 9:16
                0.500: (256, 512),  # 1:2
            },
        }
        self.ASPECT_RATIO_KEYS = {
            518: sorted(self.RESOLUTION_MAPPINGS[518].keys()),
            512: sorted(self.RESOLUTION_MAPPINGS[512].keys()),
        }
    
    def find_closest_aspect_ratio(self, aspect_ratio):
        """
        Find the closest aspect ratio from the resolution mappings using efficient key lookup.

        Args:
            aspect_ratio (float): Target aspect ratio
            resolution_set (int): Resolution set to use (518 or 512)

        Returns:
            tuple: (target_width, target_height) from the resolution mapping
        """
        aspect_keys = self.ASPECT_RATIO_KEYS[self.resolution_set]

        # Find the closest aspect ratio key using binary search approach
        closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

        return self.RESOLUTION_MAPPINGS[self.resolution_set][closest_key]

    def get_shape(self, image):
        height, width = image.shape[:2]
        
        aspect_ratios = width / height
        target_width, target_height = self.find_closest_aspect_ratio(aspect_ratios)

        return target_width, target_height

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):

        if image is not None:
            assert len(image.shape) == 3
            ori_h, ori_w, _ = image.shape
            new_width, new_height = self.get_shape(image)

            import PIL
            from PIL import Image
            image = Image.fromarray(image)
            image = image.resize(
                tuple((new_width, new_height)), resample=PIL.Image.LANCZOS
            )
            image = np.asarray(image)
            cur_h, cur_w, _ = image.shape
            ratio_h, ratio_w = 1.0 * cur_h / ori_h, 1.0 * cur_w / ori_w

            transform_info["resize_ratio_h"] = ratio_h
            transform_info["resize_ratio_w"] = ratio_w
        else:
            new_width = new_height = None

        if intrinsics is not None:  # NOTE: maybe, it is error
            intrinsics[0] = intrinsics[0] * ratio_w
            intrinsics[1] = intrinsics[1] * ratio_h
            intrinsics[2] = intrinsics[2] * ratio_w
            intrinsics[3] = intrinsics[3] * ratio_h

        if depth is not None:
            assert len(depth.shape) == 2
            if new_width is None or new_height is None:
                new_width, new_height = self.get_shape(depth)

            if self.is_lidar:
                depth = resize_depth_preserve(depth, (new_height, new_width))
            else:
                depth = cv2.resize(
                    depth,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if depth_mask is not None:
            if self.is_lidar:
                depth_mask = resize_depth_preserve(depth_mask, (new_height, new_width))
            else:
                depth_mask = cv2.resize(
                    depth_mask.astype(int),
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST,
                )

        if normal is not None:
            normal = cv2.resize(
                normal,
                dsize=(new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )

        for i, other_label in enumerate(other_labels):
            if other_label is not None:
                label_name = transform_info["other_labels"][i]
                other_labels[i] = cv2.resize(
                    other_label,
                    dsize=(new_width, new_height),
                    interpolation=cv2.INTER_NEAREST if label_name not in ["invalid_mask"] else cv2.INTER_LINEAR,
                )

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class AspectRatioCrop(object):
    def __init__(self, aspect_ratios, patch_size=14, **kwargs):
        self.aspect_ratios = aspect_ratios
        self.patch_size = patch_size

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        height, width, _ = image.shape

        aspect_ratio = 1.0 * width / height
        match_index = np.argmin([abs(aspect_ratio - r) for r in self.aspect_ratios])
        target_ratio = self.aspect_ratios[match_index]

        # 有两种可能：按高度裁宽度，或按宽度裁高度
        if aspect_ratio > target_ratio:
            # 当前图像比目标更“宽” → 需要裁剪宽度（以高度为基准）
            crop_height = height
            crop_width = int(height * target_ratio)
        else:
            # 当前图像比目标更“高” → 需要裁剪高度（以宽度为基准）
            crop_width = width
            crop_height = int(width / target_ratio)
        
        crop_width = int(np.ceil(1.0 * crop_width / self.patch_size) * self.patch_size)
        crop_height = int(np.ceil(1.0 * crop_height / self.patch_size) * self.patch_size)
        # print(aspect_ratio, width, height, crop_width, crop_height)

        CROP_HEIGHT = min(height, crop_height)
        CROP_WIDTH = min(width, crop_width)

        transform_info["fov_crop_width"] = CROP_WIDTH
        transform_info["fov_crop_height"] = CROP_HEIGHT

        top_margin = int((height - CROP_HEIGHT) / 2)
        left_margin = int((width - CROP_WIDTH) / 2)

        image = image[
            top_margin : top_margin + CROP_HEIGHT,
            left_margin : left_margin + CROP_WIDTH,
        ]

        if intrinsics is not None:
            intrinsics[2] -= left_margin
            intrinsics[3] -= top_margin

        if depth is not None:
            assert len(depth.shape) == 2
            depth = depth[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if depth_mask is not None:
            assert len(depth_mask.shape) == 2
            depth_mask = depth_mask[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if normal is not None:
            normal = normal[
                top_margin : top_margin + CROP_HEIGHT,
                left_margin : left_margin + CROP_WIDTH,
            ]

        if other_labels is not None:
            for i, other_lab in enumerate(other_labels):
                if other_lab is not None:
                    # padding if current size is not satisfied
                    assert len(other_lab.shape) == 2
                    other_labels[i] = other_lab[
                        top_margin : top_margin + CROP_HEIGHT,
                        left_margin : left_margin + CROP_WIDTH,
                    ]

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

