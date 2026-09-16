import random

import cv2
import numpy as np
from imgaug import augmenters as iaa


class RandomBlur(object):
    def __init__(
        self, aver_kernal=(2, 10), motion_kernal=(5, 15), angle=[-80, 80], prob=0.3, **kwargs
    ):

        gaussian_blur = iaa.AverageBlur(k=aver_kernal)
        motion_blur = iaa.MotionBlur(k=motion_kernal, angle=angle)
        zoom_blur = iaa.imgcorruptlike.ZoomBlur(severity=1)
        self.prob = prob
        self.blurs = [gaussian_blur, motion_blur, zoom_blur]

    def blur(self, imgs, id):
        blur_mtd = self.blurs[id]
        return blur_mtd(image=imgs)

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        if "RandomBlur" in transform_info:
            id = transform_info["RandomBlur"]
            if id != -1:
                image = self.blur(image, id)
            return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        prob = random.random()
        if prob < self.prob:
            id = random.randint(0, len(self.blurs) - 1)
            transform_info["RandomBlur"] = id
            image = self.blur(image, id)
        else:
            transform_info["RandomBlur"] = -1
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class RGBCompresion(object):
    def __init__(self, prob=0.1, compression=(0, 50), **kwargs):
        self.rgb_compress = iaa.Sequential(
            [
                iaa.JpegCompression(compression=compression),
            ],
            random_order=True,
        )
        self.prob = prob

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        if "RGBCompresion" in transform_info:
            if transform_info["RGBCompresion"]:
                image = self.rgb_compress(image=image)
            return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        if random.random() < self.prob:
            image = self.rgb_compress(image=image)
            transform_info["RGBCompresion"] = True
        else:
            transform_info["RGBCompresion"] = False
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
        to_gray_prob=0.3,
        distortion_prob=0.3,
        **kwargs
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.gray_aug = iaa.Grayscale(alpha=(0.8, 1.0))
        self.to_gray_prob = to_gray_prob
        self.distortion_prob = distortion_prob

    def convert(self, img, alpha=1.0, beta=0.0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img, beta, do):
        """Brightness distortion."""
        if do:
            # beta = random.uniform(-self.brightness_delta,
            #                         self.brightness_delta)
            img = self.convert(img, beta=beta)
        return img

    def contrast(self, img, alpha, do):
        """Contrast distortion."""
        if do:
            # alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img = self.convert(img, alpha=alpha)
        return img

    def saturation(self, img, alpha, do):
        """Saturation distortion."""
        if do:
            # alpha = random.uniform(self.saturation_lower,
            #                         self.saturation_upper)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(img[:, :, 1], alpha=alpha)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img, rand_hue, do):
        """Hue distortion."""
        if do:
            # rand_hue = random.randint(-self.hue_delta, self.hue_delta)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def rgb2gray(self, img):
        img = self.gray_aug(image=img)
        return img

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        """Call function to perform photometric distortion on image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with image distorted.
        """
        if "PhotoMetricDistortion" in transform_info.keys():
            info = transform_info["PhotoMetricDistortion"]
            brightness_beta = info["brightness_beta"]
            brightness_do = info["brightness_do"]
            contrast_alpha = info["contrast_alpha"]
            contrast_do = info["contrast_do"]
            saturate_alpha = info["saturate_alpha"]
            saturate_do = info["saturate_do"]
            rand_hue = info["rand_hue"]
            rand_hue_do = info["rand_hue_do"]
            mode = info["mode"]
            rand_to_gray_do = info["rand_to_gray_do"]
        else:
            brightness_beta = random.uniform(-self.brightness_delta, self.brightness_delta)
            brightness_do = random.random() < self.distortion_prob

            contrast_alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            contrast_do = random.random() < self.distortion_prob

            saturate_alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            saturate_do = random.random() < self.distortion_prob

            rand_hue = random.randint(-self.hue_delta, self.hue_delta)
            rand_hue_do = random.random() < self.distortion_prob

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = 1 if random.random() > 0.5 else 2

            rand_to_gray_do = random.random() < self.to_gray_prob

            transform_info["PhotoMetricDistortion"] = dict(
                brightness_beta=brightness_beta,
                brightness_do=brightness_do,
                contrast_alpha=contrast_alpha,
                contrast_do=contrast_do,
                saturate_alpha=saturate_alpha,
                saturate_do=saturate_do,
                rand_hue=rand_hue,
                rand_hue_do=rand_hue_do,
                mode=mode,
                rand_to_gray_do=rand_to_gray_do,
            )

        if rand_to_gray_do:
            image = self.rgb2gray(image)
        else:
            # random brightness
            image = self.brightness(image, brightness_beta, brightness_do)

            if mode == 1:
                image = self.contrast(image, contrast_alpha, contrast_do)

            # random saturation
            image = self.saturation(image, saturate_alpha, saturate_do)

            # random hue
            image = self.hue(image, rand_hue, rand_hue_do)

            # random contrast
            if mode == 0:
                image = self.contrast(image, contrast_alpha, contrast_do)

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class Weather(object):
    """Apply the following weather augmentations to data.
    Args:
        prob (float): probability to enforce the weather augmentation.
    """

    def __init__(self, prob=0.3, **kwargs):
        snow = iaa.FastSnowyLandscape(lightness_threshold=[50, 100], lightness_multiplier=(1.2, 2))
        cloud = iaa.Clouds()
        fog = iaa.Fog()
        snow_flakes = iaa.Snowflakes(
            flake_size=(0.2, 0.4), speed=(0.001, 0.03)
        )  # iaa.imgcorruptlike.Snow(severity=2)#
        rain = iaa.Rain(speed=(0.1, 0.3), drop_size=(0.1, 0.3))
        # rain_drops = RainDrop_Augmentor()
        self.aug_list = [
            snow,
            cloud,
            fog,
            snow_flakes,
            rain,
        ]
        self.prob = prob

    def aug_with_weather(self, imgs, id):
        weather = self.aug_list[id]
        if id < 5:
            return weather(image=imgs)
        else:
            return weather(imgs)

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        """Call function to perform photometric distortion on image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with image distorted.
        """
        if "Weather" in transform_info.keys():
            do_weather = transform_info["Weather"]["do_weather"]
            select_id = transform_info["Weather"]["select_id"]
        else:
            do_weather = random.random() < self.prob
            select_id = np.random.randint(0, high=len(self.aug_list))
            transform_info["Weather"] = dict(
                do_weather=do_weather,
                select_id=select_id,
            )
        if do_weather:
            image = self.aug_with_weather(image, select_id)
        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info


class RandomEdgeMask(object):
    """
    Random mask the input and labels.
    Args:
        images: list of RGB images.
        labels: list of depth/disparity labels.
        other labels: other labels, such as instance segmentations, semantic segmentations...
    """

    def __init__(self, mask_maxsize=32, prob=0.5, rgb_invalid=[0, 0, 0], depth_invalid=0, **kwargs):
        self.mask_maxsize = mask_maxsize
        self.prob = prob
        self.rgb_invalid = rgb_invalid
        self.depth_invalid = depth_invalid

    def mask_edge(self, image, mask_edgesize, mask_value):
        H, W = image.shape[0], image.shape[1]
        # up
        image[0 : mask_edgesize[0], :, ...] = mask_value
        # down
        image[H - mask_edgesize[1] : H, :, ...] = mask_value
        # left
        image[:, 0 : mask_edgesize[2], ...] = mask_value
        # right
        image[:, W - mask_edgesize[3] : W, ...] = mask_value

        return image

    def __call__(self, image, intrinsics, depth, depth_mask, normal, other_labels, transform_info):
        if "RandomEdgeMask" in transform_info.keys():
            prob = transform_info["RandomEdgeMask"]["prob"]
            mask_edgesize = transform_info["RandomEdgeMask"]["mask_edgesize"]
        else:
            prob = random.uniform(0, 1)
            mask_edgesize = random.sample(range(self.mask_maxsize), 4)  # [up, down, left, right]
            transform_info["RandomEdgeMask"] = dict(prob=prob, mask_edgesize=mask_edgesize)

        if prob > self.prob:
            return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info

        if image is not None:
            image = self.mask_edge(image, mask_edgesize, self.rgb_invalid)

        if depth is not None:
            depth = self.mask_edge(depth, mask_edgesize, self.depth_invalid)
            depth_mask = self.mask_edge(depth_mask, mask_edgesize, 0)

        if normal != None:
            normal = self.mask_edge(normal, mask_edgesize, mask_value=0)

        if other_labels != None:
            # other labels are like semantic segmentations, instance segmentations, instance planes segmentations...
            for i, other_label_i in enumerate(other_labels):
                other_labels[i] = self.mask_edge(other_label_i, mask_edgesize, self.depth_invalid)

        return image, intrinsics, depth, depth_mask, normal, other_labels, transform_info
