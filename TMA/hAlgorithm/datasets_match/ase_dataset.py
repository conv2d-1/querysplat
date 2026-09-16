import numpy as np

from hAlgorithm.datasets_match.base_dataset import MatchingDataset


class ASEDataset(MatchingDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def apply_de_vignette(image, alpha=1.0):
        height, width = image.shape[:2]
        X, Y = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_distance = np.sqrt(center_x**2 + center_y**2)

        # 构造一个从中心向外增强亮度的 mask
        mask = 1.0 + distance / max_distance * alpha

        # 应用到每个颜色通道
        corrected = image * mask[..., np.newaxis]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        return corrected

    def load_data(self, data_info):
        data_batch = super().load_data(data_info=data_info)
        data_batch["curr_rgb"] = self.apply_de_vignette(data_batch["curr_rgb"])
        return data_batch
