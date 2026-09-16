import os
import sys

sys.path.append(os.getcwd())

from hAlgorithm.datasets.base_dataset import BaseDataset

from hAlgorithm.datasets.kitti_dataset import KITTIDataset

class VkittiDataset(BaseDataset):
    semantic_labels = dict(
        terrain=(210, 0, 210),
        sky=(90, 200, 255),
        tree=(0, 199, 0),
        vegetation=(90, 240, 0),
        building=(140, 140, 140),
        road=(100, 60, 100),
        guardrail=(250, 100, 255),
        traffic_sign=(255, 255, 0),
        traffic_light=(200, 200, 0),
        pole=(255, 130, 0),
        misc=(80, 80, 80),
        truck=(160, 60, 60),
        car=(255, 127, 80),
        van=(0, 139, 139),
        unlabeled=(0, 0, 0),
    )

    def __init__(
        self,
        name: str = "vkitti",
        min_depth: float = 1e-3,
        max_depth: float = 150.0,
        sem_name: str = "sem",
        depth_invalid_sem_names: list = ["sky"],
        **kwargs,
    ):
        super().__init__(
            name=name,
            min_depth=min_depth,
            max_depth=max_depth,
            sem_name=sem_name,
            depth_invalid_sem_names=depth_invalid_sem_names,
            **kwargs,
        )

class VirtualKITTIDataset(KITTIDataset):
    def __init__(
        self,
        name: str = "vkitti",
        valid_mask_crop: str = None,
        depth_scale: float = 100.0,
        min_depth: float = 1e-5,
        max_depth: float = 80.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            valid_mask_crop=valid_mask_crop,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            **kwargs,
        )

        if self.phase == "train":
            self.valid_mask_crop = None


if __name__ == "__main__":

    dataset = VirtualKITTIDataset(
        phase="train",
        name="vkitti",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Marigold/vkitti2/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.KittiBenchmarkCrop"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip",
                prob=0.5,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
            # x / 255 * 2 - 1
        ],
        test_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.KittiBenchmarkCrop"),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=100.0,
        min_depth=1e-5,
        max_depth=80.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=True,
        valid_mask_crop="eigen",  # valid_mask_crop for test,
    )

    for index in range(50):
        print(index)
        dataset.__getitem__(index)