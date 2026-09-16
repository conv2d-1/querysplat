import os
import sys

sys.path.append(os.getcwd())

from hAlgorithm.datasets.base_dataset import BaseDataset


class HypersimDataset(BaseDataset):
    # nyu 40
    semantic_labels = dict(
        wall=1,
        floor=2,
        cabinet=3,
        bed=4,
        chair=5,
        sofa=6,
        table=7,
        door=8,
        window=9,
        bookshelf=10,
        picture=11,
        counter=12,
        blinds=13,
        desk=14,
        shelves=15,
        curtain=16,
        dresser=17,
        pillow=18,
        mirror=19,
        floormat=20,
        clothes=21,
        ceiling=22,
        books=23,
        refrigerator=24,
        television=25,
        paper=26,
        towel=27,
        showercurtain=28,
        box=29,
        whiteboard=30,
        person=31,
        nightstand=32,
        toilet=33,
        sink=34,
        lamp=35,
        bathtub=36,
        bag=37,
        otherstructure=38,
        otherfurniture=39,
        otherprop=40,
    )

    def __init__(
        self,
        name: str = "hypersim",
        depth_scale: float = 1.0,
        min_depth: float = 1e-5,
        max_depth: float = 65.0,
        depth_invalid_sem_names: list = ["window", "mirror", "blinds"],
        **kwargs,
    ):
        super().__init__(
            name=name,
            depth_scale=depth_scale,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_invalid_sem_names=depth_invalid_sem_names,
            **kwargs,
        )


if __name__ == "__main__":

    dataset = HypersimDataset(
        phase="train",
        # name="hypersim",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Hypersim/test.json",
        sampling_strategy="all",
        train_transforms=[
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=630,
                height=476,
            ),
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
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Resize",
                width=630,
                height=476,
            ),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1.0,
        min_depth=1e-5,
        max_depth=60.0,
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
        sparse_pattern=None,
        sparse_pattern_v2=dict(
            type="hAlgorithm.datasets.patterns.multi_patterns.MultiPatterns",
            sparse_nums=300000,
            pattern="none",
            # pattern="1.0*depth_edge",
            # rand_noise=[0.0, 0.1],
            # edge_noise=[0.0, 0.5],
            # crop_patch_ratio=1.0,
            # crop_patch_range=[0.05, 1.0],
            uv_jitter_ratio=1.0,
            uv_jitter_noise=[0.8, 0.9],
            uv_jitter_max=20,
            seed=0,
        ),
        name="hypersim_uv",
    )

    for index in range(10):
        print(index)
        dataset.__getitem__(index)
