pipeline = dict(
    type="hAlgorithm.datasets.nyu_dataset.NYUDataset",
    phase="train",
    name="nyu",
    seed=0,
    data_root="/mnt/netdata/Team/AI/datasets/TMD/",
    data_path="/mnt/netdata/Team/AI/datasets/TMD/NYUV2/test.json",
    sampling_strategy="all",
    train_transforms=[
        dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        ),
        # x / 255 * 2 - 1
    ],
    test_transforms=[
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        ),
    ],
    depth_scale=1000.0,
    min_depth=1e-3,
    max_depth=10.0,
    normalize_depth=dict(
        type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
        norm_min=-1.0,
        norm_max=1.0,
        min_max_quantile=0.02,
        clip=True,
    ),
    with_pointmap=False,
    sparse_depth_ratio=0,
    sky_index=-1,
    recalculate_normal=False,
    normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
    debug=False,
    eigen_valid_mask=True,
)
