pipeline = dict(
    type="hAlgorithm.datasets4d.base_dataset.BaseDataset4D",
    phase="train",
    name="base",
    seed=0,
    data_root="/mnt/netdata/Team/AI/datasets/TMD/",
    data_path="/mnt/netdata/Team/AI/datasets/TMD/Hypersim/data_split_marigold_v2/test.json",
    sampling_strategy="all",
    train_transforms=[
        dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
        dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
        dict(
            type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",  # NOTE
            to_gray_prob=0.1,
            distortion_prob=0.05,
        ),
        # dict(
        #     type="hAlgorithm.datasets.transforms.metric3d_transforms.Weather", prob=0.1  # NOTE
        # ),
        dict(
            type="hAlgorithm.datasets.transforms.metric3d_transforms.RandomBlur",
            prob=0.05,  # NOTE
        ),
        dict(
            type="hAlgorithm.datasets.transforms.metric3d_transforms.RGBCompresion",
            prob=0.1,
            compression=[0, 50],
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
        dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        ),
    ],
    depth_scale=1.0,
    min_depth=1e-3,
    max_depth=200.0,
    normalize_depth=dict(
        type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
        norm_min=-1.0,
        norm_max=1.0,
        min_max_quantile=0.02,
        clip=True,
    ),
    with_edge_mask=False,
    with_pointmap=False,
    sparse_depth_ratio=0,
    sky_index=-1,
    recalculate_normal=False,
    normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
    debug=False,
    size_pool=None,
    point_normalize_mode="distance",
    depth_invalid_sem_names=None,
)
