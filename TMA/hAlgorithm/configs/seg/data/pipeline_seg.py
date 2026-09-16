patch_size = 14
max_size = 840

pipeline = dict(
    type="hAlgorithm.datasets_seg.custom_seg_dataset.KosmoSegDataset",
    phase="train",
    name="kosmo_seg",
    seed=0,
    data_root="/mnt/nasTeam/Kosmo/processed_data/kosmo",
    data_path="data_seg_创意园_right_sharp.json",
    sampling_strategy="all",
    return_masks=True,
    return_boxes=True,
    mask_format="bitmap",
    box_format="xyxy",
    train_transforms=[
        dict(
            type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
            max_size=max_size,
            patch_size=patch_size,
            is_lidar=False,
            low_resolution=True,
        ),
        dict(type="hAlgorithm.datasets.transforms.transforms.RandomHorizontalFlip", prob=0.5),
        dict(
            type="hAlgorithm.datasets.transforms.metric3d_transforms.PhotoMetricDistortion",
            to_gray_prob=0.1,
            distortion_prob=0.5,
        ),
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        ),
    ],
    test_transforms=[
        dict(
            type="hAlgorithm.datasets.transforms.transforms.ResizeKeepRatio",
            max_size=max_size,
            patch_size=patch_size,
            is_lidar=False,
            low_resolution=True,
        ),
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
        ),
    ],
    debug=False,
)
