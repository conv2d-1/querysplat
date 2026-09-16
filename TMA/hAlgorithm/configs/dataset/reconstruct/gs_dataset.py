pipeline = dict(
    type="hAlgorithm.datasets_mv.gs_dataset.GSDataset",
    phase="train",
    name="gs",
    seed=0,
    normalize_cameras=True,
    data_root="/mnt/netdata/Team/AI/datasets/TMD/",
    train_transforms=[
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        ),
    ],
    test_transforms=[
        dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
        dict(
            type="hAlgorithm.datasets.transforms.transforms.Normalize",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        ),
    ],
    min_depth=1e-3,
    max_depth=1000.0,
    debug=False,
)
