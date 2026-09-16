import os
import sys

sys.path.append(os.getcwd())

from hAlgorithm.datasets.agibot_dataset import AgiBotDataset
from hAlgorithm.datasets.arkit_dataset import ARKitDataset
from hAlgorithm.datasets.blendedmvs_dataset import BlendedMVSDataset
from hAlgorithm.datasets.diml_dataset import DIMLDataset
from hAlgorithm.datasets.diode_dataset import DIODEDataset
from hAlgorithm.datasets.dynamicstereo_dataset import DynamicStereoDataset
from hAlgorithm.datasets.hypersim_dataset import HypersimDataset
from hAlgorithm.datasets.kenburns_dataset import KenBurnsDataset
from hAlgorithm.datasets.matterport3d_dataset import Matterport3DDataset
from hAlgorithm.datasets.scannet_dataset import ScanNetDataset
from hAlgorithm.datasets.std_dataset import STDDataset
from hAlgorithm.datasets.taskonomy_dataset import TaskonomyDataset

datasets = [
    BlendedMVSDataset(
        phase="test",
        name="blendedmvs",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/BlendedMVS/train.json",
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
        max_depth=1000.0,  # NOTE
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    DIMLDataset(
        phase="test",
        name="diml",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/DIML/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
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
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1000.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    DynamicStereoDataset(
        phase="test",
        name="dynamicstereo",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/DynamicStereo/train.json",
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
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    KenBurnsDataset(
        phase="test",
        name="kenburns",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/KenBurns/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
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
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=100.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    TaskonomyDataset(
        phase="test",
        name="taskonomy",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Taskonomy/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
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
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=512.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    ScanNetDataset(
        phase="test",
        name="scannet",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/ScanNet/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
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
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1000.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    HypersimDataset(
        phase="test",
        name="hypersim",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Hypersim/train.json",
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
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    DIODEDataset(
        phase="test",
        name="diode",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/DIODE/train.json",
        sampling_strategy="all",
        train_transforms=[
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
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
            dict(type="hAlgorithm.datasets.transforms.transforms.Resize", width=630, height=476),
            dict(type="hAlgorithm.datasets.transforms.transforms.ToTensor"),
            dict(
                type="hAlgorithm.datasets.transforms.transforms.Normalize",
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
            ),
        ],
        depth_scale=1.0,
        min_depth=1e-5,
        max_depth=1000.0,
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=False,
        with_pointmap=False,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=False,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    STDDataset(
        phase="test",
        name="std",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/STD/train.json",
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
        max_depth=1000.0,  # NOTE
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=True,
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    Matterport3DDataset(
        phase="test",
        name="matterport3d",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/Matterport3D/train.json",
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
        max_depth=1000.0,  # NOTE
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=True,
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    AgiBotDataset(
        phase="test",
        name="agibot",
        seed=0,
        data_root="/mnt/netdata/Team/AI/datasets/TMD/",
        data_path="/mnt/netdata/Team/AI/datasets/TMD/AgiBotWorld/train.json",
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
        depth_scale=1000.0,
        min_depth=1e-5,
        max_depth=1000.0,  # NOTE
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=True,
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
    ARKitDataset(
        phase="test",
        name="arkit",
        seed=0,
        data_root="/mnt/personal/TMD/datasets",
        data_path="/mnt/personal/TMD/datasets/ARKitScenes/train_highres_new_total.json",
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
        depth_scale=1000.0,
        min_depth=1e-5,
        max_depth=1000.0,  # NOTE
        normalize_depth=dict(
            type="hAlgorithm.datasets.transforms.depth_transforms.ScaleShiftDepthNormalizer",
            norm_min=-1.0,
            norm_max=1.0,
            min_max_quantile=0.02,
            clip=True,
        ),
        with_edge_mask=True,
        with_pointmap=True,
        sparse_depth_ratio=0.05,
        depth_invalid_sem_names=None,
        recalculate_normal=True,
        normal_transform=dict(type="hAlgorithm.datasets.transforms.norm_transforms.DepthToNormal"),
        debug=False,
    ),
]

working_dir = "./debug/data_dist/"
os.makedirs(working_dir, exist_ok=True)

for dataset in datasets:
    print(f"{dataset.name} len: {len(dataset)}.")

import numpy as np
from tqdm import tqdm

max_sample = 100000
saved_period = 1000
for dataset in datasets:
    if dataset.name not in ["arkit"]:  # std,matterport3d,agibot
        continue

    data_sample = min(max_sample, len(dataset))
    print(f"sampling {data_sample} in {dataset.name}")
    dataset_path = os.path.join(working_dir, dataset.name)
    os.makedirs(dataset_path, exist_ok=True)

    sample_ids = np.linspace(0, len(dataset) - 1, data_sample, dtype=int)

    # resume from logs
    logs_paths = [log for log in os.listdir(dataset_path) if f"{dataset.name}.npy" in log]
    if len(logs_paths) > 0:
        logs_paths.sort()
        log_path = logs_paths[-1]
        log_sample = log_path[: log_path.find("sample")].split("_")[1]
        if log_sample in ["total"]:
            log_sample = max_sample
        else:
            log_sample = int(log_sample)
        log_dist = np.load(os.path.join(dataset_path, log_path))
        print(f"Resume from {log_path}.")
    else:
        log_sample = 0
    depth_dist_total = []
    if log_sample < len(sample_ids):
        for index in tqdm(sample_ids[log_sample:]):
            try:
                data_info = dataset.data_infos[index]
                data_batch = dataset.load_data(data_info)
                depth = data_batch["curr_depth"].reshape(-1)
                depth = np.sort(depth)
                d_len = len(depth)
                dist_id = np.linspace(0, d_len - 1, 21, dtype=int)
                depth_dist_total.append(np.array([depth[d_id] for d_id in dist_id]))
            except Exception:
                continue
            if len(depth_dist_total) % saved_period == 0:
                output = np.array(depth_dist_total)
                if log_sample > 0:
                    output = np.vstack([log_dist, output])
                np.save(
                    os.path.join(
                        dataset_path,
                        f"depth_{log_sample+len(depth_dist_total)}sample_{dataset.name}.npy",
                    ),
                    output,
                )
    # save output
    output = np.array(depth_dist_total)
    if len(depth_dist_total) > 0 and log_sample > 0:
        output = np.vstack([log_dist, output])
    elif log_sample > 0:
        output = log_dist
    np.save(
        os.path.join(dataset_path, f"depth_{output.shape[0]-1}sample_{dataset.name}.npy"), output
    )
    from hAlgorithm.utils.depth_util import compute_depth_distribution, plot_distribution

    compute_depth_distribution(output, path=dataset_path, name=dataset.name)
