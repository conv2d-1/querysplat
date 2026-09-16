import bisect
import importlib
import logging
import os
import pprint
from copy import deepcopy

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from hAlgorithm.datasets.samplers import (
    MixedBatchSampler,
    MixedMaxIterBatchSampler,
    NoDuplicateDistributedSampler,
    DynamicMixedMaxIterBatchSampler,
    DebugBatchSampler,
)
from hAlgorithm.utils import dict_to_file, get_obj_from_file, instantiate_from_config
from hAlgorithm.datasets.dataloader.collate import default_collate


class CustomerConcatDataset(ConcatDataset):
    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            assert len(idx) == 2 and isinstance(idx[1], dict)
            idx, others = idx
        else:
            others = None

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if others is None:
            return self.datasets[dataset_idx][sample_idx]
        else:
            return self.datasets[dataset_idx][[sample_idx, others]]


class EmptyValDataLoader:
    """Placeholder val loader for ranks that skip inference when dataset is too small."""

    batch_size = 1

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 0

    def __iter__(self):
        return iter([])


def prepare_data_loaders(
    phase: str,
    cfg: dict,
    save_dir: str = None,
    seed: int = None,
    select_dataset: list = None,
    select_max_depth: list = None,
    is_main_process: bool = True,
    num_processes: int = None,
) -> tuple:
    """
    Prepares datasets and data loaders for the specified phase (train or test).

    This function loads dataset configurations, creates dataset instances, and sets up data loaders
    based on the provided configuration. For the training phase, it uses a mixed batch sampler to
    combine multiple datasets with different probabilities and batch sizes. For the testing phase,
    it creates separate data loaders for each dataset.

    Args:
        phase (str): The phase of the dataset, typically 'train' or 'test'.
        cfg (dict): Configuration dictionary containing dataset and dataloader settings.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        select_dataset(Optional[list(str)]): Select datasets in config by list of names. Defaults to None.
        select_max_depth(Optional[list(float)]): Overwrite max depth in select datasets. Defaults to None.

    Returns:
        tuple: A tuple containing either (concat_dataset, data_loader) for the training phase or
               (dataset_list, data_loaders) for the testing phase.
    """
    # Load the dataset configuration for the specified phase
    with open(cfg["data"][phase], "r") as file:
        yaml_data = yaml.safe_load(file)

    if f"datasets_{phase}" in yaml_data:
        dataset_configs = yaml_data[f"datasets_{phase}"]
        mix_phase_yaml = True
    else:
        dataset_configs = yaml_data.get("datasets", [])
        mix_phase_yaml = False
    set_random_size = cfg["trainer"].get("set_random_size", False)
    test_num_workers = max(cfg["trainer"].get("test_num_workers", 2), 2)
    test_num_workers = int(test_num_workers)

    # Load basic config
    basic_config = dict()
    if "basic" in cfg["data"] and cfg["data"]["basic"] is not None:
        basic_config.update(cfg["data"]["basic"])
    if f"{phase}_basic" in cfg["data"] and cfg["data"][f"{phase}_basic"] is not None:
        basic_config.update(cfg["data"][f"{phase}_basic"])

    if select_dataset is not None and select_max_depth is not None:
        assert len(select_dataset) == len(select_max_depth)

    # Initialize lists to store datasets, probabilities, and batch sizes
    dataset_list = []
    prob_list = []
    batch_size_list = []

    # Create dataset instances and extract probabilities and batch sizes
    for di, dataset_cfg in enumerate(dataset_configs):
        pipeline_ori = dataset_cfg.pop("pipeline")
        if isinstance(pipeline_ori, str):
            pipeline = get_obj_from_file(pipeline_ori, "pipeline").copy()
        else:
            pipeline = pipeline_ori.copy()

        # Skip other datasets if select_dataset is assigned
        if select_dataset is not None:
            dataset_name = pipeline["name"]
            if dataset_cfg.get("name", None) is not None:
                dataset_name = dataset_cfg["name"]
            if dataset_name not in select_dataset:
                if mix_phase_yaml:
                    yaml_data[f"datasets_{phase}"][di] = None
                else:
                    yaml_data["datasets"][di] = None
                continue
            if select_max_depth is not None:
                for data_i, dataname in enumerate(select_dataset):
                    if dataname == dataset_name:
                        dataset_cfg["max_depth"] = select_max_depth[data_i]

        dataset_cfg.update(basic_config)
        prob = dataset_cfg.pop("prob", None)
        batch_scale = dataset_cfg.pop("batch_scale", 1.0)

        pipeline["phase"] = phase
        pipeline.update(dataset_cfg)

        if not set_random_size:
            pipeline["size_pool"] = None

        dataset = instantiate_from_config(pipeline)
        assert dataset is not None, pipeline
        dataset_list.append(dataset)
        prob_list.append(prob)

        if phase == "train":
            batch_size_list.append(max(1, int(batch_scale * cfg["trainer"]["batch_size"])))
        else:
            batch_size_list.append(1)

        for key in pipeline:
            if key in dataset_cfg:
                dataset_cfg.pop(key)
        dataset_cfg["prob"] = prob
        dataset_cfg["batch_size"] = batch_size_list[-1]

        if is_main_process:
            logging.info(f"{phase} dataset {di}, {dataset.name}, len {len(dataset)}, batch_size {batch_size_list[-1]}")
            logging.debug("\n" + pprint.pformat(pipeline, compact=True))

        if save_dir is not None:
            if isinstance(pipeline_ori, str):
                save_path = os.path.join(
                    save_dir, f"{phase}_pipeline", os.path.basename(pipeline_ori)
                )
            else:
                save_path = os.path.join(save_dir, f"{phase}_pipeline/{dataset.name}.py")
            if is_main_process:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                dict_to_file(dict(pipeline=pipeline), save_path)

        if mix_phase_yaml:
            yaml_data[f"datasets_{phase}"][di]["pipeline"] = pipeline
        else:
            yaml_data["datasets"][di]["pipeline"] = pipeline

    # rm none in yaml_data
    if mix_phase_yaml:
        yaml_data[f"datasets_{phase}"] = [
            d for d in yaml_data[f"datasets_{phase}"] if d is not None
        ]
    else:
        yaml_data["datasets"] = [d for d in yaml_data["datasets"] if d is not None]

    # 定义一个类来禁用锚点和别名
    class NoAliasDumper(yaml.Dumper):
        def ignore_aliases(self, data):
            return True

    if is_main_process and save_dir is not None:
        save_path = os.path.join(save_dir, os.path.basename(cfg["data"][phase]))
        if mix_phase_yaml:
            with open(save_path, "a") as file:
                save_yaml_data = {f"datasets_{phase}": yaml_data[f"datasets_{phase}"]}
                yaml.dump(
                    save_yaml_data,
                    file,
                    default_flow_style=False,
                    encoding="utf-8",
                    allow_unicode=True,
                    Dumper=NoAliasDumper,
                )
        else:
            with open(save_path, "w") as file:
                yaml.dump(
                    yaml_data,
                    file,
                    default_flow_style=False,
                    encoding="utf-8",
                    allow_unicode=True,
                    Dumper=NoAliasDumper,
                )
        cfg["data"][phase] = save_path

    # Normalize probabilities if any are provided
    if any(prob is not None for prob in prob_list):
        prob_list = [prob if prob is not None else 1.0 for prob in prob_list]
    else:
        prob_list = None

    # Set up the random generator for reproducibility
    data_nums = [f"{dataset.name}:{len(dataset)}" for dataset in dataset_list]
    loader_generator = torch.Generator().manual_seed(seed) if seed is not None else None
    if phase == "train":
        # Create a concatenated dataset and a mixed batch sampler for training
        concat_dataset = CustomerConcatDataset(dataset_list)

        view_num_range = cfg["trainer"].pop("view_num_range", None)
        aspect_ratio_range = cfg["trainer"].pop("aspect_ratio_range", None)
        max_img_per_gpu = cfg["trainer"].pop("max_img_per_gpu", None)
        max_size_list = cfg["trainer"].pop("max_size_list", None)

        if "sampler" in cfg["trainer"] and cfg["trainer"]["sampler"] == "DynamicMixedMaxIterBatchSampler":
            mixed_sampler = DynamicMixedMaxIterBatchSampler(
                dataset_list=dataset_list,
                batch_size_list=batch_size_list,
                drop_last=False,
                prob_list=prob_list,
                shuffle=True,
                generator=loader_generator,
                set_random_size=set_random_size,
                max_iter=cfg["trainer"]["max_iter"]
                * cfg["trainer"]["gradient_accumulation_steps"]
                * num_processes,
                max_img_per_gpu=max_img_per_gpu,
                view_num_range=view_num_range,
                aspect_ratio_range=aspect_ratio_range,
                max_size_list=max_size_list,
            )
        elif "sampler" in cfg["trainer"] and cfg["trainer"]["sampler"] == "MixedMaxIterBatchSampler":
            mixed_sampler = MixedMaxIterBatchSampler(
                dataset_list=dataset_list,
                batch_size_list=batch_size_list,
                drop_last=False,
                prob_list=prob_list,
                shuffle=True,
                generator=loader_generator,
                set_random_size=set_random_size,
                max_iter=cfg["trainer"]["max_iter"]
                * cfg["trainer"]["gradient_accumulation_steps"]
                * num_processes,
                aspect_ratio_range=aspect_ratio_range,
                max_size_list=max_size_list,
            )
        elif "sampler" in cfg["trainer"] and cfg["trainer"]["sampler"] == "DebugBatchSampler":
            mixed_sampler = DebugBatchSampler(dataset_list=dataset_list)
        else:
            mixed_sampler = MixedBatchSampler(
                dataset_list=dataset_list,
                batch_size_list=batch_size_list,
                drop_last=False,
                prob_list=prob_list,
                shuffle=True,
                generator=loader_generator,
                set_random_size=set_random_size,
                aspect_ratio_range=aspect_ratio_range,
                max_size_list=max_size_list,
            )
        data_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg["trainer"]["num_workers"],
            collate_fn=default_collate,
        )
        return concat_dataset, data_loader, data_nums
    else:
        try:
            world_size = torch.distributed.get_world_size()
        except Exception:
            world_size = None
        # Create separate data loaders for each dataset in the testing phase
        if world_size is None or not get_val_distributed(cfg):
            data_loaders = [
                DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=test_num_workers,
                    collate_fn=default_collate,
                )
                for dataset in dataset_list
            ]
        else:
            data_loaders = []
            for dataset in dataset_list:
                val_num_workers = test_num_workers
                if world_size is not None and len(dataset) < world_size:
                    # Too few samples to shard across GPUs: rank0 runs full val, others skip.
                    val_num_workers = 0
                    rank = torch.distributed.get_rank()
                    if rank == 0:
                        data_loader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=val_num_workers,
                            collate_fn=default_collate,
                        )
                    else:
                        data_loader = EmptyValDataLoader(dataset)
                    if is_main_process:
                        logging.warning(
                            f"val dataset {dataset.name}: len={len(dataset)} < world_size={world_size}, "
                            f"use rank0-only validation (other ranks use EmptyValDataLoader)"
                        )
                else:
                    sampler = NoDuplicateDistributedSampler(dataset, shuffle=False)
                    data_loader = DataLoader(
                        dataset,
                        batch_size=1,
                        sampler=sampler,
                        num_workers=val_num_workers,
                        collate_fn=default_collate,
                    )
                data_loaders.append(data_loader)

        return dataset_list, data_loaders, data_nums


def get_val_distributed(cfg):
    """
    Retrieves the val_distributed attribute of the trainer.
    Args:
        cfg (dict): Configuration dictionary containing trainer info.
    Returns:
        bool: Returns the value of val_distributed if the trainer exists and has this attribute; otherwise, returns False.
    """

    trainer_type = cfg.get("trainer", {}).get("type", None)

    if trainer_type:
        # Split the trainer_type string safely
        try:
            prefix, class_name = trainer_type.rsplit(".", 1)

            # Dynamically import the module using importlib
            module = importlib.import_module(prefix)

            # Get the class from the module
            trainer_class = getattr(module, class_name, None)

            if trainer_class is not None:
                # Safely check if the class has the 'val_distributed' attribute
                return getattr(trainer_class, "val_distributed", False)
        except (ValueError, ImportError, AttributeError) as e:
            # Handle any exceptions that may arise during the process
            print(f"Error importing or accessing the trainer class: {e}")

    # Return False if no valid trainer type or val_distributed attribute was found
    return False
