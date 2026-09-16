import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import json
import logging
import math
import pprint

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    config_logging,
    config_merge_args,
    dict_to_file,
    file2dict,
    instantiate_from_config,
    parse_unknown,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="Path of checkpoint to be load.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="exp",
        help="name of experiment”",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test dataset file.",
    )
    parser.add_argument(
        "--test_vis",
        action="store_true",
        help="Show test results.",
    )
    parser.add_argument(
        "--only_vis",
        action="store_true",
        help="Show vis results, without test.",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Path to save results.",
    )
    parser.add_argument(
        "--only_save",
        action="store_true",
        help="Path to save results, without test.",
    )

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.output_dir = os.path.join(args.output_dir, args.exp, config_name + "_" + now)

    if args.test and args.save_outputs:
        args.output_dir = os.path.realpath(args.output_dir)

    if args.resume in ["None", "none", "null"]:
        args.resume = None

    if args.load_from in ["None", "none", "null"]:
        args.load_from = None

    return args, unknown_args


def get_select_dataset(trainer, select_key):
    select_dataset = None
    if trainer.get(select_key, None) is not None:
        if isinstance(trainer[select_key], str):
            if trainer[select_key] != "":
                select_dataset = [item.strip() for item in trainer[select_key].split(",")]
                select_dataset = [d for d in select_dataset if d != ""]
                if len(select_dataset) == 0:
                    select_dataset = None

    if select_dataset is None:
        trainer[select_key] = None

    return select_dataset


def data_filter_process_func(log_dir, json_paths_dict):
    from hAlgorithm.script.data_filter.json_filter import main as json_filter_main
    from hAlgorithm.script.data_filter.log_analyze import main as log_analyze_main

    csv_paths = log_analyze_main(log_dir)

    for name, json_path in json_paths_dict.items():
        json_filter_main(json_path, csv_paths[name])


def main():
    args, unknown_args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(
        timeout=datetime.timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", 1200)))
    )
    if args.mixed_precision not in ["fp8"]:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            project_config=accelerator_project_config,
            kwargs_handlers=[dist_kwargs, init_kwargs],
        )
    else:
        from accelerate.utils import FP8RecipeKwargs

        FP8_RECIPE_KWARGS = {
            "fp8_format": "HYBRID",
            "amax_history_len": 32,
            "amax_compute_algo": "max",
        }
        kwargs_handlers = FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            project_config=accelerator_project_config,
            kwargs_handlers=[dist_kwargs, init_kwargs, kwargs_handlers],
        )

    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        config_logging(out_dir=logging_dir)
    else:
        config_logging()

    logging.info(f"seed: {args.seed}")
    logging.info(f"device: {accelerator.device}")
    logging.info(torch.cuda.get_device_properties(accelerator.device))

    # git diff to code.diff
    if accelerator.is_main_process:
        os.system(f"git diff -w > {os.path.join(logging_dir, 'code.diff')}")

    # Set the training seed now.
    set_seed(args.seed)

    # Initialize directories
    ckpt_dir = os.path.join(args.output_dir, "checkpoint")
    tb_dir = os.path.join(args.output_dir, "tensorboard")
    eval_dir = os.path.join(args.output_dir, "evaluation")
    vis_dir = os.path.join(args.output_dir, "visualization")
    configs_dir = os.path.join(args.output_dir, "configs")

    if accelerator.is_main_process:
        # os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)

    # Load configuration
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    if "mem" in cfg["trainer"] and not args.test:
        try:
            mem = torch.ones(cfg["trainer"]["mem"]).cuda()
        except Exception:
            pass

    if args.test and args.test_data is not None:
        cfg["data"]["val"] = args.test_data

    cfg["trainer"]["seed"] = args.seed
    cfg["model"]["seed"] = args.seed

    if args.test:
        cfg["trainer"]["resume"] = None
        if args.load_from is not None:
            cfg["trainer"]["load_from"] = args.load_from
    else:
        assert args.resume is None or args.load_from is None
        if args.resume is not None:
            cfg["trainer"]["resume"] = args.resume
            cfg["trainer"]["load_from"] = None
        if args.load_from is not None:
            cfg["trainer"]["load_from"] = args.load_from
            cfg["trainer"]["resume"] = None

    cfg["trainer"]["output_dir"] = args.output_dir
    cfg["trainer"]["ckpt_dir"] = ckpt_dir
    cfg["trainer"]["tb_dir"] = tb_dir
    cfg["trainer"]["eval_dir"] = eval_dir
    cfg["trainer"]["vis_dir"] = vis_dir

    # Initialize datasets
    select_dataset = get_select_dataset(cfg["trainer"], "select_dataset")
    select_val_dataset = get_select_dataset(cfg["trainer"], "select_val_dataset")
    select_vis_dataset = get_select_dataset(cfg["trainer"], "select_vis_dataset")

    # Initialize select_max_depth for datasets
    select_max_depth = None
    if select_dataset is not None and cfg["trainer"].get("select_max_depth", None) is not None:
        if isinstance(cfg["trainer"]["select_max_depth"], (int, float)):
            select_max_depth = [float(cfg["trainer"]["select_max_depth"])]
        elif isinstance(cfg["trainer"]["select_max_depth"], str):
            if cfg["trainer"]["select_max_depth"] != "":
                select_max_depth = [
                    item.strip() for item in cfg["trainer"]["select_max_depth"].split(",")
                ]
                select_max_depth = [float(d) for d in select_max_depth if d != ""]
                if len(select_max_depth) == 0:
                    select_max_depth = None

    if select_max_depth is None:
        cfg["trainer"]["select_max_depth"] = None

    train_datasets, _, train_nums = prepare_data_loaders(
        "val",
        cfg,
        configs_dir,
        select_dataset=select_dataset,
        select_max_depth=select_max_depth,
        is_main_process=accelerator.is_main_process,
    )
    from torch.utils.data import ConcatDataset, DataLoader

    from hAlgorithm.datasets import CustomerConcatDataset
    from hAlgorithm.datasets.samplers import NoDuplicateDistributedSampler

    train_dataset = CustomerConcatDataset(train_datasets)
    train_sampler = NoDuplicateDistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,  # Use DistributedSampler for validation/vis datasets
        num_workers=cfg["trainer"]["num_workers"],  # Number of workers (can be increased as needed)
    )

    val_datasets, val_dataloaders, val_nums = prepare_data_loaders(
        "val",
        cfg,
        configs_dir,
        select_dataset=select_dataset,
        is_main_process=accelerator.is_main_process,
    )
    json_path_dict = {dataset.name: dataset.data_path for dataset in val_datasets}

    vis_datasets, vis_dataloaders, vis_nums = prepare_data_loaders(
        "vis",
        cfg,
        configs_dir,
        select_dataset=select_vis_dataset,
        is_main_process=accelerator.is_main_process,
    )

    # Reset epoch and iter
    max_epoch = cfg["trainer"]["max_epoch"]
    max_iter = cfg["trainer"]["max_iter"]
    batch_size = cfg["trainer"]["batch_size"]
    gradient_accumulation_steps = cfg["trainer"]["gradient_accumulation_steps"]

    total_batch_size = batch_size * gradient_accumulation_steps * accelerator.num_processes
    num_train_samples = len(train_dataset)

    if max_iter is not None:
        cfg["trainer"]["max_epoch"] = math.ceil(max_iter * total_batch_size / num_train_samples)
    else:
        cfg["trainer"]["max_iter"] = math.ceil(max_epoch * num_train_samples / total_batch_size)

    if "total_iter" in cfg["trainer"]["lr_scheduler"]:
        cfg["trainer"]["lr_scheduler"]["total_iter"] = cfg["trainer"]["max_iter"]
    if "max_iters" in cfg["trainer"]["lr_scheduler"]:
        cfg["trainer"]["lr_scheduler"]["max_iters"] = cfg["trainer"]["max_iter"]

    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")

    if accelerator.is_main_process:
        logging.info("")
        logging.info("***** Running training *****")
        logging.info(f"  Num train examples = {train_nums}, sum = {num_train_samples}")
        logging.info(
            f"  Num test examples = {val_nums}, sum = {sum([int(num.split(':')[-1]) for num in val_nums])}"
        )
        logging.info(
            f"  Num vis examples = {vis_nums}, sum = {sum([int(num.split(':')[-1]) for num in vis_nums])}"
        )
        logging.info("")
        logging.info(f"  Num train loader = {len(train_dataloader)}")
        logging.info(f"  Num Epochs = {cfg['trainer']['max_epoch']}")
        logging.info(f"  Instantaneous batch size per device = {batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(f"  Total optimization steps = {cfg['trainer']['max_iter']}")
        logging.info(f"  logging_dir = {args.output_dir}")
        logging.info("")

        # -------------------- Snapshot of code and config --------------------
        cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
        assert cfg_save_path != args.config
        dict_to_file(cfg, cfg_save_path)

    # Initialize model
    model_cfg = cfg["model"].copy()
    model = instantiate_from_config(model_cfg)
    assert model is not None
    if accelerator.is_main_process:
        with open(os.path.join(logging_dir, "model.log"), "w") as f:
            f.write(str(model))

    # Initialize trainer
    trainer_cfg = cfg["trainer"].copy()
    module_dict = dict(
        accelerator=accelerator,
        model=model,
        train_dataset=train_dataset,
        train_dataloader=train_dataloader,
        val_datasets=val_datasets,
        val_dataloaders=val_dataloaders,
        vis_datasets=vis_datasets,
        vis_dataloaders=vis_dataloaders,
    )
    trainer_cfg.update(module_dict)
    trainer = instantiate_from_config(trainer_cfg)
    assert trainer is not None

    logger = logging.getLogger()
    # Set log level to DEBUG
    logger.setLevel(logging.DEBUG)

    if args.only_vis or args.only_save:
        trainer.eval_metrics = None

    trainer.train()

    data_filter_process_func(logging_dir, json_path_dict)


if __name__ == "__main__":
    main()
