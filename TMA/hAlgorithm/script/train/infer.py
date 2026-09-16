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

from accelerate.utils import set_seed

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


def main():
    args, unknown_args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
  
    os.makedirs(logging_dir, exist_ok=True)
    config_logging(out_dir=logging_dir)

    logging.info(f"seed: {args.seed}")

    # Set the training seed now.
    set_seed(args.seed)

    # Initialize directories
    eval_dir = os.path.join(args.output_dir, "evaluation")
    vis_dir = os.path.join(args.output_dir, "visualization")
    configs_dir = os.path.join(args.output_dir, "configs")

    # os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    # Load configuration
    cfg = file2dict(args.config)
    config_merge_args(cfg, unknown_args)

    if args.test and args.test_data is not None:
        cfg["data"]["val"] = args.test_data

    cfg["trainer"]["seed"] = args.seed
    cfg["model"]["seed"] = args.seed

    if args.test or args.only_vis:
        cfg["trainer"]["resume"] = None
        if args.load_from is not None:
            if args.load_from == "latest":
                args.load_from = os.path.join(os.path.dirname(args.config), "checkpoint/latest/ckpt.pth")
            elif args.load_from == "best":
                args.load_from = os.path.join(os.path.dirname(args.config), "checkpoint/best/ckpt.pth")
            cfg["trainer"]["load_from"] = args.load_from

    cfg["trainer"]["output_dir"] = args.output_dir
    cfg["trainer"]["eval_dir"] = eval_dir
    cfg["trainer"]["vis_dir"] = vis_dir

    # Initialize datasets
    select_val_dataset = get_select_dataset(cfg["trainer"], "select_val_dataset")
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders(
        "val",
        cfg,
        configs_dir,
        select_dataset=select_val_dataset,
        is_main_process=True,
    )

    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")
    logging.info("***** Running test *****")
    logging.info(f"  Num test examples = {sum([len(dataset) for dataset in val_datasets])}")
    logging.info(f"  logging_dir = {args.output_dir}")
    logging.info("")

    # -------------------- Snapshot of config --------------------
    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config)[:-3] + "_backup.py")
    os.system(f"cp {args.config} {cfg_save_path}")

    # Initialize model
    model_cfg = cfg["model"].copy()
    model = instantiate_from_config(model_cfg)
    assert model is not None
    with open(os.path.join(logging_dir, "model.log"), "w") as f:
        f.write(str(model))

    # Initialize trainer
    trainer_cfg = cfg["trainer"].copy()
    module_dict = dict(
        accelerator=None,
        mixed_precision=args.mixed_precision,
        model=model,
        train_dataset=None,
        train_dataloader=None,
        val_datasets=val_datasets,
        val_dataloaders=val_dataloaders,
        vis_datasets=None,
        vis_dataloaders=None,
    )
    trainer_cfg.update(module_dict)
    trainer = instantiate_from_config(trainer_cfg)
    assert trainer is not None

    logger = logging.getLogger()
    # Set log level to DEBUG
    logger.setLevel(logging.DEBUG)

    if args.only_vis or args.only_save:
        trainer.eval_metrics = None

    # Start training or testing
    if args.only_save:
        trainer.validate(vis=False, save_outputs=args.only_save)
    elif args.only_vis:
        trainer.validate(vis=args.only_vis, save_outputs=args.save_outputs)
    else:
        trainer.validate(vis=args.test_vis, save_outputs=args.save_outputs)


if __name__ == "__main__":
    main()
