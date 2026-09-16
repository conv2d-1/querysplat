import os, sys

sys.path.append(os.getcwd())

import cv2
import time
import math
import json
import torch
import pprint
import logging
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from accelerate.utils import set_seed

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    config_logging,
    dict_to_file,
)
from hAlgorithm.script.quantization.quant_utils.trt_helper import TrtModel
from hAlgorithm.script.quantization.quant_utils.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Train your cute model!")
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Path to engine file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="None",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--vis_step",
        type=int,
        default=-1,
    )

    args = parser.parse_args()
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.output_dir in ["None", "none", "null"] and args.engine is not None:
        engine_abs_path = os.path.abspath(args.engine)  # 处理相对路径
        base_dir = os.path.dirname(engine_abs_path)
        args.output_dir = base_dir
    args.output_dir = os.path.join(args.output_dir, f"tensorrt_{now}")

    return args

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    # Load configuration
    cfg = file2dict(args.config)

    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")

    # -------------------- Snapshot of config --------------------
    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("quant", cfg)
    logging.info("***** Running test *****")
    logging.info(
        f"  Num test examples = {sum([len(dataset) for dataset in val_datasets])}"
    )
    logging.info(f"  logging_dir = {args.output_dir}")
    logging.info("")

    trt_model = TrtModel(args.engine)

    evaluator = Evaluator(sample_num=cfg['quantization']['eval_num'], out_dir=args.output_dir, vis_step=args.vis_step)

    data_parser_cfg = cfg['data_parser'].copy()
    data_parser = instantiate_from_config(data_parser_cfg)

    postprocessor_cfg = cfg['postprocessor'].copy()
    postprocessor = instantiate_from_config(postprocessor_cfg)

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    eval_results = {}
    frame_eval_results = []

    for dataloader in val_dataloaders:
        frame_eval_results = evaluator.eval(
            trt_model, 
            dataloader, 
            data_parser, 
            postprocessor, 
            eval_metrics
        )


if __name__ == "__main__":
    main()
