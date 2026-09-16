import os, sys

sys.path.append(os.getcwd())

import math
import json
import torch
import pprint
import logging
import datetime
import argparse

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    config_logging,
    dict_to_file
)

from tqdm import tqdm

from hAlgorithm.script.quantization.quant_utils.ptq_quantizer import PTQuantizer
from hAlgorithm.script.quantization.quant_utils.pytorch_helper import (
    get_layers,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Quant your cute model!")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path of checkpoint to be load.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="None",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.output_dir in ["None", "none", "null"] and args.ckpt not in ["None", "none", "null"]:
        ckpt_abs_path = os.path.abspath(args.ckpt)  # 处理相对路径
        base_dir = os.path.dirname(ckpt_abs_path)
        args.output_dir = base_dir
    args.output_dir = os.path.join(args.output_dir, f"quantization_{now}")

    if args.ckpt in ["None", "none", "null"]:
        args.ckpt = None

    return args

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    cfg = file2dict(args.config)
    
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")
    
    # -------------------- Snapshot of config --------------------
    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("val", cfg)
    quant_datasets, quant_dataloaders, quant_nums = prepare_data_loaders("quant", cfg)

    model_cfg = cfg["model"].copy()

    ckpt_path = args.ckpt
    
    quant_cfg = None
    quant_cfg = cfg["quantization"]
    if 'non_quant_part' not in quant_cfg:
        quant_cfg['non_quant_part'] = {}
    
    logging.info(f"  Generate Quantized Model")
    ptq_quantizer = PTQuantizer(quant_cfg['calib_config'])
    ptq_quantizer.load_model(model_cfg, ckpt_path=ckpt_path)
    non_quant_parts = get_layers(ptq_quantizer.model.model, quant_cfg['non_quant_part'])
    ptq_quantizer.set_non_quant_modules(non_quant_parts, quant_cfg['non_quant_part'])
    if quant_cfg['if_calib']:
        ptq_quantizer.quant_calib(val_dataloaders+quant_dataloaders, num_samples=quant_cfg['calib_num'])

    torch.save(ptq_quantizer.model.model.state_dict(), f"{args.output_dir}/ckpt_quant.pth")

if __name__ == "__main__":
    main()
