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
    dict_to_file,
)
from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.modules.pipelines.outputs import DepthOutput

from tqdm import tqdm

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from hAlgorithm.script.quantization.quant_utils.ptq_quantizer import PTQuantizer
from hAlgorithm.script.quantization.quant_utils.pytorch_helper import (
    TorchModel,
    TorchLayerAnalyser,
)
from hAlgorithm.script.quantization.quant_utils.evaluator import Evaluator

def main():
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

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_abs_path = os.path.abspath(args.ckpt)  # 处理相对路径
    base_dir = os.path.dirname(ckpt_abs_path)
    output_dir = os.path.join(base_dir, f"layer_analyse_pytorch_{now}")
    os.makedirs(output_dir, exist_ok=True) 

    config_logging(out_dir=output_dir)

    cfg = file2dict(args.config)

    # Initialize datasets
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")

    # -------------------- Snapshot of config --------------------
    cfg_save_path = os.path.join(output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    model_cfg = cfg["model"].copy()
    quant_cfg = cfg["quantization"].copy()
    analyse_cfg = cfg['quantization']['pytorch_analyse'].copy()
    analyse_cfg['output_dir'] = output_dir

    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("quant_val", cfg)
    logging.info("***** Running test *****")
    logging.info(f"  Num test examples = {sum([len(dataset) for dataset in val_datasets])}")
    logging.info(f"  logging_dir = {output_dir}")
    logging.info("")

    data_parser_cfg = cfg['data_parser'].copy()
    data_parser = instantiate_from_config(data_parser_cfg)

    postprocessor_cfg = cfg['postprocessor'].copy()
    postprocessor = instantiate_from_config(postprocessor_cfg)

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    evaluator = Evaluator(sample_num=analyse_cfg['data_num'], out_dir=None, vis_step=-1)
    
    ptq_quantizer = PTQuantizer()
    ptq_quantizer.load_model(model_cfg, ckpt_path=ckpt_abs_path)

    model = ptq_quantizer.model.model

    model_wrapper = TorchModel(model)

    analyser = TorchLayerAnalyser(
        model_wrapper, 
        val_dataloaders, 
        data_parser, 
        postprocessor, 
        eval_metrics, 
        evaluator, 
        analyse_cfg
    )

    analyser.analyse_layers()

if __name__ == "__main__":
    main()
