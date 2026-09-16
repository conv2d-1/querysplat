import os, sys
sys.path.append(os.getcwd())

import datetime
import onnx
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm
import json

from onnxsim import simplify

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    config_logging,
    dict_to_file,
)
from hAlgorithm.script.quantization.quant_utils.onnx_helper import (
    OnnxModel,
    SingleLayerAnalyser
)
from hAlgorithm.script.quantization.quant_utils.evaluator import Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="amax statistics")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Path to onnx path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config path",
    )
    args = parser.parse_args()

    
    config_path = args.config
    cfg = file2dict(config_path)

    onnx_path = args.onnx_path
    onnx_abs_path = os.path.abspath(onnx_path)  # 处理相对路径

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    onnx_dir = os.path.dirname(onnx_abs_path)
    analyse_dir = os.path.join(onnx_dir, f"quant_layer_analyse_{now}")
    os.makedirs(analyse_dir, exist_ok=True) 

    onnx_analyse_cfg = cfg['quantization']['onnx_analyse']
    onnx_analyse_cfg['output_dir'] = analyse_dir

    onnx_wrapper = OnnxModel(onnx_abs_path)

    # load dataloader
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("quant", cfg)
    print(f"Num test examples = {sum([len(dataset) for dataset in val_datasets])}")
    
    data_parser_cfg = cfg['data_parser'].copy()
    data_parser = instantiate_from_config(data_parser_cfg)

    postprocessor_cfg = cfg['postprocessor'].copy()
    postprocessor = instantiate_from_config(postprocessor_cfg)

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    evaluator = Evaluator(sample_num=onnx_analyse_cfg['data_num'], out_dir=None, vis_step=-1)

    analyser = SingleLayerAnalyser(
        onnx_wrapper, 
        val_dataloaders[0], 
        data_parser, 
        postprocessor,
        eval_metrics,
        evaluator,
        cfg
    )

    analyser.analyse_layers()