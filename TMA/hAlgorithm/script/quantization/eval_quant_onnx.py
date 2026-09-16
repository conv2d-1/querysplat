import os, sys
sys.path.append(os.getcwd())

import datetime
import onnx
import numpy as np
import pandas as pd
import onnxruntime as ort

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
from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.script.quantization.quant_utils.onnx_helper import OnnxModel
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
    parser.add_argument(
        "--vis_step",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    onnx_path = args.onnx_path
    config_path = args.config

    onnx_abs_path = os.path.abspath(onnx_path)  # 处理相对路径
    onnx_dir = os.path.dirname(onnx_abs_path)

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(onnx_dir, f"onnx_eval_{now}")
    os.makedirs(output_dir, exist_ok=True)

    onnx_wrapper = OnnxModel(onnx_abs_path)

    cfg = file2dict(config_path)

    onnx_analyse_cfg = cfg['quantization']['onnx_analyse']

    onnx_wrapper.modify_onnx_by_cfg(onnx_analyse_cfg)

    # load dataloader
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("quant", cfg)
    print(f"Num test examples = {sum([len(dataset) for dataset in val_datasets])}")
    
    data_parser_cfg = cfg['data_parser'].copy()
    data_parser = instantiate_from_config(data_parser_cfg)

    postprocessor_cfg = cfg['postprocessor'].copy()
    postprocessor = instantiate_from_config(postprocessor_cfg)

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    evaluator = Evaluator(sample_num=cfg['quantization']['eval_num'], out_dir=output_dir, vis_step=args.vis_step)

    eval_results = {}
    frame_eval_results = []

    for dataloader in val_dataloaders:
        frame_eval_results = evaluator.eval(
            onnx_wrapper, 
            dataloader, 
            data_parser, 
            postprocessor, 
            eval_metrics
        )

    onnx_wrapper.save(f"{output_dir}/modified_quant_model.onnx")