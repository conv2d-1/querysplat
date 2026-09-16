import os, sys
sys.path.append(os.getcwd())

import datetime
import argparse

import cv2

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    config_logging,
    dict_to_file,
)
from hAlgorithm.script.quantization.quant_utils.onnx_helper import (
    DequantAnalyse,
    OnnxModel
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

    cfg = file2dict(args.config)

    onnx_path = args.onnx_path
    onnx_abs_path = os.path.abspath(onnx_path)  # 处理相对路径

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    onnx_dir = os.path.dirname(onnx_abs_path)
    analyse_dir = os.path.join(onnx_dir, f"dequant_analyse_{now}")
    os.makedirs(analyse_dir, exist_ok=True) 

    onnx_search_cfg = cfg['quantization']['onnx_analyse']
    onnx_search_cfg['output_dir'] = analyse_dir

    onnx_wrapper = OnnxModel(onnx_abs_path)

    # load dataloader
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("quant", cfg)
    print(f"Num test examples = {sum([len(dataset) for dataset in val_datasets])}")
    
    data_parser_cfg = cfg['data_parser'].copy()
    data_parser = instantiate_from_config(data_parser_cfg)

    postprocessor_cfg = cfg['postprocessor'].copy()
    postprocessor = instantiate_from_config(postprocessor_cfg)

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    evaluator = Evaluator(sample_num=onnx_search_cfg['data_num'], out_dir=None, vis_step=-1)

    dequant_analyser = DequantAnalyse(
        onnx_wrapper, 
        val_dataloaders[0], 
        data_parser, 
        postprocessor,
        eval_metrics,
        evaluator,
        cfg
    )

    dequant_analyser.analyse() 