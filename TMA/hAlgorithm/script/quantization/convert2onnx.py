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
from hAlgorithm.utils.util import eval_dict_to_text
from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    config_logging,
    dict_to_file,
    parse_unknown,
    config_merge_args,
)

from tqdm import tqdm
import onnx
import onnxruntime as ort
from onnxsim import simplify

from hAlgorithm.script.quantization.quant_utils.ptq_quantizer import PTQuantizer
from hAlgorithm.script.quantization.quant_utils.onnx_helper import (
    OnnxModel,
)
from hAlgorithm.script.quantization.quant_utils.pytorch_helper import (
    get_layers,
    TorchModel,
    export_onnx,
)

from hAlgorithm.script.quantization.quant_utils.evaluator import Evaluator

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
        "--vis_step",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_abs_path = os.path.abspath(args.ckpt)  # 处理相对路径
    base_dir = os.path.dirname(ckpt_abs_path)
    output_dir = os.path.join(base_dir, f"convert2onnx_{now}")

    args.output_dir = output_dir

    return args

@torch.no_grad()
def quant_forward(self, image, prompt_depth, prompt_scale):
    return self.trans_onnx(image, prompt_depth, prompt_scale)

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_logging(out_dir=args.output_dir)

    cfg = file2dict(args.config)

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("val", cfg)
    quant_datasets, quant_dataloaders, quant_nums = prepare_data_loaders("quant", cfg)
    cfg_str = pprint.pformat(cfg, compact=True)
    logging.info("config:\n" + cfg_str + "\n")
    logging.info("")
    
    # -------------------- Snapshot of config --------------------
    cfg_save_path = os.path.join(args.output_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    model_cfg = cfg["model"].copy()

    ckpt_path = args.ckpt
    
    quant_cfg = cfg["quantization"]
    if 'non_quant_part' not in quant_cfg:
        quant_cfg['non_quant_part'] = []
    if 'calib_config' not in quant_cfg:
        quant_cfg['calib_config'] = None
    
    onnx_path = f"{args.output_dir}/quant_model.onnx"
    if quant_cfg['if_quant']:
        logging.info(f"  Generate Quantized Model")
        ptq_quantizer = PTQuantizer(quant_cfg['calib_config'])
        ptq_quantizer.load_model(model_cfg, ckpt_path=ckpt_path)
        non_quant_parts = get_layers(ptq_quantizer.model.model, quant_cfg['non_quant_part'])
        ptq_quantizer.set_non_quant_modules(non_quant_parts, quant_cfg['non_quant_part'])

        model = ptq_quantizer.model
    else:
        logging.info(f"  Generate Orig Model, Orig Onnx will be exported")
        model = instantiate_from_config(model_cfg)
        model.load_checkpoint(ckpt_path)
        model.eval()
        model = model.cuda()

    data_parser_cfg = cfg['data_parser'].copy()
    data_parser = instantiate_from_config(data_parser_cfg)

    postprocessor_cfg = cfg['postprocessor'].copy()
    postprocessor = instantiate_from_config(postprocessor_cfg)

    eval_metrics = instantiate_from_config(cfg["trainer"]["eval_metrics"])

    torch_model_wrapper = TorchModel(model.model)
    torch_eval_dir = f"{args.output_dir}/torch_eval"
    evaluator = Evaluator(sample_num=quant_cfg['eval_num'], out_dir=torch_eval_dir, vis_step=args.vis_step)

    # evaluate pytorch model
    for dataloader in (val_dataloaders + quant_dataloaders):
        frame_eval_results = evaluator.eval(
            torch_model_wrapper, 
            dataloader, 
            data_parser, 
            postprocessor, 
            eval_metrics
        )

    ## export to onnx model
    export_onnx(model, quant_dataloaders[0], data_parser, onnx_path, forward_func=quant_forward)

    onnx_wrapper = OnnxModel(onnx_path)
    onnx_eval_dir = f"{args.output_dir}/onnx_eval"
    evaluator.out_dir = onnx_eval_dir
    frame_eval_results = evaluator.eval(
        onnx_wrapper, 
        quant_dataloaders[0], 
        data_parser, 
        postprocessor, 
        eval_metrics
    )

    onnx_wrapper.save(onnx_path)

if __name__ == "__main__":
    main()
