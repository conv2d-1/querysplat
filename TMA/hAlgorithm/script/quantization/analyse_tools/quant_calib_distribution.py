import os, sys

sys.path.append(os.getcwd())

import math
import json
import torch
import pprint
import logging
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
    InitProcessGroupKwargs,
)

from tqdm import tqdm

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, calib
from pytorch_quantization.tensor_quant import QuantDescriptor, fake_tensor_quant

from hAlgorithm.datasets import prepare_data_loaders
from hAlgorithm.utils import (
    file2dict,
    instantiate_from_config,
    config_logging,
    dict_to_file,
    parse_unknown,
    config_merge_args,
)
from hAlgorithm.script.quantization.quant_utils.pytorch_helper import (
    get_layers,
)
from hAlgorithm.script.quantization.quant_utils.data_helper import (
    data_parser,
)

from hAlgorithm.script.quantization.quant_utils.ptq_quantizer import PTQuantizer

def parse_args():
    parser = argparse.ArgumentParser(description="")
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

    args = parser.parse_args()

    return args

def convert_to_histogram(weights, bins=128):
    flattened = weights.flatten()
    max_val = flattened.max()
    min_val = flattened.min()
    if max_val == min_val:
        return np.array([flattened.size]), np.array([min_val, max_val])

    bin_width = (max_val - min_val) / bins if bins else (max_val - min_val)
    bin_edges = np.linspace(min_val, max_val, bins+1)
    hist, _ = np.histogram(flattened, bins=bin_edges)
    return hist, bin_edges

def draw_weight_distribution(weights, layer_name, amax, bins=128):
    hist, bin_edges = convert_to_histogram(weights, bins)
    hist_log2 = np.log2(hist + 1)

    fig, ax = plt.subplots()
    width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_edges[1:], hist_log2, label=f"{layer_name}_weight", width=width, edgecolor='black')
    ax.axvline(x=amax, color='r', linestyle='--', linewidth=2, label='Amax')
    ax.text(x=amax, y=0.5, s=f'Amax: {amax.squeeze():.2f}', 
            color='red', fontsize=10,
            ha='center', va='bottom') 
    ax.axvline(x=-amax, color='r', linestyle='--', linewidth=2, label='Amax')
    ax.set_title(f"{layer_name} weight distribution")
    ax.set_xlabel('bin_edges')
    ax.set_ylabel('hist_log2')
    return fig

def draw_input_distribution(quant_module, layer_name):
    amax = quant_module._amax.cpu().numpy()
    calibrator = quant_module._calibrator
    calib_hist = calibrator._calib_hist.cpu().numpy()
    calib_hist_log2 = np.log2(calib_hist + 1)
    calib_bin_edges = calibrator._calib_bin_edges.cpu().numpy()

    fig, ax = plt.subplots()
    ax.bar(calib_bin_edges[1:], calib_hist_log2, label=f"{layer_name}_input", width=2e-5, edgecolor='black')
    ax.axvline(x=amax, color='r', linestyle='--', linewidth=2, label='Amax')
    ax.text(x=amax, y=0.5, s=f'Amax: {amax:.2f}', 
            color='red', fontsize=10,
            ha='center', va='bottom') 
    ax.set_title(f"{layer_name} input distribution")
    ax.set_xlabel('calib_bin_edges')
    ax.set_ylabel('calib_hist_log2')
    return fig

def draw_distribution(module, layer_name, output_dir):
    input_quantizer = module._input_quantizer

    layer_dir = os.path.join(output_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)

    if input_quantizer._calibrator._calib_hist is not None and \
       input_quantizer._calibrator._calib_bin_edges is not None:
        df_input_calibs = pd.DataFrame({
            'calib_hist': input_quantizer._calibrator._calib_hist.cpu().numpy(),
            'calib_bin_edges': input_quantizer._calibrator._calib_bin_edges.cpu().numpy()[1:]
        })
        df_input_calibs.to_csv(f"{layer_dir}/input_calibs.csv", index=False)

        plt.tight_layout()
        fig = draw_input_distribution(input_quantizer, layer_name)
        fig.savefig(f"{layer_dir}/{layer_name}_input.png")
        plt.close()

    weights = module.weight.detach().cpu().numpy()
    weight_quantizer = module._weight_quantizer
    weight_amaxs = weight_quantizer._amax.cpu().numpy()

    if len(weight_amaxs.shape) > 1:
        non_one_dims = [dim for dim, size in enumerate(weight_amaxs.shape) if size != 1]
        target_dim = non_one_dims[0]
        weight_amaxs = weight_amaxs.squeeze()
        for i in range(weights.shape[target_dim]):
            weight = np.take(weights, i, axis=target_dim)
            fig = draw_weight_distribution(weight, layer_name, weight_amaxs[i])
            fig.savefig(f"{layer_dir}/{layer_name}_weight{i}.png")
            plt.close()
    else:
        fig = draw_weight_distribution(weights, layer_name, weight_amaxs)
        fig.savefig(f"{layer_dir}/{layer_name}_weight.png")
        plt.close()
    
def get_quant_layers(module_part):
    layers = []
    layer_names = []
    for name, module in module_part.named_modules():
        if hasattr(module, "weight") and hasattr(module, "_input_quantizer"):
            layers.append(module)
            layer_names.append(name)
    return layers, layer_names

if __name__ == "__main__":
    args = parse_args()

    cfg = file2dict(args.config)

    # Initialize datasets
    val_datasets, val_dataloaders, val_nums = prepare_data_loaders("val", cfg)
    quant_datasets, quant_dataloaders, quant_nums = prepare_data_loaders("quant", cfg)

    model_cfg = cfg["model"].copy()

    ckpt_path = None
    if args.load_from is not None:
        ckpt_path = args.load_from
    
    quant_cfg = None
    if "quantization" in cfg:
        quant_cfg = cfg["quantization"]
    else:
        quant_cfg = dict(
            if_quant=True,
            if_calib=True,
            calib_num=None,
            non_quant_part = {}
        )
    if 'non_quant_part' not in quant_cfg:
        quant_cfg['non_quant_part'] = {}

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_abs_path = os.path.abspath(ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_abs_path)
    analyse_dir = os.path.join(ckpt_dir, f"quant_calib_distribution_{now}")
    os.makedirs(analyse_dir, exist_ok=True) 

    cfg_save_path = os.path.join(analyse_dir, os.path.basename(args.config))
    assert cfg_save_path != args.config
    dict_to_file(cfg, cfg_save_path)

    ptq_quantizer = PTQuantizer(quant_cfg['calib_config'])
    ptq_quantizer.load_model(model_cfg, ckpt_path=ckpt_path)

    model = ptq_quantizer.model.model
    # get to analyse layers
    to_analyse_layers = get_layers(model, quant_cfg['pytorch_analyse']['analyse_distribute_layers'])

    non_quant_parts = get_layers(model, quant_cfg['non_quant_part'])
    ptq_quantizer.set_non_quant_modules(non_quant_parts)
    ptq_quantizer.quant_calib(val_dataloaders+quant_dataloaders, num_samples=quant_cfg['calib_num'])

    for i in range(len(to_analyse_layers)):
        module_name = quant_cfg['pytorch_analyse']['analyse_distribute_layers'][i]
        module_name = module_name.replace("/", "-")
        module_dir = os.path.join(analyse_dir, module_name)
        os.makedirs(module_dir, exist_ok=True)
        
        layers, layer_names = get_quant_layers(to_analyse_layers[i])
        for i in range(len(layer_names)):
            layer_name = layer_names[i]
            layer_name = layer_name.replace("/", "-")
            print(f"processing layer {module_name} : {layer_name}")
            layer = layers[i]
            draw_distribution(layer, layer_name, module_dir)
