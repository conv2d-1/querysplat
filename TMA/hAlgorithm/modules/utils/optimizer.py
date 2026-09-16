import copy
import glob
import inspect
import logging
import os

import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from hAlgorithm.utils import instantiate_from_config


def register_torch_optimizers():
    torch_optimizers = {}
    for module_name in dir(torch.optim):
        if module_name.startswith("__"):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            torch_optimizers[module_name] = _optim
    return torch_optimizers


def register_bnb_optimizers():
    bnb_optimizers = {}
    for module_name in dir(bnb.optim):
        if module_name.startswith("__"):
            continue
        _optim = getattr(bnb.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, bnb.optim.optimizer.Optimizer2State):
            bnb_optimizers[module_name] = _optim
    return bnb_optimizers


TORCH_OPTIMIZER = register_torch_optimizers()
BNB_OPTIMIZER = register_bnb_optimizers()


def build_optimizer_with_cfg(optim_cfg, model):
    optim_cfg = copy.deepcopy(optim_cfg)
    optim_type = optim_cfg.pop("type", None)

    if optim_type is None:
        raise RuntimeError(f"{optim_type} is not set")
    if optim_type not in TORCH_OPTIMIZER and optim_type not in BNB_OPTIMIZER:
        raise RuntimeError(
            f"{optim_type} is not neither supported in torch {torch.__version__}, nor supported in bitsandbytes {bnb.__version__}"
        )

    def match(key1, key_list, strict_match=False):
        match_key = None
        if not strict_match:
            for k in key_list:
                if k in key1:
                    return k
        else:
            match_len = 0
            for k in key_list:
                if key1.startswith(k) and match_len < len(k):
                    match_key = k
                    match_len = len(k)
        return match_key

    if optim_type in TORCH_OPTIMIZER:
        optim_obj = TORCH_OPTIMIZER[optim_type]
    else:
        optim_obj = BNB_OPTIMIZER[optim_type]
    matching_type = optim_cfg.pop("strict_match", False)
    debug = optim_cfg.pop("debug", False)

    module_names = optim_cfg.keys()
    model_parameters = {i: [] for i in module_names}
    nongrad_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            match_key = match(key, module_names, matching_type)
            assert match_key is not None, key
            model_parameters[match_key].append(value)
            if debug:
                logging.info(f"optimizer: {match_key}, {key}")
        else:
            nongrad_parameters.append(value)
            if debug:
                logging.info(f"optimizer: nongrad, {key}")

    optims = [{"params": model_parameters[k], **optim_cfg[k]} for k in optim_cfg.keys()]
    optimizer = optim_obj(optims)

    return optimizer


def build_optimizer_and_scheduler_with_cfg(optim_cfg, lr_scheduler_cfg, model, skip_match_key=False):
    optim_cfg = copy.deepcopy(optim_cfg)
    optim_type = optim_cfg.pop("type", None)

    if optim_type is None:
        raise RuntimeError(f"{optim_type} is not set")
    if optim_type not in TORCH_OPTIMIZER and optim_type not in BNB_OPTIMIZER:
        raise RuntimeError(
            f"{optim_type} is not neither supported in torch {torch.__version__}, nor supported in bitsandbytes {bnb.__version__}"
        )

    def match(key1, key_list, strict_match=False):
        match_key = None
        if not strict_match:
            for k in key_list:
                if k in key1:
                    return k
        else:
            match_len = 0
            for k in key_list:
                if key1.startswith(k) and match_len < len(k):
                    match_key = k
                    match_len = len(k)
        return match_key

    if optim_type in TORCH_OPTIMIZER:
        optim_obj = TORCH_OPTIMIZER[optim_type]
    else:
        optim_obj = BNB_OPTIMIZER[optim_type]
    matching_type = optim_cfg.pop("strict_match", False)
    debug = optim_cfg.pop("debug", False)

    module_names = optim_cfg.keys()

    lr_lambdas = [optim_cfg[module_name].pop("lambda_str", None) for module_name in module_names]
    if any([(lr_lambda is not None) for lr_lambda in lr_lambdas]):
        base_lambda = lambda step: 1.0
        lr_lambdas = [
            (lr_lambda if lr_lambda is not None else base_lambda) for lr_lambda in lr_lambdas
        ]
    else:
        lr_lambdas = None

    parameter_name_list = [
        optim_cfg[module_name].pop("parameter_name", None) for module_name in module_names
    ]

    model_parameters = {i: [] for i in module_names}
    nongrad_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            match_key = match(key, module_names, matching_type)

            if skip_match_key and match_key is None:
                continue

            assert match_key is not None, key
            model_parameters[match_key].append(value)
            if debug:
                logging.info(f"optimizer: {match_key}, {key}")
        else:
            nongrad_parameters.append(value)
            if debug:
                logging.info(f"optimizer: nongrad, {key}")

    # optims = [{"params": model_parameters[k], **optim_cfg[k]} for k in optim_cfg.keys()]
    optims = []
    for module_name, param_name in zip(module_names, parameter_name_list):
        optim_cfg_group = {
            "params": model_parameters[module_name],
            "name": param_name if param_name is not None else module_name,
            **optim_cfg[module_name],
        }
        optims.append(optim_cfg_group)
    optimizer = optim_obj(optims)

    # LR scheduler
    if lr_lambdas is not None:
        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambdas)
    else:
        lr_func = instantiate_from_config(lr_scheduler_cfg)
        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    return optimizer, lr_scheduler
