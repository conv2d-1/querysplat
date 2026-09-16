import ast
import importlib
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
import uuid
from importlib import import_module

from tabulate import tabulate


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if config is None:
        return None
    config = config.copy()
    cur_type = config.pop("type", None)
    if cur_type is None:
        return None
    params = config["params"] if "params" in config else config
    assert isinstance(params, dict)
    return get_obj_from_str(cur_type)(**params)


def get_obj_from_file(file, name):
    module = os.path.splitext(file)[0].replace("/", ".")
    return getattr(importlib.import_module(module, package=None), name)


def validate_py_syntax(filename):
    with open(filename, encoding="utf-8") as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError("There are syntax errors in config " f"file {filename}: {e}")


def file2dict(filename):
    filename = osp.abspath(osp.expanduser(filename))
    assert filename.endswith(".py")

    fileExtname = osp.splitext(filename)[1]

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=fileExtname)
        if platform.system() == "Windows":
            temp_config_file.close()
        temp_config_name = osp.basename(temp_config_file.name)
        shutil.copyfile(filename, temp_config_file.name)

        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        validate_py_syntax(filename)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
            and not isinstance(value, types.ModuleType)
            and not isinstance(value, types.FunctionType)
        }
        # delete imported module
        del sys.modules[temp_module_name]

        # close temp file
        temp_config_file.close()

    return cfg_dict


def infer_type(x):
    """
    Attempts to convert a string argument to its most likely type.

    :param x: The input string or any other type that should not be converted.
    :return: The converted value with the inferred type.
    """
    if not isinstance(x, str):
        return x

    # Convert true false to True False
    if x in ["true"]:
        x = "True"
    elif x in ["false"]:
        x = "False"
    elif x == "none":
        x = None

    # Handle special string representations of Python literals
    if x in ["True", "False", "None"]:
        return eval(x)  # Safely evaluate these strings as they are Python literals

    # Attempt to convert to integer
    try:
        return int(x)
    except ValueError:
        pass

    # Attempt to convert to float
    try:
        return float(x)
    except ValueError:
        pass

    # If all conversions fail, return the original string
    return x


def parse_unknown(unknown_args):
    """
    Parses a list of unknown arguments from the command line and returns a nested dictionary.

    This function assumes that unknown arguments come in key=value format and can have nested keys separated by dots.

    :param unknown_args: A list of strings representing unknown command-line arguments.
    :return: A nested dictionary containing the parsed key-value pairs.
    """
    clean = []
    for arg in unknown_args:
        if "=" in arg:
            key, value = arg.split(
                "=", 1
            )  # Split only at the first '=' to handle values that contain '='
            clean.extend([key, value])
        else:
            clean.append(arg)

    # Pair up keys and values
    keys = clean[::2]
    values = clean[1::2]

    result = {}
    for key, val in zip(keys, values):
        # Remove '--' prefix and split nested keys
        key_list = key.lstrip("-").split(".")

        # Use reduce to navigate through the nested dictionaries, creating them as needed
        current_level = result
        for i, k in enumerate(key_list):
            if i == len(key_list) - 1:
                current_level[k] = infer_type(val)
            elif k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]

    return result


def config_merge_args(cfg, args):
    """
    Recursively merges two dictionaries, updating the first one with values from the second.

    If both dictionaries have a key with a dictionary value, it will recursively merge those dictionaries.
    Otherwise, it will update the value in the first dictionary with the value from the second.

    :param cfg: The target dictionary to be updated.
    :param args: The source dictionary from which to take values.
    """
    for key, val in args.items():
        if key in cfg:
            if isinstance(cfg[key], dict) and isinstance(val, dict):
                config_merge_args(cfg[key], val)  # Recursive merge for nested dictionaries
            else:
                cfg[key] = val  # Replace the value if it's not a dictionary
        else:
            cfg[key] = val  # Add new key-value pair to the target dictionary


def eval_dict_to_text(val_metrics: dict, dataset_name: str, dataset_num=None):
    if dataset_num is not None:
        eval_text = f"Evaluation metrics: on dataset {dataset_name}, nums {dataset_num}\n"
    else:
        eval_text = f"Evaluation metrics:on dataset {dataset_name}\n"

    eval_text += tabulate([val_metrics.keys(), [f"{val:.5f}" for val in val_metrics.values()]])
    eval_text += "\nFormat Text:\n" + "\t".join([f"{val:.4f}" for val in val_metrics.values()])
    return eval_text


def eval_dict_list_to_text(val_metrics: list, dataset_name: str, dataset_num=None):
    if dataset_num is not None:
        eval_text = f"Evaluation metrics: on dataset {dataset_name}, nums {dataset_num}\n"
    else:
        eval_text = f"Evaluation metrics:on dataset {dataset_name}\n"

    format_text = "\nFormat Text:\n"
    for val_metric in val_metrics:
        if len(val_metric) > 0:
            eval_text += tabulate(
                [val_metric.keys(), [f"{val:.5f}" for val in val_metric.values()]]
            )
            format_text += "\t".join([f"{val:.5f}" for val in val_metric.values()]) + "\t"
    return eval_text + format_text.rstrip()
