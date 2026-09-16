import os, sys
sys.path.append(os.getcwd())

import torch

from hAlgorithm.utils import instantiate_from_config

from itertools import islice
from tqdm import tqdm
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, calib
from pytorch_quantization.tensor_quant import QuantDescriptor, fake_tensor_quant

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(**kwargs, strict=False)
            print(f"{name:40}: {module}")
    model.cuda()

class PTQuantizer:
    def __init__(self, calib_config=None):
        if calib_config is not None:
            self.calib_config = calib_config
        else:
            self.calib_config = dict(method="histogram",
                                     param=dict(
                                         method="percentile",
                                         percentile=99.9
                                     ))

    def load_model(self, model_cfg, ckpt_path=None):
        quant_modules.initialize()
        quant_desc_input = QuantDescriptor(calib_method=self.calib_config["method"], num_bits=8)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)

        # Initialize model
        self.model = instantiate_from_config(model_cfg)
        assert self.model is not None

        if ckpt_path is not None:
            self.model.load_checkpoint(ckpt_path)

        self.model.eval()
        self.model = self.model.cuda()

        if not hasattr(self.model, "device"):
            self.model.device = 'cuda'
        if not hasattr(self.model, "dtype"):
            self.model.dtype = torch.float32

    def quant_calib(self, dataloader, num_samples=None):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        if not isinstance(dataloader, list):
            dataloader = [dataloader]

        for di, data_loader in enumerate(dataloader):
            dataset_name = (
                data_loader.dataset.name
                if hasattr(data_loader.dataset, "name")
                else "unnamed"
            )

            if num_samples is not None:
                num_samples = min(num_samples, len(data_loader))
            else:
                num_samples = len(data_loader)
            for i, batch in enumerate(islice(tqdm(data_loader, desc=f"Inference on {dataset_name}"), num_samples)):
                self.model.infer(**batch)
        
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
        if self.calib_config["method"] == "histogram":
            compute_amax(self.model, **(self.calib_config["param"]))
        elif self.calib_config["method"] == "max":
            compute_amax(self.model, method="max")
        else:
            raise NotImplementedError

    def set_non_quant_modules(self, non_quant_parts, non_quant_names=None):
        for idx, module_part in enumerate(non_quant_parts):
            for name, module in module_part.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    module.disable()
                    if non_quant_names is not None:
                        print(f"disable {non_quant_names[idx]}-{name}")
