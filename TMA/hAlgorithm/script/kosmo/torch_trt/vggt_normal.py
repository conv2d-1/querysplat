import os, sys
import torch
import tensorrt as trt
from datetime import datetime
from functools import partial

from hAlgorithm.script.kosmo.torch_trt.utils import TrtModelBase


class VGGTNormalTRT_MV(TrtModelBase):
    def __init__(self, engine=None, onnx=None, model=None, use_fp16=True):
        super().__init__(engine=engine, onnx=onnx, model=model, use_fp16=use_fp16)
        self._prepare_extra_input = model.model.fuse_encoder._onnx_prepare_extra_input
        self.patch_size = model.model.rgb_encoder.patch_size
        self.dino_norm = model.model.rgb_encoder.dinov2.norm
        self.extra_input = {}
    
    @torch.no_grad()
    def __call__(self, image: torch.Tensor, feat3: torch.Tensor, **kwargs):
        if image.ndim == 5:
            assert image.shape[0] == 1
            image = image.squeeze(0)
        B, _, H, W = image.shape

        patch_h, patch_w = (
            H // self.patch_size,
            W // self.patch_size,
        )
        if "camera_token" in self.extra_input:
            camera_token = self.extra_input["camera_token"]
            register_token = self.extra_input["register_token"]
            pos = self.extra_input["pos"]
        else:
            camera_token, register_token, pos = self._prepare_extra_input(1, 1, patch_h, patch_w)
            self.extra_input["camera_token"] = camera_token.detach().clone()
            self.extra_input["register_token"] = register_token.detach().clone()
            self.extra_input["pos"] = pos.detach().clone()

        trt_input = dict(
            camera_token=camera_token.cuda().to(dtype=self.dtype),
            register_token=register_token.cuda().to(dtype=self.dtype),
            feat3=feat3.cuda().to(dtype=self.dtype),
            pos=pos.cuda(),
        )
        output_shapes = {
            "pred_local_normal": (1, 3, H, W),
        }
        outputs = self.infer_func(input_tensors_dict=trt_input, output_shapes=output_shapes)
        
        return outputs


class VGGTNormalTRT_Dino(TrtModelBase):
    def __init__(self, engine=None, onnx=None, model=None, use_fp16=True):
        super().__init__(engine=engine, onnx=onnx, model=model, use_fp16=use_fp16)
        self.patch_size = model.model.rgb_encoder.patch_size
        self.dino_norm = model.model.rgb_encoder.dinov2.norm
        self.patch_start = 1 + model.model.rgb_encoder.dinov2.num_register_tokens

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, **kwargs):
        if image.ndim == 5:
            assert image.shape[0] == 1
            image = image.squeeze(0)
        B, _, H, W = image.shape
        
        input_tensors_dict = dict(
            image=image.cuda().to(dtype=self.dtype),
        )
        
        trt_input = {
            k: v for k, v in input_tensors_dict.items()
            if k in ['image']
        }
        output_shapes = {
            "/rgb_encoder/blocks.23/Add_1_output_0": (1, 3605, 1024),
        }
        outputs = self.infer_func(input_tensors_dict=trt_input, output_shapes=output_shapes)
        out = outputs["/rgb_encoder/blocks.23/Add_1_output_0"]
        with torch.autocast("cuda", dtype=torch.float32, enabled=True):
            out = self.dino_norm(out)
        outputs["feat3"] = out[:, self.patch_start:].to(dtype=self.dtype)
        return outputs


class VGGTNormalTRT_Split():
    def __init__(
        self,
        dino_engine=None,
        dino_onnx=None,
        mv_engine=None,
        mv_onnx=None,
        model=None,
        use_fp16=True
    ):
        self.dino = VGGTNormalTRT_Dino(dino_engine, dino_onnx, model, use_fp16)
        self.mv_dpt = VGGTNormalTRT_MV(mv_engine, mv_onnx, model, use_fp16)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, **kwargs):
        dino_out = self.dino(image=image)
        final_out = self.mv_dpt(image=image, **dino_out)
        return final_out


class VGGTNormalTRT(TrtModelBase):
    def __init__(self, engine=None, onnx=None, model=None, use_fp16=True):
        super().__init__(engine=engine, onnx=onnx, model=model, use_fp16=use_fp16)
        self._prepare_extra_input = model.model.fuse_encoder._onnx_prepare_extra_input
        self.patch_size = model.model.rgb_encoder.patch_size
        self.dino_norm = model.model.rgb_encoder.dinov2.norm
        
        self.extra_input = {}

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, **kwargs):
        if image.ndim == 5:
            assert image.shape[0] == 1
            image = image.squeeze(0)
        B, _, H, W = image.shape
        output_shapes = {
            "pred_local_normal": (B, 3, H, W),
        }
        patch_h, patch_w = (
            H // self.patch_size,
            W // self.patch_size,
        )
        if "camera_token" in self.extra_input:
            camera_token = self.extra_input["camera_token"]
            register_token = self.extra_input["register_token"]
            pos = self.extra_input["pos"]
        else:
            camera_token, register_token, pos = self._prepare_extra_input(1, 1, patch_h, patch_w)
            self.extra_input["camera_token"] = camera_token.detach().clone()
            self.extra_input["register_token"] = register_token.detach().clone()
            self.extra_input["pos"] = pos.detach().clone()
        
        trt_input = dict(
            image=image.cuda().to(dtype=self.dtype),
            camera_token=camera_token.cuda().to(dtype=self.dtype),
            register_token=register_token.cuda().to(dtype=self.dtype),
            pos=pos.cuda(),
        )
        outputs = self.infer_func(input_tensors_dict=trt_input, output_shapes=output_shapes)
        return outputs
