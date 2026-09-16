import os, sys
import torch
import tensorrt as trt
from datetime import datetime
from functools import partial

from hAlgorithm.script.kosmo.torch_trt.utils import TrtModelBase


class MVFRKosmoTRT(TrtModelBase):
    def __init__(self, engine, onnx, model, use_fp16=True):
        super().__init__(engine=engine, onnx=onnx, model=model, use_fp16=use_fp16)
        
        self._prepare_rope = model.model.fuse_encoder.pretrained._prepare_rope
        self._prepare_pos_embed = model.model.depth_head._prepare_pos_embed
        self.extra_input = {}
        self.pos_embed_size = [ # vit base channel, psize
            (96, 60),
            (192, 60),
            (384, 60),
            (768, 60),
            (64, 840),
        ]

    def __call__(self, image: torch.Tensor, prompt_scale: torch.Tensor, prompt_depth: torch.Tensor, **kwargs):
        if image.ndim == 5:
            assert image.shape[0] == 1
            prompt_scale = prompt_scale.squeeze(0)
            prompt_depth = prompt_depth.squeeze(0)
        elif image.ndim == 4:
            image = image.unsqueeze(0)
        
        B, _, _, H, W = image.shape
        output_shapes = {
            "pred_local_depth": (B, 1, H, W),
            "pred_local_conf": (B, 1, H, W),
            "pred_local_normal": (B, 3, H, W),
            "pred_local_invalid_mask": (B, 1, H, W),
        }
        if "pose" in self.extra_input:
            pos = self.extra_input["pos"]
            pos_nodiff = self.extra_input["pos_nodiff"]
            pos_embed = self.extra_input["pos_embed"]
        else:
            pos, pos_nodiff = self._prepare_rope(1, 1, H, W, image.device)
            pos_embed = {}
            for i, (channel, psize) in enumerate(self.pos_embed_size):
                pos_embed[f"pos_stage{i}"] = self._prepare_pos_embed(
                    psize, psize, channel, image, W, H
                ).to(device='cuda', dtype=self.dtype)
            self.extra_input["pos"] = pos
            self.extra_input["pos_nodiff"] = pos_nodiff
            self.extra_input["pos_embed"] = pos_embed
        
            inputs = {
                "image": image.to(device='cuda',dtype=self.dtype),
                "prompt_depth": prompt_depth.to(device='cuda',dtype=self.dtype),
                "prompt_scale": prompt_scale.to(device='cuda',dtype=self.dtype),
            }
            inputs["pos"] = pos.to(device='cuda')
            inputs["pos_nodiff"] = pos_nodiff.to(device='cuda')
            inputs.update(pos_embed)
        
        outputs = self.infer_func(input_tensors_dict=inputs, output_shapes=output_shapes)
        return outputs
