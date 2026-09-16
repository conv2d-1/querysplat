# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.streamvggt.layers.block import Block
from hAlgorithm.modules.models.vggt.heads.head_act import activate_pose
from hAlgorithm.modules.models.vggt.layers import Mlp


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        output_index: int = -1,
        pretrain: str = None,
        num_iterations: int = 4,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.num_iterations = num_iterations

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )
        self.output_index = output_index

        self.pretrain = pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"CameraHead, load pretrain {self.pretrain}")

    def forward(
        self,
        aggregated_tokens_list: list,
        num_iterations: int = 4,
        past_key_values_camera=None,
        use_cache: bool = False,
        meta_data=None,
    ) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        # pred_pose_enc_list = self.trunk_fn(pose_tokens, self.num_iterations)
        if use_cache:
            pred_pose_enc_list = self.trunk_fn(
                pose_tokens, self.num_iterations, past_key_values_camera, use_cache
            )
        else:
            pred_pose_enc_list = self.trunk_fn(pose_tokens, self.num_iterations)

        if self.output_index is not None:
            return pred_pose_enc_list[self.output_index]
        else:
            return pred_pose_enc_list

    def trunk_fn(
        self,
        pose_tokens: torch.Tensor,
        num_iterations: int,
        past_key_values_camera=None,
        use_cache: bool = False,
    ) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for ni in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(
                self.adaln_norm(pose_tokens), shift_msa, scale_msa
            )
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            if use_cache:
                assert past_key_values_camera is not None
                block_kv_list = []
                for idx in range(self.trunk_depth):
                    # if past_key_values_camera[idx] is None:
                    #     print("past_key_values_camera", idx, past_key_values_camera[idx])
                    # else:
                    #     print("past_key_values_camera", idx, past_key_values_camera[idx][0].shape, past_key_values_camera[idx][1].shape)
                    pose_tokens_modulated, block_kv = self.trunk[idx](
                        pose_tokens_modulated,
                        attn_mask=None,
                        past_key_values=past_key_values_camera[idx],
                        use_cache=True,
                    )
                    # NOTE: 为什么不是在完整 infer 后再缓存 token？
                    # past_key_values_camera[idx] = block_kv
                    # NOTE: 修改为 num_iterations 迭代后，再缓存 token
                    if ni == num_iterations - 1:
                        block_kv_list.append(block_kv)
            else:
                L = S * 1
                frame_ids = (
                    torch.arange(L, device=pose_tokens_modulated.device) // 1
                )  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = (
                    future_frame.to(pose_tokens_modulated.dtype)
                    * torch.finfo(pose_tokens_modulated.dtype).min
                )

                for idx in range(self.trunk_depth):
                    pose_tokens_modulated = self.trunk[idx](
                        pose_tokens_modulated, attn_mask=attn_mask
                    )

            # pose_tokens_modulated = self.trunk(pose_tokens_modulated)

            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        if use_cache:
            for idx in range(self.trunk_depth):
                past_key_values_camera[idx] = block_kv_list[idx]

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
