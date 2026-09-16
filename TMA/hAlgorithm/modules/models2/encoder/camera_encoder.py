# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.nn as nn

from hAlgorithm.modules.models2.external.depth_anything_3.model.utils.attention import Mlp
from hAlgorithm.modules.models2.external.depth_anything_3.model.utils.block import Block
from hAlgorithm.modules.models2.external.depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
from hAlgorithm.modules.models2.external.depth_anything_3.utils.geometry import affine_inverse


class CameraEnc(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        pretrain: str = None,
        pertrain_strict: bool = True,
        c2w: bool = True,
        normalize_translations: bool = False,
        denormalize_translations: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
        )

        self.c2w = c2w
        self.normalize_translations = normalize_translations
        self.denormalize_translations = denormalize_translations
        self.pretrain = pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=pertrain_strict,
            )
            logging.info(f"CameraEnc, load pretrain {self.pretrain}")
    
    def normalize_pose_translations(self, pose_translations):
        """
        Normalize the pose translations by the average norm of the non-zero pose translations.

        Args:
            pose_translations (torch.Tensor): Pose translations tensor of size [B, V, 3]. B is the batch size, V is the number of views.
        Returns:
            normalized_pose_translations (torch.Tensor): Normalized pose translations tensor of size [B, V, 3].
            norm_factor (torch.Tensor): Norm factor tensor of size B.
        """
        assert pose_translations.ndim == 3 and pose_translations.shape[2] == 3
        # Compute distance of all pose translations to origin
        pose_translations_dis = pose_translations.norm(dim=-1)  # [B, V]
        non_zero_pose_translations_dis = pose_translations_dis > 0  # [B, V]

        # Calculate the average norm of the translations across all views (considering only views with non-zero translations)
        sum_of_all_views_pose_translations = pose_translations_dis.sum(dim=1)  # [B]
        count_of_all_views_with_non_zero_pose_translations = (
            non_zero_pose_translations_dis.sum(dim=1)
        )  # [B]
        norm_factor = sum_of_all_views_pose_translations / (
            count_of_all_views_with_non_zero_pose_translations + 1e-8
        )  # [B]

        # Normalize the pose translations by the norm factor
        norm_factor = norm_factor.clip(min=1e-8)
        normalized_pose_translations = pose_translations / norm_factor.unsqueeze(
            -1
        ).unsqueeze(-1)

        return normalized_pose_translations

    def forward(
        self,
        w2c,
        intrinsics,
        meta_data,
        scale=None,
        **kwargs,
    ) -> tuple:
        input_width = meta_data["input_width"][0]
        input_height = meta_data["input_height"][0]
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        w2c = w2c.view(-1, frame_num * view_num, 4, 4)
        intrinsics = intrinsics.view(-1, frame_num * view_num, 3, 3)

        if self.normalize_translations:
            w2c[:, :, :3, 3] = self.normalize_pose_translations(w2c[:, :, :3, 3])
        if self.denormalize_translations and scale is not None:
            w2c = w2c.clone()
            w2c[:, :, :3, 3] *= scale.view(-1, frame_num * view_num, 1)
        
        if self.c2w:
            c2w = affine_inverse(w2c)
            pose_encoding = extri_intri_to_pose_encoding(
                c2w,
                intrinsics,
                image_size_hw=(input_height, input_width),
            )
        else:
            pose_encoding = extri_intri_to_pose_encoding(
                w2c,
                intrinsics,
                image_size_hw=(input_height, input_width),
            )
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens


class CameraEncV2(CameraEnc):
    """CameraEncV2, input c2w."""

    def __init__(self, **kwargs):
        super(CameraEncV2, self).__init__()
    
    def forward(
        self,
        c2w,
        intrinsics,
        meta_data,
        scale=None,
        **kwargs,
    ) -> tuple:
        input_width = meta_data["input_width"][0]
        input_height = meta_data["input_height"][0]
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        c2w = c2w.view(-1, frame_num * view_num, 4, 4)
        intrinsics = intrinsics.view(-1, frame_num * view_num, 3, 3)

        if self.normalize_translations:
            c2w[:, :, :3, 3] = self.normalize_pose_translations(c2w[:, :, :3, 3])
        if self.denormalize_translations and scale is not None:
            c2w = c2w.clone()
            c2w[:, :, :3, 3] *= scale.view(-1, frame_num * view_num, 1)
        
        if self.c2w:
            pose_encoding = extri_intri_to_pose_encoding(
                c2w,
                intrinsics,
                image_size_hw=(input_height, input_width),
            )
        else:
            w2c = affine_inverse(c2w)
            pose_encoding = extri_intri_to_pose_encoding(
                w2c,
                intrinsics,
                image_size_hw=(input_height, input_width),
            )
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens


class CameraTokenEncoder(nn.Module):
    """
    [NEW MODULE]
    Implements the "Camera token" strategy from the paper:
    "both Pi and Ki are passed through a linear layer to generate a camera token"
    """
    def __init__(self, embed_dim=1024):
        super().__init__()
        # Intrinsics (3x3=9) + Extrinsics/W2C (4x4=16 or 3x4=12)
        # Usually flattened. Let's assume 3x3 intrinsics and 4x4 w2c.
        input_dim = 9 + 16 
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Initialize closer to zero or identity to avoid shocking the backbone initially
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.xavier_uniform_(self.encoder[2].weight)

    def forward(self, intrinsics, w2c):
        """
        intrinsics: [B, 3, 3]
        w2c: [B, 4, 4]
        Returns: [B, 1, C]
        """
        # Flatten parameters
        K_flat = intrinsics.view(intrinsics.shape[0], -1)
        P_flat = w2c.view(w2c.shape[0], -1)
        
        # Concatenate raw params
        raw_params = torch.cat([K_flat, P_flat], dim=1)
        
        # Project to token dimension
        token = self.encoder(raw_params)
        
        # Reshape to [B, 1, C] for sequence concatenation
        return token.unsqueeze(1)