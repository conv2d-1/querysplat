# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything model class defined using UniCeption modules.
"""

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from hAlgorithm.modules.models2.external.mapanything.utils.geometry import (
    apply_log_to_norm,
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    normalize_pose_translations,
)
from hAlgorithm.modules.models2.external.uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
    PredictionHeadTokenInput,
)
from hAlgorithm.modules.models2.external.uniception.models.encoders import (
    ViTEncoderInput,
)
from hAlgorithm.modules.models2.external.uniception.models.info_sharing.base import MultiViewTransformerInput

from hAlgorithm.utils import instantiate_from_config


# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


class MapAnything(nn.Module, PyTorchModelHubMixin):
    "Modular MapAnything model class that supports input of images & optional geometric modalities (multiple reconstruction tasks)."

    def __init__(
        self,
        enc_embed_dim,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        geometric_input_config: Dict,
        fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = None,
        ignore_calibration_inputs: bool = False,
        ignore_depth_inputs: bool = False,
        ignore_pose_inputs: bool = False,
        ignore_depth_scale_inputs: bool = False,
        ignore_pose_scale_inputs: bool = False,
    ):
        """
        Multi-view model containing an image encoder fused with optional geometric modalities followed by a multi-view attention transformer and respective downstream heads.
        The goal is to output scene representation.
        The multi-view attention transformer also takes as input a scale token to predict the metric scaling factor for the predicted scene representation.

        Args:
            encoder_config (Dict): Configuration for the encoder.
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
            pred_head_config (Dict): Configuration for the prediction heads.
            geometric_input_config (Dict): Configuration for the input of optional geometric modalities.
            fusion_norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Normalization layer to use after fusion (addition) of encoder and geometric modalities. (default: partial(nn.LayerNorm, eps=1e-6))
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            load_specific_pretrained_submodules (bool): Whether to load specific pretrained submodules. (default: False)
            specific_pretrained_submodules (list): List of specific pretrained submodules to load. Must be provided when load_specific_pretrained_submodules is True. (default: None)
            torch_hub_force_reload (bool): Whether to force reload the encoder from torch hub. (default: False)
        """
        super().__init__()

        # Initalize the attributes
        self.enc_embed_dim = enc_embed_dim
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.geometric_input_config = geometric_input_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules
        self.class_init_args = {
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "geometric_input_config": self.geometric_input_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
        }

        # Initialize image encoder
        self.encoder = instantiate_from_config(self.encoder_config)

        # Initialize the encoder for ray directions
        ray_dirs_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        self.ray_dirs_encoder = instantiate_from_config(ray_dirs_encoder_config)

        # Initialize the encoder for depth (normalized per view and values after normalization are scaled logarithmically)
        depth_encoder_config = self.geometric_input_config["depth_encoder_config"]
        self.depth_encoder = instantiate_from_config(depth_encoder_config)

        # Initialize the encoder for log scale factor of depth
        depth_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        self.depth_scale_encoder = instantiate_from_config(depth_scale_encoder_config)

        # Initialize the encoder for camera rotation
        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        self.cam_rot_encoder = instantiate_from_config(cam_rot_encoder_config)

        # Initialize the encoder for camera translation (normalized across all provided camera translations)
        cam_trans_encoder_config = self.geometric_input_config[
            "cam_trans_encoder_config"
        ]
        self.cam_trans_encoder = instantiate_from_config(cam_trans_encoder_config)

        # Initialize the encoder for log scale factor of camera translation
        cam_trans_scale_encoder_config = self.geometric_input_config[
            "scale_encoder_config"
        ]
        self.cam_trans_scale_encoder = instantiate_from_config(cam_trans_scale_encoder_config)

        # Initialize the fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.enc_embed_dim)

        # Initialize the Scale Token
        # Used to scale the final scene predictions to metric scale
        # During inference extended to (B, C, T), where T is the number of tokens (i.e., 1)
        self.scale_token = nn.Parameter(torch.zeros(self.enc_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Initialize the info sharing module (multi-view transformer)
        self._initialize_info_sharing(info_sharing_config)

        # Initialize the prediction heads
        self._initialize_prediction_heads(pred_head_config)

        # Initialize the final adaptors
        self._initialize_adaptors(pred_head_config)

        # Load pretrained weights
        self._load_pretrained_weights()

        self.memory_efficient_inference: bool = False
        self.use_amp: bool = True
        self.apply_mask: bool = True
        self.mask_edges: bool = True
        self.edge_normal_threshold: float = 5.0
        self.edge_depth_threshold: float = 0.03
        self.apply_confidence_mask: bool = False
        self.confidence_percentile: float = 10
        self.ignore_calibration_inputs: bool = ignore_calibration_inputs
        self.ignore_depth_inputs: bool = ignore_depth_inputs
        self.ignore_pose_inputs: bool = ignore_pose_inputs
        self.ignore_depth_scale_inputs: bool = ignore_depth_scale_inputs
        self.ignore_pose_scale_inputs: bool = ignore_pose_scale_inputs

        # Set the model input probabilities based on input args for ignoring inputs
        self._configure_geometric_input_config(
            use_calibration=not self.ignore_calibration_inputs,
            use_depth=not self.ignore_depth_inputs,
            use_pose=not self.ignore_pose_inputs,
            use_depth_scale=not self.ignore_depth_scale_inputs,
            use_pose_scale=not self.ignore_pose_scale_inputs,
        )

        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _configure_geometric_input_config(
        self,
        use_calibration: bool,
        use_depth: bool,
        use_pose: bool,
        use_depth_scale: bool,
        use_pose_scale: bool,
    ):
        """
        Configure the geometric input configuration
        """
        # Store original config for restoration
        if not hasattr(self, "_original_geometric_config"):
            self._original_geometric_config = dict(self.geometric_input_config)

        # Set the geometric input configuration
        if not (use_calibration or use_depth or use_pose):
            # No geometric inputs (images-only mode)
            self.geometric_input_config.update(
                {
                    "overall_prob": 0.0,
                    "dropout_prob": 1.0,
                    "ray_dirs_prob": 0.0,
                    "depth_prob": 0.0,
                    "cam_prob": 0.0,
                    "sparse_depth_prob": 0.0,
                    "depth_scale_norm_all_prob": 0.0,
                    "pose_scale_norm_all_prob": 0.0,
                }
            )
        else:
            # Enable geometric inputs with deterministic behavior
            self.geometric_input_config.update(
                {
                    "overall_prob": 1.0,
                    "dropout_prob": 0.0,
                    "ray_dirs_prob": 1.0 if use_calibration else 0.0,
                    "depth_prob": 1.0 if use_depth else 0.0,
                    "cam_prob": 1.0 if use_pose else 0.0,
                    "sparse_depth_prob": 0.0,
                    "depth_scale_norm_all_prob": 0.0 if use_depth_scale else 1.0,
                    "pose_scale_norm_all_prob": 0.0 if use_pose_scale else 1.0,
                }
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _initialize_info_sharing(self, info_sharing_config):
        """
        Initialize the information sharing module based on the configuration.

        This method sets up the custom positional encoding if specified and initializes
        the appropriate multi-view transformer based on the configuration type.

        Args:
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
                Should contain 'custom_positional_encoding', 'model_type', and 'model_return_type'.

        Returns:
            None

        Raises:
            ValueError: If invalid configuration options are provided.
        """
        # Initialize Multi-View Transformer
        # Returns intermediate features and normalized last layer features
        # Initialize mulit-view transformer based on type
        self.info_sharing = instantiate_from_config(info_sharing_config["module_args"])

    def _initialize_prediction_heads(self, pred_head_config):
        """
        Initialize the prediction heads based on the prediction head configuration.

        This method configures and initializes the appropriate prediction heads based on the
        specified prediction head type (linear, DPT, or DPT+pose). It sets up the necessary
        dependencies and creates the required model components.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads.

        Returns:
            None

        Raises:
            ValueError: If an invalid pred_head_type is provided.
        """
        # Initialze Dense Predction Head for all views
        dpt_feature_head = instantiate_from_config(pred_head_config["feature_head"])
        dpt_regressor_head = instantiate_from_config(pred_head_config["regressor_head"])

        self.dense_head = nn.Sequential(dpt_feature_head, dpt_regressor_head)

        # Initialize Pose Head for all views if required
        self.pose_head = instantiate_from_config(pred_head_config["pose_head"])
        self.scale_head = instantiate_from_config(pred_head_config["scale_head"])

    def _initialize_adaptors(self, pred_head_config):
        """
        Initialize the adaptors based on the prediction head configuration.

        This method sets up the appropriate adaptors for different scene representation types,
        such as pointmaps, ray maps with depth, or ray directions with depth and pose.

        Args:
            pred_head_config (Dict): Configuration for the prediction heads including adaptor type.

        Returns:
            None

        Raises:
            ValueError: If an invalid adaptor_type is provided.
            AssertionError: If ray directions + depth + pose is used with an incompatible head type.
        """
        self.scene_rep_type = "raydirs+depth+pose+confidence+mask"
        self.dense_adaptor = instantiate_from_config(pred_head_config["dpt_adaptor"])
        self.pose_adaptor = instantiate_from_config(pred_head_config["pose_adaptor"])
        self.scale_adaptor = instantiate_from_config(pred_head_config["scale_adaptor"])

    def _load_pretrained_weights(self):
        """
        Load pretrained weights from a checkpoint file.

        If load_specific_pretrained_submodules is True, only loads weights for the specified submodules.
        Otherwise, loads all weights from the checkpoint.

        Returns:
            None
        """
        if self.pretrained_checkpoint_path is not None:
            if self.pretrained_checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                ckpt = load_file(self.pretrained_checkpoint_path)
                strict = False
            else:
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                strict = True
            if not self.load_specific_pretrained_submodules:
                logging.info(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} ..."
                )
                logging.info(self.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=strict))
            else:
                logging.info(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} for specific submodules: {self.specific_pretrained_submodules} ..."
                )
                
                filtered_ckpt = {}
                for ckpt_key, ckpt_value in ckpt["model"].items():
                    for submodule in self.specific_pretrained_submodules:
                        if ckpt_key.startswith(submodule):
                            filtered_ckpt[ckpt_key] = ckpt_value
                logging.info(self.load_state_dict(filtered_ckpt, strict=False))

    def _encode_and_fuse_ray_dirs(
        self,
        ray_dirs,
        batch_size_per_view, num_views, height, width,
        all_encoder_features_across_views,
        per_sample_ray_dirs_input_mask,
    ):
        """
        Encode the ray directions for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_ray_dirs_input_mask (torch.Tensor): Tensor containing the per sample ray direction input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        if ray_dirs is None:
            ray_dirs = torch.zeros(
                (batch_size_per_view*num_views, 3, height, width),
                dtype=all_encoder_features_across_views.dtype,
                device=all_encoder_features_across_views.device,
            )

        # Encode the ray directions
        ray_dirs_features_across_views = self.ray_dirs_encoder(ray_dirs)
        ray_dirs_features_across_views = ray_dirs_features_across_views.permute(0, 2, 1).contiguous().view(batch_size_per_view * num_views, *all_encoder_features_across_views.shape[-3:])

        # Fuse the ray direction features with the other encoder features (zero out the features where the ray direction input mask is False)
        ray_dirs_features_across_views = (
            ray_dirs_features_across_views
            * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        all_encoder_features_across_views = (
            all_encoder_features_across_views + ray_dirs_features_across_views
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_depths(
        self,
        prompt_depth,
        num_views,
        batch_size_per_view,
        height, 
        width,
        all_encoder_features_across_views,
        per_sample_depth_input_mask,
    ):
        """
        Encode the z depths for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            per_sample_depth_input_mask (torch.Tensor): Tensor containing the per sample depth input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        # Get the device and height and width of the images
        device = all_encoder_features_across_views.device

        # Get the depths for all the views
        depths = torch.zeros(
            (batch_size_per_view*num_views, height, width, 1),
            dtype=all_encoder_features_across_views.dtype,
            device=device,
        )
        depth_norm_factors = torch.zeros(
            (batch_size_per_view*num_views),
            dtype=all_encoder_features_across_views.dtype,
            device=device,
        )
        metric_scale_depth_mask = torch.zeros(
            (batch_size_per_view*num_views),
            dtype=torch.bool,
            device=device,
        )

        # Stack the depths for all the views and permute to (B * V, C, H, W)
        # depths = torch.cat(depth_list, dim=0)  # (B * V, H, W, 1)
        depths = apply_log_to_norm(
            depths
        )  # Scale logarithimically (norm is computed along last dim)
        depths = depths.permute(0, 3, 1, 2).contiguous()  # (B * V, 1, H, W)
        # Encode the depths using the depth encoder
        depth_features_across_views = self.depth_encoder(depths)
        depth_features_across_views = depth_features_across_views.permute(0, 2, 1).contiguous().view(batch_size_per_view * num_views, *all_encoder_features_across_views.shape[-3:])
        # Zero out the depth features where the depth input mask is False
        depth_features_across_views = depth_features_across_views * per_sample_depth_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Stack the depth norm factors for all the views
        # depth_norm_factors = torch.cat(depth_norm_factors_list, dim=0)  # (B * V, )
        # Encode the depth norm factors using the log scale encoder for depth
        log_depth_norm_factors = torch.log(depth_norm_factors + 1e-8)  # (B * V, )
        depth_scale_features_across_views = self.depth_scale_encoder(log_depth_norm_factors.unsqueeze(-1))
        # Zero out the depth scale features where the depth input mask is False
        depth_scale_features_across_views = depth_scale_features_across_views * per_sample_depth_input_mask.unsqueeze(-1)
        
        # Stack the metric scale mask for all the views
        # metric_scale_depth_mask = torch.cat(
        #     metric_scale_depth_mask_list, dim=0
        # )  # (B * V, )
        # Zero out the depth scale features where the metric scale mask is False
        # Scale encoding is only provided for metric scale samples
        depth_scale_features_across_views = (
            depth_scale_features_across_views * metric_scale_depth_mask.unsqueeze(-1)
        )

        # Fuse the depth features & depth scale features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + depth_features_across_views
            + depth_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_cam_quats_and_trans(
        self,
        num_views,
        batch_size_per_view,
        all_encoder_features_across_views,
        pose_quats_across_views,
        pose_trans_across_views,
        per_sample_cam_input_mask,
    ):
        """
        Encode the camera quats and trans for all the views and fuse it with the other encoder features in a single forward pass.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            num_views (int): Number of views.
            batch_size_per_view (int): Batch size per view.
            all_encoder_features_across_views (torch.Tensor): Tensor containing the encoded features for all N views.
            pose_quats_across_views (torch.Tensor): Tensor containing the pose quats for all the views in the frame of the reference view 0. (batch_size_per_view * view, 4)
            pose_trans_across_views (torch.Tensor): Tensor containing the pose trans for all the views in the frame of the reference view 0. (batch_size_per_view * view, 3)
            per_sample_cam_input_mask (torch.Tensor): Tensor containing the per sample camera input mask.

        Returns:
            torch.Tensor: A tensor containing the encoded features for all the views.
        """
        if pose_quats_across_views is None or pose_trans_across_views is None:
            # Initialize the pose quats and trans for all views as identity
            pose_quats_across_views = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=all_encoder_features_across_views.dtype, device=all_encoder_features_across_views.device
            ).repeat(batch_size_per_view * num_views, 1)  # (q_x, q_y, q_z, q_w)
            pose_trans_across_views = torch.zeros(
                (batch_size_per_view * num_views, 3), dtype=all_encoder_features_across_views.dtype, device=all_encoder_features_across_views.device
            )

        # Encode the pose quats
        pose_quats_features_across_views = self.cam_rot_encoder(pose_quats_across_views)
        # Zero out the pose quat features where the camera input mask is False
        pose_quats_features_across_views = (
            pose_quats_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # Get the metric scale mask for all samples
        device = all_encoder_features_across_views.device
        metric_scale_pose_trans_mask = torch.zeros(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )
        for view_idx in range(num_views):
            metric_scale_mask = torch.zeros(
                batch_size_per_view, dtype=torch.bool, device=device
            )
            metric_scale_pose_trans_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ] = metric_scale_mask

        # Turn off indication of metric scale samples based on the pose_scale_norm_all_prob
        pose_norm_all_mask = (
            torch.rand(batch_size_per_view * num_views)
            < self.geometric_input_config["pose_scale_norm_all_prob"]
        )
        if pose_norm_all_mask.any():
            metric_scale_pose_trans_mask[pose_norm_all_mask] = False

        # Get the scale norm factor for all the samples and scale the pose translations
        pose_trans_across_views = torch.split(
            pose_trans_across_views, batch_size_per_view, dim=0
        )  # Split into num_views chunks
        pose_trans_across_views = torch.stack(
            pose_trans_across_views, dim=1
        )  # Stack the views along a new dimension (batch_size_per_view, num_views, 3)
        scaled_pose_trans_across_views, pose_trans_norm_factors = (
            normalize_pose_translations(
                pose_trans_across_views, return_norm_factor=True
            )
        )

        # Resize the pose translation back to (batch_size_per_view * num_views, 3) and extend the norm factor to (batch_size_per_view * num_views, 1)
        scaled_pose_trans_across_views = scaled_pose_trans_across_views.unbind(
            dim=1
        )  # Convert back to list of views, where each view has batch_size_per_view tensor
        scaled_pose_trans_across_views = torch.cat(
            scaled_pose_trans_across_views, dim=0
        )  # Concatenate back to (batch_size_per_view * num_views, 3)
        pose_trans_norm_factors_across_views = pose_trans_norm_factors.unsqueeze(
            -1
        ).repeat(num_views, 1)  # (B, ) -> (B * V, 1)

        # Encode the pose trans
        if self.cam_trans_encoder is not None:
            pose_trans_features_across_views = self.cam_trans_encoder(scaled_pose_trans_across_views)
            # Zero out the pose trans features where the camera input mask is False
            pose_trans_features_across_views = (
                pose_trans_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
            )
        else:
            pose_trans_features_across_views = 0

        if self.cam_trans_scale_encoder is not None:
            # Encode the pose translation norm factors using the log scale encoder for pose trans
            log_pose_trans_norm_factors_across_views = torch.log(
                pose_trans_norm_factors_across_views + 1e-8
            )
            pose_trans_scale_features_across_views = self.cam_trans_scale_encoder(log_pose_trans_norm_factors_across_views)
            # Zero out the pose trans scale features where the camera input mask is False
            pose_trans_scale_features_across_views = (
                pose_trans_scale_features_across_views
                * per_sample_cam_input_mask.unsqueeze(-1)
            )
            # Zero out the pose trans scale features where the metric scale mask is False
            # Scale encoding is only provided for metric scale samples
            pose_trans_scale_features_across_views = (
                pose_trans_scale_features_across_views
                * metric_scale_pose_trans_mask.unsqueeze(-1)
            )
        else:
            pose_trans_scale_features_across_views = 0

        # Fuse the pose quat features, pose trans features, pose trans scale features and pose trans type PE features with the other encoder features
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_optional_geometric_inputs(
        self, batch_size_per_view, num_views, height, width, all_encoder_features_across_views, prompt_depth=None, ray_directions=None, camera_quats=None, camera_trans=None,
    ):
        """
        Encode all the input optional geometric modalities and fuses it with the image encoder features in a single forward pass.
        Assumes all the input views have the same shape and batch size.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
            all_encoder_features_across_views (List[torch.Tensor]): List of tensors containing the encoded image features for all N views.

        Returns:
            List[torch.Tensor]: A list containing the encoded features for all N views.
        """
        device = all_encoder_features_across_views.device
        dtype = all_encoder_features_across_views.dtype

        # Get the overall input mask for all the views
        overall_geometric_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["overall_prob"]
        )
        overall_geometric_input_mask = overall_geometric_input_mask.repeat(num_views)

        # Get the per sample input mask after dropout
        # Per sample input mask is in view-major order so that index v*B + b in each mask corresponds to sample b of view v: (B * V)
        per_sample_geometric_input_mask = torch.rand(
            batch_size_per_view * num_views, device=device
        ) < (1 - self.geometric_input_config["dropout_prob"])
        per_sample_geometric_input_mask = (
            per_sample_geometric_input_mask & overall_geometric_input_mask
        )

        # Get the ray direction input mask
        per_sample_ray_dirs_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["ray_dirs_prob"]
        )
        per_sample_ray_dirs_input_mask = per_sample_ray_dirs_input_mask.repeat(
            num_views
        )
        per_sample_ray_dirs_input_mask = (
            per_sample_ray_dirs_input_mask & per_sample_geometric_input_mask
        )

        # Get the depth input mask
        per_sample_depth_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["depth_prob"]
        )
        per_sample_depth_input_mask = per_sample_depth_input_mask.repeat(num_views)
        per_sample_depth_input_mask = (
            per_sample_depth_input_mask & per_sample_geometric_input_mask
        )

        # Get the camera input mask
        per_sample_cam_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["cam_prob"]
        )
        per_sample_cam_input_mask = per_sample_cam_input_mask.repeat(num_views)
        per_sample_cam_input_mask = (
            per_sample_cam_input_mask & per_sample_geometric_input_mask
        )

        # Compute the pose quats and trans for all the non-reference views in the frame of the reference view 0
        # Returned pose quats and trans represent identity pose for views/samples where the camera input mask is False
        if camera_quats is not None and camera_trans is not None:
            pose_quats_across_views = camera_quats * per_sample_cam_input_mask.unsqueeze(-1)
            pose_trans_across_views = camera_trans * per_sample_cam_input_mask.unsqueeze(-1)
        else:
            pose_trans_across_views = pose_trans_across_views = None

        # Encode the ray directions and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            ray_directions,
            batch_size_per_view, num_views, height, width,
            all_encoder_features_across_views,
            per_sample_ray_dirs_input_mask,
        )

        # Encode the depths and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_depths(
            None,
            num_views,
            batch_size_per_view,
            height,
            width,
            all_encoder_features_across_views,
            per_sample_depth_input_mask,
        )

        # Encode the cam quat and trans and fuse with the image encoder features
        all_encoder_features_across_views = self._encode_and_fuse_cam_quats_and_trans(
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            pose_quats_across_views,
            pose_trans_across_views,
            per_sample_cam_input_mask,
        )

        # Normalize the fused features (permute -> normalize -> permute)
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 2, 3, 1
        ).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(
            all_encoder_features_across_views
        )
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 3, 1, 2
        ).contiguous()

        # Split the batched views into individual views
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views

    def _compute_adaptive_minibatch_size(
        self,
        memory_safety_factor: float = 0.95,
    ) -> int:
        """
        Compute adaptive minibatch size based on available PyTorch memory.

        Args:
            memory_safety_factor: Safety factor to avoid OOM (0.95 = use 95% of available memory)

        Returns:
            Computed minibatch size
        """
        device = self.device

        if device.type == "cuda":
            # Get available GPU memory
            torch.cuda.empty_cache()
            available_memory = torch.cuda.mem_get_info()[0]  # Free memory in bytes
            usable_memory = (
                available_memory * memory_safety_factor
            )  # Use safety factor to avoid OOM
        else:
            # For non-CUDA devices, use conservative default
            print(
                "Non-CUDA device detected. Using conservative default minibatch size of 1 for memory efficient dense prediction head inference."
            )
            return 1

        # Determine minibatch size based on available memory
        max_estimated_memory_per_sample = (
            680 * 1024 * 1024
        )  # 680 MB per sample (upper bound profiling using a 518 x 518 input)
        computed_minibatch_size = int(usable_memory / max_estimated_memory_per_sample)
        if computed_minibatch_size < 1:
            computed_minibatch_size = 1

        return computed_minibatch_size

    def downstream_dense_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        img_shape: Tuple[int, int],
    ):
        """
        Run the downstream dense prediction head
        """
        dense_head_outputs = self.dense_head(
            PredictionHeadLayeredInput(
                list_features=dense_head_inputs,
                target_output_shape=img_shape,
            )
        )
        dense_final_outputs = self.dense_adaptor(
            AdaptorInput(
                adaptor_feature=dense_head_outputs.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        return dense_final_outputs

    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        scale_head_inputs: torch.Tensor,
        img_shape: Tuple[int, int],
        memory_efficient_inference: bool = False,
    ):
        """
        Run Prediction Heads & Post-Process Outputs
        """
        # Get device
        device = self.device

        # Use mini-batch inference to run the dense prediction head (the memory bottleneck)
        # This saves memory and is slower than running the dense prediction head in one go
        if memory_efficient_inference:
            # Obtain the batch size of the dense head inputs
            batch_size = dense_head_inputs[0].shape[0]

            # Compute the mini batch size and number of mini batches adaptively based on available memory
            minibatch = self._compute_adaptive_minibatch_size()
            num_batches = (batch_size + minibatch - 1) // minibatch

            # Run prediction for each mini-batch
            dense_final_outputs_list = []
            pose_final_outputs_list = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * minibatch
                end_idx = min((batch_idx + 1) * minibatch, batch_size)

                # Get the inputs for the current mini-batch
                dense_head_inputs_batch = [
                    x[start_idx:end_idx] for x in dense_head_inputs
                ]

                # Dense prediction (mini-batched)
                dense_final_outputs_batch = self.downstream_dense_head(
                    dense_head_inputs_batch, img_shape
                )
                dense_final_outputs_list.append(dense_final_outputs_batch)

                # Pose prediction (mini-batched)
                pose_head_inputs_batch = dense_head_inputs[-1][start_idx:end_idx]
                pose_head_outputs_batch = self.pose_head(
                    PredictionHeadInput(last_feature=pose_head_inputs_batch)
                )
                pose_final_outputs_batch = self.pose_adaptor(
                    AdaptorInput(
                        adaptor_feature=pose_head_outputs_batch.decoded_channels,
                        output_shape_hw=img_shape,
                    )
                )
                pose_final_outputs_list.append(pose_final_outputs_batch)

            # Concatenate the dense prediction head outputs from all mini-batches
            available_keys = dense_final_outputs_batch.__dict__.keys()
            dense_pred_data_dict = {
                key: torch.cat(
                    [getattr(output, key) for output in dense_final_outputs_list], dim=0
                )
                for key in available_keys
            }
            dense_final_outputs = dense_final_outputs_batch.__class__(
                **dense_pred_data_dict
            )

            # Concatenate the pose prediction head outputs from all mini-batches
            available_keys = pose_final_outputs_batch.__dict__.keys()
            pose_pred_data_dict = {
                key: torch.cat(
                    [getattr(output, key) for output in pose_final_outputs_list],
                    dim=0,
                )
                for key in available_keys
            }
            pose_final_outputs = pose_final_outputs_batch.__class__(
                **pose_pred_data_dict
            )

            # Clear CUDA cache for better memory efficiency
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            # Run prediction for all (batch_size * num_views) in one go
            # Dense prediction
            dense_final_outputs = self.downstream_dense_head(
                dense_head_inputs, img_shape
            )

            # Pose prediction
            pose_head_outputs = self.pose_head(
                PredictionHeadInput(last_feature=dense_head_inputs[-1])
            )
            pose_final_outputs = self.pose_adaptor(
                AdaptorInput(
                    adaptor_feature=pose_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )

        # Scale prediction is lightweight, so we can run it in one go
        scale_head_output = self.scale_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)
        )
        scale_final_output = self.scale_adaptor(
            AdaptorInput(
                adaptor_feature=scale_head_output.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        scale_final_output = scale_final_output.value.squeeze(-1)  # (B, 1, 1) -> (B, 1)

        # Clear CUDA cache for better memory efficiency
        if memory_efficient_inference and device.type == "cuda":
            torch.cuda.empty_cache()

        return dense_final_outputs, pose_final_outputs, scale_final_output

    def forward(
        self,
        rgb,
        prompt_depth=None,
        ray_directions=None,
        camera_quats=None,
        camera_trans=None,
        meta_data=None,
        **kwargs,
    ):
        # Get input shape of the images, number of views, and batch size per view
        batch_size_per_view, num_views, _, height, width = rgb.shape
        img_shape = (int(height), int(width))

        # Run the image encoder on all the input views
        rgb = rgb.view(batch_size_per_view * num_views, -1, height, width)
        rgb = ((rgb + 1) * 0.5 - self._mean) / self._std

        encoder_input = ViTEncoderInput(image=rgb, data_norm_type="dinov2")
        all_encoder_features_across_views = self.encoder(encoder_input).features

        if prompt_depth is not None:
            prompt_depth = prompt_depth.view(
                batch_size_per_view * num_views, -1, height, width
            )

        if ray_directions is not None:
            ray_directions = ray_directions.view(
                batch_size_per_view * num_views, -1, height, width
            )
        
        if camera_quats is not None:
            camera_quats = camera_quats.view(batch_size_per_view * num_views, 4)
        
        if camera_trans is not None:
            camera_trans = camera_trans.view(batch_size_per_view * num_views, 3)

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    batch_size_per_view, num_views, height, width, all_encoder_features_across_views, prompt_depth=prompt_depth, ray_directions=ray_directions, camera_quats=camera_quats, camera_trans=camera_trans,
                )
            )

        # Expand the scale token to match the batch size
        input_scale_token = (
            self.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens=input_scale_token,
        )
        (
            final_info_sharing_multi_view_feat,
            intermediate_info_sharing_multi_view_feat,
        ) = self.info_sharing(info_sharing_input)

        # Get the list of features for all views
        dense_head_inputs_list = []
        # Stack all the image encoder features for all views
        stacked_encoder_features = torch.cat(
            all_encoder_features_across_views, dim=0
        )
        dense_head_inputs_list.append(stacked_encoder_features)
        # Stack the first intermediate features for all views
        stacked_intermediate_features_1 = torch.cat(
            intermediate_info_sharing_multi_view_feat[0].features, dim=0
        )
        dense_head_inputs_list.append(stacked_intermediate_features_1)
        # Stack the second intermediate features for all views
        stacked_intermediate_features_2 = torch.cat(
            intermediate_info_sharing_multi_view_feat[1].features, dim=0
        )
        dense_head_inputs_list.append(stacked_intermediate_features_2)
        # Stack the last layer features for all views
        stacked_final_features = torch.cat(
            final_info_sharing_multi_view_feat.features, dim=0
        )
        dense_head_inputs_list.append(stacked_final_features)

        with torch.autocast("cuda", enabled=False):
            # Prepare inputs for the downstream heads
            dense_head_inputs = dense_head_inputs_list
            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Run the downstream heads
            dense_final_outputs, pose_final_outputs, scale_final_output = (
                self.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=self.memory_efficient_inference,
                )
            )

            # Reshape output dense rep to (B * V, H, W, C)
            output_dense_rep = dense_final_outputs.value.permute(
                0, 2, 3, 1
            ).contiguous()
            # Get the predicted ray directions and depths along rays
            output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                [3, 1], dim=-1
            )
            # Get the predicted camera translations and quaternions
            output_cam_translations, output_cam_quats = (
                pose_final_outputs.value.split([3, 4], dim=-1)
            )
            # Get the predicted pointmaps in world frame and camera frame
            output_pts3d = (
                convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                    output_ray_directions,
                    output_depth_along_ray,
                    output_cam_translations,
                    output_cam_quats,
                )
            )
            output_pts3d_cam = output_ray_directions * output_depth_along_ray
            # Split the predicted quantities back to their respective views
            output_ray_directions_per_view = output_ray_directions.chunk(
                num_views, dim=0
            )
            output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                num_views, dim=0
            )
            output_cam_translations_per_view = output_cam_translations.chunk(
                num_views, dim=0
            )
            output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
            output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
            output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
            # Pack the output as a list of dictionaries
            res = []
            for i in range(num_views):
                res.append(
                    {
                        "pts3d": output_pts3d_per_view[i]
                        * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                        "pts3d_cam": output_pts3d_cam_per_view[i]
                        * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                        "ray_directions": output_ray_directions_per_view[i],
                        "depth_along_ray": output_depth_along_ray_per_view[i]
                        * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                        "cam_trans": output_cam_translations_per_view[i]
                        * scale_final_output,
                        "cam_quats": output_cam_quats_per_view[i],
                        "metric_scaling_factor": scale_final_output,
                    }
                )
            
            # Get the output confidences for all views (if available) and add them to the result
            output_confidences = dense_final_outputs.confidence
            # Reshape confidences to (B * V, H, W)
            output_confidences = (
                output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            # Split the predicted confidences back to their respective views
            output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
            # Add the confidences to the result
            for i in range(num_views):
                res[i]["conf"] = output_confidences_per_view[i]

            # Get the output masks (and logits) for all views (if available) and add them to the result
            # Get the output masks
            output_masks = dense_final_outputs.mask
            # Reshape masks to (B * V, H, W)
            output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            # Threshold the masks at 0.5 to get binary masks (0: ambiguous, 1: non-ambiguous)
            output_masks = output_masks > 0.5
            # Split the predicted masks back to their respective views
            output_masks_per_view = output_masks.chunk(num_views, dim=0)
            # Get the output mask logits (for loss)
            output_mask_logits = dense_final_outputs.logits
            # Reshape mask logits to (B * V, H, W)
            output_mask_logits = (
                output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            )
            # Split the predicted mask logits back to their respective views
            output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
            # Add the masks and logits to the result
            for i in range(num_views):
                res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
                res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

            total = {
                "depth": torch.stack([data["pts3d_cam"] for data in res], dim=1),
                "confidence": torch.stack([data["conf"] for data in res], dim=1),
                "global_points": torch.stack([data["pts3d"] for data in res], dim=1),
                "global_confidence": torch.stack([data["conf"] for data in res], dim=1),
                "ray": torch.stack([data["ray_directions"] for data in res], dim=1),
                "cam_quats": torch.stack([data["cam_quats"] for data in res], dim=1),
                "cam_trans": torch.stack([data["cam_trans"] for data in res], dim=1),
                "metric_scaling_factor": torch.stack([data["metric_scaling_factor"] for data in res], dim=1),
                "invalid_mask": torch.stack([data["non_ambiguous_mask"] for data in res], dim=1),
                "invalid_mask_logits": torch.stack([data["non_ambiguous_mask_logits"] for data in res], dim=1),
            }

        return total
  