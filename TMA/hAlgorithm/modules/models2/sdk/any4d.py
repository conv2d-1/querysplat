"""
Any4D model wrapper for integration with internal framework.
Supports scene flow prediction in addition to standard multi-view reconstruction.
"""

import logging
from functools import partial
from typing import Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import PyTorchModelHubMixin

from hAlgorithm.modules.models2.external.any4d.utils.geometry import (
    apply_log_to_norm,
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    normalize_pose_translations,
)
from hAlgorithm.modules.models2.external.uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
    PredictionHeadTokenInput,
    RegressionAdaptorOutput,
    UniCeptionAdaptorBase,
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


class SceneFlowAdaptor(UniCeptionAdaptorBase):
    """Adaptor for 3D Scene Flow prediction in Any4D."""

    def __init__(
        self,
        name: str,
        mode: str = "linear",
        vmin: float = -np.inf,
        vmax: float = np.inf,
        *args,
        **kwargs,
    ):
        """
        Adaptor for 3D scene flow prediction.

        Args:
            name (str): Name of the adaptor.
            mode (str): Mode of the scene flow, either "linear", "square" or "exp".
            vmin (float): Minimum value of the scene flow after scaling.
            vmax (float): Maximum value of the scene flow after scaling.
        """
        super().__init__(name, required_channels=3, *args, **kwargs)

        self.mode = mode
        self.vmin = vmin
        self.vmax = vmax
        self.no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    def forward(self, adaptor_input: AdaptorInput):
        """
        Forward pass for the SceneFlowAdaptor.

        Args:
            adaptor_input (AdaptorInput): Input to the adaptor. (B x 3 x H x W)

        Returns:
            RegressionAdaptorOutput: Output of the adaptor containing 3D scene flow.
        """
        x = adaptor_input.adaptor_feature

        if self.mode == "linear":
            output_scene_flow = x
        elif self.mode == "square":
            output_scene_flow = torch.sign(x) * x.abs() ** 2
        elif self.mode == "exp":
            output_scene_flow = torch.sign(x) * (torch.exp(x.abs()) - 1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self.no_bounds:
            output_scene_flow = output_scene_flow.clip(self.vmin, self.vmax)

        return RegressionAdaptorOutput(value=output_scene_flow)


class Any4D(nn.Module, PyTorchModelHubMixin):
    """
    Any4D model wrapper that supports multi-view reconstruction with scene flow prediction.
    
    This model outputs:
    - pts3d: 3D pointmap in world frame
    - pts3d_cam: 3D pointmap in camera frame
    - scene_flow: Scene flow vectors (for temporal frames)
    - ray_directions, depth_along_ray: Ray-based depth representation
    - cam_trans, cam_quats: Camera poses
    - metric_scaling_factor: Scale factor for metric reconstruction
    """

    def __init__(
        self,
        enc_embed_dim: int,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        scene_flow_pred_head_config: Dict,
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
        Initialize Any4D model.

        Args:
            enc_embed_dim: Embedding dimension for encoder
            encoder_config: Configuration for the image encoder
            info_sharing_config: Configuration for multi-view attention transformer
            pred_head_config: Configuration for prediction heads
            scene_flow_pred_head_config: Configuration for scene flow prediction head
            geometric_input_config: Configuration for optional geometric inputs
            fusion_norm_layer: Normalization layer after fusion
            pretrained_checkpoint_path: Path to pretrained checkpoint
            load_specific_pretrained_submodules: Whether to load specific submodules only
            specific_pretrained_submodules: List of submodules to load
            ignore_*: Flags to ignore specific geometric inputs
        """
        super().__init__()

        # Store config
        self.enc_embed_dim = enc_embed_dim
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.scene_flow_pred_head_config = scene_flow_pred_head_config
        self.geometric_input_config = geometric_input_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules

        # Initialize image encoder
        self.encoder = instantiate_from_config(self.encoder_config)

        # Initialize geometric encoders
        ray_dirs_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        self.ray_dirs_encoder = instantiate_from_config(ray_dirs_encoder_config)

        depth_encoder_config = self.geometric_input_config["depth_encoder_config"]
        self.depth_encoder = instantiate_from_config(depth_encoder_config)

        depth_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        self.depth_scale_encoder = instantiate_from_config(depth_scale_encoder_config)

        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        self.cam_rot_encoder = instantiate_from_config(cam_rot_encoder_config)

        cam_trans_encoder_config = self.geometric_input_config["cam_trans_encoder_config"]
        self.cam_trans_encoder = instantiate_from_config(cam_trans_encoder_config)

        cam_trans_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        self.cam_trans_scale_encoder = instantiate_from_config(cam_trans_scale_encoder_config)

        # Initialize fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.enc_embed_dim)

        # Initialize Scale Token
        self.scale_token = nn.Parameter(torch.zeros(self.enc_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Initialize info sharing module (multi-view transformer)
        self.info_sharing = instantiate_from_config(info_sharing_config["module_args"])

        # Initialize prediction heads
        self._initialize_prediction_heads(pred_head_config)

        # Initialize scene flow prediction head
        self._initialize_scene_flow_head(scene_flow_pred_head_config)

        # Initialize adaptors
        self._initialize_adaptors(pred_head_config)

        # Load pretrained weights
        self._load_pretrained_weights()

        # Runtime settings
        self.memory_efficient_inference: bool = False
        self.use_amp: bool = True
        self.ignore_calibration_inputs: bool = ignore_calibration_inputs
        self.ignore_depth_inputs: bool = ignore_depth_inputs
        self.ignore_pose_inputs: bool = ignore_pose_inputs
        self.ignore_depth_scale_inputs: bool = ignore_depth_scale_inputs
        self.ignore_pose_scale_inputs: bool = ignore_pose_scale_inputs

        # Configure geometric inputs
        self._configure_geometric_input_config(
            use_calibration=not self.ignore_calibration_inputs,
            use_depth=not self.ignore_depth_inputs,
            use_pose=not self.ignore_pose_inputs,
            use_depth_scale=not self.ignore_depth_scale_inputs,
            use_pose_scale=not self.ignore_pose_scale_inputs,
        )

        # Image normalization buffers
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
        """Configure geometric input probabilities."""
        if not hasattr(self, "_original_geometric_config"):
            self._original_geometric_config = dict(self.geometric_input_config)

        if not (use_calibration or use_depth or use_pose):
            self.geometric_input_config.update({
                "overall_prob": 0.0,
                "dropout_prob": 1.0,
                "ray_dirs_prob": 0.0,
                "depth_prob": 0.0,
                "cam_prob": 0.0,
                "sparse_depth_prob": 0.0,
                "depth_scale_norm_all_prob": 0.0,
                "pose_scale_norm_all_prob": 0.0,
            })
        else:
            self.geometric_input_config.update({
                "overall_prob": 1.0,
                "dropout_prob": 0.0,
                "ray_dirs_prob": 1.0 if use_calibration else 0.0,
                "depth_prob": 1.0 if use_depth else 0.0,
                "cam_prob": 1.0 if use_pose else 0.0,
                "sparse_depth_prob": 0.0,
                "depth_scale_norm_all_prob": 0.0 if use_depth_scale else 1.0,
                "pose_scale_norm_all_prob": 0.0 if use_pose_scale else 1.0,
            })

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _initialize_prediction_heads(self, pred_head_config):
        """Initialize the dense and pose prediction heads."""
        dpt_feature_head = instantiate_from_config(pred_head_config["feature_head"])
        dpt_regressor_head = instantiate_from_config(pred_head_config["regressor_head"])
        self.dense_head = nn.Sequential(dpt_feature_head, dpt_regressor_head)

        self.pose_head = instantiate_from_config(pred_head_config["pose_head"])
        self.scale_head = instantiate_from_config(pred_head_config["scale_head"])

    def _initialize_scene_flow_head(self, scene_flow_pred_head_config):
        """Initialize the scene flow prediction head."""
        scene_flow_feature_head = instantiate_from_config(scene_flow_pred_head_config["feature_head"])
        scene_flow_regressor_head = instantiate_from_config(scene_flow_pred_head_config["regressor_head"])
        self.scene_flow_dense_head = nn.Sequential(scene_flow_feature_head, scene_flow_regressor_head)
        self.scene_flow_dense_adaptor = instantiate_from_config(scene_flow_pred_head_config["dpt_adaptor"])

    def _initialize_adaptors(self, pred_head_config):
        """Initialize output adaptors."""
        self.scene_rep_type = "raydirs+depth+pose+scene_flow+confidence+mask"
        self.dense_adaptor = instantiate_from_config(pred_head_config["dpt_adaptor"])
        self.pose_adaptor = instantiate_from_config(pred_head_config["pose_adaptor"])
        self.scale_adaptor = instantiate_from_config(pred_head_config["scale_adaptor"])

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint."""
        if self.pretrained_checkpoint_path is not None:
            if self.pretrained_checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                ckpt = load_file(self.pretrained_checkpoint_path)
                strict = False
            else:
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
                strict = True

            if not self.load_specific_pretrained_submodules:
                logging.info(f"Loading pretrained Any4D weights from {self.pretrained_checkpoint_path} ...")
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                logging.info(self.load_state_dict(state_dict, strict=strict))
            else:
                logging.info(f"Loading pretrained Any4D weights for specific submodules: {self.specific_pretrained_submodules} ...")
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
        """Encode ray directions and fuse with encoder features."""
        if ray_dirs is None:
            ray_dirs = torch.zeros(
                (batch_size_per_view * num_views, 3, height, width),
                dtype=all_encoder_features_across_views.dtype,
                device=all_encoder_features_across_views.device,
            )

        ray_dirs_features = self.ray_dirs_encoder(ray_dirs)
        ray_dirs_features = ray_dirs_features.permute(0, 2, 1).contiguous().view(
            batch_size_per_view * num_views, *all_encoder_features_across_views.shape[-3:]
        )

        ray_dirs_features = (
            ray_dirs_features * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        all_encoder_features_across_views = all_encoder_features_across_views + ray_dirs_features

        return all_encoder_features_across_views

    def _encode_and_fuse_depths(
        self,
        prompt_depth,
        num_views, batch_size_per_view, height, width,
        all_encoder_features_across_views,
        per_sample_depth_input_mask,
    ):
        """Encode depths and fuse with encoder features."""
        device = all_encoder_features_across_views.device

        depths = torch.zeros(
            (batch_size_per_view * num_views, height, width, 1),
            dtype=all_encoder_features_across_views.dtype,
            device=device,
        )
        depth_norm_factors = torch.zeros(
            (batch_size_per_view * num_views),
            dtype=all_encoder_features_across_views.dtype,
            device=device,
        )
        metric_scale_depth_mask = torch.zeros(
            (batch_size_per_view * num_views),
            dtype=torch.bool,
            device=device,
        )

        depths = apply_log_to_norm(depths)
        depths = depths.permute(0, 3, 1, 2).contiguous()
        depth_features = self.depth_encoder(depths)
        depth_features = depth_features.permute(0, 2, 1).contiguous().view(
            batch_size_per_view * num_views, *all_encoder_features_across_views.shape[-3:]
        )
        depth_features = depth_features * per_sample_depth_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        log_depth_norm_factors = torch.log(depth_norm_factors + 1e-8)
        depth_scale_features = self.depth_scale_encoder(log_depth_norm_factors.unsqueeze(-1))
        depth_scale_features = depth_scale_features * per_sample_depth_input_mask.unsqueeze(-1)
        depth_scale_features = depth_scale_features * metric_scale_depth_mask.unsqueeze(-1)

        all_encoder_features_across_views = (
            all_encoder_features_across_views + depth_features + depth_scale_features.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_cam_quats_and_trans(
        self,
        num_views, batch_size_per_view,
        all_encoder_features_across_views,
        pose_quats_across_views,
        pose_trans_across_views,
        per_sample_cam_input_mask,
    ):
        """Encode camera poses and fuse with encoder features."""
        if pose_quats_across_views is None or pose_trans_across_views is None:
            pose_quats_across_views = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=all_encoder_features_across_views.dtype, device=all_encoder_features_across_views.device
            ).repeat(batch_size_per_view * num_views, 1)
            pose_trans_across_views = torch.zeros(
                (batch_size_per_view * num_views, 3), dtype=all_encoder_features_across_views.dtype, device=all_encoder_features_across_views.device
            )

        pose_quats_features = self.cam_rot_encoder(pose_quats_across_views)
        pose_quats_features = pose_quats_features * per_sample_cam_input_mask.unsqueeze(-1)

        device = all_encoder_features_across_views.device
        metric_scale_pose_trans_mask = torch.zeros(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )

        pose_norm_all_mask = (
            torch.rand(batch_size_per_view * num_views)
            < self.geometric_input_config["pose_scale_norm_all_prob"]
        )
        if pose_norm_all_mask.any():
            metric_scale_pose_trans_mask[pose_norm_all_mask] = False

        pose_trans_split = torch.split(pose_trans_across_views, batch_size_per_view, dim=0)
        pose_trans_stacked = torch.stack(pose_trans_split, dim=1)
        scaled_pose_trans, pose_trans_norm_factors = normalize_pose_translations(
            pose_trans_stacked, return_norm_factor=True
        )

        scaled_pose_trans = torch.cat(scaled_pose_trans.unbind(dim=1), dim=0)
        pose_trans_norm_factors = pose_trans_norm_factors.unsqueeze(-1).repeat(num_views, 1)

        if self.cam_trans_encoder is not None:
            pose_trans_features = self.cam_trans_encoder(scaled_pose_trans)
            pose_trans_features = pose_trans_features * per_sample_cam_input_mask.unsqueeze(-1)
        else:
            pose_trans_features = 0

        if self.cam_trans_scale_encoder is not None:
            log_pose_trans_norm = torch.log(pose_trans_norm_factors + 1e-8)
            pose_trans_scale_features = self.cam_trans_scale_encoder(log_pose_trans_norm)
            pose_trans_scale_features = pose_trans_scale_features * per_sample_cam_input_mask.unsqueeze(-1)
            pose_trans_scale_features = pose_trans_scale_features * metric_scale_pose_trans_mask.unsqueeze(-1)
        else:
            pose_trans_scale_features = 0

        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + pose_quats_features.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_features.unsqueeze(-1).unsqueeze(-1)
            + pose_trans_scale_features.unsqueeze(-1).unsqueeze(-1)
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_optional_geometric_inputs(
        self, batch_size_per_view, num_views, height, width,
        all_encoder_features_across_views,
        prompt_depth=None, ray_directions=None, camera_quats=None, camera_trans=None,
    ):
        """Encode and fuse all optional geometric inputs."""
        device = all_encoder_features_across_views.device

        overall_mask = (
            torch.rand(batch_size_per_view, device=device) < self.geometric_input_config["overall_prob"]
        ).repeat(num_views)

        per_sample_mask = torch.rand(batch_size_per_view * num_views, device=device) < (
            1 - self.geometric_input_config["dropout_prob"]
        )
        per_sample_mask = per_sample_mask & overall_mask

        ray_dirs_mask = (
            torch.rand(batch_size_per_view, device=device) < self.geometric_input_config["ray_dirs_prob"]
        ).repeat(num_views) & per_sample_mask

        depth_mask = (
            torch.rand(batch_size_per_view, device=device) < self.geometric_input_config["depth_prob"]
        ).repeat(num_views) & per_sample_mask

        cam_mask = (
            torch.rand(batch_size_per_view, device=device) < self.geometric_input_config["cam_prob"]
        ).repeat(num_views) & per_sample_mask

        if camera_quats is not None and camera_trans is not None:
            pose_quats = camera_quats * cam_mask.unsqueeze(-1)
            pose_trans = camera_trans * cam_mask.unsqueeze(-1)
        else:
            pose_quats = pose_trans = None

        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            ray_directions, batch_size_per_view, num_views, height, width,
            all_encoder_features_across_views, ray_dirs_mask,
        )

        all_encoder_features_across_views = self._encode_and_fuse_depths(
            None, num_views, batch_size_per_view, height, width,
            all_encoder_features_across_views, depth_mask,
        )

        all_encoder_features_across_views = self._encode_and_fuse_cam_quats_and_trans(
            num_views, batch_size_per_view, all_encoder_features_across_views,
            pose_quats, pose_trans, cam_mask,
        )

        all_encoder_features_across_views = all_encoder_features_across_views.permute(0, 2, 3, 1).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(all_encoder_features_across_views)
        all_encoder_features_across_views = all_encoder_features_across_views.permute(0, 3, 1, 2).contiguous()

        return all_encoder_features_across_views.chunk(num_views, dim=0)

    def downstream_dense_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        img_shape: Tuple[int, int],
    ):
        """Run downstream dense prediction head."""
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

    def downstream_scene_flow_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        img_shape: Tuple[int, int],
    ):
        """Run downstream scene flow prediction head."""
        scene_flow_head_outputs = self.scene_flow_dense_head(
            PredictionHeadLayeredInput(
                list_features=dense_head_inputs,
                target_output_shape=img_shape,
            )
        )
        scene_flow_final_outputs = self.scene_flow_dense_adaptor(
            AdaptorInput(
                adaptor_feature=scene_flow_head_outputs.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        return scene_flow_final_outputs

    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],
        scale_head_inputs: torch.Tensor,
        img_shape: Tuple[int, int],
        memory_efficient_inference: bool = False,
    ):
        """Run all prediction heads."""
        dense_final_outputs = self.downstream_dense_head(dense_head_inputs, img_shape)
        scene_flow_final_outputs = self.downstream_scene_flow_head(dense_head_inputs, img_shape)

        pose_head_outputs = self.pose_head(
            PredictionHeadInput(last_feature=dense_head_inputs[-1])
        )
        pose_final_outputs = self.pose_adaptor(
            AdaptorInput(
                adaptor_feature=pose_head_outputs.decoded_channels,
                output_shape_hw=img_shape,
            )
        )

        scale_head_output = self.scale_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)
        )
        scale_final_output = self.scale_adaptor(
            AdaptorInput(
                adaptor_feature=scale_head_output.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        scale_final_output = scale_final_output.value.squeeze(-1)

        return dense_final_outputs, scene_flow_final_outputs, pose_final_outputs, scale_final_output

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
        """
        Forward pass of Any4D model.

        Args:
            rgb: Input images, shape (B, V, C, H, W)
            prompt_depth: Optional prompt depth
            ray_directions: Optional ray directions
            camera_quats: Optional camera quaternions
            camera_trans: Optional camera translations
            meta_data: Optional metadata

        Returns:
            Dictionary containing:
            - depth: Pointmap in camera frame (B, V, H, W, 3)
            - confidence: Confidence scores (B, V, H, W)
            - global_points: Pointmap in world frame (B, V, H, W, 3)
            - scene_flow: Scene flow vectors (B, V, H, W, 3)
            - ray: Ray directions (B, V, 3, H, W)
            - cam_quats: Camera quaternions (B, V, 4)
            - cam_trans: Camera translations (B, V, 3)
            - metric_scaling_factor: Scale factor (B, V, 1)
        """
        batch_size_per_view, num_views, _, height, width = rgb.shape
        img_shape = (int(height), int(width))

        # Encode images
        rgb_flat = rgb.view(batch_size_per_view * num_views, -1, height, width)
        rgb_normalized = ((rgb_flat + 1) * 0.5 - self._mean) / self._std

        encoder_input = ViTEncoderInput(image=rgb_normalized, data_norm_type="dinov2")
        all_encoder_features = self.encoder(encoder_input).features

        # Reshape optional inputs
        if prompt_depth is not None:
            prompt_depth = prompt_depth.view(batch_size_per_view * num_views, -1, height, width)
        if ray_directions is not None:
            ray_directions = ray_directions.view(batch_size_per_view * num_views, -1, height, width)
        if camera_quats is not None:
            camera_quats = camera_quats.view(batch_size_per_view * num_views, 4)
        if camera_trans is not None:
            camera_trans = camera_trans.view(batch_size_per_view * num_views, 3)

        # Encode and fuse geometric inputs
        with torch.autocast("cuda", enabled=False):
            all_encoder_features = self._encode_and_fuse_optional_geometric_inputs(
                batch_size_per_view, num_views, height, width,
                all_encoder_features,
                prompt_depth=prompt_depth,
                ray_directions=ray_directions,
                camera_quats=camera_quats,
                camera_trans=camera_trans,
            )

        # Multi-view information sharing
        input_scale_token = (
            self.scale_token.unsqueeze(0).unsqueeze(-1).repeat(batch_size_per_view, 1, 1)
        )

        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features,
            additional_input_tokens=input_scale_token,
        )
        final_features, intermediate_features = self.info_sharing(info_sharing_input)

        # Prepare dense head inputs
        dense_head_inputs_list = []
        dense_head_inputs_list.append(torch.cat(all_encoder_features, dim=0))
        dense_head_inputs_list.append(torch.cat(intermediate_features[0].features, dim=0))
        dense_head_inputs_list.append(torch.cat(intermediate_features[1].features, dim=0))
        dense_head_inputs_list.append(torch.cat(final_features.features, dim=0))

        # Run downstream heads
        with torch.autocast("cuda", enabled=False):
            dense_final_outputs, scene_flow_final_outputs, pose_final_outputs, scale_final_output = (
                self.downstream_head(
                    dense_head_inputs=dense_head_inputs_list,
                    scale_head_inputs=final_features.additional_token_features,
                    img_shape=img_shape,
                    memory_efficient_inference=self.memory_efficient_inference,
                )
            )

            # Process dense outputs
            output_dense_rep = dense_final_outputs.value.permute(0, 2, 3, 1).contiguous()
            output_ray_directions, output_depth_along_ray = output_dense_rep.split([3, 1], dim=-1)

            # Process scene flow outputs
            output_scene_flow = scene_flow_final_outputs.value.permute(0, 2, 3, 1).contiguous()

            # Process pose outputs
            output_cam_trans, output_cam_quats = pose_final_outputs.value.split([3, 4], dim=-1)

            # Compute pointmaps
            output_pts3d = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                output_ray_directions,
                output_depth_along_ray,
                output_cam_trans,
                output_cam_quats,
            )
            output_pts3d_cam = output_ray_directions * output_depth_along_ray

            # Split by views
            output_ray_directions_per_view = output_ray_directions.chunk(num_views, dim=0)
            output_depth_along_ray_per_view = output_depth_along_ray.chunk(num_views, dim=0)
            output_cam_trans_per_view = output_cam_trans.chunk(num_views, dim=0)
            output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
            output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
            output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
            output_scene_flow_per_view = output_scene_flow.chunk(num_views, dim=0)

            # Pack results
            res = []
            for i in range(num_views):
                res.append({
                    "pts3d": output_pts3d_per_view[i] * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    "pts3d_cam": output_pts3d_cam_per_view[i] * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    "ray_directions": output_ray_directions_per_view[i],
                    "depth_along_ray": output_depth_along_ray_per_view[i] * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    "cam_trans": output_cam_trans_per_view[i] * scale_final_output,
                    "cam_quats": output_cam_quats_per_view[i],
                    "scene_flow": output_scene_flow_per_view[i] * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                    "metric_scaling_factor": scale_final_output,
                })

            # Add confidence if available
            if hasattr(dense_final_outputs, 'confidence') and dense_final_outputs.confidence is not None:
                output_confidences = dense_final_outputs.confidence.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
                for i in range(num_views):
                    res[i]["conf"] = output_confidences_per_view[i]

            # Add mask if available
            if hasattr(dense_final_outputs, 'mask') and dense_final_outputs.mask is not None:
                output_masks = dense_final_outputs.mask.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                output_masks = output_masks > 0.5
                output_masks_per_view = output_masks.chunk(num_views, dim=0)
                for i in range(num_views):
                    res[i]["non_ambiguous_mask"] = output_masks_per_view[i]

            # Construct final output
            total = {
                "depth": torch.stack([data["pts3d_cam"] for data in res], dim=1),
                "confidence": torch.stack([data.get("conf", torch.ones_like(data["pts3d_cam"][..., 0])) for data in res], dim=1),
                "global_points": torch.stack([data["pts3d"] for data in res], dim=1),
                "global_confidence": torch.stack([data.get("conf", torch.ones_like(data["pts3d"][..., 0])) for data in res], dim=1),
                "scene_flow": torch.stack([data["scene_flow"] for data in res], dim=1),
                "ray": torch.stack([data["ray_directions"] for data in res], dim=1),
                "cam_quats": torch.stack([data["cam_quats"] for data in res], dim=1),
                "cam_trans": torch.stack([data["cam_trans"] for data in res], dim=1),
                "metric_scaling_factor": torch.stack([data["metric_scaling_factor"] for data in res], dim=1),
            }

            if "non_ambiguous_mask" in res[0]:
                total["invalid_mask"] = torch.stack([data["non_ambiguous_mask"] for data in res], dim=1)

        return total