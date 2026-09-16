# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

from hAlgorithm.modules.models.vggt.heads.track_modules.base_track_predictor import (
    BaseTrackerPredictor, BaseTrackerPredictorV2
)

from .vggt_dpt_head import VGGTDPTHead


class TrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
        intermediate_layer_idx=[0, 1, 2, 3],
        max_scale=518,
        use_spaceatt=True,
        tracker_depth=6,
        output_index=None,
        pretrain=None,
        disable_amp=True,
        tracker_type='base',
        **kwargs,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size
        self.down_ratio = stride
        self.features = features
        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = VGGTDPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=self.down_ratio,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
            intermediate_layer_idx=intermediate_layer_idx,
        )

        # Tracker module that predicts point trajectories
        # Takes feature maps and predicts coordinates and visibility
        if tracker_type in ['base']:
            self.tracker = BaseTrackerPredictor(
                latent_dim=features,  # Match the output_dim of feature extractor
                predict_conf=predict_conf,
                stride=stride,
                corr_levels=corr_levels,
                corr_radius=corr_radius,
                hidden_size=hidden_size,
                use_spaceatt=use_spaceatt,
                depth=tracker_depth,
                max_scale=max_scale,
                **kwargs
            )
        elif tracker_type in ['v2']:
            self.tracker = BaseTrackerPredictorV2(
                latent_dim=features,  # Match the output_dim of feature extractor
                predict_conf=predict_conf,
                stride=stride,
                corr_levels=corr_levels,
                corr_radius=corr_radius,
                hidden_size=hidden_size,
                use_spaceatt=use_spaceatt,
                depth=tracker_depth,
                max_scale=max_scale,
                **kwargs
            )
        else:
            raise NotImplementedError(f"tracker_type must be [base, v2], but got {tracker_type}")

        self.iters = iters
        self.output_index = output_index
        self.disable_amp = disable_amp

        self.pretrain = pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"TrackHead, load pretrain {self.pretrain}")

    # def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
    def forward(
        self,
        aggregated_tokens_list,
        query_points=None,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        **kwargs,
    ):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        # B, S, _, H, W = images.shape

        if self.disable_amp:
            with torch.cuda.amp.autocast(enabled=False):
                # Extract features from tokens
                # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
                feature_maps = self.feature_extractor(
                    features=aggregated_tokens_list,
                    prompt_features=None,  # NOTE Debug VGG Head
                    patch_h=patch_h,
                    patch_w=patch_w,
                    meta_data=meta_data,
                )

                # Perform tracking using the extracted features
                B = aggregated_tokens_list[0].shape[0]
                feature_maps = feature_maps.reshape(B, -1, *feature_maps.shape[1:])
                coord_preds, vis_scores, conf_scores = self.tracker(
                    query_points=query_points,
                    fmaps=feature_maps,
                    iters=self.iters,
                )
        else:
            # Extract features from tokens
            # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
            feature_maps = self.feature_extractor(
                features=aggregated_tokens_list,
                prompt_features=None,  # NOTE Debug VGG Head
                patch_h=patch_h,
                patch_w=patch_w,
                meta_data=meta_data,
            )

            # Perform tracking using the extracted features
            B = aggregated_tokens_list[0].shape[0]
            feature_maps = feature_maps.reshape(B, -1, *feature_maps.shape[1:])
            coord_preds, vis_scores, conf_scores = self.tracker(
                query_points=query_points,
                fmaps=feature_maps,
                iters=self.iters,
            )

        if self.output_index is None:
            return coord_preds, vis_scores, conf_scores
        else:
            return (
                coord_preds[self.output_index],
                vis_scores[self.output_index],
                conf_scores[self.output_index],
            )

class TrackHeadV2(TrackHead):
    def __init__(self, depth_feature_dim=0, depth_fusion_type = "add", **kwargs):
        super().__init__(**kwargs)
        
        self.depth_feature_dim = depth_feature_dim
        self.depth_fusion_type = depth_fusion_type
        assert self.depth_fusion_type in ['add', 'concat']
        if self.depth_feature_dim > 0:
            if self.depth_fusion_type in ['add']:
                self.depth_convs = nn.Conv2d(
                    self.depth_feature_dim, self.features,
                    kernel_size=self.down_ratio, stride=self.down_ratio, padding=0,
                )
                # 将权重初始化为 0
                torch.nn.init.constant_(self.depth_convs.weight, 0)
                # 将偏置初始化为 0（如果存在）
                if self.depth_convs.bias is not None:
                    torch.nn.init.constant_(self.depth_convs.bias, 0)
                logging.info(f"TrackHead, Zero Init depth_convs.")
            else:
                self.depth_convs = nn.Conv2d(
                    self.depth_feature_dim, self.features,
                    kernel_size=self.down_ratio, stride=self.down_ratio, padding=0,
                )
                self.feat_convs = nn.Conv2d(
                    self.features*2, self.features,
                    kernel_size=1, stride=1, padding=0,
                )

    def forward_track(
        self,
        aggregated_tokens_list,
        query_points=None,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        **kwargs,
    ):
        # Extract features from tokens
        # feature_maps has shape (B*S, C, H//2, W//2) due to down_ratio=2
        feature_maps = self.feature_extractor(
            features=aggregated_tokens_list,
            prompt_features=None,  # NOTE Debug VGG Head
            patch_h=patch_h,
            patch_w=patch_w,
            meta_data=meta_data,
        )
        depth_features = kwargs.get("local_feature_maps", None)
        if depth_features is not None and self.depth_feature_dim > 0:
            if self.depth_fusion_type in ['add']:
                # depth_features has shape (B*S, C, H//2, W//2) due to down_ratio=2
                depth_features = self.depth_convs(depth_features)
                feature_maps = torch.add(feature_maps, depth_features)
            else:
                # depth_features has shape (B*S, C, H//2, W//2) due to down_ratio=2
                depth_features = self.depth_convs(depth_features)
                feature_maps = torch.concat([feature_maps, depth_features], dim=1)
                feature_maps = self.feat_convs(feature_maps)

        # Perform tracking using the extracted features
        B = aggregated_tokens_list[0].shape[0]
        feature_maps = feature_maps.reshape(B, -1, *feature_maps.shape[1:])

        if query_points.shape[-1] == 3:
            query_frames = query_points[..., -1]
            query_points = query_points[..., 0:2]
        else:
            query_frames = None

        coord_preds, vis_scores, conf_scores = self.tracker(
            query_points=query_points,
            query_frames=query_frames,
            fmaps=feature_maps,
            pose_enc=kwargs.get("pose_enc", None),
            local_depth=kwargs.get("local_depth", None),
            intrinsics=kwargs.get("intrinsics", None),
            iters=self.iters,
        )
        
        if self.output_index is None:
            return coord_preds, vis_scores, conf_scores
        else:
            return (
                coord_preds[self.output_index],
                vis_scores[self.output_index],
                conf_scores[self.output_index],
            )

    def forward(
        self,
        aggregated_tokens_list,
        query_points=None,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        **kwargs,
    ):
        if self.disable_amp:
            with torch.cuda.amp.autocast(enabled=False):
                return self.forward_track(
                    aggregated_tokens_list=aggregated_tokens_list,
                    query_points=query_points,
                    prompt_features=prompt_features,
                    patch_h=patch_h, patch_w=patch_w,
                    meta_data=meta_data, **kwargs
                )
        else:
            return self.forward_track(
                aggregated_tokens_list=aggregated_tokens_list,
                query_points=query_points,
                prompt_features=prompt_features,
                patch_h=patch_h, patch_w=patch_w,
                meta_data=meta_data, **kwargs
            )
    