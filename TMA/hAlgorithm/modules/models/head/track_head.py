# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.vggt.heads.track_modules.base_track_predictor import (
    BaseTrackerPredictor,
)
from hAlgorithm.utils import cuda_timing_context, instantiate_from_config


class TrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
        max_scale=518,
        use_spaceatt=True,
        tracker_depth=6,
        feature_extractor=None,
        with_local_feature=False,
        with_glb_feature=False,
        with_fuse_feature=False,
        pretrain=None,
        feature_extractor_pretrain=None,
        tracker_pretrain=None,
        output_index=None,
        with_sdpa=False,
        timing=False,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = instantiate_from_config(feature_extractor)

        # Tracker module that predicts point trajectories
        # Takes feature maps and predicts coordinates and visibility
        self.stride = stride
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
            with_sdpa=with_sdpa,
        )

        self.iters = iters
        self.output_index = output_index
        self.timing = timing

        self.with_local_feature = with_local_feature
        self.with_glb_feature = with_glb_feature
        self.with_fuse_feature = with_fuse_feature

        self.pretrain = pretrain
        self.feature_extractor_pretrain = feature_extractor_pretrain
        self.tracker_pretrain = tracker_pretrain

        if self.pretrain is not None:
            self.load_state_dict(
                torch.load(self.pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"TrackHead, load pretrain {self.pretrain}")

        elif self.feature_extractor_pretrain is not None:
            res = self.feature_extractor.load_state_dict(
                torch.load(self.feature_extractor_pretrain, map_location="cpu", weights_only=False),
                strict=False,
            )
            logging.info(
                f"TrackHead, load feature_extractor_pretrain {self.feature_extractor_pretrain}"
            )
            logging.info(f"unexpected_keys: {res.unexpected_keys}")
            logging.info(f"missing_keys: {res.missing_keys}")

        elif self.tracker_pretrain is not None:
            self.tracker.load_state_dict(
                torch.load(self.tracker_pretrain, map_location="cpu", weights_only=False),
                strict=True,
            )
            logging.info(f"TrackHead, load tracker_pretrain {self.tracker_pretrain}")

    def forward(
        self,
        aggregated_tokens_list,
        query_points=None,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        local_feature_maps=None,
        glb_feature_maps=None,
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
        if self.feature_extractor is not None:
            # Extract features from tokens
            # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
            feature_maps = self.feature_extractor(
                features=aggregated_tokens_list,
                prompt_features=None,  # NOTE Debug VGG Head
                patch_h=patch_h,
                patch_w=patch_w,
                meta_data=meta_data,
            )
        else:
            if self.with_local_feature:
                feature_maps = local_feature_maps
            elif self.with_glb_feature:
                feature_maps = glb_feature_maps
            elif self.with_fuse_feature:
                feature_maps = torch.cat([local_feature_maps, glb_feature_maps], dim=-3)

            feature_maps = F.interpolate(
                feature_maps,
                (
                    feature_maps.shape[-2] // self.tracker.stride,
                    feature_maps.shape[-1] // self.tracker.stride,
                ),
                mode="bilinear",
                align_corners=True,
            )

        with cuda_timing_context("track_infer", self.timing):
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
            return coord_preds[self.output_index], vis_scores, conf_scores


class TrackHeadV2(TrackHead):
    """return dict outputs."""

    def __init__(self, return_features=False, **kwargs):
        super(TrackHeadV2, self).__init__(**kwargs)

        self.return_features = return_features

    def forward(
        self,
        aggregated_tokens_list,
        query_points=None,
        prompt_features=None,
        patch_h=None,
        patch_w=None,
        meta_data=None,
        local_feature_maps=None,
        glb_feature_maps=None,
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
        if self.feature_extractor is not None:
            # Extract features from tokens
            # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
            feature_maps = self.feature_extractor(
                features=aggregated_tokens_list,
                prompt_features=None,  # NOTE Debug VGG Head
                patch_h=patch_h,
                patch_w=patch_w,
                meta_data=meta_data,
            )
        else:
            if self.with_local_feature:
                feature_maps = local_feature_maps
            elif self.with_glb_feature:
                feature_maps = glb_feature_maps
            elif self.with_fuse_feature:
                feature_maps = torch.cat([local_feature_maps, glb_feature_maps], dim=-3)

            feature_maps = F.interpolate(
                feature_maps,
                (
                    feature_maps.shape[-2] // self.tracker.stride,
                    feature_maps.shape[-1] // self.tracker.stride,
                ),
                mode="bilinear",
                align_corners=True,
            )

        with cuda_timing_context("track_infer", self.timing):
            # Perform tracking using the extracted features
            B = aggregated_tokens_list[0].shape[0]
            feature_maps = feature_maps.reshape(B, -1, *feature_maps.shape[1:])
            coord_preds, vis_scores, conf_scores = self.tracker(
                query_points=query_points,
                fmaps=feature_maps,
                iters=self.iters,
            )

        output = dict()

        if self.output_index is None:
            output["track"] = coord_preds
        else:
            output["track"] = coord_preds[self.output_index]

        output["track_vis"] = vis_scores
        output["track_confidence"] = conf_scores

        if self.return_features:
            output["features"] = feature_maps

        return output
