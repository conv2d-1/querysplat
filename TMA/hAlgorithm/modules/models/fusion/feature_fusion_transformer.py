import torch
import torch.nn as nn

from .flow_base import FlowBase
from .mf_transformer import MultiViewFeatureTransformer


class FeatureFusionBlockTransformer(nn.Module, FlowBase):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        mf_frame_num,
        num_layers=6,
        nhead=1,
        no_cross_attn=False,
        attn_num_splits=1,
        low_mem=True,
        **kwargs,
    ):
        nn.Module.__init__(self)
        FlowBase.__init__(self, mf_frame_num)

        self.attn_num_splits = attn_num_splits
        self.mf_frame_num = mf_frame_num
        self.low_mem = low_mem

        self.temporal_conv0 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.temporal_conv1 = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.mf_transformer = MultiViewFeatureTransformer(
            num_layers=num_layers,
            d_model=hidden_channels,
            nhead=nhead,
            no_cross_attn=no_cross_attn,
            **kwargs,
        )

    def forward(self, feature_list, flow_mode=False):
        # feature_list [B,C,H,W] * T
        if flow_mode:
            feature_list = self.prepare_flow(feature_list)

        feature_list = [self.temporal_conv0(feat) for feat in feature_list]  # [B,C_hidden,H,W] * T

        b, n = feature_list[0].shape[0], len(feature_list)
        if self.low_mem:
            attn_feat_list = []  # [[1,C_hidden,H,W]*T]*B
            for i in range(b):
                temporal_feat = [
                    feat[i].unsqueeze(0) for feat in feature_list
                ]  # [1,C_hidden,H,W]*T
                attn_feat_list.append(
                    self.mf_transformer(temporal_feat, attn_num_splits=self.attn_num_splits)
                )
            feature_list = [
                torch.concat([attn_feat[i] for attn_feat in attn_feat_list], dim=0)
                for i in range(n)
            ]
        else:
            feature_list = self.mf_transformer(feature_list, attn_num_splits=self.attn_num_splits)

        feature_list = [self.temporal_conv1(feat) for feat in feature_list]  # [B,C,H,W] * T

        return feature_list


class FeatureFusionNetTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        mf_frame_num,
        num_layers=6,
        nhead=1,
        no_cross_attn=False,
        attn_num_splits=1,
        low_mem=True,
        share_block=False,
        **kwargs,
    ):
        super().__init__()

        self.attn_num_splits = attn_num_splits
        self.mf_frame_num = mf_frame_num
        self.low_mem = low_mem
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.share_block = share_block

        if self.share_block:
            assert len(in_channels) == 1

        if isinstance(self.hidden_channels, (list, tuple)):
            self.fusion_layers = nn.ModuleList(
                [
                    FeatureFusionBlockTransformer(
                        in_channels=in_channel,
                        hidden_channels=out_channel,
                        mf_frame_num=mf_frame_num,
                        num_layers=num_layers,
                        nhead=nhead,
                        no_cross_attn=no_cross_attn,
                        attn_num_splits=attn_num_splits,
                        low_mem=low_mem,
                        **kwargs,
                    )
                    for in_channel, out_channel in zip(self.in_channels, self.hidden_channels)
                ]
            )
        else:
            self.fusion_layers = nn.ModuleList(
                [
                    FeatureFusionBlockTransformer(
                        in_channels=in_channel,
                        hidden_channels=self.hidden_channels,
                        mf_frame_num=mf_frame_num,
                        num_layers=num_layers,
                        nhead=nhead,
                        no_cross_attn=no_cross_attn,
                        attn_num_splits=attn_num_splits,
                        low_mem=low_mem,
                        **kwargs,
                    )
                    for in_channel in self.in_channels
                ]
            )

    def clear_hidden(self):
        for i in range(len(self.fusion_layers)):
            self.fusion_layers[i].clear_hidden()

    def forward(self, features_list, flow_mode=False):
        # feature_list [feat_list] * n
        if not self.share_block:
            # split block
            feat_lists = [
                [feat[i] for feat in features_list] for i in range(len(features_list[0]))
            ]  # [[feat*n],...]

            for i, feats in enumerate(feat_lists):
                feat_lists[i] = self.fusion_layers[i](feats, flow_mode=flow_mode)

            feat_lists = [[feat[i] for feat in feat_lists] for i in range(len(features_list))]
        else:
            # share block
            feat_size = len(features_list[0])
            feat_lists = [torch.concat(feat) for feat in features_list]  # [feat*b,c,h,w] * n
            feat_lists = self.fusion_layers[0](feat_lists, flow_mode=flow_mode)
            feat_lists = [feat.chunk(feat_size) for feat in feat_lists]
        return feat_lists
