import torch
import torch.nn as nn

from .flow_base import FlowBase


class FeatureFusionBlockConv(nn.Module, FlowBase):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        mf_frame_num,
    ):
        nn.Module.__init__(self)
        FlowBase.__init__(self, mf_frame_num)

        self.fusion_conv0 = nn.Conv2d(
            in_channels=in_channels * mf_frame_num,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.fusion_conv1 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fusion_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * mf_frame_num,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.fuser = nn.Sequential(
            self.fusion_conv0,
            nn.ReLU(),
            self.fusion_conv1,
            nn.ReLU(),
            self.fusion_conv2,
        )

    def forward(self, feat_list, flow_mode=False):
        if flow_mode:
            feat_list = self.prepare_flow(feat_list)

        b, c, h, w = feat_list[0].shape
        # feat_list [b,c,h,w]*n
        feat = torch.concat(feat_list, dim=1)  # [b,c*n,h,w]
        feat = self.fuser(feat)
        feat_list = [f for f in feat.split(c, dim=1)]
        return feat_list


class FeatureFusionNetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        mf_frame_num,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.mf_frame_num = mf_frame_num

        if isinstance(self.in_channels, (list, tuple)):
            self.fusion_layers = nn.ModuleList(
                [
                    FeatureFusionBlockConv(
                        in_channels=in_channel,
                        hidden_channels=out_channel,
                        mf_frame_num=mf_frame_num,
                    )
                    for in_channel, out_channel in zip(self.in_channels, self.hidden_channels)
                ]
            )
        else:
            self.fusion_layers = nn.ModuleList(
                [
                    FeatureFusionBlockConv(
                        in_channels=self.in_channels,
                        hidden_channels=out_channel,
                        mf_frame_num=mf_frame_num,
                    )
                    for out_channel in self.hidden_channels
                ]
            )

    def clear_hidden(self):
        for i in range(len(self.hidden_channels)):
            self.fusion_layers[i].clear_hidden()

    def forward(self, features_list, flow_mode=False):
        # feature_list [feat_list] * n
        feat_lists = [
            [feat[i] for feat in features_list] for i in range(len(features_list[0]))
        ]  # [[feat*n],...]

        for i, feats in enumerate(feat_lists):
            feat_lists[i] = self.fusion_layers[i](feats, flow_mode=flow_mode)
        feat_lists = [[feat[i] for feat in feat_lists] for i in range(len(features_list))]
        return feat_lists
