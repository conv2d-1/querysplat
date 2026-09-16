import torch
import torch.nn as nn


class DinoFusionBlockConcat(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mf_frame_num,
        use_clstoken=False,
        project_rn_first=False,
        **kwargs
    ):
        super().__init__()
        self.use_clstoken = use_clstoken
        self.only_last_frame = True
        original_out_channels = out_channels
        self.mf_frame_num = mf_frame_num

        self.project_rn_first = project_rn_first
        if not self.project_rn_first:
            in_channels = int(in_channels * mf_frame_num)
            out_channels = [int(channel * mf_frame_num) for channel in out_channels]
            self.project_rn = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=out_channel,
                        out_channels=original_out_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for i, out_channel in enumerate(out_channels)
                ]
            )
        else:
            self.project_rn = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=int(in_channels * mf_frame_num),
                        out_channels=in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for out_channel in out_channels
                ]
            )

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        if self.use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )
        self.hidden = []

    def infer_flow(self, feature_list, patch_h, patch_w):
        assert len(feature_list) == 1
        feat_flow = []
        self.hidden = self.hidden[-self.mf_frame_num + 1 :]
        self.hidden.extend(feature_list)
        feat_flow.extend(self.hidden)
        while len(feat_flow) < self.mf_frame_num:
            feat_flow.extend(feature_list)
        return self.forward(feat_flow, patch_h, patch_w)

    def forward(self, feature_list, patch_h, patch_w):
        """
        - feature_list: [[[(B,H,W),(B,H,W)]*4] * T
        """
        out_list = []  # [[B,C,H,W]*4]*T
        for feature in feature_list:
            out = []
            for i, x in enumerate(feature):
                if self.use_clstoken:
                    x, cls_token = x[0], x[1]
                    readout = cls_token.unsqueeze(1).expand_as(x)
                    x = self.readout_projects[i](torch.cat((x, readout), -1))
                else:
                    x = x[0]

                x = (
                    x.permute(0, 2, 1)
                    .reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
                    .contiguous()
                )

                out.append(x)
            out_list.append(out)
        path_num = len(out_list[0])
        fusion_feat_list = [
            torch.concat([feat[path_i] for feat in out_list], dim=1) for path_i in range(path_num)
        ]  # [[B,C*T,H,W]*4]

        out_list = []
        # [[B,C,H,W]*4] * 1
        for i, x in enumerate(fusion_feat_list):
            if not self.project_rn_first:
                x = self.projects[i](x)
                x = self.resize_layers[i](x)
                x = self.project_rn[i](x)
            else:
                x = self.project_rn[i](x)
                x = self.projects[i](x)
                x = self.resize_layers[i](x)
            out_list.append(x)
        return [out_list]  # T=1


class FeatureFusionBlockConcat(nn.Module):
    def __init__(self, mf_frame_num, head_features_1, **kwargs):
        super().__init__()
        self.only_last_frame = True
        self.project_output = nn.Conv2d(
            in_channels=int(head_features_1 * mf_frame_num),
            out_channels=head_features_1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.mf_frame_num = mf_frame_num
        self.hidden = []

    def infer_flow(self, feature_list):
        assert len(feature_list) == 1
        feat_flow = []
        self.hidden = self.hidden[-self.mf_frame_num + 1 :]
        self.hidden.extend(feature_list)
        feat_flow.extend(self.hidden)
        while len(feat_flow) < self.mf_frame_num:
            feat_flow.extend(feature_list)
        return self.forward(feat_flow)

    def forward(self, out_list):
        """
        -out_list: [B,C,H,W] * T
        """
        fusion_feat_list = [torch.concat(out_list, dim=1)]
        # [[B, C*T, H, W]]
        for i, feat in enumerate(fusion_feat_list):
            fusion_feat_list[i] = self.project_output(feat)
        # [[B, C, H, W]]
        return fusion_feat_list
