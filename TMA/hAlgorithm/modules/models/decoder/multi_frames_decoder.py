import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.croco.blocks import PositionGetter
from hAlgorithm.utils import instantiate_from_config

from .blocks import _make_fusion_block, _make_scratch


class MFDecoder(nn.Module):
    def __init__(
        self,
        features,
        in_channels,
        mid_channels,
        use_bn=False,
        use_dino_clstoken=False,
        prompt_inchannel=0,
        prompt_fusion_mode="add",
        prompt_project_cfg=None,
        prompt_flag=None,
        refinenet_out_cfg=None,
        rgb_fusion_cfg=None,
        prompt_fusion_cfg=None,
        path_fusion_cfg=None,
        **kwargs,
    ):
        """
        初始化MFDecoder模块。

        参数:
        - features: 用于细化特征的scratch层的数量。
        - use_bn: 是否使用批归一化（Batch Normalization）。
        - prompt_inchannel: 提示特征的输入通道数。
        - prompt_fusion_mode: 融合提示特征的方法 ('add', 'concat' 等)。
        - prompt_project_cfg: 提示处理的配置字典。
        - prompt_flag: 布尔标志列表，指示是否在每个阶段使用提示特征，默认为 [True, True, True, True]。
        - refinenet_out_cfg: 融合块中输出卷积的配置。
        """
        super().__init__()

        self.features = features
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.use_bn = use_bn
        self.use_dino_clstoken = use_dino_clstoken

        self.prompt_inchannel = prompt_inchannel
        self.prompt_fusion_mode = prompt_fusion_mode
        self.prompt_project_cfg = prompt_project_cfg
        self.prompt_flag = prompt_flag if prompt_flag is not None else [True, True, True, True]

        self.refinenet_out_cfg = refinenet_out_cfg

        self.rgb_fusion_cfg = rgb_fusion_cfg
        self.prompt_fusion_cfg = prompt_fusion_cfg
        self.path_fusion_cfg = path_fusion_cfg

        self.build_mf_fusion()
        self.build_act_postprocess()
        self.build_dpt_adapter()

    def build_mf_fusion(self):
        self.rgb_fuser = instantiate_from_config(self.rgb_fusion_cfg)
        self.prompt_fuser = instantiate_from_config(self.prompt_fusion_cfg)
        self.path_fuser = instantiate_from_config(self.path_fusion_cfg)

    def build_act_postprocess(self):
        if isinstance(self.in_channels, (list, tuple)):
            self.projects = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for in_channel, out_channel in zip(self.in_channels, self.mid_channels)
                ]
            )
        else:
            self.projects = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for out_channel in self.mid_channels
                ]
            )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.mid_channels[0],
                    out_channels=self.mid_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=self.mid_channels[1],
                    out_channels=self.mid_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=self.mid_channels[3],
                    out_channels=self.mid_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if self.use_dino_clstoken:
            self.readout_projects = nn.ModuleList()
            for i in range(len(self.projects)):
                if isinstance(self.in_channels, (list, tuple)):
                    self.readout_projects.append(
                        nn.Sequential(
                            nn.Linear(2 * self.in_channels[i], self.in_channels[i]), nn.GELU()
                        )
                    )
                else:
                    self.readout_projects.append(
                        nn.Sequential(nn.Linear(2 * self.in_channels, self.in_channels), nn.GELU())
                    )

    def build_dpt_adapter(self):
        # 初始化用于特征细化的scratch层
        self.scratch = _make_scratch(
            self.mid_channels,
            self.features,
            groups=1,
            expand=False,
        )

        # 设置stem transpose为None（可选组件）
        self.scratch.stem_transpose = None

        # 根据是否使用提示特征初始化refinenet层
        self.scratch.refinenet1 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[0] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )
        self.scratch.refinenet2 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[1] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )
        self.scratch.refinenet3 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[2] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )
        self.scratch.refinenet4 = _make_fusion_block(
            self.features,
            self.use_bn,
            prompt_inchannel=self.prompt_inchannel if self.prompt_flag[3] else 0,
            mode=self.prompt_fusion_mode,
            depth_cfg=self.prompt_project_cfg,
            out_conv_cfg=self.refinenet_out_cfg,
        )

    def forward(
        self, rgb_features, prompt_features=None, meta_data=None, grad_index=None, flow_mode=False
    ):
        """
        前向传播过程。

        参数:
        - rgb_features: 来自不同层的RGB特征张量列表。[b,n,c,d]*4 or [b,c,d]*4
        - prompt_features: 包含提示特征的可选张量。[b,n,c,h,w] or [b,c,h,w]
        - meta_data: 元数据信息，包含帧数和视图数等。

        返回:
        - path_1: 经过所有细化阶段后的处理特征张量。
        """
        assert meta_data is not None
        patch_h = meta_data["patch_h"]
        patch_w = meta_data["patch_w"]

        if rgb_features[0].ndim == 3:
            rgb = self.rgb_pre_single(
                rgb_features,
                patch_h=patch_h,
                patch_w=patch_w,
            )
            rgb = self.rgb_post_single(rgb)
            path = self.rgb_prompt_single(
                rgb,
                prompt_features,
            )
            return path

        b, n = rgb_features[0].shape[0], rgb_features[0].shape[1]
        if grad_index is None:
            grad_index = range(n)

        rgb_features_list = []  # [[b,c,h,w]*4]*n
        for frame_i in range(n):
            if frame_i in grad_index:
                rgb_features_list.append(
                    self.rgb_pre_single(
                        [rgb_feat[:, frame_i, ...] for rgb_feat in rgb_features],
                        patch_h=patch_h,
                        patch_w=patch_w,
                    )
                )
            else:
                with torch.no_grad():
                    rgb_features_list.append(
                        self.rgb_pre_single(
                            [rgb_feat[:, frame_i, ...] for rgb_feat in rgb_features],
                            patch_h=patch_h,
                            patch_w=patch_w,
                        )
                    )

        if self.rgb_fuser is not None:
            rgb_features_list = self.rgb_fuser(rgb_features_list, flow_mode=flow_mode)

        if prompt_features is not None:
            prompt_list = [prompt_features[:, i, ...] for i in range(n)]  # [b,c,h,w]*n
            if self.prompt_fuser is not None:
                prompt_list = self.prompt_fuser(prompt_list, flow_mode=flow_mode)
        else:
            prompt_list = [None for i in range(n)]

        path_list = []  # [b,c,h,w]*n
        for frame_i in range(n):
            rgb_features_list[frame_i] = self.rgb_post_single(rgb_features_list[frame_i])
            path_list.append(
                self.rgb_prompt_single(
                    rgb_features=rgb_features_list[frame_i],
                    prompt_features=prompt_list[frame_i],
                )
            )

        if self.path_fuser is not None:
            path_list = self.path_fuser(path_list, flow_mode=flow_mode)
        path_list = torch.stack(path_list, dim=1)  # b,n,c,h,w

        return path_list

    def clear_hidden(self):
        logging.info("Clearing Hidden info due to scene change.")
        if self.rgb_fuser is not None:
            self.rgb_fuser.clear_hidden()
        if self.prompt_fuser is not None:
            self.prompt_fuser.clear_hidden()
        if self.path_fuser is not None:
            self.path_fuser.clear_hidden()

    def rgb_pre_single(self, rgb_features, patch_h, patch_w):
        # rgb preprocess
        out = []
        for i, x in enumerate(rgb_features):
            if self.use_dino_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()
            out.append(x)
        return out

    def rgb_post_single(self, rgb_features):
        # rgb postprocess
        out = []
        for i, x in enumerate(rgb_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out

    def rgb_prompt_single(self, rgb_features, prompt_features):

        layer_1, layer_2, layer_3, layer_4 = rgb_features
        # 对RGB特征应用降维网络
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # 使用refinenet层细化特征，可选择性地结合提示特征
        path_4 = self.scratch.refinenet4(
            layer_4_rn,
            size=layer_3_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[3] else None,
        )
        path_3 = self.scratch.refinenet3(
            path_4,
            layer_3_rn,
            size=layer_2_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[2] else None,
        )
        path_2 = self.scratch.refinenet2(
            path_3,
            layer_2_rn,
            size=layer_1_rn.shape[2:],
            prompt_depth=prompt_features if self.prompt_flag[1] else None,
        )
        path_1 = self.scratch.refinenet1(
            path_2,
            layer_1_rn,
            prompt_depth=prompt_features if self.prompt_flag[0] else None,
        )

        return path_1
