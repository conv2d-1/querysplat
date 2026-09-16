from functools import partial

import torch
import torch.nn as nn

from .spconv_utils import replace_feature, spconv, get_voxel_centers
from .spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, in_channels, out_channels, mid_channels=16, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, mid_channels, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(mid_channels),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(mid_channels, mid_channels, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [mid_channels00, 1408, 41] <- [800, 704, 21]
            block(mid_channels, mid_channels * 2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(mid_channels * 2, mid_channels * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(mid_channels * 2, mid_channels * 2, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(mid_channels * 2, mid_channels * 4, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(mid_channels * 4, mid_channels * 4, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(mid_channels * 4, mid_channels * 4, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(mid_channels * 4, mid_channels * 4, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(mid_channels * 4, mid_channels * 4, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(mid_channels * 4, mid_channels * 4, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(mid_channels * 4, mid_channels * 4, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(mid_channels * 8, mid_channels * 4, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(mid_channels * 4, mid_channels * 4, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(mid_channels * 4, mid_channels * 4, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(mid_channels * 8, mid_channels * 4, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(mid_channels * 4, mid_channels * 2, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [mid_channels00, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(mid_channels * 2, mid_channels * 2, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(mid_channels * 4, mid_channels * 2, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(mid_channels * 2, mid_channels, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [mid_channels00, 1408, 41] <- [mid_channels00, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(mid_channels, mid_channels, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(mid_channels * 2, mid_channels, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(mid_channels, out_channels, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = out_channels

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_size, sparse_shape, voxel_features, voxel_coords):
        """
        Args:
            batch_size: int
            voxel_features: (num_voxels, C)
            voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [mid_channels00, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [mid_channels00, 1408, 41] <- [mid_channels00, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        return x_up1.features
