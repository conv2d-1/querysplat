import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, "only odd kernel is supported but kernel = {}".format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def _make_fusion_block(
    features,
    use_bn,
    size=None,
    prompt_inchannel=1,
    sin_enc=None,
    mode="add",
    depth_cfg=None,
    out_conv_cfg=None,
):
    return FeatureFusionDepthBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        prompt_inchannel=prompt_inchannel,
        sin_enc=sin_enc,
        mode=mode,
        depth_cfg=depth_cfg,
        out_conv_cfg=out_conv_cfg,
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(
            torch.tensor([2**i for i in range(min_deg, max_deg)]), requires_grad=False
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        x_ret = torch.sin(x_enc)
        x_ret = x_ret.permute(0, 3, 1, 2)
        return x_ret


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class FeatureFusionControlBlock(FeatureFusionBlock):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super.__init__(features, activation, deconv, bn, expand, align_corners, size)
        self.copy_block = FeatureFusionBlock(
            features, activation, deconv, bn, expand, align_corners, size
        )

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureFusionDepthBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        prompt_inchannel=1,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        sin_enc=None,
        mode="add",
        depth_cfg=None,
        out_conv_cfg=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionDepthBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.mode = mode
        assert self.mode in ["cat", "add", "cat_add"]

        self.prompt_inchannel = prompt_inchannel
        self.groups = 1
        self.depth_cfg = depth_cfg
        self.out_conv_cfg = out_conv_cfg

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        if self.out_conv_cfg is None:
            self.out_conv = nn.Conv2d(
                features,
                out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                groups=1,
            )
        else:
            self.out_conv_cfg["input_channel"] = features
            self.out_conv = nn.Sequential(
                instantiate_from_config(self.out_conv_cfg),
                nn.Conv2d(
                    features,
                    out_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    groups=1,
                ),
            )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        if prompt_inchannel is not None and prompt_inchannel > 0:
            if sin_enc is not None:
                min_deg, max_deg = sin_enc
                self.pos_emb_fn = PositionalEncoding(min_deg, max_deg)
                prompt_inchannel_mul = (max_deg - min_deg) * 2
            else:
                self.pos_emb_fn = None
                prompt_inchannel_mul = 1

            if self.depth_cfg is None:
                self.resConfUnit_depth = nn.Sequential(
                    nn.Conv2d(
                        prompt_inchannel * prompt_inchannel_mul,
                        features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        groups=1,
                    ),
                    activation,
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        groups=1,
                    ),
                    activation,
                    zero_module(
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            groups=1,
                        )
                    ),
                )

            else:
                self.depth_cfg["output_channel"][-1] = features
                self.resConfUnit_depth = nn.Sequential(
                    instantiate_from_config(self.depth_cfg),
                    zero_module(
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            groups=1,
                        )
                    ),
                )

            if self.mode == "add":
                self.resFusion_depth = nn.Identity()
            elif self.mode == "cat":
                self.resFusion_depth = nn.Conv2d(
                    features * 2,
                    features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                    groups=1,
                )
            elif self.mode == "cat_add":
                self.resFusion_depth = nn.Sequential(
                    nn.Conv2d(
                        features * 2,
                        features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        groups=1,
                    ),
                    activation,
                    zero_module(
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            groups=1,
                        )
                    ),
                )
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, prompt_depth=None, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if prompt_depth is not None:
            prompt_depth = F.interpolate(
                prompt_depth, output.shape[2:], mode="bilinear", align_corners=False
            )
            if self.pos_emb_fn is not None:
                prompt_depth = self.pos_emb_fn(prompt_depth)
            res = self.resConfUnit_depth(prompt_depth)
            if self.mode == "add":
                output = self.skip_add.add(output, res)
            elif self.mode == "cat":
                output = self.skip_add.cat([output, res], dim=1)
                output = self.resFusion_depth(output)
            elif self.mode == "cat_add":
                res = self.resFusion_depth(torch.cat((output, res), 1))
                output = self.skip_add.add(output, res)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class AuxModule(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride, bn, act, act_type="relu"):
        super().__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = kernel
        self.stride = stride
        self.bn = bn
        self.act = act
        self.act_type = act_type
        if isinstance(act_type, str):
            self.act_type = [self.act_type] * len(self.kernel)

        self.layer_num = len(self.output_channel)

        # build_conv_layer
        layers = []
        input_channel = self.input_channel
        for idx in range(self.layer_num):
            output_channel = self.output_channel[idx]
            layers.append(
                nn.Conv2d(
                    input_channel,
                    output_channel,
                    kernel_size=self.kernel[idx],
                    stride=self.stride[idx],
                    padding=self.kernel[idx] // 2,
                    bias=not self.bn[idx],
                    groups=1,
                )
            )
            input_channel = output_channel
            if self.bn[idx]:
                layers.append(nn.BatchNorm2d(output_channel))
            if self.act[idx]:
                if self.act_type[idx] == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(self.act_type[idx])

        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class AuxResnetModule(nn.Module):
    def __init__(self, layer_num, input_channel, use_bn):
        super().__init__()

        self.layer_num = layer_num
        self.input_channel = input_channel
        self.use_bn = use_bn

        layers = []
        for idx in range(self.layer_num):
            layers.append(
                ResidualConvUnit(features=self.input_channel, activation=nn.ReLU(), bn=self.use_bn)
            )
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class ResidualConvUnitV2(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, kernel_size=3):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.kernel_size = kernel_size
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class AuxResnetModuleV2(nn.Module):
    def __init__(self, layer_num, input_channel, use_bn, output_channel=None, kernel_size=3):
        super().__init__()

        self.layer_num = layer_num
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.use_bn = use_bn
        self.kernel_size = kernel_size

        layers = []
        if self.output_channel is None:
            for idx in range(self.layer_num):
                layers.append(
                    ResidualConvUnitV2(
                        features=self.input_channel,
                        activation=nn.ReLU(),
                        bn=self.use_bn,
                        kernel_size=kernel_size,
                    )
                )
        else:
            if isinstance(self.output_channel, int):
                layers.append(
                    nn.Conv2d(
                        self.input_channel,
                        self.output_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=True,
                        groups=1,
                    )
                )
                layers.append(nn.ReLU())
                for idx in range(self.layer_num):
                    layers.append(
                        ResidualConvUnitV2(
                            features=self.output_channel,
                            activation=nn.ReLU(),
                            bn=self.use_bn,
                            kernel_size=kernel_size,
                        )
                    )
            else:
                input_channel = self.input_channel
                for output_channel in self.output_channel:
                    layers.append(
                        nn.Conv2d(
                            input_channel,
                            output_channel,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2,
                            bias=True,
                            groups=1,
                        )
                    )
                    layers.append(nn.ReLU())

                    for idx in range(self.layer_num):
                        layers.append(
                            ResidualConvUnitV2(
                                features=output_channel, activation=nn.ReLU(), bn=self.use_bn
                            )
                        )
                    input_channel = output_channel

        assert len(layers) > 0
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class AuxResnetModuleV3(nn.Module):
    def __init__(self, layer_num, input_channel, use_bn, output_channel=None, kernel_size=3):
        super().__init__()

        self.layer_num = layer_num
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.use_bn = use_bn
        self.kernel_size = kernel_size

        layers = []
        if self.output_channel is None:
            for idx in range(self.layer_num):
                layers.append(
                    ResidualConvUnitV2(
                        features=self.input_channel, activation=nn.ReLU(), bn=self.use_bn
                    )
                )
        else:
            if isinstance(self.output_channel, int):
                layers.append(
                    nn.Conv2d(
                        self.input_channel,
                        self.output_channel,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        bias=True,
                        groups=1,
                    )
                )
                # layers.append(nn.ReLU())
                for idx in range(self.layer_num):
                    layers.append(
                        ResidualConvUnitV2(
                            features=self.output_channel, activation=nn.ReLU(), bn=self.use_bn
                        )
                    )
            else:
                input_channel = self.input_channel
                for output_channel in self.output_channel:
                    layers.append(
                        nn.Conv2d(
                            input_channel,
                            output_channel,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size // 2,
                            bias=True,
                            groups=1,
                        )
                    )
                    # layers.append(nn.ReLU())

                    for idx in range(self.layer_num):
                        layers.append(
                            ResidualConvUnitV2(
                                features=output_channel, activation=nn.ReLU(), bn=self.use_bn
                            )
                        )
                    input_channel = output_channel

        assert len(layers) > 0
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)
