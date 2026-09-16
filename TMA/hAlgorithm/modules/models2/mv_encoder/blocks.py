import collections
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.utils import instantiate_from_config


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


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
        self.scales = nn.Parameter(torch.tensor([2**i for i in range(min_deg, max_deg)]), requires_grad=False)

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

        if self.bn:
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
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
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
        if self.expand:
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

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

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
        self.copy_block = FeatureFusionBlock(features, activation, deconv, bn, expand, align_corners, size)

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

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

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
        if self.expand:
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
            prompt_depth = F.interpolate(prompt_depth, output.shape[2:], mode="bilinear", align_corners=False)
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

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


class ModLN(nn.Module):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]
