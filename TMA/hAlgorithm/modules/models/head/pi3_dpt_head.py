from copy import deepcopy

import torch.nn as nn

from hAlgorithm.modules.models.pi3.models.layers.pos_embed import RoPE2D
from hAlgorithm.modules.models.pi3.models.layers.transformer_head import (
    LinearPts3d,
    TransformerDecoder,
)

from .blocks import Exp, InverseLog, MogeActivation


class Pi3DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size=14,
        embed_dim=1024,
        num_heads=16,
        output_dim=3,
        act="moge",
        pos_type="rope100",
        return_features=False,
        features_only=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.act = act
        self.return_features = return_features
        self.features_only = features_only

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):  # eg rope100
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError

        # ----------------------
        #  Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=self.in_channels,
            dec_embed_dim=self.embed_dim,
            dec_num_heads=self.num_heads,
            out_dim=self.embed_dim,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(
            patch_size=self.patch_size,
            dec_embed_dim=self.embed_dim,
            output_dim=self.output_dim,
            permute=False,
        )
        if self.act == "moge":
            self.point_act = MogeActivation()
        elif self.act == "exp":
            self.point_act = Exp()
        elif self.act == "inverse_log":
            self.point_act = InverseLog()
        else:
            self.point_act = nn.Identity()

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(
            patch_size=self.patch_size, dec_embed_dim=1024, output_dim=1, permute=False
        )

    def forward(
        self,
        features,
        pos,
        patch_start_idx,
        patch_h=None,
        patch_w=None,
        return_dict=False,
        meta_data=None,
        **kwargs,
    ):
        if patch_h is None or patch_w is None:
            patch_h = meta_data["patch_h"]
            patch_w = meta_data["patch_w"]

        if isinstance(features, (list, tuple)):
            features = features[-1]

        B, N, L, C = features.shape
        features = features.reshape(B * N, L, C)
        pos = pos.reshape(B * N, L, 2)

        point_hidden = self.point_decoder(features, xpos=pos)
        conf_hidden = self.conf_decoder(features, xpos=pos)

        if self.features_only:
            return point_hidden

        W = patch_w * self.patch_size
        H = patch_h * self.patch_size

        points = self.point_head([point_hidden[:, patch_start_idx:]], (H, W)).reshape(
            B, N, self.output_dim, H, W
        )
        points = self.point_act(points)

        conf = self.conf_head([conf_hidden[:, patch_start_idx:]], (H, W)).reshape(B, N, 1, H, W)

        return_dict = dict(pointmap=points, confidence=conf)

        if self.return_features:
            return_dict["features"] = point_hidden

        return return_dict
