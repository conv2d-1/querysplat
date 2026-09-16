import torch.nn as nn

from hAlgorithm.modules.models.pi3.models.layers.camera_head import CameraHead, CameraHeadV2
from hAlgorithm.modules.models.pi3.models.layers.pos_embed import RoPE2D
from hAlgorithm.modules.models.pi3.models.layers.transformer_head import (
    TransformerDecoder,
)


class Pi3CameraHead(nn.Module):
    def __init__(
        self,
        dim_in,
        pos_type="rope100",
        pred_fov=False,
    ):
        super().__init__()

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
        #  Camera Pose Decoder
        # ----------------------
        self.decoder = TransformerDecoder(
            in_dim=dim_in,
            dec_embed_dim=1024,
            dec_num_heads=16,  # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False,
        )

        self.pred_fov = pred_fov
        if self.pred_fov:
            self.head = CameraHeadV2(dim=512)
        else:
            self.head = CameraHead(dim=512)

    def forward(
        self, hidden, pos, patch_start_idx, patch_h=None, patch_w=None, meta_data=None, **kwargs
    ):
        if patch_h is None or patch_w is None:
            patch_h = meta_data["patch_h"]
            patch_w = meta_data["patch_w"]

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        B, N, L, C = hidden.shape
        hidden = hidden.reshape(B * N, L, C)
        pos = pos.reshape(B * N, L, 2)

        hidden = self.decoder(hidden, xpos=pos)
        if self.pred_fov:
            poses, fov = self.head(hidden[:, patch_start_idx:], patch_h, patch_w)
            poses = poses.reshape(B, N, 4, 4)
            fov = fov.reshape(B, N, 2)
        else:
            poses = self.head(hidden[:, patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
            fov = None

        return poses, fov
