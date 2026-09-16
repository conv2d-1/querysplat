import torch
import torch.nn as nn
from hAlgorithm.modules.models2.external.cut3r.dust3r.heads import head_factory
from hAlgorithm.modules.models2.external.cut3r.dust3r.utils.misc import transpose_to_landscape

from hAlgorithm.modules.models2.external.cut3r.croco.models.pos_embed import RoPE2D

class Cut3rDptHead(nn.Module):
    def __init__(
        self, 
        output_mode, 
        head_type, 
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        pos_embed,
        patch_size,
        **kwargs
    ):
        super().__init__()
        self.enc_embed_dim=enc_embed_dim
        self.dec_embed_dim=dec_embed_dim
        self.dec_num_heads=dec_num_heads
        self.patch_size=patch_size
        if pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)
        
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.set_downstream_head(
            self.output_mode,
            self.head_type,
            landscape_only,
            self.depth_mode,
            self.conf_mode,
            self.pose_mode,
            depth_head,
            rgb_head,
            pose_conf_head,
            pose_head,
        )

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        **kw,
    ):
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )
    
    def forward(self, decout, img_shape, pos:torch.Tensor, **kwargs):
        B,S,P,C = decout[-1].shape
        img_shape = torch.tensor(img_shape[-2:]).unsqueeze(0).repeat(B*S, 1)
        decout = [
            out.reshape(B*S, *out.shape[2:]).contiguous() for out in decout
        ]
        pos = pos.reshape(B*S, *pos.shape[2:]).contiguous()
        out = self.head(decout, img_shape, pos=pos)
        out["depth"] = out.pop("pts3d_in_self_view").permute(0,3,1,2)
        out["confidence"] = out.pop("conf_self")
        out["global_points"] = out.pop("pts3d_in_other_view").permute(0,3,1,2)
        out["global_confidence"] = out.pop("conf")
        out["pose_enc"] = out.pop("camera_pose")
        return out
