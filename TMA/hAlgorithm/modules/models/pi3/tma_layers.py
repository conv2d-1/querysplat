from .models.layers.transformer_head import TransformerDecoder, LinearPts3d
import torch
import torch.nn.functional as F
import torch.nn as nn
from hAlgorithm.modules.models.facebookresearch_dinov2_main.dinov2.layers import PatchEmbed
from hAlgorithm.modules.models.head.blocks import Exp, InverseLog, PointmapExp

class PromptTransformerHead(nn.Module):
    def __init__(
        self,
        in_dim,
        prompt_in_chans,
        dec_out_dim,
        dec_embed_dim=512,
        prompt_embed_dim=512,
        fuse_embed_dim=0,
        fuse_type='cat',
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        need_project=True,
        patch_size=14,
        prompt_patch_size=None,
        output_act="point_exp",
        final_outchannel=3,
        pred_confidence=False,
    ):
        super().__init__()

        self.prompt_in_chans = prompt_in_chans
        self.prompt_embed_dim = prompt_embed_dim
        self.fuse_embed_dim = fuse_embed_dim
        self.patch_size = patch_size
        self.prompt_patch_size = prompt_patch_size or self.patch_size
        self.fuse_type = fuse_type

        if self.prompt_in_chans > 0:
            if fuse_embed_dim == 0:
                if self.fuse_type in ['cat', 'concat']:
                    fuse_embed_dim = in_dim + self.prompt_embed_dim
                elif self.fuse_type in ['add']:
                    fuse_embed_dim = in_dim
            self.prompt_patch_embed = PatchEmbed(
                img_size=518,
                patch_size=self.prompt_patch_size,
                in_chans=self.prompt_in_chans,
                embed_dim=self.prompt_embed_dim,
            )
            if self.fuse_embed_dim > 0:
                if self.fuse_type in ['cat', 'concat']:
                    self.fuse_project = nn.Linear(
                        in_dim + self.prompt_embed_dim, self.fuse_embed_dim, bias=True
                    )
                elif self.fuse_type in ['add']:
                    self.fuse_project_rgb = nn.Linear(
                        in_dim , self.fuse_embed_dim, bias=True
                    )
                    self.fuse_project_prompt = nn.Linear(
                        self.prompt_embed_dim , self.fuse_embed_dim, bias=True
                    )
            assert self.fuse_type in ['cat', 'concat', 'add'], f"fuse_type must be add or cat, but got {self.fuse_type}."
        else:
            if fuse_embed_dim == 0:
                fuse_embed_dim = in_dim
        self.embed_dim = fuse_embed_dim

        self.decoder = TransformerDecoder(
            in_dim=fuse_embed_dim, out_dim=dec_out_dim, dec_embed_dim=dec_embed_dim,
            depth=depth, dec_num_heads=dec_num_heads, mlp_ratio=mlp_ratio,
            rope=None, need_project=need_project, use_checkpoint=False,
        )
        
        self.head = LinearPts3d(
            patch_size=patch_size, dec_embed_dim=dec_out_dim, 
            output_dim=final_outchannel, permute=False
        )
        self.pred_confidence=pred_confidence
        if pred_confidence:
            self.conf_head = LinearPts3d(
                patch_size=patch_size, dec_embed_dim=dec_out_dim, 
                output_dim=1, permute=False
            )

        if output_act == "sigmoid":
            act_func = nn.Sigmoid()
        elif output_act == "relu":
            act_func = nn.ReLU(True)
        elif output_act == "exp":
            act_func = Exp()
        elif output_act == "inverse_log":
            act_func = InverseLog()
        elif output_act == "point_exp":
            act_func = PointmapExp()
        else:
            act_func = nn.Identity()
        self.act_func = act_func

    def prepare_tokens(self, rgb_tokens, prompt_features, meta_data, **kwargs):
        image_h, image_w = meta_data["image_h"], meta_data["image_w"]
        if isinstance(rgb_tokens, (list, tuple)):
            rgb_tokens = rgb_tokens[-1]

        if self.prompt_in_chans > 0 and prompt_features is not None:
            prompt_features = F.interpolate(prompt_features, (image_h, image_w), mode="bilinear", align_corners=True)
            prompt_tokens = self.prompt_patch_embed(prompt_features)
            if self.fuse_type in ['cat', 'concat']:
                patch_tokens = torch.cat([rgb_tokens, prompt_tokens], dim=-1)
                if self.fuse_embed_dim > 0:
                    patch_tokens = self.fuse_project(patch_tokens)
            elif self.fuse_type in ['add']:
                if self.fuse_embed_dim > 0:
                    rgb_tokens = self.fuse_project_rgb(rgb_tokens)
                    prompt_tokens = self.fuse_project_rgb(prompt_tokens)
                patch_tokens = torch.add(rgb_tokens, prompt_tokens)
        else:
            patch_tokens = rgb_tokens
        return patch_tokens

    def forward(self, rgb_tokens, prompt_features, meta_data, return_dict=True, **kwargs):
        image_h, image_w = meta_data["image_h"], meta_data["image_w"]

        patch_tokens = self.prepare_tokens(rgb_tokens, prompt_features, meta_data, **kwargs)
        
        patch_tokens = self.decoder(patch_tokens, xpos=None)
        patch_tokens = [patch_tokens]

        pointmap = self.head(patch_tokens, (image_h, image_w))
        pointmap = self.act_func(pointmap)

        if self.pred_confidence:
            confidence = self.conf_head(patch_tokens, (image_h, image_w))
        else:
            confidence = None

        if return_dict:
            output = dict(
                pointmap=pointmap,
                confidence=confidence,
            )
        else:
            output = (pointmap, confidence)
        return output

class PromptTransformerHeadV2(PromptTransformerHead):
    def __init__(self, 
        in_dim,
        prompt_in_chans,
        dec_out_dim,
        use_scale_token=False,
        num_register_tokens=0,
        **kwargs
    ):
        super().__init__(in_dim=in_dim, prompt_in_chans=prompt_in_chans, dec_out_dim=dec_out_dim, **kwargs)
        
        self.patch_start_idx = 0
        
        self.use_scale_token = use_scale_token
        if self.use_scale_token:
            self.scale_proj = nn.Linear(1, self.embed_dim)
            self.patch_start_idx += 1
        
        self.num_register_tokens = num_register_tokens
        if self.num_register_tokens > 0:
            self.register_token = nn.Parameter(torch.randn(1, 1, self.num_register_tokens, self.embed_dim))
            nn.init.normal_(self.register_token, std=1e-6)
            self.patch_start_idx += self.num_register_tokens
    
    def forward(self, rgb_tokens, prompt_features, meta_data, return_dict=True, **kwargs):
        image_h, image_w = meta_data["image_h"], meta_data["image_w"]

        patch_tokens = self.prepare_tokens(rgb_tokens, prompt_features, meta_data, **kwargs)
        
        if self.use_scale_token:
            prompt_scale = meta_data["prompt_scale"].squeeze(-1) # B, 1, 1
            scale_token = self.scale_proj(prompt_scale) # B, 1, embed
            patch_tokens = torch.cat([scale_token, patch_tokens], dim=1)

        patch_tokens = self.decoder(patch_tokens, xpos=None)
        patch_tokens = [patch_tokens[:,self.patch_start_idx:,...]]

        pointmap = self.head(patch_tokens, (image_h, image_w))
        pointmap = self.act_func(pointmap)

        if self.pred_confidence:
            confidence = self.conf_head(patch_tokens, (image_h, image_w))
        else:
            confidence = None

        if return_dict:
            output = dict(
                pointmap=pointmap,
                confidence=confidence,
            )
        else:
            output = (pointmap, confidence)
        return output
