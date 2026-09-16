import torch
import torch.nn as nn
from einops import rearrange

from hAlgorithm.modules.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .prompt_mix_vit import MixVIT, slice_expand_and_flatten


class RayMixVIT(MixVIT):
    def __init__(self, **kwargs):
        super(RayMixVIT, self).__init__(**kwargs)

    def compute_rays(self, c2w, fxfycxcy, h, w, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d

    def forward(self, patch_tokens, pose_enc, meta_data, **kwargs):
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        if isinstance(pose_enc, (list, tuple)):
            pose_enc = pose_enc[-1]

        if patch_tokens.ndim == 3:
            BS, P, C = patch_tokens.shape
            S = frame_num * view_num
            B = BS // S
        else:
            B, S, P, C = patch_tokens.shape
            patch_tokens = patch_tokens.reshape(B * S, P, C)

        with torch.no_grad():
            dtype, device = pose_enc.dtype, pose_enc.device
            h, w = self.patch_size * patch_h, self.patch_size * patch_w
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_encoding=pose_enc.detach().float(),
                image_size_hw=(h, w),
                pose_encoding_type="absT_quaR_FoV",
                build_intrinsics=True,
                translation_scale=kwargs["prompt_scale"],
            )
            ray_o, ray_d = self.compute_rays(
                c2w=extrinsics.inverse(),
                fxfycxcy=intrinsics[:, :, [0, 1, 0, 1], [0, 1, 2, 2]],
                h=h,
                w=w,
                device=device,
            )
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            prompt_features = torch.cat([o_cross_d, ray_d], dim=2)
            prompt_features = prompt_features.reshape(B * S, *prompt_features.shape[-3:]).to(dtype)

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, S)

        if self.prompt_in_chans > 0 and prompt_features is not None:
            prompt_tokens = self.prompt_patch_embed(prompt_features)
            patch_tokens = torch.cat([patch_tokens, prompt_tokens], dim=-1)
            if self.fuse_embed_dim > 0:
                patch_tokens = self.fuse_project(patch_tokens)

        # Concatenate special tokens with patch tokens
        if self.num_register_tokens > 0:
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        else:
            tokens = torch.cat([camera_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, patch_h, patch_w, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        camera_output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(global_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                if self.output_frame:
                    concat_inter = frame_intermediates[i]
                elif self.output_global:
                    concat_inter = global_intermediates[i]
                else:
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )

                output_list.append(concat_inter[:, :, self.patch_start_idx :])
                camera_output_list.append(concat_inter[:, :, : self.patch_start_idx])

        # del concat_inter
        # del frame_intermediates
        # del global_intermediates

        if self.debug:
            self.vis_vit_features(
                [output_list[hook] for hook in self.hooks], patch_h, patch_w, meta_data
            )

        if self.hooks is not None:
            return [output_list[hook] for hook in self.hooks], [
                camera_output_list[hook] for hook in self.hooks
            ]
        else:
            return output_list, camera_output_list
