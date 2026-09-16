import torch
import torch.nn as nn

from .blocks import slice_expand_and_flatten
from .pi3_mix_vit import MixVIT


class MeMMixVIT(MixVIT):
    def __init__(self, flow=False, **kwargs):
        super(MeMMixVIT, self).__init__(**kwargs)
        self.fuse_project = None
        self.flow = flow
        self.mem_tokens = None

    def forward(self, patch_tokens, prompt_features, meta_data, **kwargs):
        patch_w = meta_data["input_width"][0].item() // self.patch_size
        patch_h = meta_data["input_height"][0].item() // self.patch_size
        frame_num = meta_data["frames"][0].item()
        view_num = meta_data["views"][0].item()

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        BS, P, C = patch_tokens.shape
        B = BS // (frame_num * view_num)

        if self.flow:
            mem_tokens = self.mem_tokens
        else:
            mem_tokens = None

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, view_num)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, view_num)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * view_num, patch_h, patch_w, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * view_num, self.patch_start_idx, 2)
                .to(patch_tokens.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        total_outputs = []
        for fi in range(frame_num):
            curr_patch_tokens = patch_tokens.view(B, frame_num, view_num, *patch_tokens.shape[-2:])[
                :, fi
            ].view(-1, *patch_tokens.shape[-2:])

            if self.prompt_in_chans > 0 and prompt_features is not None:
                curr_prompt_tokens = self.prompt_patch_embed(
                    prompt_features.view(B, frame_num, view_num, *prompt_features.shape[-3:])[
                        :, fi
                    ].view(-1, *prompt_features.shape[-3:])
                )
                curr_patch_tokens = curr_patch_tokens + curr_prompt_tokens

            # Concatenate special tokens with patch tokens
            if self.num_register_tokens > 0:
                tokens = torch.cat([camera_token, register_token, curr_patch_tokens], dim=1)
            else:
                tokens = torch.cat([camera_token, curr_patch_tokens], dim=1)

            if mem_tokens is not None:
                tokens = tokens + mem_tokens

            # update P because we added special tokens
            _, P, C = tokens.shape

            frame_idx = 0
            global_idx = 0
            output_list = []

            for _ in range(self.aa_block_num):
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, view_num, P, C, frame_idx, pos=pos
                        )
                    elif attn_type == "global":
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, view_num, P, C, global_idx, pos=pos
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                for i in range(len(global_intermediates)):
                    # concat frame and global intermediates, [B x view_num x P x 2C]
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)

            mem_tokens = tokens.view(B * view_num, P, C)

            if self.hooks is None:
                total_outputs.append(output_list[-1])
            else:
                total_outputs.append([output_list[hook] for hook in self.hooks])

        if self.flow:
            self.mem_tokens = mem_tokens

        if self.hooks is None:
            total_outputs = torch.stack(total_outputs, dim=1).view(
                B, frame_num * view_num, P, 2 * C
            )
            return total_outputs, pos, self.patch_start_idx
        else:
            merge_outputs = []
            for i in range(len(total_outputs[0])):
                merge_outputs.append(
                    torch.stack([total_outputs[fi][i] for fi in range(frame_num)], dim=1).view(
                        B, frame_num * view_num, P, 2 * C
                    )
                )
            return merge_outputs, pos, self.patch_start_idx


class MeMMixVIT2(MeMMixVIT):
    """camera_token 替换为时刻0的 camera_token."""

    def __init__(self, flow=False, **kwargs):
        super(MeMMixVIT2, self).__init__(**kwargs)
        self.fuse_project = None
        self.flow = flow
        self.mem_tokens = None
        self.first_camera_token = None

    def forward(self, patch_tokens, prompt_features, meta_data, **kwargs):
        patch_w = meta_data["input_width"][0].item() // self.patch_size
        patch_h = meta_data["input_height"][0].item() // self.patch_size
        frame_num = meta_data["frames"][0].item()
        view_num = meta_data["views"][0].item()

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        BS, P, C = patch_tokens.shape
        B = BS // (frame_num * view_num)

        if self.flow:
            mem_tokens = self.mem_tokens
            first_camera_token = self.first_camera_token
        else:
            mem_tokens = None
            first_camera_token = None

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, view_num)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, view_num)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * view_num, patch_h, patch_w, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * view_num, self.patch_start_idx, 2)
                .to(patch_tokens.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        total_outputs = []
        for fi in range(frame_num):
            curr_patch_tokens = patch_tokens.view(B, frame_num, view_num, *patch_tokens.shape[-2:])[
                :, fi
            ].view(-1, *patch_tokens.shape[-2:])

            if self.prompt_in_chans > 0 and prompt_features is not None:
                curr_prompt_tokens = self.prompt_patch_embed(
                    prompt_features.view(B, frame_num, view_num, *prompt_features.shape[-3:])[
                        :, fi
                    ].view(-1, *prompt_features.shape[-3:])
                )
                curr_patch_tokens = curr_patch_tokens + curr_prompt_tokens

            if mem_tokens is not None:
                curr_patch_tokens = curr_patch_tokens + mem_tokens

            # Concatenate special tokens with patch tokens
            if first_camera_token is not None:
                if self.num_register_tokens > 0:
                    tokens = torch.cat(
                        [first_camera_token, register_token, curr_patch_tokens], dim=1
                    )
                else:
                    tokens = torch.cat([first_camera_token, curr_patch_tokens], dim=1)
            else:
                if self.num_register_tokens > 0:
                    tokens = torch.cat([camera_token, register_token, curr_patch_tokens], dim=1)
                else:
                    tokens = torch.cat([camera_token, curr_patch_tokens], dim=1)

            # update P because we added special tokens
            _, P, C = tokens.shape

            frame_idx = 0
            global_idx = 0
            output_list = []

            for _ in range(self.aa_block_num):
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, view_num, P, C, frame_idx, pos=pos
                        )
                    elif attn_type == "global":
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, view_num, P, C, global_idx, pos=pos
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                for i in range(len(global_intermediates)):
                    # concat frame and global intermediates, [B x view_num x P x 2C]
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)

            mem_tokens = tokens.view(B * view_num, P, C)[:, self.patch_start_idx :, :]

            if fi == 0 and first_camera_token is None:
                first_camera_token = tokens.view(B, view_num, P, C)[:, :, 0:1, :].view(
                    B * view_num, 1, C
                )

            if self.hooks is None:
                total_outputs.append(output_list[-1])
            else:
                total_outputs.append([output_list[hook] for hook in self.hooks])

        if self.flow:
            self.mem_tokens = mem_tokens
            self.first_camera_token = first_camera_token

        if self.hooks is None:
            total_outputs = torch.stack(total_outputs, dim=1).view(
                B, frame_num * view_num, P, 2 * C
            )
            return total_outputs, pos, self.patch_start_idx
        else:
            merge_outputs = []
            for i in range(len(total_outputs[0])):
                merge_outputs.append(
                    torch.stack([total_outputs[fi][i] for fi in range(frame_num)], dim=1).view(
                        B, frame_num * view_num, P, 2 * C
                    )
                )
            return merge_outputs, pos, self.patch_start_idx


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MeMMixVIT3(MeMMixVIT):
    def __init__(self, fuse="add", fuse_fc=None, fuse_mlp=None, fuse_gru=None, **kwargs):
        super(MeMMixVIT3, self).__init__(**kwargs)

        self.fuse = fuse
        self.fuse_fc = fuse_fc
        self.fuse_mlp = fuse_mlp
        self.fuse_gru = fuse_gru

        if self.fuse_fc is not None:
            self.fuse_fc = nn.Linear(self.fuse_fc["dim"], self.fuse_fc["hidden_dim"], bias=True)

        if self.fuse_mlp is not None:
            self.fuse_mlp = Mlp(
                in_features=self.fuse_mlp["dim"],
                hidden_features=self.fuse_mlp["hidden_dim"],
            )

    def forward(self, patch_tokens, prompt_features, meta_data, **kwargs):
        patch_w = meta_data["input_width"][0].item() // self.patch_size
        patch_h = meta_data["input_height"][0].item() // self.patch_size
        frame_num = meta_data["frames"][0].item()
        view_num = meta_data["views"][0].item()

        if isinstance(patch_tokens, (list, tuple)):
            patch_tokens = patch_tokens[-1]

        BS, P, C = patch_tokens.shape
        B = BS // (frame_num * view_num)

        if self.flow:
            mem_tokens = self.mem_tokens
        else:
            mem_tokens = None

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, view_num)
        if self.num_register_tokens > 0:
            register_token = slice_expand_and_flatten(self.register_token, B, view_num)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * view_num, patch_h, patch_w, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * view_num, self.patch_start_idx, 2)
                .to(patch_tokens.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        total_outputs = []
        for fi in range(frame_num):
            curr_patch_tokens = patch_tokens.view(B, frame_num, view_num, *patch_tokens.shape[-2:])[
                :, fi
            ].view(-1, *patch_tokens.shape[-2:])

            if self.prompt_in_chans > 0 and prompt_features is not None:
                curr_prompt_tokens = self.prompt_patch_embed(
                    prompt_features.view(B, frame_num, view_num, *prompt_features.shape[-3:])[
                        :, fi
                    ].view(-1, *prompt_features.shape[-3:])
                )
                curr_patch_tokens = curr_patch_tokens + curr_prompt_tokens

            # Concatenate special tokens with patch tokens
            if self.num_register_tokens > 0:
                tokens = torch.cat([camera_token, register_token, curr_patch_tokens], dim=1)
            else:
                tokens = torch.cat([camera_token, curr_patch_tokens], dim=1)

            if mem_tokens is not None:
                if self.fuse == "add":
                    tokens = tokens + mem_tokens
                elif self.fuse == "concat":
                    tokens = torch.cat([tokens, mem_tokens], dim=-1)
                else:
                    raise NotImplementedError

                if self.fuse_fc is not None:
                    tokens = self.fuse_fc(tokens)
                elif self.fuse_mlp is not None:
                    tokens = self.fuse_mlp(tokens)

            # update P because we added special tokens
            _, P, C = tokens.shape

            frame_idx = 0
            global_idx = 0
            output_list = []

            for _ in range(self.aa_block_num):
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                            tokens, B, view_num, P, C, frame_idx, pos=pos
                        )
                    elif attn_type == "global":
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, view_num, P, C, global_idx, pos=pos
                        )
                    else:
                        raise ValueError(f"Unknown attention type: {attn_type}")

                for i in range(len(global_intermediates)):
                    # concat frame and global intermediates, [B x view_num x P x 2C]
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)

            mem_tokens = tokens.view(B * view_num, P, C)

            if self.hooks is None:
                total_outputs.append(output_list[-1])
            else:
                total_outputs.append([output_list[hook] for hook in self.hooks])

        if self.flow:
            self.mem_tokens = mem_tokens

        if self.hooks is None:
            total_outputs = torch.stack(total_outputs, dim=1).view(
                B, frame_num * view_num, P, 2 * C
            )
            return total_outputs, pos, self.patch_start_idx
        else:
            merge_outputs = []
            for i in range(len(total_outputs[0])):
                merge_outputs.append(
                    torch.stack([total_outputs[fi][i] for fi in range(frame_num)], dim=1).view(
                        B, frame_num * view_num, P, 2 * C
                    )
                )
            return merge_outputs, pos, self.patch_start_idx
