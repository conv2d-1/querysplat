import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models2.external.cut3r.dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
    PositionGetter
)  # noqa
from hAlgorithm.modules.models2.sdk.cut3r import LocalMemory

from torch.utils.checkpoint import checkpoint
from functools import partial
from hAlgorithm.modules.models2.external.cut3r.croco.models.pos_embed import RoPE2D

class Cut3rFlowEncoder(nn.Module):
    def __init__(
        self,
        state_dec_num_heads,
        state_size,
        state_pe,
        local_mem_size,
        enc_embed_dim=1024,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        mlp_ratio=4,
        patch_size=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_im2_in_dec=True,
        gradient_checkpointing=True,
        pos_embed="RoPE100",
        fixed_input_length=True,
        **kwargs
    ):
        super().__init__()
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.enc_embed_dim = enc_embed_dim
        self.dec_num_heads = dec_num_heads
        self.state_dec_num_heads = state_dec_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_im2_in_dec = norm_im2_in_dec
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.fixed_input_length=fixed_input_length
        if pos_embed is None:
            self.enc_pos_embed = None
            self.rope = None
        elif pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)
        # mix decoder
        self._set_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            self.dec_num_heads,
            self.dec_depth,
            self.mlp_ratio,
            self.norm_layer,
            self.norm_im2_in_dec,
        )
        
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            self.state_dec_num_heads,
            self.dec_depth,
            self.mlp_ratio,
            self.norm_layer,
            self.norm_im2_in_dec,
        )
        
        # local mem
        self.local_mem_size = local_mem_size
        if self.local_mem_size > 0:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=self.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
            #
        self.state_size = state_size
        self.state_pe = state_pe
        
        self.register_tokens = nn.Embedding(state_size, self.enc_embed_dim)
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.patch_start_idx = 0

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def _encode_state(self, batch_size, dtype, device):
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=dtype,
                    device=device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=dtype,
                    device=device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none" or self.state_pe is None:
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _init_state(self, batch_size, dtype, device):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(batch_size, dtype, device)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)
        if self.local_mem_size > 0:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)
        final_output.append((f_state, f_img))
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and torch.is_grad_enabled()
            ):
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
            final_output.append((f_state, f_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
    ):
        new_state_feat, dec = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec

    def _decoder_onnx(
        self,
        f_state,
        pos_state,
        f_img,
        pos_img,
        f_pose,
        pos_pose,
        hooks=[-1],
    ):
        # === 输入检查（替换 assert）===
        if f_state.shape[-1] != self.dec_embed_dim:
            raise ValueError("f_state embedding dim mismatch")

        # === 图像嵌入 ===
        f_img = self.decoder_embed(f_img)

        # === 条件拼接（保持静态图）===
        if self.local_mem_size > 0:
            if f_pose is None or pos_pose is None:
                raise ValueError("f_pose/pos_pose required when local_mem_size > 0")
            f_img = torch.cat([f_pose, f_img], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)

        # === 构建所有层输出 ===
        num_layers = len(self.dec_blocks_state)
        total_levels = num_layers + 1  # level 0: input; level 1~L: after each block

        # 标准化 hooks（支持负索引）
        normalized_hooks = []
        for h in hooks:
            if h < 0:
                h = total_levels + h
            if h < 0 or h >= total_levels:
                raise IndexError(f"Hook {h} out of range [0, {total_levels})")
            normalized_hooks.append(h)

        # 我们将收集所有需要的 (state, feat) 对
        needed_levels = set(normalized_hooks)

        # 初始化
        current_state = f_state
        current_feat = f_img

        # 存储需要的输出
        collected_states = {}
        collected_feats = {}

        # Level 0: 初始输入（未经过任何 block）
        if 0 in needed_levels:
            collected_states[0] = current_state
            collected_feats[0] = current_feat

        # 逐层处理 blocks
        for i in range(num_layers):
            blk_state = self.dec_blocks_state[i]
            blk_img = self.dec_blocks[i]

            # 前向（无 checkpoint）
            next_state, _ = blk_state(current_state, current_feat, pos_state, pos_img)
            next_feat, _ = blk_img(current_feat, current_state, pos_img, pos_state)

            current_state = next_state
            current_feat = next_feat

            level = i + 1  # 当前是第 (i+1) 层
            if level in needed_levels:
                collected_states[level] = current_state
                collected_feats[level] = current_feat

        # === 应用 LayerNorm（仅最后一层）===
        last_level = total_levels - 1
        if last_level in collected_states:
            collected_states[last_level] = self.dec_norm_state(collected_states[last_level])
        if last_level in collected_feats:
            collected_feats[last_level] = self.dec_norm(collected_feats[last_level])

        # === 按原始 hooks 顺序提取结果 ===
        hooked_states = []
        hooked_feats = []
        for h in hooks:
            idx = h if h >= 0 else total_levels + h
            hooked_states.append(collected_states[idx])
            hooked_feats.append(collected_feats[idx])

        # 返回：两个列表，每个元素对应一个 hook 层
        return hooked_states, hooked_feats

    def forward_flow(
        self,
        patch_tokens: torch.Tensor,
        pos: torch.Tensor,
        state_feat: torch.Tensor,
        state_pos: torch.Tensor,
        hooks=[-1],
    ):
        with_mem = self.local_mem_size > 0
        assert not with_mem, "Mem Flow Mode Not Supported"
        new_state_feat, output_tokens = self._decoder_onnx(
            state_feat,
            state_pos,
            patch_tokens,
            pos,
            None,
            None,
            hooks=hooks,
        )
        new_state_feat = new_state_feat[-1]
        return output_tokens, new_state_feat

    def forward(
        self, 
        patch_tokens: torch.Tensor, 
        pos: torch.Tensor, 
        meta_data,
        **kwargs
    ):
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        
        BS, P, C = patch_tokens.shape
        S = frame_num * view_num
        B = BS // S
        patch_tokens = patch_tokens.view(B, S, P, C).permute(1, 0, 2, 3).contiguous() # S,B,P,C
        if pos is not None:
            pos = pos.view(B, S, *pos.shape[-2:]).permute(1, 0, 2, 3).contiguous() # S, B, 2
        
        with_mem = self.local_mem_size > 0
        state_feat, state_pos = self._init_state(B, torch.long, patch_tokens.device)
        init_state_feat = state_feat.clone()
        if with_mem:
            mem = self.pose_retriever.mem.expand(B, -1, -1)
            init_mem = mem.clone()
        
        
        # during training, update for every frame, never reset
        img_masks = torch.ones(S, B).to(patch_tokens.device).bool()
        reset_masks = torch.zeros(S, B).to(patch_tokens.device).bool()
        update_masks = torch.ones(S, B).to(patch_tokens.device).bool()
        
        head_inputs = [None, None, None, None]
        for i in range(S):
            feat_i = patch_tokens[i]
            pos_i = pos[i] if pos is not None else None
            
            if with_mem:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = pose_pos_i = None
            
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            if with_mem:
                new_mem = self.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )
            assert len(dec) == self.dec_depth + 1
            
            if with_mem:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4].float(),
                    dec[self.dec_depth * 3 // 4].float(),
                    dec[self.dec_depth].float(),
                ]
            if i == 0:
                for _hook in range(len(head_input)):
                    head_inputs[_hook] = [head_input[_hook]]
            else:
                for _hook in range(len(head_input)):
                    head_inputs[_hook].append(head_input[_hook])
            img_mask = img_masks[i]
            update = update_masks[i]
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            if with_mem:
                mem = new_mem * update_mask + mem * (
                    1 - update_mask
                )  # then update local state
                reset_mask = reset_masks[i]
                if reset_mask is not None:
                    reset_mask = reset_mask[:, None, None].float()
                    state_feat = init_state_feat * reset_mask + state_feat * (
                        1 - reset_mask
                    )
                    mem = init_mem * reset_mask + mem * (1 - reset_mask)
        
        head_inputs = [
            torch.stack(outputs, dim=1) for outputs in head_inputs
        ]# B,S,P,C
        if pos is not None:
            pos = pos.permute(1, 0, 2, 3)
        return head_inputs, pos, self.patch_start_idx

class PromptCut3rFlowEncoder(Cut3rFlowEncoder):
    def __init__(
        self,
        prompt_in_channel=0,
        prompt_mode='add',
        zero_conv=False,
        **kwargs
    ):
        super(PromptCut3rFlowEncoder, self).__init__(**kwargs)
        self.prompt_in_channel = prompt_in_channel
        self.prompt_mode = prompt_mode
        if prompt_in_channel > 0:
            patch_hw = (self.patch_size, self.patch_size)
            
            self.prompt_patch_embed = nn.Conv2d(
                in_channels=prompt_in_channel, 
                out_channels=prompt_in_channel,
                kernel_size=patch_hw, 
                stride=patch_hw
            )
            if self.prompt_mode in ['add']:
                self.prompt_proj = nn.Linear(prompt_in_channel, self.enc_embed_dim)
            elif self.prompt_mode in ['cat', 'concat']:
                self.prompt_proj = nn.Linear(prompt_in_channel+self.enc_embed_dim, self.enc_embed_dim)
            else:
                raise NotImplementedError()
        self.zero_conv = zero_conv
        if self.zero_conv:
            self.zero_init_proj()

    def zero_init_proj(self):
        if self.prompt_in_channel > 0 and self.prompt_mode in ['add']:
            nn.init.constant_(self.prompt_proj.weight, 0)
            nn.init.constant_(self.prompt_proj.bias, 0)

    def fuse_prompt(
        self, 
        patch_tokens: torch.Tensor, 
        image_h,
        image_w,
        prompt_depth : torch.Tensor = None ,
        **kwargs
    ):
        if prompt_depth is not None and self.prompt_in_channel > 0:
            prompt_depth = F.interpolate(prompt_depth, size=(image_h, image_w), mode='bilinear', align_corners=True)
            prompt_depth = self.prompt_patch_embed(prompt_depth)
            prompt_depth = prompt_depth.flatten(2).transpose(1, 2) # B, P, C_prompt
            if self.prompt_mode in ['add']:
                prompt_depth = self.prompt_proj(prompt_depth)
                patch_tokens = torch.add(patch_tokens, prompt_depth)
            elif self.prompt_mode in ['cat', 'concat']:
                patch_tokens = torch.concat(patch_tokens, prompt_depth, dim=-1)
                patch_tokens = self.prompt_proj(patch_tokens)
            else:
                raise NotImplementedError()
        return patch_tokens

    def forward(
        self, 
        patch_tokens: torch.Tensor, 
        pos: torch.Tensor, 
        meta_data,
        prompt_depth : torch.Tensor = None ,
        **kwargs
    ):
        image_h, image_w = meta_data["input_height"][0], meta_data["input_width"][0]
        patch_tokens = self.fuse_prompt(patch_tokens, image_h=image_h, image_w=image_w, prompt_depth=prompt_depth)
        return super().forward(patch_tokens, pos, meta_data, **kwargs)

    def forward_flow(
        self,
        patch_tokens: torch.Tensor,
        pos: torch.Tensor,
        state_feat: torch.Tensor,
        state_pos: torch.Tensor,
        image_h, 
        image_w,
        hooks=[-1],
        prompt_depth: torch.Tensor = None,
    ):
        patch_tokens = self.fuse_prompt(patch_tokens, image_h=image_h, image_w=image_w, prompt_depth=prompt_depth)
        return super().forward_flow(patch_tokens, pos, state_feat, state_pos, hooks=hooks)

class DinoFlowEncoder(PromptCut3rFlowEncoder):
    def __init__(
        self,
        dino_layers=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.dino_layers = dino_layers if isinstance(dino_layers, (list, tuple)) else [dino_layers]
        if self.rope is not None:
            self.position_getter = PositionGetter()

    def forward(self, dino_tokens, meta_data, prompt_depth = None, **kwargs):
        # dino tokens should be a list of tokens
        assert isinstance(dino_tokens, (list, tuple))
        patch_w = meta_data["input_width"][0] // self.patch_size
        patch_h = meta_data["input_height"][0] // self.patch_size
        frame_num = meta_data["frames"][0]
        view_num = meta_data["views"][0]
        BS, P, C = dino_tokens[-1].shape
        S = frame_num * view_num
        B = BS // S
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, patch_h, patch_w, device=dino_tokens[-1].device)
        
        out = []
        for dino_idx, patch_tokens in enumerate(dino_tokens):
            if dino_idx in self.dino_layers:
                out_token, _, _ = super().forward(patch_tokens, pos, meta_data, prompt_depth, **kwargs)
                if self.local_mem_size > 0:
                    out_token = out_token[-1][1:]
                else:
                    out_token = out_token[-1]
                out.append(out_token.reshape(BS, *out_token.shape[-2:]).contiguous())
            else:
                out.append(patch_tokens)
        return out, pos, 0

    def forward_flow(
        self, 
        dino_tokens: torch.Tensor,
        pos: torch.Tensor,
        state_feat: torch.Tensor,
        state_pos: torch.Tensor,
        image_h, 
        image_w,
        hooks=[-1],
        prompt_depth: torch.Tensor = None,
    ):
        out = []
        update_states = []
        for dino_idx, patch_tokens in enumerate(dino_tokens):
            if dino_idx in self.dino_layers:
                out_token, update_state = super().forward_flow(
                    patch_tokens, 
                    pos, 
                    state_feat, 
                    state_pos, 
                    image_h, 
                    image_w,
                    hooks=hooks,
                    prompt_depth=prompt_depth,
                )
                if self.local_mem_size > 0:
                    out_token = out_token[-1][1:]
                else:
                    out_token = out_token[-1]
                out.append(out_token)
                update_states.append(update_state)
            else:
                out.append(patch_tokens)
        return out, update_states[-1]

