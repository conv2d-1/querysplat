import torch
import torch.nn as nn

from .unimatch.utils import merge_splits, split_feature


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** 0.5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def single_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C] for 2-view
    # for multi-view cross-attention, q: [B, L, C], k, v: [B, N-1, L, C]

    # multi(>2)-view corss-attention
    if not (q.dim() == k.dim() == v.dim() == 3):
        assert k.dim() == v.dim() == 4
        assert h is not None and w is not None
        assert q.size(1) == h * w

        m = k.size(1)  # m + 1 is num of views

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, m, h, w, c)  # [B, N-1, H, W, C]
        v = v.view(b, m, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))

        q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
        k = split_feature(
            k.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]
        v = split_feature(
            v.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]

        k = (
            k.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 3, 1, 2, 4)
            .reshape(b_new, c, -1)
        )  # [B*K*K, C, H/K*W/K*(N-1)]
        v = (
            v.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 1, 2, 4, 3)
            .reshape(b_new, -1, c)
        )  # [B*K*K, H/K*W/K*(N-1), C]

        scores = (
            torch.matmul(q.view(b_new, -1, c), k) / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K*(N-1)]

        if with_shift:
            scores += attn_mask.repeat(b, 1, m)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)
    else:
        # 2-view self-attention or cross-attention
        assert q.dim() == k.dim() == v.dim() == 3

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

        q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
        k = split_feature(k, num_splits=num_splits, channel_last=True)
        v = split_feature(v, num_splits=num_splits, channel_last=True)

        scores = (
            torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K]

        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)

    return out


def multi_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
    num_head=1,
):
    """Multi-head scaled dot-product attention
    Args:
        q: [N, L, D]
        k: [N, S, D]
        v: [N, S, D]
    Returns:
        out: (N, L, D)
    """

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    assert c % num_head == 0

    scale_factor = (c // num_head) ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits)
    v = split_feature(v, num_splits=num_splits)

    # multi-head attn
    q = q.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)  # [B, N, H*W, C]
    k = k.view(b_new, -1, num_head, c // num_head).permute(0, 2, 3, 1)  # [B, N, C, H*W]
    scores = torch.matmul(q, k) / scale_factor  # [B*K*K, N, H/K*W/K, H/K*W/K]

    if with_shift:
        scores += attn_mask.unsqueeze(1).repeat(b, num_head, 1, 1)

    attn = torch.softmax(scores, dim=-1)  # [B*K*K, N, H/K*W/K, H/K*W/K]

    out = torch.matmul(
        attn, v.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)
    )  # [B*K*K, N, H/K*W/K, C]

    out = merge_splits(
        out.permute(0, 2, 1, 3).reshape(b_new, h // num_splits, w // num_splits, c),
        num_splits=num_splits,
    )  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        no_ffn=False,
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        **kwargs,
    ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.add_per_view_attn = add_per_view_attn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        if "attn_type" in kwargs:
            attn_type = kwargs["attn_type"]
        else:
            attn_type = self.attention_type

        # source, target: [B, L, C] for 2-view
        # for multi-view cross-attention, source: [B, L, C], target: [B, N-1, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C] or [B, N-1, L, C]
        value = self.v_proj(value)  # [B, L, C] or [B, N-1, L, C]

        if attn_type == "swin" and attn_num_splits > 1:
            if self.nhead > 1:
                message = multi_head_split_window_attention(
                    query,
                    key,
                    value,
                    num_splits=attn_num_splits,
                    with_shift=self.with_shift,
                    h=height,
                    w=width,
                    attn_mask=shifted_window_attn_mask,
                    num_head=self.nhead,
                )
            else:
                if self.add_per_view_attn:
                    assert query.dim() == 3 and key.dim() == 4 and value.dim() == 4
                    b, l, c = query.size()
                    query = query.unsqueeze(1).repeat(1, key.size(1), 1, 1)  # [B, N-1, L, C]
                    query = query.view(-1, l, c)  # [B*(N-1), L, C]
                    key = key.view(-1, l, c)
                    value = value.view(-1, l, c)
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
                    # [B, L, C]  # add per view attn
                    message = message.view(b, -1, l, c).sum(1)
                else:
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        self.no_cross_attn = no_cross_attn

        if no_cross_attn:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )
        else:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                no_ffn=True,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
            )

            self.cross_attn_ffn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(
            source,
            source,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        if self.no_cross_attn:
            return source

        # cross attention and ffn
        source = self.cross_attn_ffn(
            source,
            target,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        return source


def batch_features(features):
    num_views = len(features)
    stacked = torch.stack(features, dim=0)  # [N, B, ...]

    q_list = []
    kv_list = []

    for i in range(num_views):
        # 提取当前视角作为查询（q）
        q_i = stacked[i]  # 形状: [B, ...]

        # 构造键值对（kv），排除当前视角
        mask = torch.ones(num_views, dtype=torch.bool)
        mask[i] = False
        kv_i = stacked[mask]  # 形状: [N-1, B, ...]

        # 调整kv_i的维度为 [B, N-1, ...]
        kv_i = kv_i.transpose(0, 1)  # 转换前两个维度

        q_list.append(q_i)
        kv_list.append(kv_i)

    # 拼接所有查询和键值对
    q = torch.cat(q_list, dim=0)  # 形状: [N*B, ...]
    kv = torch.cat(kv_list, dim=0)  # 形状: [N*B, N-1, ...]

    return q, kv


class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=128,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(MultiViewFeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=(True if attention_type == "swin" and i % 2 == 1 else False),
                    add_per_view_attn=add_per_view_attn,
                    no_cross_attn=no_cross_attn,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # zero init layers beyond 6
        if num_layers > 6:
            for i in range(6, num_layers):
                self.layers[i].self_attn.norm1.weight.data.zero_()
                self.layers[i].self_attn.norm1.bias.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.weight.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.bias.data.zero_()

    def forward(
        self,
        multi_view_features,
        attn_num_splits=1,
        **kwargs,
    ):
        # multi_view_features: list of [B, C, H, W]
        b, c, h, w = multi_view_features[0].shape
        assert self.d_model == c

        num_views = len(multi_view_features)

        if self.attention_type == "swin" and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=multi_view_features[0].device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # [N*B, C, H, W], [N*B, N-1, C, H, W]
        concat0, concat1 = batch_features(multi_view_features)
        concat0 = concat0.reshape(num_views * b, c, -1).permute(0, 2, 1)  # [N*B, H*W, C]
        concat1 = concat1.reshape(num_views * b, num_views - 1, c, -1).permute(
            0, 1, 3, 2
        )  # [N*B, N-1, H*W, C]

        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                height=h,
                width=w,
                shifted_window_attn_mask=shifted_window_attn_mask,
                attn_num_splits=attn_num_splits,
            )

            if i < len(self.layers) - 1:
                # list of features
                features = list(concat0.chunk(chunks=num_views, dim=0))
                # [N*B, H*W, C], [N*B, N-1, H*W, C]
                concat0, concat1 = batch_features(features)

        features = concat0.chunk(chunks=num_views, dim=0)
        features = [f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous() for f in features]

        return features


class DinoFusionBlockTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mf_frame_num,
        use_clstoken=False,
        only_last_frame=True,
        use_attn=True,
        temporal_feat_channel=256,
        num_layers=6,
        nhead=1,
        no_cross_attn=False,
        attn_num_splits=1,
        **kwargs,
    ):
        super().__init__()

        self.only_last_frame = only_last_frame
        self.use_attn = use_attn
        self.attn_num_splits = attn_num_splits
        self.use_clstoken = use_clstoken

        if self.use_attn:
            self.temporal_conv0 = nn.Conv2d(
                in_channels,
                temporal_feat_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.temporal_conv1 = nn.Conv2d(
                temporal_feat_channel,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.mf_transformer = MultiViewFeatureTransformer(
                num_layers=num_layers,
                d_model=temporal_feat_channel,
                nhead=nhead,
                no_cross_attn=no_cross_attn,
                **kwargs,
            )

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )
        self.mf_frame_num = mf_frame_num
        self.hidden = []

    def infer_flow(self, feature_list, patch_h, patch_w):
        assert len(feature_list) == 1
        feat_flow = []
        self.hidden = self.hidden[-self.mf_frame_num + 1 :]
        self.hidden.extend(feature_list)
        feat_flow.extend(self.hidden)
        while len(feat_flow) < self.mf_frame_num:
            feat_flow.extend(feature_list)
        return self.forward(feat_flow, patch_h, patch_w)

    def temporal_feature(self, feature_list, batch_num):
        """
        feature_list: [B,C,H,W] * T
        """
        attn_feat = []
        # attn_feat: [[1,C,H,W]*T]*B
        for b in range(batch_num):
            temporal_feat = [out[b].unsqueeze(0) for out in feature_list]
            # temporal_feat: [1,C,H,W]*T
            attn_feat.append(
                self.mf_transformer(
                    temporal_feat,
                    attn_num_splits=self.attn_num_splits,
                )
            )
        for i, _ in enumerate(feature_list):
            feature_list[i] = torch.concat([f[i] for f in attn_feat])
            feature_list[i] = self.temporal_conv1(feature_list[i])
        # feature_list: [B,C,H,W] * T

    def forward(self, feature_list, patch_h, patch_w):
        """
        - feature_list: [[[(B,H,W),(B,H,W)]*4] * T

        Returns:
        -out_list: [[B,C,H,W]*4]
        """
        out_list = []  # [[B,C,H,W]*4]*T
        for feature in feature_list:
            out = []
            for i, x in enumerate(feature):
                if self.use_clstoken:
                    x, cls_token = x[0], x[1]
                    readout = cls_token.unsqueeze(1).expand_as(x)
                    x = self.readout_projects[i](torch.cat((x, readout), -1))
                else:
                    x = x[0]

                x = (
                    x.permute(0, 2, 1)
                    .reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
                    .contiguous()
                )

                out.append(x)
            out_list.append(out)

        fusion_feat_list = []
        # fusion_feat_list: [4*B,C,H,W] * T
        for i, feature in enumerate(out_list):
            # feature: [B,C,H,W] * 4
            if self.use_attn:
                out_feat = torch.concat(feature)  # out_feat: [4*B,C,H,W]
                out_feat = self.temporal_conv0(out_feat)
                fusion_feat_list.append(out_feat)
            else:
                fusion_feat_list.append(torch.concat(feature))

        if self.use_attn:
            attn_num = fusion_feat_list[0].shape[0]
            self.temporal_feature(fusion_feat_list, batch_num=attn_num)

        chunk_size = len(feature_list[0])
        if self.only_last_frame:
            fusion_feat_list = [fusion_feat_list[-1].chunk(chunk_size)]  # [((B,C,H,W)*4)]
        else:
            fusion_feat_list = [
                feat.chunk(chunk_size) for feat in fusion_feat_list
            ]  # [((B,C,H,W)*4)] * T

        out_list = []
        for feat in fusion_feat_list:
            out = []
            for i, x in enumerate(feat):
                x = self.projects[i](x)
                x = self.resize_layers[i](x)
                out.append(x)
            out_list.append(out)

        return out_list


class FeatureFusionBlockTransformer(nn.Module):
    def __init__(
        self,
        head_features_1,
        mf_frame_num,
        only_last_frame=True,
        use_attn=True,
        temporal_feat_channel=128,
        num_layers=6,
        nhead=1,
        no_cross_attn=False,
        attn_num_splits=1,
        **kwargs,
    ):
        super().__init__()
        self.only_last_frame = only_last_frame
        self.use_attn = use_attn
        self.attn_num_splits = attn_num_splits

        if self.use_attn:
            self.temporal_conv0 = nn.Conv2d(
                head_features_1,
                temporal_feat_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.temporal_conv1 = nn.Conv2d(
                temporal_feat_channel,
                head_features_1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.mf_transformer = MultiViewFeatureTransformer(
                num_layers=num_layers,
                d_model=temporal_feat_channel,
                nhead=nhead,
                no_cross_attn=no_cross_attn,
                **kwargs,
            )
        self.mf_frame_num = mf_frame_num
        self.hidden = []

    def infer_flow(self, feature_list):
        assert len(feature_list) == 1
        feat_flow = []
        self.hidden = self.hidden[-self.mf_frame_num + 1 :]
        self.hidden.extend(feature_list)
        feat_flow.extend(self.hidden)
        while len(feat_flow) < self.mf_frame_num:
            feat_flow.extend(feature_list)
        return self.forward(feat_flow)

    def temporal_feature(self, feature_list, batch_num):
        """
        feature_list: [B,C,H,W] * T
        """
        attn_feat = []
        # attn_feat: [[1,C,H,W]*T]*B
        for b in range(batch_num):
            temporal_feat = [out[b].unsqueeze(0) for out in feature_list]
            # temporal_feat: [1,C,H,W]*T
            attn_feat.append(
                self.mf_transformer(
                    temporal_feat,
                    attn_num_splits=self.attn_num_splits,
                )
            )
        for i, _ in enumerate(feature_list):
            feature_list[i] = torch.concat([f[i] for f in attn_feat])
            feature_list[i] = self.temporal_conv1(feature_list[i])
        # feature_list: [B,C,H,W] * T

    def forward(self, out_list):
        # out_list: [B,C,H,W] * T
        if self.use_attn:
            batch_num = out_list[0].shape[0]
            attn_feat = []
            # attn_feat: [[1,C,H,W]*T]*B
            for b in range(batch_num):
                temporal_feat = [out[b].unsqueeze(0) for out in out_list]
                # temporal_feat: [1,C,H,W]*T
                attn_feat.append(
                    self.mf_transformer(
                        temporal_feat,
                        attn_num_splits=self.attn_num_splits,
                    )
                )
            for i, _ in enumerate(out_list):
                out_list[i] = torch.concat([f[i] for f in attn_feat])
                out_list[i] = self.temporal_conv1(out_list[i])
            # out_list: [B,C,H,W] * T
        if self.only_last_frame:
            out_list = [out_list[-1]]
        return out_list
