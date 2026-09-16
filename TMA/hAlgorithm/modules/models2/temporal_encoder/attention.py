import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope.float() if rope is not None else None

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]  # each: (B, num_heads, N, head_dim)

        q_type = q.dtype
        k_type = k.dtype
        if self.rope is not None:
            q = q.float()
            k = k.float()
            with torch.autocast(device_type="cuda", enabled=False):
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)
            q = q.to(q_type)
            k = k.to(k_type)

        # === 替换 scaled_dot_product_attention 为显式计算 ===
        q = q * self.scale  # 缩放查询
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 在 eval 模式下自动跳过

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):

    def __init__(
        self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope.float() if rope is not None else None

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = (
            self.projq(query)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (B, num_heads, Nq, head_dim)
        k = (
            self.projk(key)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (B, num_heads, Nk, head_dim)
        v = (
            self.projv(value)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # (B, num_heads, Nv, head_dim)

        q_type = q.dtype
        k_type = k.dtype
        if self.rope is not None:
            if qpos is not None:
                q = q.float()
                with torch.autocast(device_type="cuda", enabled=False):
                    q = self.rope(q, qpos)
                q = q.to(q_type)

            if kpos is not None:
                k = k.float()
                with torch.autocast(device_type="cuda", enabled=False):
                    k = self.rope(k, kpos)
                k = k.to(k_type)

        q = q * self.scale  # 等价于在 attn score 上除 sqrt(d)
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, Nq, Nk)

        # Softmax + Dropout
        attn = attn.softmax(dim=-1)
        
        # ⚠️ 注意：在 eval 模式下，Dropout 自动禁用，ONNX 导出安全
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x