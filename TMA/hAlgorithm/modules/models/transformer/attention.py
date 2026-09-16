import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWindowAttention(nn.Module):
    def __init__(
        self,
        original_attn,
        input_resolution,
        window_size=7,
        shift_size=3,
        with_cls_token=True,
        with_multi_frames=False,
    ):
        """Adaptive Window-based Multi-head Self Attention Module.

        This attention mechanism adapts an existing attention module to use window-based attention
        while preserving its original projection matrices. It's particularly useful for modifying
        pre-trained attention modules to use window-based attention without retraining from scratch.

        Args:
            original_attn (nn.Module): Original attention module to get projection matrices from
            input_resolution (tuple[int]): Input resolution (H, W)
            window_size (int): Window size for local attention
            shift_size (int): Shift size for shifted window attention
        """
        super().__init__()
        # Keep original projection matrices
        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.attn_drop = original_attn.attn_drop

        # Window attention parameters
        self.H, self.W = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale

        self.with_cls_token = with_cls_token
        self.with_multi_frames = with_multi_frames

        # Cache for attention masks
        self.attn_mask_dict = {}

    def window_partition(self, x, window_size):
        """Partition feature map into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows, B

    def window_reverse(self, windows, window_size, H, W, B):
        """Reverse window partition."""
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def get_attn_mask(self, H, W, device):
        """Generate attention mask for shifted window attention."""
        key = f"{H}_{W}_{self.window_size}_{self.shift_size}"
        if key in self.attn_mask_dict:
            return self.attn_mask_dict[key]

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows, _ = self.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        self.attn_mask_dict[key] = attn_mask
        return attn_mask

    def forward(self, x, pos=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor of shape [B, L, C]

        Returns:
            Tensor: Output tensor of shape [B, L, C]
        """
        B, L, C = x.shape

        # Handle cls token
        if self.with_cls_token:
            cls_token, x = x[:, 0:1, :], x[:, 1:, :]  # Split cls_token and patch tokens
            assert self.H * self.W == (
                L - 1
            ), f"Input feature size {L-1} doesn't match H*W ({self.H}*{self.W})"

        if self.with_multi_frames:
            assert not self.with_cls_token
            assert (
                L % (self.H * self.W) == 0
            ), f"Input feature size {L} doesn't match T*H*W ({self.H}*{self.W})"
            # Reshape to image format
            T = L // (self.H * self.W)
        else:
            T = 1

        # Reshape to image format
        x = x.view(B, T * self.H, self.W, C)

        # Add padding if needed
        pad_h = (self.window_size - self.H % self.window_size) % self.window_size
        pad_w = (self.window_size - self.W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # Determine whether to use shifted windows
        _, pad_H, pad_W, _ = x.shape
        shift_size = self.shift_size if min(pad_H, pad_W) > self.window_size else 0

        # Apply cyclic shift and get attention mask if needed
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
            attn_mask = self.get_attn_mask(pad_H, pad_W, x.device)
        else:
            attn_mask = None

        # Window partition
        x_windows, orig_B = self.window_partition(x, self.window_size)

        # Multi-head self attention
        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(
            -1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add attention mask for shifted windows
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(
                orig_B,
                nW,
                self.num_heads,
                self.window_size * self.window_size,
                self.window_size * self.window_size,
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(
                -1,
                self.num_heads,
                self.window_size * self.window_size,
                self.window_size * self.window_size,
            )

        # Apply softmax and dropout
        attn = self.attn_drop(attn.softmax(dim=-1))

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj_drop(self.proj(x))

        # Reverse window partition
        x = self.window_reverse(x, self.window_size, pad_H, pad_W, orig_B)

        # Reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, : T * self.H, : self.W, :].contiguous()

        # Reshape and concatenate with cls token
        x = x.view(B, T * self.H * self.W, C)

        if self.with_cls_token:
            x = torch.cat([cls_token, x], dim=1)

        return x


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim=None, dropout=0.0, bias=True):
        """
        Group Query Attention implementation.

        Args:
            hidden_size: Total hidden size dimension.
            num_heads: Number of query heads.
            num_kv_heads: Number of key/value heads (must be divisible into num_heads).
            head_dim: Dimension of each attention head. If None, computed as hidden_size // num_heads.
            dropout: Dropout probability for attention weights.
            bias: Whether to use bias in projection layers.
        """
        super().__init__()

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Determine head dimension (embedding dimension per head)
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim

        # Total dimensions for query projection and for key/value projection
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Query projection
        self.q_proj = nn.Linear(hidden_size, self.q_dim, bias=bias)
        # Combined key and value projection; output dimension is 2 * kv_dim:
        # first half for keys, second half for values.
        self.kv_proj = nn.Linear(hidden_size, 2 * self.kv_dim, bias=bias)
        # Output projection
        self.o_proj = nn.Linear(self.q_dim, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _repeat_kv_for_query_groups(self, k, v):
        """
        Repeat key and value tensors for query groups.

        Args:
            k: Keys tensor of shape [B, N, num_kv_heads, head_dim].
            v: Values tensor of shape [B, N, num_kv_heads, head_dim].

        Returns:
            k_expanded, v_expanded: Tensors of shape [B, N, num_heads, head_dim].
        """
        B, N, num_kv_heads, head_dim = k.shape

        # Expand keys and values for each query in the group
        k_expanded = k.unsqueeze(3).expand(B, N, num_kv_heads, self.num_queries_per_kv, head_dim)
        v_expanded = v.unsqueeze(3).expand(B, N, num_kv_heads, self.num_queries_per_kv, head_dim)

        # Reshape to [B, N, num_heads, head_dim]
        k_expanded = k_expanded.reshape(B, N, self.num_heads, head_dim)
        v_expanded = v_expanded.reshape(B, N, self.num_heads, head_dim)

        return k_expanded, v_expanded

    def forward(self, x, attention_bias=None):
        """
        Forward pass for Group Query Attention using F.scaled_dot_product_attention.

        Args:
            x: Input tensor of shape [B, N, hidden_size].
            attention_mask: Optional mask tensor of shape [B, 1, 1, N] or [B, 1, N, N].
            causal_mask: Whether to apply causal masking for autoregressive models.

        Returns:
            Output tensor of shape [B, N, hidden_size].
        """
        B, N, _ = x.shape

        # Compute query projection and reshape to [B, N, num_heads, head_dim]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Compute combined key and value projections and reshape:
        # kv has shape [B, N, 2 * kv_dim] -> reshape to [B, N, 2, num_kv_heads, head_dim]
        kv = self.kv_proj(x).view(B, N, 2, self.num_kv_heads, self.head_dim)
        k, v = torch.unbind(kv, dim=2)  # Each of shape [B, N, num_kv_heads, head_dim]

        # Expand keys and values to match the number of query heads
        k, v = self._repeat_kv_for_query_groups(k, v)

        # Transpose tensors to [B, num_heads, N, head_dim] for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot product attention using the built-in function.
        # Note: dropout probability is taken from self.dropout.p.
        attn_out = F.scaled_dot_product_attention(q, k, v, attention_bias)

        # Reshape output back to [B, N, q_dim] and project back to hidden_size
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.q_dim)
        output = self.o_proj(attn_out)

        return output


class MultiQueryAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, num_kv_heads=1, head_dim=None, dropout=0.0, bias=True
    ):
        """
        Multi-Query Attention implementation

        Args:
            hidden_size: Total hidden size dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads (default: 1)
            head_dim: Dimension of each attention head. If None, computed as hidden_size // num_heads
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in projection layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Determine head dimension (embedding dimension per head)
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim

        # Total dimensions for Q, K, V
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.head_dim  # Only one KV head

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.q_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=bias)
        self.o_proj = nn.Linear(self.q_dim, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = 1 / math.sqrt(self.head_dim)

    def _repeat_kv_for_query_heads(self, k, v):
        """
        Repeat single key and value head for all query heads

        Args:
            k: Keys tensor of shape [batch_size, seq_len, head_dim]
            v: Values tensor of shape [batch_size, seq_len, head_dim]

        Returns:
            k_expanded: Keys repeated to match query heads
            v_expanded: Values repeated to match query heads
        """
        batch_size, seq_len, head_dim = k.shape

        # Expand single KV head to all query heads
        # Shape: [batch_size, seq_len, num_heads, head_dim]
        k_expanded = k.unsqueeze(2).expand(batch_size, seq_len, self.num_heads, head_dim)
        v_expanded = v.unsqueeze(2).expand(batch_size, seq_len, self.num_heads, head_dim)

        return k_expanded, v_expanded

    def forward(self, x, attention_mask=None, causal_mask=True):
        """
        Forward pass for Multi-Query Attention

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
            causal_mask: Whether to apply causal masking for autoregressive models

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.head_dim)  # Single K head
        v = self.v_proj(x).view(batch_size, seq_len, self.head_dim)  # Single V head

        # Expand k and v to match number of query heads
        k, v = self._repeat_kv_for_query_heads(k, v)

        # Transpose for batched matrix multiplication
        # [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale

        # Apply causal mask if needed (for autoregressive models)
        if causal_mask:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
            )
            attn_scores.masked_fill_(causal_mask, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)

        # Reshape and project back to hidden size
        # [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.q_dim)
        output = self.o_proj(output)

        return output


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, latent_dim=64, head_dim=None, dropout=0.0, bias=True
    ):
        """
        Standard Multi-Head Attention implementation

        Args:
            hidden_size: Total hidden size dimension
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head. If None, computed as hidden_size // num_heads
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in projection layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # Determine head dimension (embedding dimension per head)
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim

        # Total dimensions for Q, K, V
        self.dim = self.num_heads * self.head_dim

        # Linear projections for queries, keys, and values
        self.qkv_latent = nn.Linear(hidden_size, self.latent_dim, bias=bias)
        self.qkv = nn.Linear(self.latent_dim, 3 * self.dim, bias=bias)

        # Output projection
        self.proj = nn.Linear(self.dim, hidden_size, bias=bias)

        # Dropout layer
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        """
        Forward pass for Multi-Head Latent Attention

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, 1, 1, seq_len]
            causal_mask: Whether to apply causal masking (rarely used with MLA)

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        B, N, C = x.shape

        # Original QKV projection
        qkv = self.qkv(self.qkv_latent(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # Extract q, k, v but keep k, v with only 1 head
        # For MQA: multiple query heads, but only one key and value head
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, H, N, C//H)
        k = (
            qkv[:, :, 1, 0:1].permute(0, 2, 1, 3).expand(-1, self.num_heads, -1, -1)
        )  # Expand single k head
        v = (
            qkv[:, :, 2, 0:1].permute(0, 2, 1, 3).expand(-1, self.num_heads, -1, -1)
        )  # Expand single v head

        # Using SDPA for efficient computation
        x = F.scaled_dot_product_attention(q, k, v, attn_bias)

        # Reshape back to original dimensions
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim=None, dropout=0.0, bias=True):
        """
        Linear Attention implementation that computes attention as Q*(K^T*V) directly
        to reduce memory and computational complexity from O(N^2) to O(N).

        Args:
            hidden_size: Total hidden size dimension
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head. If None, computed as hidden_size // num_heads
            dropout: Dropout probability
            bias: Whether to use bias in projection layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Determine head dimension
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim
        self.dim = self.num_heads * self.head_dim

        # Linear projections
        self.q_proj = nn.Linear(hidden_size, self.dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.dim, bias=bias)
        self.o_proj = nn.Linear(self.dim, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, x, attn_bias=None):
        """
        Forward pass computing Q*(K^T*V) directly.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attn_bias: Optional attention bias (not used in linear attention)

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        B, N, C = x.shape

        # Project to queries, keys and values
        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.k_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, N, D]
        v = self.v_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, H, N, D]

        # Scale q and k
        q = q * self.scale

        # Compute K^T * V first [B, H, D, N] @ [B, H, N, D] -> [B, H, D, D]
        kv = torch.matmul(k.transpose(-2, -1), v)

        # Then compute Q * (K^T * V) [B, H, N, D] @ [B, H, D, D] -> [B, H, N, D]
        out = torch.matmul(q, kv)

        # Reshape and project back to hidden_size
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim)
        out = self.o_proj(self.dropout(out))

        return out


class MultiHeadLatentAttentionOnnx(nn.Module):
    def __init__(
        self, hidden_size, num_heads, latent_dim=64, head_dim=None, dropout=0.0, bias=True
    ):
        """
        标准的多头注意力实现，并经过修改以支持ONNX导出

        参数:
            hidden_size: 总的隐藏层维度
            num_heads: 注意力头的数量
            latent_dim: 中间投影维度
            head_dim: 每个头的维度。如果为None，则自动计算为 hidden_size // num_heads
            dropout: 注意力投影后的dropout概率
            bias: 是否在全连接层中使用bias
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # 确定每个头的维度
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim

        # 总维度
        self.dim = self.num_heads * self.head_dim

        # 先对输入进行latent映射，再映射到q, k, v
        self.qkv_latent = nn.Linear(hidden_size, self.latent_dim, bias=bias)
        self.qkv = nn.Linear(self.latent_dim, 3 * self.dim, bias=bias)

        # 输出投影
        self.proj = nn.Linear(self.dim, hidden_size, bias=bias)

        # dropout层
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        """
        前向传播

        参数:
            x: [batch_size, seq_len, hidden_size] 的输入张量
            attn_bias: (可选) [batch_size, num_heads, seq_len, seq_len] 的注意力偏置

        返回:
            [batch_size, seq_len, hidden_size] 的输出张量
        """
        B, N, C = x.shape

        # 先做latent映射，再做q, k, v的映射，并reshape为 [B, N, 3, num_heads, head_dim]
        qkv = self.qkv_latent(x)
        qkv = self.qkv(qkv).reshape(B, N, 3, self.num_heads, self.head_dim)

        # 对于多头查询，但是只使用一个key和value头（然后在头维度上扩展）
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = (
            qkv[:, :, 1, 0:1].permute(0, 2, 1, 3).expand(-1, self.num_heads, -1, -1)
        )  # (B, num_heads, N, head_dim)
        v = (
            qkv[:, :, 2, 0:1].permute(0, 2, 1, 3).expand(-1, self.num_heads, -1, -1)
        )  # (B, num_heads, N, head_dim)

        d_k = self.head_dim
        # 手动实现缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, num_heads, N, N)
        if attn_bias is not None:
            scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)

        # 还原为原始形状
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AdaptiveLinearAttention(nn.Module):
    def __init__(self, original_attn):
        """Adaptive Linear Attention Module.

        This attention mechanism adapts an existing attention module to use linear attention
        while preserving its original projection matrices. It's particularly useful for modifying
        pre-trained attention modules to use linear attention without retraining from scratch.

        Args:
            original_attn (nn.Module): Original attention module to get projection matrices from
        """
        super().__init__()
        # Keep original projection matrices
        self.qkv = original_attn.qkv
        self.proj = original_attn.proj
        self.proj_drop = original_attn.proj_drop
        self.attn_drop = original_attn.attn_drop

        # Attention parameters
        self.num_heads = original_attn.num_heads
        self.scale = original_attn.scale

    def forward(self, x, attn_bias=None):
        """
        Forward pass computing Q*(K^T*V) directly.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attn_bias: Optional attention bias (not used in linear attention)

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        B, N, C = x.shape

        # Project to queries, keys and values
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each of shape [B, H, N, D]

        # Scale q and k
        q = q * self.scale
        k = k * self.scale

        # Compute K^T * V first [B, H, D, N] @ [B, H, N, D] -> [B, H, D, D]
        # Add numerical stability by normalizing k and v
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        k = k / (k_norm + 1e-6)
        v = v / (v_norm + 1e-6)

        kv = torch.matmul(k.transpose(-2, -1), v)

        # Then compute Q * (K^T * V) [B, H, N, D] @ [B, H, D, D] -> [B, H, N, D]
        # Add numerical stability by normalizing q
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q = q / (q_norm + 1e-6)

        out = torch.matmul(q, kv)

        # Reshape and project back to hidden_size
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))

        return out
