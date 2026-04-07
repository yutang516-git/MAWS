import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from networkx.utils.misc import groups

from ...common import Adapter,MOEAdapter

class AdapterFusionBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        args,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            # adapter_scale = 0.1

        )

        if(args.mid_dim != None):
            adapter_dim = args.mid_dim
        else:
            adapter_dim = dim

        self.MLPx_Adapter = Adapter(adapter_dim, skip_connect=False)  # MLP-adapter, no skip connection
        self.MLPy_Adapter = Adapter(adapter_dim, skip_connect=False)  # MLP-adapter, no skip connection
        self.Img_Adapter = Adapter(adapter_dim)  # with skip connection
        self.DSM_Adapter = Adapter(adapter_dim)  # with skip connection
        self.scale = scale
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.wx_Adapter = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.wy_Adapter = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.wx_Adapter.data.fill_(0.5)
        self.wy_Adapter.data.fill_(0.5)

        self.window_size = window_size


        self.fusion_Adapter = nn.Sequential(
            nn.Linear(dim*2,dim//4),
            nn.GELU(),
            nn.Linear(dim//4,dim*2)
        )
        self.moe_Adapter = MOEAdapter(
            D_features=1536,
            num_experts= 4,
            top_k = 2,
            mlp_ratio = 0.25,
            act_layer = nn.GELU,
            skip_connect= False,
            expert_capacity = None,
        )
        self.sa_Adapter = Shuffle_Adapter(channels=1536,groups = 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
    # def forward(self, i, index, x: torch.Tensor, y: torch.Tensor):
        shortcutx = x
        shortcuty = y

        concat = torch.cat([x,y],dim = -1)
        fused = self.fusion_Adapter(concat)
        fused = self.sa_Adapter(fused)
        fused_x,fused_y = fused.chunk(2, dim=-1)


        x = self.norm1(x)
        y = self.norm1(y)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hwx = window_partition(x, self.window_size)
            y, pad_hwy = window_partition(y, self.window_size)

        x = self.attn(x)
        y = self.attn(y)
        #### without Adapter
        x = self.Img_Adapter(x)
        y = self.DSM_Adapter(y)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hwx, (H, W))
            y = window_unpartition(y, self.window_size, pad_hwy, (H, W))

        x = shortcutx + x + fused_x
        y = shortcuty + y + fused_y


        xn = self.norm2(x)
        yn = self.norm2(y)

        cat = torch.cat([xn, yn], dim=-1)
        moe = self.moe_Adapter(cat)
        moe_x ,moe_y = moe.chunk(2,dim = -1)

        #### MMAdapter
        # adax = self.MLPx_Adapter(xn)
        # aday = self.MLPy_Adapter(yn)
        #
        # x = x + self.mlp(xn) + self.wx_Adapter * adax + (1 - self.wx_Adapter) * aday
        # y = y + self.mlp(yn) + self.wy_Adapter * aday + (1 - self.wy_Adapter) * adax


        x = x + self.mlp(xn) + moe_x
        y = y + self.mlp(yn) + moe_y

        return x, y


def channel_shuffle(x, groups):
    B, H, W, C = x.shape
    assert C % groups == 0
    x = x.view(B, H, W, groups, C // groups)
    x = x.permute(0, 1, 2, 4, 3).contiguous()  # [B, H, W, C//g, g]
    x = x.view(B, H, W, C)
    return x

def ShuffleAttention(x, cw, cb, sw, sb, G):
    B, H, W, C = x.shape
    assert C % (2 * G) == 0, f"Channel {C} must be divisible by 2*G={2*G}"
    shortcut = x
    x = x.view(B * G, H, W, C // G)
    x_0, x_1 = x.chunk(2, dim=-1)  # each: [B*G, H, W, C//(2G)]

    # Channel Attention
    xn = torch.mean(x_0, dim=[1, 2], keepdim=True)
    xn = cw * xn + cb
    xn = x_0 * torch.sigmoid(xn)

    # Spatial Attention
    x1_perm = x_1.permute(0, 3, 1, 2).contiguous()
    xs_norm = F.group_norm(x1_perm, num_groups=1)
    xs_norm = xs_norm.permute(0, 2, 3, 1).contiguous()
    xs = sw * xs_norm + sb
    xs = x_1 * torch.sigmoid(xs)

    out = torch.cat([xn, xs], dim=-1)
    out = out.view(B, H, W, C)
    out = channel_shuffle(out, 2)
    return out



class Shuffle_Adapter(nn.Module):
    def __init__(self, channels, groups=1):
        super().__init__()
        self.groups = groups
        assert channels % (2 * groups) == 0, "channels must be divisible by 2*groups"
        hidden_dim = channels // (2 * groups)

        # 可学习参数：形状 [1, 1, 1, hidden_dim]
        self.cw = nn.Parameter(torch.ones(1, 1, 1, hidden_dim))
        self.cb = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        self.sw = nn.Parameter(torch.ones(1, 1, 1, hidden_dim))
        self.sb = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        # x: [B, H, W, C]
        return ShuffleAttention(x, self.cw, self.cb, self.sw, self.sb, self.groups)



# class Attention(nn.Module):
#     """Multi-head Attention block with relative position embeddings and q/k/v Adapters."""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = True,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         input_size: Optional[Tuple[int, int]] = None,
#         adapter_scale: float = 0.3,  # 新增：Adapter 缩放系数初始值
#            # 新增：Adapter 下采样比例
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             use_rel_pos (bool): If True, add relative positional embeddings.
#             rel_pos_zero_init (bool): If True, zero initialize relative pos params.
#             input_size (tuple(int, int) or None): Input resolution for rel pos.
#             adapter_scale (float): Initial scale factor for Adapter outputs.
#             reduction_factor (int): Downscaling factor for Adapter bottleneck.
#         """
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#         self.use_rel_pos = use_rel_pos
#         if self.use_rel_pos:
#             assert (
#                 input_size is not None
#             ), "Input size must be provided if using relative positional encoding."
#             # initialize relative positional embeddings
#             self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
#             self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
#         reduction_factor = 4
#         # === Adapter for q, k, v ===
#         adapter_dim = dim // reduction_factor
#
#         def make_adapter():
#             return nn.Sequential(
#                 nn.Linear(dim, adapter_dim),
#                 nn.GELU(),
#                 nn.Linear(adapter_dim, dim)
#             )
#
#         self.q_Adapter = make_adapter()
#         self.k_Adapter = make_adapter()
#         self.v_Adapter = make_adapter()
#
#         # 可学习的缩放因子（标量，也可改为 nn.ParameterList 分别控制）
#         self.adapter_scale = nn.Parameter(torch.tensor(adapter_scale))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, H, W, _ = x.shape
#         # qkv with shape (3, B, H*W, dim)
#         qkv = self.qkv(x)  # (B, H*W, 3*dim)
#         qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # (3, B, nHead, HW, C)
#         # Split into q, k, v: each (B * nHead, HW, C)
#         q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
#
#         # Reshape back to (B, HW, dim) for Adapter (since Adapter operates on full dim, not per-head)
#         q_full = q.reshape(B, H * W, -1)  # (B, HW, dim)
#         k_full = k.reshape(B, H * W, -1)
#         v_full = v.reshape(B, H * W, -1)
#
#         # Apply Adapters with residual connection scaled by adapter_scale
#         # q_full = q_full + self.adapter_scale * self.q_Adapter(q_full)
#         k_full = k_full + self.adapter_scale * self.k_Adapter(k_full)
#         v_full = v_full + self.adapter_scale * self.v_Adapter(v_full)
#
#         # Reshape back to multi-head format
#         q = q_full.view(B * self.num_heads, H * W, -1)
#         k = k_full.view(B * self.num_heads, H * W, -1)
#         v = v_full.view(B * self.num_heads, H * W, -1)
#
#         # Compute attention
#         attn = (q * self.scale) @ k.transpose(-2, -1)
#
#         if self.use_rel_pos:
#             attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))
#
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
#         x = self.proj(x)
#
#         return x
class Attention(nn.Module):
    """带有Adapter的多头注意力块，Adapter并联在QKV计算前"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
            adapter_scale: float = 0.1,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            adapter_reduction (int): Adapter下采样比例
            adapter_scale (float): Adapter输出的缩放因子
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 原始的QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 相对位置编码
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

        # ========== 添加Adapter ==========
        # 选项1: 直接在QKV投影后添加Adapter（更简单）

        self.adapter_scale = adapter_scale

        # QKV Adapter: 对Q、K、V分别添加Adapter
        # 或者对拼接后的QKV整体添加Adapter
        self.qkv_Adapter = nn.Sequential(
            nn.Linear(dim * 3, (dim * 3) // 4),
            nn.GELU(),
            nn.Linear((dim * 3) // 4, dim * 3)
        )

        # 可学习的Adapter权重
        self.Adapter_weight = nn.Parameter(torch.tensor(adapter_scale))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # 1. 原始QKV投影
        qkv_orig = self.qkv(x)  # (B, H*W, 3*dim)

        # 2. 并行：应用Adapter到QKV
        qkv_adapter = self.qkv_Adapter(qkv_orig)

        # 3. 融合原始QKV和Adapter输出
        # 使用可学习权重控制Adapter贡献
        qkv_fused = qkv_orig + self.Adapter_weight * qkv_adapter

        # 4. 重形状并分离Q、K、V
        qkv = qkv_fused.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # 5. 注意力计算（与原始相同）
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
# class Attention(nn.Module):
#     """Multi-head Attention block with relative position embeddings."""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         qkv_bias: bool = True,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         input_size: Optional[Tuple[int, int]] = None,
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads.
#             qkv_bias (bool):  If True, add a learnable bias to query, key, value.
#             rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             input_size (tuple(int, int) or None): Input resolution for calculating the relative
#                 positional parameter size.
#         """
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#
#         self.use_rel_pos = use_rel_pos
#         if self.use_rel_pos:
#             assert (
#                 input_size is not None
#             ), "Input size must be provided if using relative positional encoding."
#             # initialize relative positional embeddings
#             self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
#             self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
#
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, H, W, _ = x.shape
#         # qkv with shape (3, B, nHead, H * W, C)
#         qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         # q, k, v with shape (B * nHead, H * W, C)
#         q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
#
#         attn = (q * self.scale) @ k.transpose(-2, -1)
#
#         if self.use_rel_pos:
#             attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))
#
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
#         x = self.proj(x)
#
#         return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

def closest_numbers(target):
    a = int(target ** 0.5)
    b = a + 1
    while True:
        if a * b == target:
            return (a, b)
        elif a * b < target:
            b += 1
        else:
            a -= 1


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))