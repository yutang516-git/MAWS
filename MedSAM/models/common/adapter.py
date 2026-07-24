import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from einops import rearrange
from typing import Optional,List


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class SingleAdapter(nn.Module):

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x



class MOEAdapter(nn.Module):


    def __init__(
            self,
            D_features: int,
            num_experts: int = 4,
            top_k: int = 2,
            mlp_ratio: float = 0.25,
            act_layer: nn.Module = nn.GELU,
            skip_connect: bool = True,
            expert_capacity: Optional[int] = None,
    ):
        super().__init__()

        self.D_features = D_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.skip_connect = skip_connect

        # 专家网络
        self.experts = nn.ModuleList([
            SingleAdapter(D_features, mlp_ratio, act_layer, skip_connect=False)
            for _ in range(num_experts)
        ])

        # 路由网络
        self.router = nn.Linear(D_features, num_experts)

        # 初始化
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # 将 [B, H, W, C] -> [B, H*W, C]
        x = x.reshape(B, H * W, C)

        batch_size, seq_len, dim = x.shape

        # 路由决策
        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)
        # print("Softmax 之前的 router_logits:", router_logits[0, 0, :])
        router_weights =F.softmax(router_logits,dim=-1)

        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(
            router_weights, self.top_k, dim=-1
        )

        # 归一化权重
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 为每个专家分配token
        final_output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # 创建mask选择当前expert的token
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)

            if expert_mask.any():
                # 获取这些token的输入和权重
                expert_input = x[expert_mask]  # (num_selected, dim)

                # 计算每个token对该expert的总权重
                token_weights = torch.zeros(
                    expert_mask.shape[0], expert_mask.shape[1],
                    device=x.device
                )
                for k in range(self.top_k):
                    mask_k = (top_k_indices[..., k] == expert_idx)
                    weights_k = top_k_weights[..., k]
                    token_weights[mask_k] = token_weights[mask_k] + weights_k[mask_k]

                expert_weights = token_weights[expert_mask].unsqueeze(-1)

                # expert处理
                expert_output = self.experts[expert_idx](expert_input)

                # 加权输出
                weighted_output = expert_output * expert_weights

                # 累积到最终输出
                final_output[expert_mask] = final_output[expert_mask] + weighted_output

        x_out = final_output.reshape(B, H, W, C)

        return x_out

