# Reference: Lucidrains ST-MoE Github, Google Flaxformer GitHub

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale
    

class GEGLU(nn.Module):
    def __init__(self, dim, mult_bias=True):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x * self.mult_bias


class Experts(nn.Module):
    def __init__(self, num_experts: int, experts: List[nn.Module]) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(experts)
    
    def forward(self, x):
        [B, E, N, D] = x.shape
        x = torch.permute(x, (1, 0, 2, 3))
        outputs = []

        for i in range(self.num_experts):
            per_expert_output = self.experts[i](x[i])
            outputs.append(per_expert_output.unsqueeze(0))

        outputs = torch.stack(outputs).view(E, B, N, D)
        return outputs.view(B, E, N, D)


class MHSA(nn.Module):
    def __init__(self, dim, num_heads, scaling=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = dim
        self.dropout = nn.Dropout(dropout)
        self.scaling = scaling

        self.queries_keys_values = nn.Linear(dim, 3*dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        splits = rearrange(self.queries_keys_values(
            x), 'b n (h d qkv) -> (qkv) b h n d', qkv=3, h=self.num_heads)
        queries, keys, values = splits[0], splits[1], splits[2]

        attention = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        attention = nn.functional.softmax(attention, dim=-1)
        if self.scaling:
            attention = attention / (self.embedding_dim**0.5)

        attention = self.dropout(attention)

        output = torch.einsum('bhad, bhdv -> bhav', attention, values)
        output = rearrange(output, 'b h a v -> b a (h v)')
        return self.projection(output)



def load_balancing_loss(router_probs, expert_indices):
    [B, N, E] = router_probs.shape
    expert_mask = F.one_hot(expert_indices, E)
    expert_mask, _ = expert_mask.max(dim=-2)
    
    tokens_per_group_and_expert = expert_mask.float().mean(dim=-2).float()
    router_probs_per_group_and_expert = router_probs.mean(dim=-2).float()
    
    loss = (tokens_per_group_and_expert * router_probs_per_group_and_expert) * E**2
    return loss


def router_z_loss(router_logits):
    [B, N, E] = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    loss = log_z**2
    return loss.sum().float() / (B * N) 


def _one_hot(indices, num_classes, dtype=torch.int32):
    """
    Augments the default `torch.nn.functional.one_hot` as follows:
    1. ignores negative elements in the input `indices`
    2. ignores elements with value greater than `num_classes-1`.
    """
    mask1 = indices < 0
    mask2 = indices >= num_classes
    new_indices = indices
    new_indices[mask1] = 0
    new_indices[mask2] = 0
    res = F.one_hot(new_indices, num_classes)
    res[mask1] = 0
    res[mask2] = 0
    return res.to(dtype)