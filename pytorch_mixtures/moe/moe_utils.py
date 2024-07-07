# Reference: Lucidrains ST-MoE Github, Google Flaxformer GitHub

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


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
        # shape of x: [B, E, N, D]
        [B, E, N, D] = x.shape
        x = x.view(E, B, N, D)
        outputs = []

        for i in range(self.num_exerts):
            per_expert_output = self.experts[i](x[i])
            outputs.append(per_expert_output.unsqueeze(0))

        outputs = torch.stack(outputs).view(E, B, N, D)
        return outputs.view(B, E, N, D)


def load_balancing_loss(router_probs, expert_indices):
    [B, N, E] = router_probs.shape
    expert_mask = F.one_hot(expert_indices, E)
    expert_mask = expert_mask.max(dim=-2)
    
    tokens_per_group_and_expert = expert_mask.mean(dim=-2).float()
    router_probs_per_group_and_expert = router_probs.mean(dim=-2).float()
    
    loss = (tokens_per_group_and_expert * router_probs_per_group_and_expert) * E**2
    return loss


def router_z_loss(router_logits):
    [B, N, E] = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    loss = log_z**2
    return loss.sum().float() / (B * N) 
    