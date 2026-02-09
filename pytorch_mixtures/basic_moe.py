import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_mixtures.utils import _assert_no_nan, EXPERT_REGISTRY
from pytorch_mixtures.configs import MoEConfig


class MoE(ABC, nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.cfg = config
        self.weight = nn.Parameter(torch.empty(config.hidden_dim, config.num_experts))
        self.experts = nn.ModuleList([
            EXPERT_REGISTRY[config.expert_fn](config) for _ in range(config.num_experts)
        ])
    
    @abstractmethod 
    def compute_routing(self, router_logits: torch.Tensor) -> tp.Tuple[torch.Tensor]:
        raise NotImplementedError("Must be implemented specific to one's desired routing scheme.")
    
    @abstractmethod 
    def apply_routing_on_experts(self, x: torch.Tensor, weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented specific to one's desired routing scheme.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.reshape(-1, D)
        router_logits = x @ self.weight
        _assert_no_nan(router_logits, "router_logits")
        weights, selected = self.compute_routing(router_logits)
        output = self.apply_routing_on_experts(x, weights, selected)
        output = output.reshape(B, L, D)
        return output
