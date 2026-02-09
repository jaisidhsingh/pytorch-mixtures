from typing import *

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from pytorch_mixtures.configs import MoEConfig


def _round_to_4_decimals(unrounded: Union[List[float], float]) -> List[float]:
    if not isinstance(unrounded, list):
        unrounded = [unrounded]
    return [round(item, 4) for item in unrounded]


def _assert_no_nan(t: Tensor, name: str) -> None: 
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"


def _router_z_loss(router_logits: Tensor) -> Tensor:
    [B, N, E] = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    loss = log_z**2
    return loss.sum().float() / (B * N) 


class SwiGLU(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.cfg = config
        self.gate_up = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.act = getattr(F, config.expert_act) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.act(g) * u)


class FeedForward(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.cfg = config
        self.up = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.down = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.act = getattr(F, config.expert_act)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


EXPERT_REGISTRY = {"swiglu": SwiGLU, "ff": FeedForward}
