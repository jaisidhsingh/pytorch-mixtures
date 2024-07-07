import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from pytorch_mixtures.utils import Experts


class MoELayer(nn.Module):
    def __init__(self, num_experts: int, router: nn.Module, experts: List[nn.Module], capacity_factor: float) -> None:
        super().__init__()
        self.router = router
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.experts = Experts(num_experts, experts)
    
    def forward(self, token_inputs):
        [B, N, D] = token_inputs.shape
        expert_capacity = int((N * self.capacity_factor) // self.num_experts)
        # send tokens to router
        routing_instructions = self.router(token_inputs, expert_capacity)
        # dispatch to experts
        expert_inputs = torch.einsum("bnd,bnec->becd", token_inputs, routing_instructions.dispatch_tensor)
        # processing by experts
        expert_outputs = self.experts(expert_inputs)
        # combine expert outputs
        output = torch.einsum("becd,bnec->bnd", expert_outputs, routing_instructions.combine_tensor)
        return output
 