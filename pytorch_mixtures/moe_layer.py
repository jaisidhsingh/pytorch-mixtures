import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from pytorch_mixtures.utils import Experts
from pytorch_mixtures.routing import AutonomousRouter


class MoELayer(nn.Module):
    def __init__(self, router: nn.Module, experts: List[nn.Module], capacity_factor: float) -> None:
        super().__init__()
        self.router = router
        self.experts = Experts(experts)
        self.num_experts = len(experts)
        self.capacity_factor = capacity_factor
    
    def forward(self, token_inputs):
        [B, N, D] = token_inputs.shape
        expert_capacity = int((N * self.capacity_factor) // self.num_experts)
        
        # send tokens to router
        if not isinstance(self.router, AutonomousRouter):
            routing_instructions = self.router(token_inputs, expert_capacity)
        else:
            routing_instructions = self.router(token_inputs, expert_capacity, self.experts.experts)
        
        # dispatch to experts
        expert_inputs = torch.einsum(
            "bnd,bnec->becd", 
            token_inputs, 
            routing_instructions["dispatch_tensor"]
        )
        # processing by experts
        expert_outputs = self.experts(expert_inputs)
        # combine expert outputs
        output = torch.einsum(
            "becd,bnec->bnd", 
            expert_outputs, 
            routing_instructions["combine_tensor"]
        )
        return output, routing_instructions["aux_loss"], routing_instructions["router_z_loss"]
 