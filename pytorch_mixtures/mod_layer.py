# Like routing to one expert which is a attention/transformer block
# Expert Choice Routing
# NOTE: This has not been tested via training.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import *


class MoDLayer(nn.Module):
    def __init__(self, router: nn.Module, attn_fn: nn.Module, capacity_factor: float) -> None:
        super().__init__()
        self.router = router
        self.capacity_factor = capacity_factor
        self.attn_fn = attn_fn
    
    def forward(self, token_inputs: Tensor) -> Tensor:
        [B, N, D] = token_inputs.shape
        expert_capacity = int((N * self.capacity_factor))
        # send tokens to router
        routing_instructions = self.router(token_inputs, expert_capacity)
        # dispatch to experts
        attn_inputs = torch.einsum(
            "bnd,bnec->becd", 
            token_inputs, 
            routing_instructions["dispatch_tensor"]
        )
        # processing by attention layer
        attn_inputs = attn_inputs.squeeze(1)
        attn_outputs = self.attn_fn(attn_inputs)
        attn_outputs = attn_outputs.unsqueeze(1)
        # combine expert outputs
        output = torch.einsum(
            "becd,bnec->bnd", 
            attn_outputs, 
            routing_instructions["combine_tensor"]
        )
        return output, routing_instructions["router_z_loss"]
