import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import *

from .moe_utils import load_balancing_loss, router_z_loss


class RouterOutput():
    def __init__(self, dispatch_tensor: Tensor, combine_tensor: Tensor, aux_loss: Tensor) -> None:
        self.dispatch_tensor = dispatch_tensor
        self.combine_tensor = combine_tensor
        self.aux_loss = aux_loss


class RouterWeights(nn.Module):
    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty(dim, num_experts))
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bnd,de->bne", x, self.w_gate)


class Router(nn.Modulde):
    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.weights = RouterWeights(dim, num_experts)
    
    def forward(self, token_inputs: Tensor, expert_capacity: int) -> Tensor:
        router_logits = self.weights(token_inputs)
        router_probs = F.softmax(router_logits, dim=-1)
        routing_instructions = self.compute_routing_instructions(router_probs, expert_capacity)
        return RouterOutput(**routing_instructions)
    

class ExpertChoiceRouter(Router):
    def compute_routing_instructions(self, router_probs: Tensor, expert_capacity: int) -> dict:
        [B, N, E] = router_probs.shape
        # shape = [B, E, N]
        transposed_router_probs = torch.permute(router_probs, (0, 2, 1))
        # shape = [B, E, C]
        expert_gate, expert_index = torch.topk(transposed_router_probs, k=expert_capacity, dim=-1)
        
        # make the dispatch tensor
        # shape = [B, E, C, N]
        dispatch_tensor = F.one_hot(expert_index, N)
        # shape = [B, N, E, C]
        dispatch_tensor = torch.permute(dispatch_tensor, (0, 3, 1, 2))
        
        # make the combine tensor
        # shape = [B, N, E, C]
        combine_tensor = torch.einsum(
			"bec,bnec->bnec",
			expert_gate,
			dispatch_tensor
		)
        aux_loss = torch.tensor(0.0).to(router_probs.device)
        return {"dispatch_tensor": dispatch_tensor, "combine_tensor": combine_tensor, "aux_loss": aux_loss}


class TopkRouter(Router):
    def __init__(self, topk: int):
        self.topk = topk
        
    def compute_routing_instructions(self, router_probs, expert_capacity):
        [B, N, E] = router_probs.shape
        gate_weights, gate_indices = router_probs.topk(k=self.topk, dim=-1)
        
        aux_loss = load_balancing_loss(router_probs, gate_indices)
        
        gate_weights_reshaped = gate_weights.view(self.topk, B, N)
        gate_indices_reshaped = gate_indices.view(self.topk, B, N)
        
        # start making masks
        one_hot_indices = F.one_hot(gate_indices_reshaped, E)
        mask = one_hot_indices.float()
        
        # normalize topk expert weights
        denom = gate_weights_reshaped.sum(dim=0).view(1, B, N)
        gate_weights_reshaped = gate_indices_reshaped / denom     
     