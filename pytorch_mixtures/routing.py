# Reference:
# 1. Google Flaxformer GitHub: https://github.com/google/flaxformer

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import *
from pytorch_mixtures.utils import load_balancing_loss, router_z_loss, _one_hot


class RouterWeights(nn.Module):
    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(dim, num_experts))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bnd,de->bne", x, self.w_gate)


class Router(nn.Module):
    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.weights = RouterWeights(dim, num_experts)

    def forward(self, token_inputs: Tensor, expert_capacity: int) -> dict:
        router_logits = self.weights(token_inputs)
        z_loss = router_z_loss(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)

        routing_instructions = self.compute_routing_instructions(router_probs, expert_capacity)
        routing_instructions.update({"router_z_loss": z_loss})
        return routing_instructions


class ExpertChoiceRouter(Router):
    def compute_routing_instructions(self, router_probs: Tensor, expert_capacity: int) -> dict:
        [B, N, E] = router_probs.shape
        # shape = [B, E, N]
        transposed_router_probs = torch.permute(router_probs, (0, 2, 1))
        # shape = [B, E, C]
        expert_gate, expert_index = torch.topk(transposed_router_probs, k=expert_capacity, dim=-1)
        # make the dispatch tensor
        # shape = [B, E, C, N]
        dispatch_tensor = _one_hot(expert_index, N)
        # shape = [B, N, E, C]
        dispatch_tensor = torch.permute(dispatch_tensor, (0, 3, 1, 2)).to(torch.float32)

        # make the combine tensor
        # shape = [B, N, E, C]
        combine_tensor = torch.einsum(
			"bec,bnec->bnec",
			expert_gate,
			dispatch_tensor
		).to(torch.float32)
        
        aux_loss = torch.tensor(0.0).to(router_probs.device)
        return {"dispatch_tensor": dispatch_tensor, "combine_tensor": combine_tensor, "aux_loss": aux_loss}


class TopkRouter(Router):
    def __init__(self, dim: int, num_experts: int, topk: int) -> None:
        # self.dim = dim
        # self.num_experts = num_experts
        super().__init__(dim, num_experts)
        self.topk = topk

    def compute_routing_instructions(self, router_probs: Tensor, expert_capacity: int) -> dict:
        [B, N, E] = router_probs.shape
        expert_gate, expert_indices = router_probs.topk(k=self.topk, dim=-1)

        aux_loss = load_balancing_loss(router_probs, expert_indices)

        # shape: [B, self.topk, N]
        expert_indices = torch.permute(expert_indices, (0, 2, 1))
        # shape: [B, self.topk * N]
        expert_indices = expert_indices.reshape(B, self.topk * N)

        # shape: [B, N * self.topk, E]
        expert_mask = _one_hot(expert_indices, E).to(torch.int32)
        # shape: [B, self.topk * N, E]
        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0
        # shape: [B, self.topk, N, E]
        token_priority = token_priority.view(B, self.topk, N, E)
        # shape: [B, N, self.topk, E]
        token_priority = torch.permute(token_priority, (0, 2, 1, 3))
        # shape: [B, N, E]
        token_priority, _ = torch.max(token_priority, dim=2)
        token_priority = token_priority.long()

        # make the dispatch and combine tensors
        # shape: [B, N, E, expert_capacity]
        dispatch_tensor = _one_hot(token_priority, expert_capacity).float()
        # shape: [B, N, E, expert_capacity]
        combine_tensor = torch.einsum(
            "...te,...tec->...tec",
            router_probs,
            dispatch_tensor
        )
        return {"dispatch_tensor": dispatch_tensor, "combine_tensor": combine_tensor, "aux_loss": aux_loss}
