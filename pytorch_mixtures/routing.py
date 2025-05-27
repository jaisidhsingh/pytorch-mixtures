import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import *
from pytorch_mixtures.utils import load_balancing_loss, router_z_loss, _one_hot


class SparseRouterWeights(nn.Module):
    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(dim, num_experts))

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bnd,de->bne", x, self.w_gate)


class SparseRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int) -> None:
        super().__init__()
        self.weights = SparseRouterWeights(dim, num_experts)

    def forward(self, token_inputs: Tensor, expert_capacity: int) -> dict:
        router_logits = self.weights(token_inputs)
        z_loss = router_z_loss(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)

        routing_instructions = self.compute_routing_instructions(router_probs, expert_capacity)
        routing_instructions.update({"router_z_loss": z_loss})
        return routing_instructions


class ExpertChoiceRouter(SparseRouter):
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


class TopkRouter(SparseRouter):
    def __init__(self, dim: int, num_experts: int, topk: int) -> None:
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


class SoftRouterWeights(nn.Module):
    def __init__(self, dim: int, num_experts: int, seq_len: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.seq_len = seq_len
        self.num_slots = seq_len // num_experts 
        self.w_gate = nn.Parameter(torch.randn(dim, num_experts, self.num_slots))
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("bnd,des->bnes", x, self.w_gate)


class SoftRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, seq_len: int) -> None:
        super().__init__()
        self.weights = SoftRouterWeights(dim, num_experts, seq_len)

    def forward(self, token_inputs: Tensor, expert_capacity: int = None) -> dict:
        # just for consistency
        expert_capacity = self.weights.num_slots
        router_logits = self.weights(token_inputs)

        dispatch_tensor = F.softmax(router_logits, dim=1)
        combine_tensor = F.softmax(F.softmax(router_logits, dim=2), dim=3)
        
        aux_loss = torch.tensor(0.0)
        router_z_loss = torch.tensor(0.0)
        
        return {"dispatch_tensor": dispatch_tensor, "combine_tensor": combine_tensor, "aux_loss": aux_loss, "router_z_loss": router_z_loss}


class AutonomousRouter(nn.Module):
    """
    Autonomous Router is just top-k routing based on the norm of the pre-gate bottleneck activations in the experts. There are no learnable router weights.
    Reference: https://arxiv.org/pdf/2501.13074

    This should follow the same unit test as TopkRouter, i.e., when all the experts have the same weights, the output of the MoE Layer should be the same as
    the output of any one expert applied to the input tokens in a dense (non-MoE) feed-forward layer. 
    """
    def __init__(self, topk: int) -> None:
        super().__init__()
        self.topk = topk
    
    def forward(self, token_inputs: Tensor, expert_capacity: int, experts: nn.ModuleList) -> dict:
        [B, N, D] = token_inputs.shape
        E = len(experts)

        # shape: [E, dim, bottleneck_dim]
        stacked_bottlnecks = torch.stack([expert.bottleneck_in.weight.t() for expert in experts], dim=0).to(token_inputs.device)
        # shape: [dim, E, bottleneck_dim]
        stacked_bottlnecks = torch.permute(stacked_bottlnecks, (1, 0, 2))

        # shape: [B, N, E, bottleneck_dim]
        activation_cache = torch.einsum("bnd,dec->bnec", token_inputs, stacked_bottlnecks)

        # shape: [B, N, E]
        router_logits = activation_cache.norm(dim=-1)
        assert router_logits.shape == torch.Size([B, N, E]), f"Expected `router_logits` shape [B, N, E], got {router_logits.shape}."

        # values, _ = router_logits.topk(k=self.topk, dim=-1)
        # router_probs = F.softmax(values, dim=-1)
        router_probs = F.softmax(router_logits, dim=-1)

        routing_instructions = self.compute_routing_instructions(router_probs, expert_capacity)
        routing_instructions.update({"router_z_loss": 0.0})
        return routing_instructions

    def compute_routing_instructions(self, router_probs: Tensor, expert_capacity: int) -> dict:
        [B, N, E] = router_probs.shape
        expert_gate, expert_indices = router_probs.topk(k=self.topk, dim=-1)

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
        return {"dispatch_tensor": dispatch_tensor, "combine_tensor": combine_tensor, "aux_loss": 0.0}
