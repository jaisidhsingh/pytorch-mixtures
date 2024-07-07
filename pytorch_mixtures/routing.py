# Reference: Google Flaxformer GitHub

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import *
from pytorch_mixtures.utils import load_balancing_loss, router_z_loss, _one_hot


class RouterOutput():
    def __init__(self, dispatch_tensor: Tensor, combine_tensor: Tensor, aux_loss: Tensor, router_z_loss: Tensor) -> None:
        self.dispatch_tensor = dispatch_tensor
        self.combine_tensor = combine_tensor
        self.aux_loss = aux_loss
        self.router_z_loss = router_z_loss


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

    def forward(self, token_inputs: Tensor, expert_capacity: int) -> Tensor:
        router_logits = self.weights(token_inputs)
        z_loss = router_z_loss(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)

        routing_instructions = self.compute_routing_instructions(router_probs, expert_capacity)
        routing_instructions.update({"router_z_loss": z_loss})

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
    def __init__(self, dim: int, num_experts: int, topk: int):
        # self.dim = dim
        # self.num_experts = num_experts
        super().__init__(dim, num_experts)
        self.topk = topk

    def compute_routing_instructions(self, router_logits: Tensor, expert_capacity: int) -> dict:
        router_probs = F.softmax(router_logits, dim=-1)
        [B, N, E] = router_probs.shape
        gate_weights, gate_indices = router_probs.topk(k=self.topk, dim=-1)

        aux_loss = load_balancing_loss(router_probs, gate_indices)

        gate_weights_reshaped = gate_weights.view(self.topk, B, N)
        gate_indices_reshaped = gate_indices.view(self.topk, B, N)

        # start making masks
        one_hot_indices = _one_hot(gate_indices_reshaped, E)
        mask = one_hot_indices.float()

        # normalize topk expert weights
        denom = gate_weights_reshaped.sum(dim=0).view(1, B, N)
        gate_weights_reshaped = gate_indices_reshaped / denom

        preferred_experts = torch.permute(gate_indices, (0, 2, 1))
        preferred_experts = preferred_experts.reshape(B, N * self.topk)

        expert_mask = _one_hot(preferred_experts, E)
        # incorporate token priority into forward pass
        # shape: [B, N * self.topk, E]
        token_priority = torch.cumsum(expert_mask, dim=1) * expert_mask - 1.0

        # shape: [B, self.topk, N, E]
        token_priority = token_priority.view(B, self.topk, N, E)
        # shape: [B, N, self.topk, E]
        token_priority = torch.permute(token_priority, (0, 2, 1, 3))
        # shape: [B, N, self.topk]
        token_priority, _ = token_priority.max(dim=-1)

        # reshape preferred experts to the default shape for this stage
        preferred_experts = preferred_experts.view(B, self.topk, N)
        preferred_experts = torch.permute(preferred_experts, (0, 2, 1))

        # make combine tensor
        combine_tensor = gate_weights
        combine_tensor *= token_priority < expert_capacity
        # make dispatch tensor
        dispatch_tensor = torch.cat([preferred_experts, token_priority], dim=-1)

        print(dispatch_tensor.shape, combine_tensor.shape)
        return {"dispatch_tensor": dispatch_tensor, "combine_tensor": combine_tensor, "aux_loss": aux_loss}



        """
            # Shape: [num_groups, tokens_per_group * num_selected_experts, num_experts].
    token_priority = jnp.cumsum(expert_mask, axis=1) * expert_mask - 1.0
    # Shape: [num_groups, num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape(
        (num_groups, self.num_selected_experts, -1, num_experts))
    # Shape: [num_groups, tokens_per_group, num_selected_experts, num_experts].
    token_priority = jnp.swapaxes(token_priority, 1, 2)
    # For each token, across all experts, select the only non-negative
    # (unmasked) priority. Shape: [num_groups, tokens_per_group,
    # num_selected_experts].
    token_priority = jnp.max(token_priority, axis=-1)

    # Return to original index shape.
    preferred_experts = preferred_experts.reshape(num_groups,
                                                  self.num_selected_experts,
                                                  tokens_per_group)
    # Shape: [num_groups, tokens_per_group, num_selected_experts]
    preferred_experts = jnp.swapaxes(preferred_experts, 1, 2)
        """

# Note: Need to integrate all routings into the MoELayer Module
# and need to test it using "the same N experts" strategy.
# For this, work out the mathematics for Expert Choice routing, for
# you already know this works for TopkRouting.
