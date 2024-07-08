# Referece: Mixture of Depths is a Vibe (Huggingface blog)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import *
from pytorch_mixtures.utils import load_balancing_loss


class TestMoDLayer(nn.Module):
    """
    Do not use for now!
    """
    def __init__(self, router: nn.Module, attn_fn: nn.Module, capacity_factor: float) -> None:
        super().__init__()
        self.router = router
        self.capacity_factor = capacity_factor
        self.attn_fn = attn_fn
    
    def forward(self, token_inputs):
        [B, N, D] = token_inputs.shape
        expert_capacity = int((N * self.capacity_factor))
        # send tokens to router
        routing_instructions = self.router(token_inputs, expert_capacity)
        # dispatch to experts
        attn_inputs = torch.einsum("bnd,bnec->becd", token_inputs, routing_instructions.dispatch_tensor)
        # processing by attention layer
        attn_inputs = attn_inputs.squeeze(1)
        attn_outputs = self.attn_fn(attn_inputs)
        attn_outputs = attn_outputs.unsqueeze(1)
        # combine expert outputs
        output = torch.einsum("becd,bnec->bnd", attn_outputs, routing_instructions.combine_tensor)
        return output



class MoDLayer(nn.Module):
    def __init__(self, dim: int, capacity_factor: float, attention_block: nn.Module) -> None:
        super().__init__()
        self.dim = dim
        self.capacity_factor = capacity_factor

        self.attention = attention_block
        self.router = nn.Linear(self.dim, 1, bias=False) # analogous to routing to only 1 expert
        self.aux_router = nn.Sequential(
            nn.Linear(self.dim, self.dim//2),
            nn.SiLU(),
            nn.Linear(self.dim//2, 1)
        )

    def forward(self, x: Tensor, auxiliary_loss=False):
        [B, N, D] = x.shape

        topk = int(N * self.capacity_factor)
        # shape: [B, N, 1]
        router_logits = self.router(x)
        gate_logits, gate_indices = torch.topk(router_logits, topk, dim=1, sorted=False)

        # find the C indices to route to the attention block
        selected_tokens, selected_indices = torch.sort(gate_indices, dim=1)
        indices_expanded = selected_tokens.expand(-1, -1, D)
        filtered_x = torch.gather(input=x, dim=1, index=indices_expanded)  # -> batch, capacity, dim

        attn_output = self.attention(x)
        # routing probs
        token_weights = F.softmax(token_weights, dim=1) 

        routing_weights = torch.gather(token_weights, dim=1, index=selected_indices)
        weighted_attn_output = routing_weights * attn_output
        out = torch.scatter_add(input=x, dim=1, index=indices_expanded, src=weighted_attn_output)

        aux_router_probs = self.aux_router(x).softmax(dim=1)
        aux_loss = load_balancing_loss(auxiliary_loss, selected_tokens)
        return out, aux_loss
