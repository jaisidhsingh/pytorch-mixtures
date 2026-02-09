import typing as tp

import torch
import torch.nn.functional as F

from pytorch_mixtures.utils import _assert_no_nan
from pytorch_mixtures.basic_moe import MoE


class TopkMoE(MoE):
    def compute_routing(self, router_logits: torch.Tensor) -> tp.Tuple[torch.Tensor]:
        weights, selected_experts = torch.topk(router_logits, self.cfg.topk, dtype=self.cfg.dtype)
        weights = F.softmax(weights, dim=-1, dtype=self.cfg.dtype)
        _assert_no_nan(weights, "topk_weights")
        return weights, selected_experts
    
    def apply_routing_on_experts(self, x: torch.Tensor, weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        final_output = torch.zeros_like(x).to(x.dtype)
        for i, expert in enumerate(self.experts):
            token_idx, expert_idx = torch.where(selected == i)
            final_output[token_idx] += weights[token_idx, expert_idx, None] * expert(x[token_idx])
        return final_output
