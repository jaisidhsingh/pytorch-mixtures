import typing as tp

import torch
import torch.nn.functional as F

from pytorch_mixtures.utils import _assert_no_nan
from pytorch_mixtures.basic_moe import MoE


class ExpertChoiceMoE(MoE):
    def compute_routing(self, router_logits: torch.Tensor) -> tp.Tuple[torch.Tensor]:
        topk = int(self.cfg.capacity_factor * router_logits.shape[0] / self.cfg.num_experts)
        weights, selected_tokens = torch.topk(router_logits.T, k=topk, dtype=self.cfg.dtype)
        weights = F.softmax(weights, dim=-1, dtype=self.cfg.dtype)
        _assert_no_nan(weights, "ec_weights")
        return weights, selected_tokens
    
    def apply_routing_on_experts(self, x: torch.Tensor, weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        final_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            token_idx = selected[i]
            final_output[token_idx] += weights[i, :, None] * expert(x[token_idx])
        return final_output
