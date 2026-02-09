import typing as tp

import torch
import torch.nn.functional as F

from pytorch_mixtures.utils import _assert_no_nan
from pytorch_mixtures.basic_moe import MoE


class SoftMoE(MoE):
    def compute_routing(self, router_logits: torch.Tensor) -> tp.Tuple[torch.Tensor]:
        """Returns dispatch and combine tensors instead of weights and indices like `TopkMoE` and `ExpertChoiceMoE`"""
        dispatch_tensor = F.softmax(router_logits, dim=0)
        _assert_no_nan(dispatch_tensor, "soft_dispatch")
        combine_tensor = F.softmax(F.softmax(router_logits, dim=1), dim=2)
        _assert_no_nan(combine_tensor, "soft_combine")
        return combine_tensor, dispatch_tensor
    
    def apply_routing_on_experts(self, x: torch.Tensor, weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """`weights` is `combine_tensor` and `selected` is `dispatch_tensor` returned from `SoftMoE.compute_routing()`"""
        dispatched_inputs = torch.einsum("md,mnp->npd", x, selected)
        expertwise_outputs = torch.stack([f(dispatched_inputs[i, :, :]) for i, f in enumerate(self.experts)], dim=0)
        combined_output = torch.einsum("npd,mnp->md", expertwise_outputs, weights)
        return combined_output
