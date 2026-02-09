import torch
import typing as tp
from dataclasses import dataclass


@dataclass
class MoEConfig:
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    expert_fn: str
    expert_act: str
    router_fn: tp.Literal["ec", "topk", "soft"] | None
    capacity_factor: float | None
    topk: int | None
    dtype: torch.dtype
 