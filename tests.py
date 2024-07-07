from pytorch_mixtures.routing import TopkRouter, ExpertChoiceRouter
from pytorch_mixtures.moe_layer import MoELayer
from pytorch_mixtures.mod_layer import MoDLayer
from pytorch_mixtures.utils import MHSA
import torch
import torch.nn as nn


def difference_within_bound(a, b, eps=1e-4):
    x = a-b
    return x.all() < eps

def print_report(a, b):
    bounds = [1e-4, 1e-5, 1e-6, 1e-7]
    best_bound = None
    for ep in bounds:
        result = difference_within_bound(a, b, eps=ep)
        if result:
            best_bound = ep
        
    if best_bound is not None:
        print(f"The difference between the two tensors is 0 within {best_bound} across across elements.")
    else:
        print(f"The two tensors cannot be reliably called equal.")

@torch.no_grad()
def test_expert_choice_moe_with_identical_routers():
    B, N, D = 1, 128, 64
    E = 8
    C = 1
    expert_capacity = int((N * C) // E)
    print(expert_capacity)
    x = torch.randn(B, N, D)

    # set up an expert to use as reference
    reference_expert = nn.Linear(64, 64)
    many_experts = [nn.Linear(64, 64) for _ in range(E)]
    # make all the experts the same as the reference expert
    for i in range(E):
        many_experts[i].load_state_dict(reference_expert.state_dict())
    
    router = ExpertChoiceRouter(dim=64, num_experts=E)
    moe = MoELayer(num_experts=E, router=router, experts=many_experts, capacity_factor=C)
    y = moe(x)
    z = reference_expert(x)

    print_report(y, z)

@torch.no_grad()
def test_topk_moe_with_identical_routers():
    B, N, D = 1, 128, 64
    E = 8
    C = 1
    K = 2
    expert_capacity = int((N * C) // E)
    print(expert_capacity)
    x = torch.randn(B, N, D)

    # set up an expert to use as reference
    reference_expert = nn.Linear(64, 64)
    many_experts = [nn.Linear(64, 64) for _ in range(E)]
    # make all the experts the same as the reference expert
    for i in range(E):
        many_experts[i].load_state_dict(reference_expert.state_dict())
    
    router = TopkRouter(dim=64, num_experts=E, topk=K)
    moe = MoELayer(num_experts=E, router=router, experts=many_experts, capacity_factor=C)
    y = moe(x)
    z = reference_expert(x)

    print_report(y, z)


@torch.no_grad()
def test_expert_choice_mod():
    B, N, D = 1, 128, 64
    E = 8
    C = 1
    expert_capacity = int((N * C) // E)
    print(expert_capacity)
    x = torch.randn(B, N, D)
    
    router = ExpertChoiceRouter(dim=D, num_experts=1)
    attn_fn = MHSA(dim=D, num_heads=E)
    moe = MoDLayer(router=router, attn_fn=attn_fn, capacity_factor=C)
    y = moe(x)
    z = attn_fn(x)

    print(y[0, 0, :])
    print(z[0, 0, :])
    print(y.shape)
    print_report(y, z)


if __name__ == "__main__":
    test_expert_choice_moe_with_identical_routers()
