from .moe_layer import MoELayer
from .mod_layer import MoDLayer
from .routing import ExpertChoiceRouter, TopkRouter
from .utils import MHSA

import torch
import torch.nn as nn

from absl.testing import absltest
from absl.testing import parameterized


def difference_within_bound(a, b, eps=1e-4):
    x = a-b
    return x.all() < eps

def print_report(a, b):
    bounds = [1e-5, 1e-6, 1e-7]
    best_bound = None
    for ep in bounds:
        result = difference_within_bound(a, b, eps=ep)
        if result:
            best_bound = ep
        
    if best_bound is not None:
        print(f"The difference between the two tensors is 0 within {best_bound} across across elements.")
        return True
    else:
        print(f"The two tensors cannot be reliably called equal.")
        return False


class PyTorchMixturesTests(parameterized.TestCase):
    @torch.no_grad()
    def test_expert_choice_moe_with_identical_routers(self):
        B, N, D = 1, 128, 64
        E = 8
        C = 1
        expert_capacity = int((N * C) // E)
        x = torch.randn(B, N, D)

        # set up an expert to use as reference
        reference_expert = nn.Linear(64, 64)
        many_experts = [nn.Linear(64, 64) for _ in range(E)]
        # make all the experts the same as the reference expert
        for i in range(E):
            many_experts[i].load_state_dict(reference_expert.state_dict())
        
        router = ExpertChoiceRouter(dim=64, num_experts=E)
        # aux_loss is 0 for Expert Choice Routers
        moe, aux_loss, router_z_loss = MoELayer(num_experts=E, router=router, experts=many_experts, capacity_factor=C)
        y = moe(x)
        z = reference_expert(x)

        print(" ")
        print("Test details:")
        print("--------------------------------------------------------------------------")
        print("1. All experts are initialized as the same network.")
        print("2. The router follows Expert-Choice protocol and is randomy initialized.")
        print("3. Since all experts are equal, the weighted sum of the expert outputs")
        print("   should equal the input tensor passed through any expert.")
        print("4. This test reports whether the different b/w the MoE output and the")
        print("   input tensor passed through any expert is 0 (sufficiently).")
        print(" ")
        
        report_return = print_report(y, z)
        self.assertEqual(report_return, True)

    @torch.no_grad()
    def test_topk_moe_with_identical_routers(self):
        B, N, D = 1, 128, 64
        E = 8
        C = 1
        K = 2
        expert_capacity = int((N * C) // E)
        x = torch.randn(B, N, D)

        # set up an expert to use as reference
        reference_expert = nn.Linear(64, 64)
        many_experts = [nn.Linear(64, 64) for _ in range(E)]
        # make all the experts the same as the reference expert
        for i in range(E):
            many_experts[i].load_state_dict(reference_expert.state_dict())
        
        router = TopkRouter(dim=64, num_experts=E, topk=K)
        moe, aux_loss, router_z_loss = MoELayer(num_experts=E, router=router, experts=many_experts, capacity_factor=C)
        y = moe(x)
        z = reference_expert(x)

        print(" ")
        print("Test details:")
        print("--------------------------------------------------------------------------")
        print("1. All experts are initialized as the same network.")
        print("2. The router follows Top-2 protocol and is randomy initialized.")
        print("3. Since all experts are equal, the weighted sum of the expert outputs")
        print("   should equal the input tensor passed through any expert.")
        print("4. This test reports whether the different b/w the MoE output and the")
        print("   input tensor passed through any expert is 0 (sufficiently).")
        print(" ")
        
        report_return = print_report(y, z)
        self.assertEqual(report_return, True)

    @torch.no_grad()
    def test_expert_choice_mod(self):
        B, N, D = 1, 128, 64
        E = 8
        C = 1
        expert_capacity = int((N * C) // E)
        x = torch.randn(B, N, D)
        
        router = ExpertChoiceRouter(dim=D, num_experts=1)
        attn_fn = MHSA(dim=D, num_heads=E)
        mod, router_z_loss = MoDLayer(router=router, attn_fn=attn_fn, capacity_factor=C)
        y = mod(x)

        print(" ")
        print("Test details:")
        print("-----------------------------------------------------------------")
        print("Only checks for an error-free forward pass through the MoD layer.")
        print(" ")

def run_tests():
    absltest.main(__name__)
