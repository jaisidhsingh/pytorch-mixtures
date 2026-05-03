from .configs import MoEConfig
from .topk_moe import TopkMoE
from .expert_choice_moe import ExpertChoiceMoE


def run_tests():
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tests.tests import main
    from dataclasses import dataclass

    @dataclass
    class TestingConfig:
        seed: int = 123
        B: int = 16
        T: int = 32
        D: int = 128
        d_hidden: int = 128
        n_experts: int = 4
        k: int = 2
        capacity_factor: float = 5
        fast_implementation: bool = False
        match_routing_type: str = "topk"
        expert_fn: str = "ff"
        expert_act: str = "silu"

    main(TestingConfig())
