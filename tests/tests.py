import tyro
import torch
from time import perf_counter
from dataclasses import dataclass

from pytorch_mixtures import TopkMoE, ExpertChoiceMoE, MoEConfig
from pytorch_mixtures.utils import FeedForward, SwiGLU, _assert_no_nan, EXPERT_REGISTRY


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


def _timeit(callable, some_input):
    start = perf_counter()
    result = callable(some_input)
    return result, perf_counter() - start


def _print_diagnostics(d, l):
    for k, v in d.items():
        print(k, "\t", v)
    for k, v in l.items():
        print(k, "\t", v)


def _end_test():
    print("\n\n")


def _show_description():
    description = """Executes two tests for Mixture-of-Experts (MoE) implementations:
    -> [TEST 1]: _implementation_equivalence_test:
    \t Test basic MoE forward pass.
    
    -> [TEST 2]: _expert_equivalence_test:
    \t Verify that MoE outputs match a base expert when all experts are identical to the base expert. 
    \t Note: this returns True for ExpertChoiceRouting only when capacity is large enough.
    """
    print(description)
    print("\n\n")


def _implementation_equivalence_test(config):
    print("[TEST 1] - Basic MoE forward pass test")
    torch.manual_seed(config.seed)
    routing_type = config.match_routing_type

    x = torch.randn(config.B, config.T, config.D)

    moe_config = MoEConfig(
        hidden_dim=config.D,
        intermediate_dim=config.d_hidden,
        num_experts=config.n_experts,
        expert_fn=config.expert_fn,
        expert_act=config.expert_act,
        router_fn=routing_type,
        capacity_factor=config.capacity_factor,
        topk=config.k if routing_type == "topk" else None,
        dtype=torch.float32,
    )

    moe_ref = TopkMoE if routing_type == "topk" else ExpertChoiceMoE
    moe = moe_ref(moe_config)

    result, t = _timeit(moe, x)

    print(f"MoE with `routing_type = {routing_type}` forward pass successful")
    print(f"Implementation took {t} seconds.")

    _end_test()


def _expert_equivalence_test(config):
    print("[TEST 2]")
    torch.manual_seed(config.seed)

    x = torch.randn(config.B, config.T, config.D)

    moe_config = MoEConfig(
        hidden_dim=config.D,
        intermediate_dim=config.d_hidden,
        num_experts=config.n_experts,
        expert_fn=config.expert_fn,
        expert_act=config.expert_act,
        router_fn="topk",
        capacity_factor=config.capacity_factor,
        topk=config.k,
        dtype=torch.float32,
    )

    base = FeedForward(moe_config)

    topk = TopkMoE(moe_config)

    with torch.no_grad():
        for i in range(config.n_experts):
            for layer, base_layer in zip(
                [topk.experts[i].up, topk.experts[i].down], [base.up, base.down]
            ):
                layer.weight.copy_(base_layer.weight)
                layer.bias.copy_(base_layer.bias)

        topk.weight.fill_(1.0)

    ref = base(x)
    y_topk = topk(x)

    logged_topk = torch.allclose(y_topk, ref)
    print("TopKMoE equals single expert output", logged_topk)
    if not logged_topk:
        print(
            "[TEST FAILED, IMPORTANT NOTICE] Implementation is suspected to be incorrect."
        )

    print("--" * 30)
    print(" ")

    torch.testing.assert_close(y_topk, ref, rtol=1e-5, atol=1e-6)

    _end_test()


def main(config):
    _show_description()
    _implementation_equivalence_test(config)
    _expert_equivalence_test(config)


if __name__ == "__main__":
    config = tyro.cli(TestingConfig, default=vars(TestingConfig()))
    main(config)
