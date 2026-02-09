import tyro
import torch
from time import perf_counter
from dataclasses import dataclass

from pytorch_mixtures import TopkMoE, ExpertChoiceMoE, SoftMoE
from pytorch_mixtures.utils import FeedForward, SwiGLU, _assert_no_nan 


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


def _timeit(callable, some_input):
    start = perf_counter()
    result = callable(some_input)
    return result, perf_counter()-start


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
    router_kwargs = dict(
        d_model=config.D,
        n_experts=config.n_experts,
        d_hidden=config.d_hidden,
        capacity_factor=config.capacity_factor
    )
    if routing_type == "topk":
        router_kwargs["k"] = config.k

    moe_ref = TopkMoE if routing_type == "topk" else ExpertChoiceMoE
    moe = moe_ref(**router_kwargs)

    result, t = _timeit(moe, x)

    print(f"MoE with `routing_type = {routing_type}` forward pass successful")
    print(f"Implementation took {t} seconds.")
    print(" ")
    print("Diagnostics:")
    _print_diagnostics(result[2], result[1])

    _end_test()


def _expert_equivalence_test(config):
    print("[TEST 2]")
    torch.manual_seed(config.seed)

    x = torch.randn(config.B, config.T, config.D)
    base = FeedForward(config.D, config.d_hidden)

    topk = TopkMoE(config.D, config.n_experts, config.d_hidden, k=config.k, capacity_factor=config.capacity_factor)
    ec = ExpertChoiceMoE(config.D, config.n_experts, config.d_hidden, capacity_factor=config.capacity_factor)

    # Make all experts identical to `base`
    with torch.no_grad():
        for i in range(config.n_experts):
            for layer, base_layer in zip([topk.experts[i].fc1, topk.experts[i].fc2],
                                         [base.fc1, base.fc2]):
                layer.weight.copy_(base_layer.weight)
                layer.bias.copy_(base_layer.bias)
            for layer, base_layer in zip([ec.experts[i].fc1, ec.experts[i].fc2],
                                         [base.fc1, base.fc2]):
                layer.weight.copy_(base_layer.weight)
                layer.bias.copy_(base_layer.bias)

        # Uniform gating (all ones → softmax uniform). Ensures routing weights sum to 1
        # and with identical experts the MoE output matches the single-expert output.
        topk.gate.weight.fill_(1.0)
        ec.gate.weight.fill_(1.0)

    ref = base(x)
    y_topk, l_topk, d_topk = topk(x)
    y_ec, l_ec, d_ec = ec(x)

    # check topk routing
    logged_topk = torch.allclose(y_topk, ref)
    print("TopKMoE equals single expert output", logged_topk)
    if not logged_topk:
        print("[TEST FAILED, IMPORTANT NOTICE] Implementation is suspected to be incorrect.")

    print("--"*30)
    print("TopKMoE diagnostics")
    _print_diagnostics(d_topk, l_topk)
    print(" ")

    # raise test-failed error after printing moe diagnostics
    torch.testing.assert_close(y_topk, ref, rtol=1e-5, atol=1e-6)

    # check expert-choice routing
    logged_ec = torch.allclose(y_ec, ref)
    print("ExpertChoiceMoE equals single expert output", logged_ec)
    if not logged_ec:
        print(f"[TEST FAILED, IMPORTANT NOTICE] Check diagnostics for sufficient capacity before changing routing implementation.")
        print("ExpertChoiceMoE outputs match the single expert when no tokens are dropped.")

    print("--"*30)
    print("ExpertChoiceMoE diagnostics")
    _print_diagnostics(d_ec, l_ec)
    print(" ")

    # ec_moe will not match the base expert if capacity is low.
    # raise test-failed error after printing moe diagnostics to verify capacity (and dropped tokens).
    torch.testing.assert_close(y_ec, ref, rtol=1e-5, atol=1e-6)

    _end_test()


def main(config):
    _show_description()
    _implementation_equivalence_test(config)
    _expert_equivalence_test(config)


if __name__ == "__main__":
    config = tyro.cli(TestingConfig, default=vars(TestingConfig()))
    main(config)
