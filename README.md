# PyTorch Mixtures <a href="https://pypi.org/project/pytorch-mixtures/">[PyPi]</a>

A plug-and-play module for <a href="https://arxiv.org/abs/2202.08906">Mixture-of-Experts</a> and <a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths</a> in PyTorch. Your one-stop solution for inserting MoE/MoD layers into custom neural networks effortlessly!

<img src="assets/moe_pic.png" width="350" height="200"> -- <img src="assets/mod_pic.png" width="150" height="200">

Sources:

1. <a href="https://arxiv.org/abs/1701.06538">Sparse Mixture of Experts, 2017</a>
2. <a href="https://arxiv.org/abs/2404.02258">Mixture of Depths, 2024</a>

## Features/Todo

- [x] Mixture of Experts
    - [x] Top-k Routing
    - [x] Expert Choice Routing
    - [x] router-z loss
    - [x] load-balancing loss
    - [x] Testing of all MoE protocols - finished
- [x] Mixture of Depths
    - [x] capacity-based routing around attention layer
    - [x] Testing of MoD protocol - finished

## Installation

Simply using `pip3 install pytorch-mixtures` will install this package. Note that this requires `torch` and `einops` to be pre-installed as dependencies. If you would like to build this package from source, run the following command:

```bash
git clone https://github.com/jaisidhsingh/pytorch-mixtures.git
cd pytorch-mixtures
pip3 install .
```

## Usage

`pytorch-mixtures` is designed to effortlessly integrate into your existing code for any neural network of your choice, for example

```python
from pytorch_mixtures.routing import ExpertChoiceRouter
from pytorch_mixtures.moe_layer import MoELayer

import torch
import torch.nn as nn


# define some config
BATCH_SIZE = 16
SEQ_LEN = 128
DIM = 768
NUM_EXPERTS = 8
CAPACITY_FACTOR = 1.25

# first initialize the router
router = ExpertChoiceRouter(dim=DIM, num_experts=NUM_EXPERTS)

# choose the experts you want: pytorch-mixtures just needs a list of `nn.Module` experts
# for e.g. our experts are just linear layers
experts=[nn.Linear(DIM, DIM) for _ in range(NUM_EXPERTS)]

# supply the router and experts to the MoELayer for modularity
moe = MoELayer(
    num_experts=NUM_EXPERTS, 
    router=router, 
    experts=experts, 
    capacity_factor=CAPACITY_FACTOR
)

# initialize some test input
x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)

# pass through moe
moe_output, aux_loss, router_z_loss = moe(x) # shape: [BATCH_SIZE, SEQ_LEN, DIM]
```

You can also use this easily within your own `nn.Module` classes

```python
from pytorch_mixtures.routing import ExpertChoiceRouter
from pytorch_mixtures.moe import MoELayer
from pytorch_mixtures.utils import MHSA # multi-head self-attention layer provided for ease
import torch
import torch.nn as nn


class CustomMoEAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, num_experts, capacity_factor, experts):
        super().__init__()
        self.attn = MHSA(dim, num_heads)
        router = ExpertChoiceRouter(dim, num_experts)
        self.moe = MoELayer(dim, router, experts, capacity_factor)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.norm1(self.attn(x) + x)
        moe_output, aux_loss, router_z_loss = self.moe(x)
        x = self.norm2(moe_output + x)
        return x, aux_loss, router_z_loss


experts = [nn.Linear(768, 768) for _ in range(8)]
my_block = CustomMoEAttentionBlock(
    dim=768,
    num_heads=8,
    num_experts=8,
    capacity_factor=1.25,
    experts=experts
)

# some test input
x = torch.randn(16, 128, 768)
output, aux_loss, router_z_loss = my_block(x) # output shape: [16, 128, 768]
```

# Testing

This package provides the user to run a simple yet reliable `absl test` for the MoE code. If all experts are initialized as the same module, the output of the `MoELayer` should be equal to the input tensor passed through any expert. Both `ExpertChoiceRouter` and `TopkRouter` are tested thusly, and succeed in the tests. The users can run these tests for themselves by running the following:

```python
from pytorch_mixtures import run_tests

run_tests()
```
Note: All tests pass correctly. If a test fails, it is likely due to an edge case in the random initializations. Try again, and it will pass.

# Citation

If you found this package useful, please cite it in your work:

```bib
@misc{JaisidhSingh2024,
  author = {Singh, Jaisidh},
  title = {pytorch-mixtures},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jaisidhsingh/pytorch-mixtures}},
}
```

# References

This package was built with the help of open-source code mentioned below:
1. <a href="https://github.com/google/flaxformer">Google Flaxformer</a>
2. <a href="https://github.com/lucidrains/st-moe-pytorch">ST-MoE by Lucidrains</a>
