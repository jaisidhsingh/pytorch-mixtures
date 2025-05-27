from typing import *
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BasicMLP(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(dim * expansion_factor)
        self.fc1 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.fc2 = nn.Linear(self.hidden_dim, dim, bias=False)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.fc1(x))
        return self.dropout(self.fc2(x))


class GatedMLP(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(dim * expansion_factor)
        self.gate = nn.Linear(dim, self.hidden_dim, bias=False)
        self.proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, dim, bias=False)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.gate(x)) * self.proj(x)
        return self.dropout(self.out(x))


class BottleneckedGatedMLP(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4, bottleneck_factor: float = 0.25, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = int(dim * expansion_factor)
        self.bottleneck_dim = int(dim * bottleneck_factor)

        self.bottleneck_in = nn.Linear(dim, self.bottleneck_dim, bias=False)
        self.bottleneck_out = nn.Linear(self.bottleneck_dim, self.hidden_dim, bias=False)
        self.proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, dim, bias=False)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.bottleneck_out(self.bottleneck_in(x))) * self.proj(x)
        return self.dropout(self.out(x))
 