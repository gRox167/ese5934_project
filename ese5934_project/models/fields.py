import torch
from torch import nn


class Grid(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.grid = nn.Parameter(torch.zeros(size), requires_grad=True)

    def forward(self, coordinates):
        return self.grid.clone()


class TensoRF(nn.Module):
    def __init__(self, size):
        super().__init__()

    def forward(self, coordinates):
        ...
