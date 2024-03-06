import torch
from torch import nn


class Grid(nn.Module):
    def __init__(self, size, mean, std):
        super().__init__()
        self.grid = nn.Parameter(torch.zeros(size + (2,)), requires_grad=True)

    def forward(self, coordinates):
        return self.grid
