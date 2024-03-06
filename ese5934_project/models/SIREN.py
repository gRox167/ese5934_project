import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset


def get_coordinates(size, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = [torch.linspace(-1, 1, s) for s in size]
    # tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    # mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        size,
        mean,
        std,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

        self.flatten = Rearrange("h w dim -> (h w) dim")
        self.unflatten = Rearrange("(h w) complex -> h w complex", h=size[0], w=size[1])

    def forward(self, coords):
        batched_coords = self.flatten(coords)
        output = self.net(batched_coords)
        output = self.unflatten(output)
        output = output * self.std + self.mean
        return output
        # return output

    # def forward_origianl(self, coords):
    #     coords = (
    #         coords.clone().detach().requires_grad_(True)
    #     )  # allows to take derivative w.r.t. input
    #     output = self.net(coords)
    #     return output, coords

    # def forward_with_activations(self, coords, retain_grad=False):
    #     """Returns not only model output, but also intermediate activations.
    #     Only used for visualizing activations later!"""
    #     activations = OrderedDict()

    #     activation_count = 0
    #     x = coords.clone().detach().requires_grad_(True)
    #     activations["input"] = x
    #     for i, layer in enumerate(self.net):
    #         if isinstance(layer, SineLayer):
    #             x, intermed = layer.forward_with_intermediate(x)

    #             if retain_grad:
    #                 x.retain_grad()
    #                 intermed.retain_grad()

    #             activations[
    #                 "_".join((str(layer.__class__), "%d" % activation_count))
    #             ] = intermed
    #             activation_count += 1
    #         else:
    #             x = layer(x)

    #             if retain_grad:
    #                 x.retain_grad()

    #         activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
    #         activation_count += 1

    #     return activations


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.0
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(
            y[..., i], x, torch.ones_like(y[..., i]), create_graph=True
        )[0][..., i : i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


if __name__ == "__main__":
    img_siren = Siren(
        in_features=2,
        out_features=1,
        hidden_features=256,
        hidden_layers=3,
        outermost_linear=True,
    )
