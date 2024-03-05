from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torchopt
from fastmri import complex_abs_sq
from matplotlib import pyplot as plt
from torch.func import functional_call, grad
from torch.nn import functional as F

from ese5934_project.models.GridField import Grid
from ese5934_project.models.operators import ForwardModel, MaskedForwardModel

import torch.optim as optim

def reconstruct_L(
    field,
    coordinates,
    kspace_masked,
    csm,
    mask,
    masked_forward_model=MaskedForwardModel(),
    iterations=500,
    device=torch.device("cpu"),
    params_init=None,
    kspace_normalization=False,
):
    field.to(device)
    coordinates = coordinates.to(device)
    kspace_masked = kspace_masked.to(device)
    csm = csm.to(device)
    mask = mask.to(device)
    masked_forward_model.to(device)
    params = dict(field.named_parameters()) if params_init is None else params_init

    # Define the loss function
    def loss_fn(
        params,
        coordinates,
        kspace_masked,
        csm,
        mask,
        kspace_normalization=False,
    ):
        image = functional_call(field, params, (coordinates,))
        kspace_hat = masked_forward_model(image, csm, mask)
        loss = complex_abs_sq(kspace_hat - kspace_masked).mean()
        return loss,(loss,image,kspace_hat)
  

    # Define the optimizer as conjugate gradient descent
    optimizer = optim.LBFGS(params.values(), lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss, _ = loss_fn(params, coordinates, kspace_masked, csm, mask)
        loss.backward()
        return loss
    image_show_list = []
    # Run optimization
    for t in range(1, iterations + 1):
        optimizer.step(closure)
        loss, image, kspace_hat = loss_fn(params, coordinates, kspace_masked, csm, mask)
        print(f"Iteration {t}, loss: {loss.item()}")
        if t % 50 == 0:
            print(f"Iteration {t}")
            image_show_list.append(image.detach().cpu())

    return params, image_show_list
