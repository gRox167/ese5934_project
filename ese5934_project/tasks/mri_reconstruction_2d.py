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


def reconstruct(
    field,
    coordinates,
    kspace_masked,
    csm,
    mask,
    masked_forward_model=MaskedForwardModel(),
    optimizer=torchopt.adam(0.1),
    iterations=500,
    device=torch.device("cpu"),
    params_init=None,
):
    field.to(device)
    coordinates = coordinates.to(device)
    kspace_masked = kspace_masked.to(device)
    csm = csm.to(device)
    mask = mask.to(device)
    masked_forward_model.to(device)
    params = dict(field.named_parameters()) if params_init is None else params_init
    opt_state = optimizer.init(params)

    # define functionanl loss
    def loss_fn(
        params,
        coordinates,
        kspace_masked,
        csm,
        mask,
    ):
        image = functional_call(field, params, (coordinates,))
        kspace_hat = masked_forward_model(image, csm, mask)
        loss = complex_abs_sq(kspace_hat - kspace_masked).sum()
        return loss, (loss, image, kspace_hat)

    image_show_list = []
    for t in range(1, iterations + 1):
        grad_loss_fn = grad(loss_fn, has_aux=True)
        grads, aux = grad_loss_fn(
            params,
            coordinates,
            kspace_masked,
            csm,
            mask,
        )
        loss, image, kspace_hat = aux
        print(f"iteration {t}, loss: {loss.item()}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = torchopt.apply_updates(params, updates)
        params = {k: v.detach() for k, v in params.items()}  # detach params
        # check for results
        if t % 50 == 0:
            print(f"iteration {t}")
            image_show_list.append(image.detach().cpu())
    return params, image_show_list
