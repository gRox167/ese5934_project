from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torchopt
from matplotlib import pyplot as plt
from torch.func import functional_call, grad
from torch.nn import functional as F

from ese5934_project.models.fields import Grid
from ese5934_project.models.operators import ForwardModel, MaskedForwardModel


def reconstruct(
    field,
    kspace_masked,
    csm,
    mask,
    masked_forward_model=MaskedForwardModel(),
    optimizer=torchopt.adam(0.1),
    iterations=500,
    device="mps",
):
    field.to(device)
    params = dict(field.named_parameters())
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
        loss = ((kspace_hat - kspace_masked).abs() ** 2).sum()
        return loss, (loss, image, kspace_hat)

    image_show_list = []
    for t in range(iterations):
        coordinates = None

        grads, aux = grad(loss_fn, has_aux=True)(
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

        # check for results
        if t % 50 == 0:
            print(f"iteration {t}")
            image_show_list.append(image.numpy(force=True))
    return image_show_list
