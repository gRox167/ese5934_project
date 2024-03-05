import torch
import numpy as np

def reconstruct_cg_torchoptim(
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
        return loss

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(params, coordinates, kspace_masked, csm, mask)
        loss.backward()
        return loss

    def compute_Hv(v):
        optimizer.zero_grad()
        loss = loss_fn(params, coordinates, kspace_masked, csm, mask)
        grad_params = torch.autograd.grad(loss, params.values(), create_graph=True)
        Hv = torch.autograd.grad(grad_params, params.values(), grad_outputs=v)
        return Hv

    optimizer = torch.optim.LBFGS(params.values(), lr=0.1, max_iter=20, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100)

    image_show_list = []
    for t in range(1, iterations + 1):
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(params, coordinates, kspace_masked, csm, mask)
            loss.backward()
            return loss

        optimizer.step(closure)

        # check for results
        if t % 50 == 0:
            print(f"iteration {t}")
            image_show_list.append(image.detach().cpu())
    return params, image_show_list
