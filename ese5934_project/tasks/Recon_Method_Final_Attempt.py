

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

############################

'''def conjugate_gradient_reconstruct(A_fn, b, x0, iterations, tolerance=1e-10, checkpoint_interval=50):
    x = x0.clone()
    r = b - A_fn(x)
    d = r.clone()
    delta_new = torch.sum(r.conj() * r, dim=(-2, -1))
    delta_0 = delta_new.clone()
    
    image_show_list = []  # List to store images at checkpoints

    for i in range(iterations):
        q = A_fn(d)
        alpha = delta_new / torch.sum(d.conj() * q, dim=(-2, -1))
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # Make alpha broadcastable
        x = x + alpha * d
        r = r - alpha * q
        delta_old = delta_new
        delta_new = torch.sum(r.conj() * r, dim=(-2, -1))
        beta = delta_new / delta_old
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # Make beta broadcastable
        d = r + beta * d
        residual_norm = torch.norm(r)  # Update residual norm at each iteration
        #if i % checkpoint_interval == 0 or i == iterations - 1:
            #print(f"Iteration {i+1}, Residual norm: {torch.max(delta_new).item()**0.5}")
            #image_show_list.append(x.clone().detach().cpu())  # Add current reconstruction
        
        #if torch.max(delta_new) < tolerance**2 * torch.max(delta_0):
            #break  # Convergence criterion met for all slices/channels

    #return x, image_show_list"
        print(f"Iteration {i+1}, Residual norm: {residual_norm.item()}")
       
        if i % checkpoint_interval == 0: #or i == iterations - 1:
            #print(f"Iteration {i+1}, Residual norm: {residual_norm.item()}")
            image_show_list.append(x.clone().detach().cpu())  # Add current reconstruction
        
        if residual_norm < tolerance:
            break  # Convergence criterion met for all slices/channels

    return x, image_show_list

############################################

def reconstruct_cg(
    kspace_masked, 
    csm, mask, 
    masked_forward_model=MaskedForwardModel(), 
    iterations=500, 
    device=torch.device("cpu"), 
    checkpoint_interval=50, 
    ):
    #kspace_masked, csm, mask = kspace_masked.to(device), csm.to(device), mask.to(device)

    kspace_masked = kspace_masked.to(device)
    csm = csm.to(device)
    mask = mask.to(device)
    masked_forward_model.to(device)
    

    def A_fn(x):
        return masked_forward_model(x, csm, mask)
    
    #x0 = torch.zeros_like(kspace_masked)#Not sure about the initialization
    
    x0 = torch.fft.ifft2(kspace_masked).abs()  # Take the absolute value to ensure magnitude image
    
    reconstructed_image, image_show_list = conjugate_gradient_reconstruct(
        A_fn=A_fn, 
        b=kspace_masked, 
        x0=x0, 
        iterations=iterations,
        tolerance=1e-10,
        checkpoint_interval=checkpoint_interval
    )
    
    return reconstructed_image, image_show_list'''



################################################################################
def conjugate_gradient_reconstruct(A_fn, b, x0, iterations, tolerance=1e-10, checkpoint_interval=50):
    x = x0.clone()
    r = -b + A_fn(x)
    d = -(r.clone())
    delta_new = torch.sum(r.conj() * r, dim=(-2, -1))
    delta_0 = delta_new.clone()
    
    image_show_list = []  # List to store images at checkpoints

    for i in range(iterations):
        q = A_fn(d)
        alpha = delta_new / torch.sum(d.conj() * q, dim=(-2, -1))
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)  # Make alpha broadcastable
        x = x + alpha * d
        r = r + alpha * q
        delta_old = delta_new
        delta_new = torch.sum(r.conj() * r, dim=(-2, -1))
        beta = delta_new / delta_old
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # Make beta broadcastable
        d = -r + beta * d
        residual_norm = torch.norm(r)  # Update residual norm at each iteration
        #if i % checkpoint_interval == 0 or i == iterations - 1:
            #print(f"Iteration {i+1}, Residual norm: {torch.max(delta_new).item()**0.5}")
            #image_show_list.append(x.clone().detach().cpu())  # Add current reconstruction
        
        #if torch.max(delta_new) < tolerance**2 * torch.max(delta_0):
            #break  # Convergence criterion met for all slices/channels

    #return x, image_show_list"
        print(f"Iteration {i+1}, Residual norm: {residual_norm.item()}")
       
        if i % checkpoint_interval == 0: #or i == iterations - 1:
            #print(f"Iteration {i+1}, Residual norm: {residual_norm.item()}")
            image_show_list.append(x.clone().detach().cpu())  # Add current reconstruction
        
        if residual_norm < tolerance:
            break  # Convergence criterion met for all slices/channels

    return x, image_show_list

############################################

def reconstruct_cg(
    kspace_masked, 
    csm, mask, 
    masked_forward_model=MaskedForwardModel(), 
    iterations=500, 
    device=torch.device("cpu"), 
    checkpoint_interval=50, 
    ):
    #kspace_masked, csm, mask = kspace_masked.to(device), csm.to(device), mask.to(device)

    kspace_masked = kspace_masked.to(device)
    csm = csm.to(device)
    mask = mask.to(device)
    masked_forward_model.to(device)
    

    def A_fn(x):
        return masked_forward_model(x, csm, mask)
    
    #x0 = torch.zeros_like(kspace_masked)#Not sure about the initialization
    
    x0 = torch.fft.ifft2(kspace_masked).abs()  # Take the absolute value to ensure magnitude image
    
    reconstructed_image, image_show_list = conjugate_gradient_reconstruct(
        A_fn=A_fn, 
        b=kspace_masked, 
        x0=x0, 
        iterations=iterations,
        tolerance=1e-10,
        checkpoint_interval=checkpoint_interval
    )
    
    return reconstructed_image, image_show_list




