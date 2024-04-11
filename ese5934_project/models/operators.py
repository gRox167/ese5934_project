import torch
from einops import rearrange, repeat
from fastmri import complex_abs, complex_abs_sq, complex_conj, complex_mul
from fastmri import fft2c as fft2c
from fastmri import ifft2c as ifft2c
from torch import nn


class C(nn.Module):
    """
    Sensitivity forward operator to do SENS expansion
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, csm):
        return complex_mul(x, csm)


class C_adj(nn.Module):
    """
    Sensitivity adjoint operator to do SENS expansion
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, csm):
        return complex_mul(x, complex_conj(csm)).sum(dim=0, keepdim=True)


class F(nn.Module):
    """
    Fourier forward operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return fft2c(x, norm="ortho")


class F_adj(nn.Module):
    """
    Fourier adjoint operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ifft2c(x, norm="ortho")


class M(nn.Module):
    """
    Masking forward operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x * mask


class MaskedForwardModel(nn.Module):
    """
    MR forward model to do SENS expansion and Fourier transform
    """

    def __init__(self):
        super().__init__()
        self.C = C()
        self.F = F()
        self.M = M()

    def forward(self, x, csm, mask):
        x = self.C(x, csm)
        x = self.F(x)
        x = self.M(x, mask)
        return x


class ForwardModel(nn.Module):
    """
    MR forward model to do SENS expansion and Fourier transform
    """

    def __init__(self):
        super().__init__()
        self.C = C()
        self.F = F()

    def forward(self, x, csm, mask):
        x = self.C(x, csm)
        x = self.F(x)
        return x
