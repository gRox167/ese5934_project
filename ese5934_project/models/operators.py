import torch
from einops import rearrange, repeat
from torch import nn


class C(nn.module):
    """
    Sensitivity forward operator to do SENS expansion
    """

    def __init__(self):
        super(self).__init__()

    def forward(self, x, csm):
        return x * csm


class C_adj(nn.module):
    """
    Sensitivity adjoint operator to do SENS expansion
    """

    def __init__(self):
        super(self).__init__()

    def forward(self, x, csm):
        return (x * csm.conj()).sum(dim=1)


class F(nn.module):
    """
    Fourier forward operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
        # fftshift
        x = torch.fft.fftshift(x, dim=(-2, -1))
        # x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class F_adj(nn.module):
    """
    Fourier adjoint operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifftn(x, dim=(-2, -1), norm="ortho")
        # fftshift
        x = torch.fft.fftshift(x, dim=(-2, -1))
        # x = rearrange(x, 'b (h w) c -> b c h w', h=self.n, w=self.m)
        return x


class M(nn.module):
    """
    Masking forward operator
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        return x * mask


class MRForwardModel(nn.module):
    """
    MR forward model to do SENS expansion and Fourier transform
    """

    def __init__(self):
        super(self).__init__()
        self.C = C()
        self.F = F()
        self.M = M()

    def forward(self, x, csm, mask):
        x = self.C(x, csm)
        x = self.F(x)
        x = self.M(x, mask)
        return x
