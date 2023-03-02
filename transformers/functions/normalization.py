"""Defines a Range of different Normalization Functions"""

import torch
from torch import nn


class ScaleNorm(nn.Module):
    """Layer that can compute both ScaleNorm and FixNorm"""

    def __init__(
        self, affine: bool = True, scale_factor: float = 1.0, dtype=torch.float32
    ):
        super().__init__()
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, dtype=dtype))
        else:
            self.register_buffer("scale", torch.ones(scale_factor, dtype=dtype))

    def forward(self, x):
        # divide scale by l2 norm of input
        scale = self.scale / torch.norm(x, p=2, dim=-1, keepdim=True)
        # return pointwise multiplication
        return x * scale


def FixNorm(scale_factor: float = 1.0, dtype=torch.float32):
    """Wrapper to create a FixNorm layer"""
    return ScaleNorm(affine=False, scale_factor=scale_factor, dtype=dtype)


class WeightNorm(nn.Module):
    """Layer that can compute both WeightNorm and FixNorm"""

    # TODO: check if this is really correct?

    def __init__(
        self, affine: bool = True, scale_factor: float = 1.0, dtype=torch.float32
    ):
        super().__init__()
        self.affine = affine
        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, dtype=dtype))
        else:
            self.register_buffer("scale", torch.ones(scale_factor, dtype=dtype))

    def forward(self, x):
        # divide scale by l2 norm of input
        scale = self.scale / torch.norm(x, p=2, dim=0, keepdim=True)
        # return pointwise multiplication
        return x * scale
