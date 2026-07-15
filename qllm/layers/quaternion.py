"""Structured real, complex, and quaternion linear layers.

Inputs and outputs are ordinary real tensors. Complex and quaternion structure is
only imposed on the weights, so the layers are drop-in replacements for Linear.
"""

import math
from typing import Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RealLinear(nn.Linear):
    pass


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features % 2 or out_features % 2:
            raise ValueError("complex dimensions must be divisible by 2")
        self.in_features, self.out_features = in_features, out_features
        shape = (out_features // 2, in_features // 2)
        self.weight_re = nn.Parameter(torch.empty(shape))
        self.weight_im = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Rayleigh magnitude + uniform phase (Trabelsi-style polar init).
        fan_in = self.in_features // 2
        sigma = 1.0 / math.sqrt(2.0 * fan_in)
        with torch.no_grad():
            u = torch.rand_like(self.weight_re).clamp_min_(1e-7)
            magnitude = sigma * torch.sqrt(-2.0 * torch.log(u))
            phase = torch.empty_like(magnitude).uniform_(-math.pi, math.pi)
            self.weight_re.copy_(magnitude * torch.cos(phase))
            self.weight_im.copy_(magnitude * torch.sin(phase))
            if self.bias is not None:
                self.bias.zero_()

    def structured_weight(self) -> Tensor:
        return torch.cat(
            [torch.cat([self.weight_re, -self.weight_im], dim=1),
             torch.cat([self.weight_im, self.weight_re], dim=1)],
            dim=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.structured_weight(), self.bias)


class QuaternionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = "parcollet"):
        super().__init__()
        if in_features % 4 or out_features % 4:
            raise ValueError("quaternion dimensions must be divisible by 4")
        self.in_features, self.out_features = in_features, out_features
        shape = (out_features // 4, in_features // 4)
        self.weight_r = nn.Parameter(torch.empty(shape))
        self.weight_x = nn.Parameter(torch.empty(shape))
        self.weight_y = nn.Parameter(torch.empty(shape))
        self.weight_z = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init == "gaussian":
            # Explicit Stage-D ablation: independent real components.
            std = 1.0 / math.sqrt(self.in_features)
            with torch.no_grad():
                for weight in (self.weight_r, self.weight_x, self.weight_y, self.weight_z):
                    weight.normal_(0.0, std)
                if self.bias is not None:
                    self.bias.zero_()
            return
        if self.init != "parcollet":
            raise ValueError(f"unknown quaternion initialization: {self.init}")
        # Chi(4) magnitude, random unit pure direction, and uniform phase.
        fan_in = self.in_features // 4
        sigma = 1.0 / math.sqrt(2.0 * fan_in)
        with torch.no_grad():
            magnitude = torch.linalg.vector_norm(
                torch.randn(*self.weight_r.shape, 4, device=self.weight_r.device), dim=-1
            ) * sigma
            direction = torch.randn(*self.weight_r.shape, 3, device=self.weight_r.device)
            direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min_(1e-7)
            phase = torch.empty_like(magnitude).uniform_(-math.pi, math.pi)
            self.weight_r.copy_(magnitude * torch.cos(phase))
            imaginary = magnitude.unsqueeze(-1) * direction * torch.sin(phase).unsqueeze(-1)
            self.weight_x.copy_(imaginary[..., 0])
            self.weight_y.copy_(imaginary[..., 1])
            self.weight_z.copy_(imaginary[..., 2])
            if self.bias is not None:
                self.bias.zero_()

    def structured_weight(self) -> Tensor:
        r, x, y, z = self.weight_r, self.weight_x, self.weight_y, self.weight_z
        rows = [
            torch.cat([r, -x, -y, -z], dim=1),
            torch.cat([x, r, -z, y], dim=1),
            torch.cat([y, z, r, -x], dim=1),
            torch.cat([z, -y, x, r], dim=1),
        ]
        return torch.cat(rows, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.structured_weight(), self.bias)

    def naive_forward(self, inputs: Tensor) -> Tensor:
        """Component-wise Hamilton reference (intentionally unfused)."""
        ir, ix, iy, iz = inputs.chunk(4, dim=-1)
        r, x, y, z = self.weight_r, self.weight_x, self.weight_y, self.weight_z
        output = torch.cat((
            F.linear(ir, r) - F.linear(ix, x) - F.linear(iy, y) - F.linear(iz, z),
            F.linear(ir, x) + F.linear(ix, r) - F.linear(iy, z) + F.linear(iz, y),
            F.linear(ir, y) + F.linear(ix, z) + F.linear(iy, r) - F.linear(iz, x),
            F.linear(ir, z) - F.linear(ix, y) + F.linear(iy, x) + F.linear(iz, r),
        ), dim=-1)
        return output if self.bias is None else output + self.bias


def linear_class(kind: str) -> Type[nn.Module]:
    try:
        return {"real": RealLinear, "complex": ComplexLinear, "quaternion": QuaternionLinear}[kind]
    except KeyError as exc:
        raise ValueError(f"unknown layer kind: {kind}") from exc
