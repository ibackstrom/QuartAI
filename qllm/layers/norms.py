"""Norm choice is deliberately shared: real LayerNorm over all components."""

from torch import nn

LayerNorm = nn.LayerNorm
