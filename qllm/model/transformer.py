"""A compact Karpathy-style decoder shared by every experimental arm."""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from qllm.layers import linear_class


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    n_layers: int = 6
    n_heads: int = 8
    width: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.1
    layer_kind: str = "real"
    attention_layer_kind: Optional[str] = None
    ffn_layer_kind: Optional[str] = None
    quaternion_init: str = "parcollet"


def _linear_factory(kind: str, config: ModelConfig) -> Callable[..., nn.Module]:
    Linear = linear_class(kind)
    if kind == "quaternion":
        return lambda *args, **kwargs: Linear(*args, init=config.quaternion_init, **kwargs)
    return Linear


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % config.n_heads:
            raise ValueError("width must be divisible by n_heads")
        Linear = _linear_factory(config.attention_layer_kind or config.layer_kind, config)
        self.q = Linear(config.width, config.width, bias=False)
        self.k = Linear(config.width, config.width, bias=False)
        self.v = Linear(config.width, config.width, bias=False)
        self.proj = Linear(config.width, config.width, bias=False)
        self.n_heads = config.n_heads
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels = x.shape
        def heads(t: Tensor) -> Tensor:
            return t.view(batch, time, self.n_heads, channels // self.n_heads).transpose(1, 2)
        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        # Attention scores and softmax remain real-valued in every arm.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        Linear = _linear_factory(config.ffn_layer_kind or config.layer_kind, config)
        hidden = config.width * config.mlp_ratio
        self.up = Linear(config.width, hidden, bias=False)
        self.down = Linear(hidden, config.width, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Split GELU: independently applied to each real component.
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.width)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.width)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.lm_head.weight = self.token_embedding.weight  # real-valued, tied head

    def forward(self, tokens: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        _, time = tokens.shape
        if time > self.config.context_length:
            raise ValueError("sequence exceeds context length")
        positions = torch.arange(time, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: Tensor, new_tokens: int, temperature: float = 0.8) -> Tensor:
        for _ in range(new_tokens):
            logits, _ = self(tokens[:, -self.config.context_length:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probs, 1)), dim=1)
        return tokens


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
