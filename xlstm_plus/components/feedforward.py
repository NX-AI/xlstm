# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

from ..utils import UpProjConfigMixin
from .init import small_init_init_, wang_init_

_act_fn_registry = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "relu^2": lambda x: torch.square(nn.functional.relu(x)),
    "sigmoid": nn.functional.sigmoid,
    "swish": nn.functional.silu,
    "selu": nn.functional.selu,
}


def get_act_fn(act_fn_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if act_fn_name in _act_fn_registry:
        return _act_fn_registry[act_fn_name]
    else:
        assert (
            False
        ), f'Unknown activation function name "{act_fn_name}". Available activation functions are: {str(_act_fn_registry.keys())}'


@dataclass
class FeedForwardConfig(UpProjConfigMixin):
    proj_factor: float = 1.3
    act_fn: str = "gelu"
    embedding_dim: int = -1
    dropout: float = 0.0
    bias: bool = False
    ff_type: Literal["ffn_gated"] = "ffn_gated"

    _num_blocks: int = 1

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        assert (
            self.act_fn in _act_fn_registry
        ), f"Unknown activation function {self.act_fn}"


class GatedFeedForward(nn.Module):
    config_class = FeedForwardConfig

    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self.config = config

        self.proj_up = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=2 * self.config._proj_up_dim,
            bias=self.config.bias,
        )
        self.proj_down = nn.Linear(
            in_features=self.config._proj_up_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.bias,
        )

        self.act_fn = get_act_fn(self.config.act_fn)

        self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        gate_preact, up_proj = self.proj_up(x).split(self.config._proj_up_dim, dim=-1)
        x = self.dropout(self.proj_down(self.act_fn(gate_preact) * up_proj))
        return x

    def reset_parameters(self):
        small_init_init_(self.proj_up.weight, dim=self.config.embedding_dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        wang_init_(
            self.proj_down.weight,
            dim=self.config.embedding_dim,
            num_blocks=self.config._num_blocks,
        )
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)


def create_feedforward(config: FeedForwardConfig) -> nn.Module:
    if config.ff_type == "ffn_gated":
        return GatedFeedForward(config)
    else:
        raise ValueError(f"Unknown feedforward type {config.ff_type}")
