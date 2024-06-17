# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
from dataclasses import dataclass, field
from typing import Optional

import torch

# from einops import rearrange
from torch import nn


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


def conv1d_step(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    B: batch size
    S: sequence length
    D: feature dimension
    KS: kernel size
    Args:
        x (torch.Tensor): (B, S, D)
        conv_state (torch.Tensor): (B, KS, D)
        conv1d_weight (torch.Tensor): (KS, D)
    """
    assert (
        x.shape[0] == conv_state.shape[0]
    ), f"x has batch size {x.shape[0]} but conv_state has batch size {conv_state.shape[0]}"
    assert (
        x.shape[2] == conv_state.shape[2]
    ), f"x has feature dimension {x.shape[2]} but conv_state has feature dimension {conv_state.shape[2]}"
    assert x.shape[1] == 1, f"x has sequence length {x.shape[1]} but it should be 1"
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=1))
    conv_state[:, -1:, :] = x
    y = torch.sum(conv_state * conv1d_weight, dim=1, keepdim=True)
    if conv1d_bias is not None:
        y += conv1d_bias
    return y, conv_state
