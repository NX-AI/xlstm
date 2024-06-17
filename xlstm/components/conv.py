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


class CausalConv1d(nn.Module):
    config_class = CausalConv1dConfig
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, config: CausalConv1dConfig):
        super().__init__()
        self.config = config
        self.groups = self.config.feature_dim
        if self.config.channel_mixing:
            self.groups = 1
        if self.config.kernel_size == 0:
            self.conv = None  # Noop
        else:
            self.pad = (
                self.config.kernel_size - 1
            )  # padding of this size assures temporal causality.
            self.conv = nn.Conv1d(
                in_channels=self.config.feature_dim,
                out_channels=self.config.feature_dim,
                kernel_size=self.config.kernel_size,
                padding=self.pad,
                groups=self.groups,
                bias=self.config.causal_conv_bias,
                **self.config.conv1d_kwargs,
            )
        # B, C, L
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        self.conv.reset_parameters()

    def _create_weight_decay_optim_groups(
        self,
    ) -> tuple[set[nn.Parameter], set[nn.Parameter]]:
        if self.config.kernel_size == 0:
            return (), ()
        else:
            weight_decay = (self.conv.weight,)
            no_weight_decay = ()
            if self.config.causal_conv_bias:
                no_weight_decay += (self.conv.bias,)
            return weight_decay, no_weight_decay

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        return_last_state: bool = False,
    ) -> torch.Tensor:
        if conv_state is not None:
            x = torch.cat([conv_state, x], dim=1)

        if self.config.kernel_size == 0:
            return x
        y = x.transpose(2, 1)  # (B,F,T) tensor - now in the right shape for conv layer.
        y = self.conv(y)  # (B,F,T+pad) tensor
        if conv_state is not None:
            y = y[:, :, conv_state.shape[1] :]

        if return_last_state:
            return y[:, :, : -self.pad].transpose(2, 1), x[:, -self.pad :]
        else:
            return y[:, :, : -self.pad].transpose(2, 1)

    def step(
        self,
        x: torch.Tensor,
        conv_state: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:

        if self.config.kernel_size == 0:
            return x, conv_state

        B, S, D = x.shape

        if conv_state is None:
            conv_state = (
                torch.zeros(
                    size=(B, self.config.kernel_size, D),
                    device=self.conv.weight.device,
                    dtype=self.conv.weight.dtype,
                ),
            )

        y, conv_state = conv1d_step(
            x,
            conv_state[0],
            self.conv.weight[:, 0, :].transpose(0, 1),  # rearrange(, "D 1 KS -> KS D")
            conv1d_bias=self.conv.bias if self.config.causal_conv_bias else None,
        )
        return y, (conv_state,)
