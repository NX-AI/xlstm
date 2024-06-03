# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

import torch
from torch import nn

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init_init_, wang_init_
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...utils import UpProjConfigMixin
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1

    _num_blocks: int = 1
    _inner_embedding_dim: int = None

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim


class mLSTMLayer(nn.Module):
    config_class = mLSTMLayerConfig

    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.proj_up = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=2 * self.config._inner_embedding_dim,
            bias=self.config.bias,
        )

        num_proj_heads = round(self.config._inner_embedding_dim // self.config.qkv_proj_blocksize)
        self.q_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )
        self.k_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )
        self.v_proj = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
            )
        )

        self.conv1d = CausalConv1d(
            config=CausalConv1dConfig(
                feature_dim=self.config._inner_embedding_dim,
                kernel_size=self.config.conv1d_kernel_size,
            )
        )
        self.conv_act_fn = nn.SiLU()
        self.mlstm_cell = mLSTMCell(
            config=mLSTMCellConfig(
                context_length=self.config.context_length,
                embedding_dim=self.config._inner_embedding_dim,
                num_heads=self.config.num_heads,
            )
        )
        self.ogate_act_fn = nn.SiLU()

        self.learnable_skip = nn.Parameter(torch.ones(self.config._inner_embedding_dim, requires_grad=True))

        self.proj_down = nn.Linear(
            in_features=self.config._inner_embedding_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.bias,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = x.shape

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.split(x_inner, split_size_or_sections=self.config._inner_embedding_dim, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = self.dropout(self.proj_down(h_state))
        return y

    def step(
        self,
        x: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        conv_state: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        B, S, _ = x.shape

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.split(x_inner, split_size_or_sections=self.config._inner_embedding_dim, dim=-1)

        # mlstm branch
        x_mlstm_conv, conv_state = self.conv1d.step(x_mlstm, conv_state=conv_state)
        x_mlstm_conv_act = self.conv_act_fn(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state, mlstm_state = self.mlstm_cell.step(q=q, k=k, v=v, mlstm_state=mlstm_state)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * self.ogate_act_fn(z)

        # down-projection
        y = self.dropout(self.proj_down(h_state))
        return y, {"mlstm_state": mlstm_state, "conv_state": conv_state}

    def reset_parameters(self):
        # init inproj
        small_init_init_(self.proj_up.weight, dim=self.config.embedding_dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj
        wang_init_(self.proj_down.weight, dim=self.config.embedding_dim, num_blocks=self.config._num_blocks)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_init_(qkv_proj.weight, dim=self.config.embedding_dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()
