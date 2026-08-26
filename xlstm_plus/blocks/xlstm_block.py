# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Modifications Copyright (c) 2026 LeZeez
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..components.feedforward import FeedForwardConfig, create_feedforward
from ..components.ln import LayerNorm
from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from .slstm.layer import sLSTMLayer, sLSTMLayerConfig



@dataclass
class xLSTMBlockConfig:
    mlstm: Optional[mLSTMLayerConfig] = None
    slstm: Optional[sLSTMLayerConfig] = None

    feedforward: Optional[FeedForwardConfig] = None
    
    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        assert self.mlstm is not None or self.slstm is not None, "Either mlstm or slstm must be provided"
        assert self.mlstm is None or self.slstm is None, "Only one of mlstm or slstm can be provided"
        embedding_dim = self.mlstm.embedding_dim if self.mlstm is not None else self.slstm.embedding_dim
        if self.mlstm:
            self.mlstm._num_blocks = self._num_blocks
            self.mlstm._block_idx = self._block_idx
        if self.slstm:
            self.slstm._num_blocks = self._num_blocks
            self.slstm._block_idx = self._block_idx
        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()


class xLSTMBlock(nn.Module):
    """An xLSTM block can be either an sLSTM Block or an mLSTM Block.

    It contains the pre-LayerNorms and the skip connections.
    """

    config_class = xLSTMBlockConfig

    def __init__(self, config: xLSTMBlockConfig) -> None:
        super().__init__()
        self.config = config
        embedding_dim = (
            self.config.mlstm.embedding_dim if self.config.mlstm is not None else self.config.slstm.embedding_dim
        )

        self.xlstm_norm = LayerNorm(ndim=embedding_dim, weight=True, bias=False)

        if self.config.mlstm is not None:
            self.xlstm = mLSTMLayer(config=self.config.mlstm)
        elif self.config.slstm is not None:
            self.xlstm = sLSTMLayer(config=self.config.slstm)
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        if self.config.feedforward is not None:
            self.ffn_norm = LayerNorm(ndim=self.config.feedforward.embedding_dim, weight=True, bias=False)
            self.ffn = create_feedforward(config=self.config.feedforward)
        else:
            self.ffn_norm = None
            self.ffn = None

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        return_last_state: bool = False,
        return_detached_states: bool = False,
        **kwargs,
    ):
        """Forward pass of the xLSTM block.

        Args:
            x: (B, S, D)
            boundaries: Optional bool (B, S) for document boundary reset.
            state: Optional carry state dict from previous chunk.
            return_last_state: Return state after this block.
            return_detached_states: Detach state tensors before returning.
        """
        need_state = return_last_state or (state is not None)
        # Unpack state whether supplied as dict (sLSTM) or tuple (mLSTM)
        if isinstance(state, dict):
            conv_state  = state.get("conv_state",  None)
            slstm_state = state.get("slstm_state", None)
            mlstm_state = state.get("mlstm_state", None)
        elif isinstance(state, tuple):
            conv_state  = None
            slstm_state = None
            mlstm_state = state
        else:
            conv_state  = None
            slstm_state = None
            mlstm_state = None

        if need_state:
            xlstm_out, xlstm_state = self.xlstm(
                self.xlstm_norm(x),
                boundaries=boundaries,
                # mLSTM layer uses `state` kwarg; sLSTM uses positional conv/slstm_state
                state=mlstm_state,
                conv_state=conv_state,
                slstm_state=slstm_state,
                return_last_state=True,
                return_detached_states=return_detached_states,
                **kwargs,
            )
        else:
            xlstm_out = self.xlstm(
                self.xlstm_norm(x),
                boundaries=boundaries,
                **kwargs,
            )
            xlstm_state = None

        x = x + xlstm_out
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x))

        if need_state:
            return x, xlstm_state
        return x

    def step(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        x_xlstm, xlstm_state = self.xlstm.step(self.xlstm_norm(x), **kwargs)
        x = x + x_xlstm
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_norm(x), **kwargs)
        return x, xlstm_state

    def reset_parameters(self) -> None:
        self.xlstm.reset_parameters()
        self.xlstm_norm.reset_parameters()
        if self.ffn is not None:
            self.ffn.reset_parameters()
            self.ffn_norm.reset_parameters()
