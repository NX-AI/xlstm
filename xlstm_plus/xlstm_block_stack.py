# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
# Modifications Copyright (c) 2026 LeZeez
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
from torch import nn

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
from .components.ln import LayerNorm


@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: Optional[mLSTMBlockConfig] = None
    slstm_block: Optional[sLSTMBlockConfig] = None

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: Union[list[int], Literal["all"]] = field(default_factory=list)

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    _block_map: str = None

    @property
    def block_map(self) -> list[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        """Creates the block map, that specifies which block is used at which position."""
        block_map = [0] * self.num_blocks

        for slstm_position_idx in self.slstm_at:
            assert slstm_position_idx < self.num_blocks, f"Invalid slstm position {slstm_position_idx}"
            block_map[slstm_position_idx] = 1

        block_map_str = ",".join(map(str, block_map))

        return block_map_str

    def __post_init__(self):
        if self.mlstm_block is None:
            self.slstm_at = "all"
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))

        if self.mlstm_block is not None:
            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length
            
            self.mlstm_block._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()

        self._block_map = self._create_block_map()


class xLSTMBlockStack(nn.Module):
    config_class = xLSTMBlockStackConfig

    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config

        self.blocks = self._create_blocks(config=config)
        if config.add_post_blocks_norm:
            self.post_blocks_norm = LayerNorm(ndim=config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

    def _create_blocks(self, config: xLSTMBlockStackConfig):

        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(mLSTMBlock(config=config))
            elif block_type_int == 1:
                config = deepcopy(self.config.slstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(sLSTMBlock(config=config))
            else:
                raise ValueError(f"Invalid block type {block_type_int}")

        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
        return_last_states: bool = False,
        return_detached_states: bool = False,
        **kwargs,
    ):
        """Forward pass of the xLSTM block stack.

        Args:
            x: (B, S, D)
            boundaries: Optional bool (B, S) document-boundary mask.
            state: Optional ``dict[str, dict]`` keyed ``"block_0"``, ``"block_1"`` …
                carrying per-block states from a previous chunk.
            return_last_states: Return a state dict after the forward pass.
            return_detached_states: Detach all state tensors before returning.

        Returns:
            x (B, S, D), or ``(x, state_dict)`` when states are requested.
        """
        need_state = return_last_states or (state is not None)
        if state is None:
            state = {}

        new_state: dict = {}
        for block_idx, block in enumerate(self.blocks):
            b_key = f"block_{block_idx}"
            b_state = state.get(b_key, None)

            if need_state:
                x, b_state_new = block(
                    x,
                    boundaries=boundaries,
                    state=b_state,
                    return_last_state=True,
                    return_detached_states=return_detached_states,
                    **kwargs,
                )
                new_state[b_key] = b_state_new
            else:
                x = block(x, boundaries=boundaries, **kwargs)

        x = self.post_blocks_norm(x)

        if need_state:
            return x, new_state
        return x

    def step(
        self, x: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        if state is None:
            state = {}

        for block_idx, block in enumerate(self.blocks):
            x, state[f"block_{block_idx}"] = block.step(x, **state.get(f"block_{block_idx}", {}))

        x = self.post_blocks_norm(x)

        return x, state
