# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass, field

from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig:
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


class mLSTMBlock(xLSTMBlock):
    config_class = mLSTMBlockConfig

    def __init__(self, config: mLSTMBlockConfig) -> None:
        super().__init__(
            config=xLSTMBlockConfig(
                mlstm=config.mlstm,
                slstm=None,
                feedforward=None,
                _num_blocks=config._num_blocks,
                _block_idx=config._block_idx,
            )
        )
