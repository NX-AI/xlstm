# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel

from dataclasses import dataclass, field
from typing import Optional

from ...components.feedforward import FeedForwardConfig
from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import sLSTMLayerConfig


@dataclass
class sLSTMBlockConfig:
    slstm: sLSTMLayerConfig = field(default_factory=sLSTMLayerConfig)
    feedforward: Optional[FeedForwardConfig] = field(default_factory=FeedForwardConfig)

    _num_blocks: int = 1
    _block_idx: int = 0

    def __post_init__(self):
        self.slstm._block_idx = self._block_idx
        self.slstm._num_blocks = self._num_blocks
        self.slstm.__post_init__()
        if self.feedforward is not None:
            self.feedforward.__post_init__()


class sLSTMBlock(xLSTMBlock):
    config_class = sLSTMBlockConfig

    def __init__(self, config: sLSTMBlockConfig):
        super().__init__(
            xLSTMBlockConfig(
                mlstm=None,
                slstm=config.slstm,
                feedforward=config.feedforward,
                _block_idx=config._block_idx,
                _num_blocks=config._num_blocks,
            )
        )
