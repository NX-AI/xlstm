# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .components.init import small_init_init_
from .utils import WeightDecayOptimGroupMixin
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False


class xLSTMLMModel(WeightDecayOptimGroupMixin, nn.Module):
    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, **kwargs):
        super().__init__()
        self.config = config

        self.xlstm_block_stack = xLSTMBlockStack(config=config)
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.emb_dropout = nn.Dropout(config.dropout) if config.add_embedding_dropout else nn.Identity()

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()

        small_init_init_(self.token_embedding.weight, dim=self.config.embedding_dim)

        if not self.config.tie_weights:
            small_init_init_(self.lm_head.weight, dim=self.config.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x = self.xlstm_block_stack(x)
        logits = self.lm_head(x)
        return logits

    def step(
        self, idx: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        # remove token embedding and add it to the correct group, accrording to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weight,)
        else:
            no_weight_decay += (self.token_embedding.weight,)

        return weight_decay, no_weight_decay
