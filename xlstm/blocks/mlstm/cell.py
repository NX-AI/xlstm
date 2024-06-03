# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

import torch
from torch import nn

from ...components.init import bias_linspace_init_
from ...components.ln import MultiHeadLayerNorm
from .backends import parallel_stabilized_simple, recurrent_step_stabilized_simple


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1


class mLSTMCell(nn.Module):
    config_class = mLSTMCellConfig

    def __init__(self, config: mLSTMCellConfig):
        super().__init__()
        self.config = config

        self.backend_fn = parallel_stabilized_simple
        self.backend_fn_step = recurrent_step_stabilized_simple

        self.igate = nn.Linear(3 * config.embedding_dim, config.num_heads)
        self.fgate = nn.Linear(3 * config.embedding_dim, config.num_heads)

        self.outnorm = MultiHeadLayerNorm(ndim=config.embedding_dim, weight=True, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        h_state = self.backend_fn(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=self.causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mlstm_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, S, _ = q.shape  # (B, S, H)
        assert S == 1, f"mLSTMCell.step only supports sequence length S=1, but got S={S}."

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        _, _, NH, DH = q.shape

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)

        if mlstm_state is None:
            c_state = torch.zeros(size=(B, NH, DH, DH), device=q.device, dtype=q.dtype)
            n_state = torch.zeros(size=(B, NH, DH, 1), device=q.device, dtype=q.dtype)
            m_state = torch.zeros(size=(B, NH, 1, 1), device=q.device, dtype=q.dtype)
        else:
            c_state, n_state, m_state = mlstm_state
            c_state = c_state.to(device=q.device, dtype=q.dtype)
            n_state = n_state.to(device=q.device, dtype=q.dtype)
            m_state = m_state.to(device=q.device, dtype=q.dtype)

        assert c_state.shape == (B, NH, DH, DH), f"Expected c_state shape {(B, NH, DH, DH)}, but got {c_state.shape}."
        assert n_state.shape == (B, NH, DH, 1), f"Expected n_state shape {(B, NH, DH, 1)}, but got {n_state.shape}."
        assert m_state.shape == (B, NH, 1, 1), f"Expected m_state shape {(B, NH, 1, 1)}, but got {m_state.shape}."

        h_state, mlstm_state = self.backend_fn_step(
            c_state=c_state,
            n_state=n_state,
            m_state=m_state,
            q=q,
            k=k,
            v=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
        )  # (B, NH, 1 DH), ((B, NH, DH, DH), (B, NH, DH, 1), (B, NH, 1, 1))

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm, mlstm_state

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
