# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbininan Pöppel

from typing import Callable
import torch

from .slstm import slstm_forward_pointwise as slstm_forward_pointwise_slstm
from .lstm import slstm_forward_pointwise as slstm_forward_pointwise_lstm


slstm_pointwise_function_registry: dict[str, Callable] = {
    "slstm": slstm_forward_pointwise_slstm,
    "lstm": slstm_forward_pointwise_lstm,
}


def slstm_forward(
    x: torch.Tensor,  # [S, B, G*I]
    states: torch.Tensor,  # [4, B, H] only the first is used for recurrence!
    R: torch.Tensor,  # [K, R*H, H] - K num_heads
    b: torch.Tensor,  # [T*H]
    pointwise_forward: Callable[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]],
        tuple[torch.Tensor, torch.Tensor],
    ],
    constants: dict[str, float] = {},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    num_gates_r = (
        R.shape[1] // R.shape[2]
    )  # this only works for a fully-connected RNN, for a hin change this
    hidden_dim = R.shape[2] * R.shape[0]
    num_gates_t = b.shape[0] // hidden_dim
    batch_dim = x.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    g = torch.zeros(
        [sequence_dim + 1, num_gates_t, batch_dim, hidden_dim],
        device=x.device,
        dtype=x.dtype,
    )

    states_all = torch.zeros(
        [num_states, sequence_dim + 1, batch_dim, hidden_dim],
        device=x.device,
        dtype=x.dtype,
    )
    states_all[:, 0] = states
    for i, Wx_t in enumerate(x.unbind(dim=0)):
        Ry = (
            states[0]
            .reshape(batch_dim, num_heads, 1, -1)
            .matmul(
                R.transpose(1, 2).reshape(
                    1, num_heads, head_dim, num_gates_r * head_dim
                )
            )
            .reshape(batch_dim, num_heads, num_gates_r, -1)
            .transpose(1, 2)
            .reshape(batch_dim, -1)
        )
        sdtype = states.dtype
        states, gates = pointwise_forward(Wx_t, Ry, b, states, constants=constants)
        states = states.to(dtype=sdtype)
        g[i] = gates
        states_all[:, i + 1] = states

    # shapes ([S, B, H], ([B,H], [B,H], [B,H]), [S, B, 4*H])
    return states_all, states, g


def slstm_forward_step(
    x: torch.Tensor,  # [B, G*I]
    states: torch.Tensor,  # [4, B, H] only the first is used for recurrence!
    R: torch.Tensor,  # [K, R*H, H] - K num_heads
    b: torch.Tensor,  # [T*H]
    pointwise_forward: Callable[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]],
        tuple[torch.Tensor, torch.Tensor],
    ],
    constants: dict[str, float] = {},
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_states = states.shape[0]
    sequence_dim = x.shape[0]
    num_gates_r = (
        R.shape[1] // R.shape[2]
    )  # this only works for a fully-connected RNN, for a hin change this
    hidden_dim = R.shape[2] * R.shape[0]
    num_gates_t = b.shape[0] // hidden_dim
    batch_dim = x.shape[1]
    num_heads = R.shape[0]
    head_dim = R.shape[2]

    assert batch_dim == states.shape[1]
    assert hidden_dim == states.shape[2]

    g = torch.zeros(
        [sequence_dim + 1, num_gates_t, batch_dim, hidden_dim],
        device=x.device,
        dtype=x.dtype,
    )

    states_all = torch.zeros(
        [num_states, sequence_dim + 1, batch_dim, hidden_dim],
        device=x.device,
        dtype=x.dtype,
    )
    states_all[:, 0] = states
    Ry = (
        states[0]
        .reshape(batch_dim, num_heads, 1, -1)
        .matmul(
            R.transpose(1, 2).reshape(1, num_heads, head_dim, num_gates_r * head_dim)
        )
        .reshape(batch_dim, num_heads, num_gates_r, -1)
        .transpose(1, 2)
        .reshape(batch_dim, -1)
    )
    sdtype = states.dtype
    states, gates = pointwise_forward(x[0], Ry, b, states, constants=constants)
    states = states.to(dtype=sdtype)

    # shapes ([S, B, H], ([B,H], [B,H], [B,H]), [S, B, 4*H])
    return states[:, None, ...], g[:, None, ...]
