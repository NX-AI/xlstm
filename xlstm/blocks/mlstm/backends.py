# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import math
from typing import Optional

import torch


def parallel_stabilized_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    stabilize_rowwise: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        lower_triangular_matrix (torch.Tensor, optional): (S,S). Defaults to None.
        stabilize_rowwise (bool, optional): Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


def recurrent_step_stabilized_simple(
    c_state: torch.Tensor,
    n_state: torch.Tensor,
    m_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        c_state (torch.Tensor): (B, NH, DH, DH)
        n_state (torch.Tensor): (B, NH, DH, 1)
        m_state (torch.Tensor): (B, NH, 1, 1)
        q (torch.Tensor): (B, NH, 1, DH)
        k (torch.Tensor): (B, NH, 1, DH)
        v (torch.Tensor): (B, NH, 1, DH)
        igate_preact (torch.Tensor): (B, NH, 1, 1)
        fgate_preact (torch.Tensor): (B, NH, 1, 1)

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            (hidden_state [B, NH, DH], (c_state_new [B, NH, DH, DH], n_state_new [B, NH, DH, 1]], m_state_new [B, NH, 1, 1]))
    """
    B, NH, S, DH = q.shape
    # projections
    q, k, v = q.squeeze_(2).unsqueeze(-1), k.squeeze_(2).unsqueeze(-1), v.squeeze_(2).unsqueeze(-1)  # (B, NH, DH, 1)

    # gates
    log_fg_act = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, 1, 1)

    # update rule
    m_state_new = torch.max(log_fg_act + m_state, igate_preact)  # (B, NH, 1, 1)

    fg_act = torch.exp(log_fg_act + m_state - m_state_new)  # (B, NH, 1, 1)
    ig_act = torch.exp(igate_preact - m_state_new)  # (B, NH, 1, 1)

    k_scaled = k / math.sqrt(DH)

    c_state_new = fg_act * c_state + ig_act * (k_scaled @ v.transpose(-1, -2))  # (B, NH, DH, DH)
    n_state_new = fg_act * n_state + ig_act * k_scaled  # (B, NH, DH, 1)

    h_num = q.transpose(-1, -2) @ c_state_new  # (B, NH, 1, DH)

    qn_dotproduct = q.transpose(-1, -2) @ n_state_new  # (B, NH, 1, 1)
    max_val = torch.exp(-m_state_new)  # (B, NH, 1, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + eps
    h = h_num / h_denom  # (B, NH, 1, DH) / (B, NH, 1, 1) = (B, NH, 1, DH)

    return h, (c_state_new, n_state_new, m_state_new)


def chunkwise_simple(
    queries: torch.Tensor,
    keys: torch.Tensor,  # B, NH, S, DH
    values: torch.Tensor,  # B, NH, S, DH
    igate_preact: torch.Tensor,  # B, NH, S
    fgate_preact: torch.Tensor,  # B, NH, S
    initial_C: Optional[torch.Tensor] = None,  # B, NH, DH, DH
    initial_n: Optional[torch.Tensor] = None,  # B, NH, DH
    initial_m: Optional[torch.Tensor] = None,  # B, NH, 1
    chunk_size: int = 64,  # optimize this
    return_last_state: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    B, NH, S, DH = queries.shape
    NS, CS = S // chunk_size, chunk_size
    _dtype, _device = queries.dtype, queries.device

    # form chunks
    q = queries.view(B, NH, NS, CS, DH) / math.sqrt(DH)
    k = keys.view(B, NH, NS, CS, DH)
    v = values.view(B, NH, NS, CS, DH)

    # forget gates
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact).view(B, NH, NS, CS)
    log_fgates_acc = log_fgates.cumsum(dim=3)
    igate_preact = igate_preact.view(B, NH, NS, CS)

    loggates = (igate_preact - log_fgates_acc)[:, :, :, :, None]
    m_loc, _ = torch.max(
        loggates + log_fgates_acc[:, :, :, -1, None, None], dim=3, keepdim=True
    )
    loggates = loggates + log_fgates_acc[:, :, :, -1, None, None] - m_loc

    kv = k.transpose(-1, -2) @ (v * (loggates).exp())
    ksum = (k * (loggates).exp()).sum(dim=-2)
    C = torch.zeros((B, NH, NS + 1, DH, DH), device=kv.device, dtype=kv.dtype)
    n = torch.zeros((B, NH, NS + 1, DH), device=kv.device, dtype=kv.dtype)
    if initial_C is not None:
        C[:, :, 0] = initial_C
    if initial_n is not None:
        n[:, :, 0] = initial_n

    m = torch.zeros((B, NH, NS + 1, 1, 1), device=kv.device, dtype=kv.dtype)
    if initial_m is not None:
        m[:, :, 0] = initial_m[:, :, None, None]

    for i in range(1, NS + 1):
        m[:, :, i] = torch.maximum(
            log_fgates_acc[:, :, i - 1, -1, None, None] + m[:, :, i - 1],
            m_loc[:, :, i - 1],
        )
        C[:, :, i] = (
            C[:, :, i - 1].clone()
            * (
                log_fgates_acc[:, :, i - 1, -1, None, None]
                + m[:, :, i - 1]
                - m[:, :, i]
            ).exp()
            + kv[:, :, i - 1] * (m_loc[:, :, i - 1] - m[:, :, i]).exp()
        )
        n[:, :, i] = (
            n[:, :, i - 1].clone()
            * (
                log_fgates_acc[:, :, i - 1, -1, None]
                + m[:, :, i - 1, 0]
                - m[:, :, i, 0]
            ).exp()
            + ksum[:, :, i - 1] * (m_loc[:, :, i - 1, 0] - m[:, :, i, 0]).exp()
        )

    log_fgates_rep = log_fgates_acc[:, :, :, :, None].repeat(1, 1, 1, 1, CS)
    log_fg_matrix = (
        log_fgates_rep
        - log_fgates_rep.transpose(-1, -2)
        - torch.triu(float("inf") * torch.ones([1, 1, 1, CS, CS]).to(q), diagonal=1)
    )

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact[:, :, :, :, None].transpose(
        -2, -1
    )  # (B, NH, NS, CS, CS)
    D_max, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)

    stab = torch.maximum(D_max, m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None])
    inter_C = (
        q * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
    ) @ C[:, :, :-1]
    inter_n = (
        q * (m[:, :, :-1, :] + log_fgates_acc[:, :, :, :, None] - stab).exp()
    ) @ n[:, :, :-1, :, None]

    # D matrix stabilization
    log_D_matrix_stabilized = log_D_matrix - stab  # (B, NH, NS, CS, CS)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, NS, CS, CS)

    # combination matrix C
    qk_matrix = q @ k.transpose(-2, -1)  # (B, NH, NS, CS, CS)
    E_matrix = qk_matrix * D_matrix  # (B, NH, NS, CS, CS)

    normalizer = torch.maximum(
        (E_matrix.sum(dim=-1, keepdim=True) + inter_n).abs(),
        torch.exp(-stab),
    )  # (B, NH, NS, CS, 1)

    E_matrix_normalized = E_matrix / (normalizer + eps)

    # retrieved values
    intra = E_matrix_normalized @ v  # (B, NH, S, DH)
    inter = inter_C / (normalizer + eps)

    if return_last_state:
        return (intra + inter).view((B, NH, S, DH)), (C[:, :, -1], n[:, :, -1], m[:, :, -1])
    else:
        return (intra + inter).view((B, NH, S, DH))

