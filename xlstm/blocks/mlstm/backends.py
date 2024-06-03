# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math

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
