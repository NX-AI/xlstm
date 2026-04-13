# Copyright (c) NXAI GmbH and its affiliates 2024
import torch

def stc_sparse_update(
    c_state: torch.Tensor,
    k_q: torch.Tensor,
    v_q: torch.Tensor,
    fg_act: torch.Tensor,
    ig_act: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse Ternary Covariance (STC) update.
    
    C_new = fg_act * C_prev + ig_act * (k_q @ v_q^T)
    
    If c_state is None, returns (k_q @ v_q^T).
    """
    # outer product (B, NH, DH, DH)
    outer = k_q @ v_q.transpose(-1, -2)
    
    if c_state is None:
        return outer
    
    return fg_act * c_state + ig_act * outer
