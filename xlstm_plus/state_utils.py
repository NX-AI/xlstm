# Copyright (c) NXAI GmbH and its affiliates 2024
# Modifications Copyright (c) 2026 LeZeez (xlstm-plus) - Apache-2.0
"""State management utilities for xlstm-plus.

Provides:
    _BOUNDARY_RESET_LOGF – sentinel used to zero the forget gate at document boundaries.
    detach_states        – recursively detach every tensor in a nested state structure.
    zero_rows            – in-place zero selected batch rows across a nested state.

State-format compatibility
--------------------------
xlstm_large   : dict[int, tuple[Tensor, Tensor, Tensor]]   (C, n, m) per block index
classic stack : dict[str, dict[str, tuple[Tensor, ...]]]   keyed "block_0", "block_1", …
Both formats are handled transparently by detach_states / zero_rows because they recurse
through arbitrary dict / list / tuple nesting before touching individual tensors.

Type aliases (re-exported for user convenience)
-----------------------------------------------
mLSTMLayerStateType   = tuple[Tensor, Tensor, Tensor]          # (C, n, m)
mLSTMStateType        = dict[int, mLSTMLayerStateType]         # xlstm_large
ClassicBlockStateType = dict[str, tuple[Tensor, ...]]          # single classic block
ClassicStackStateType = dict[str, ClassicBlockStateType]       # full classic stack
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

# ---------------------------------------------------------------------------
# Shared constant
# ---------------------------------------------------------------------------

_BOUNDARY_RESET_LOGF: float = -1000.0
"""Sentinel value applied to the forget-gate pre-activation at document boundaries.

logsigmoid(-1000) ≈ -1000  →  exp(-1000) ≈ 0, so the memory carry (C, n) is wiped out
cleanly across backends without cross-document attention leakage.
"""

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

mLSTMLayerStateType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = Dict[int, mLSTMLayerStateType]
ClassicBlockStateType = Dict[str, Tuple[torch.Tensor, ...]]
ClassicStackStateType = Dict[str, ClassicBlockStateType]

# ---------------------------------------------------------------------------
# detach_states
# ---------------------------------------------------------------------------

def detach_states(
    state: Union[Dict, List, Tuple, torch.Tensor, Any],
) -> Union[Dict, List, Tuple, torch.Tensor, Any]:
    """Recursively detach every tensor in a nested state structure.

    Supports:
        * ``dict``  – keys are preserved; values are recursed.
        * ``list`` / ``tuple`` – element-wise recursion; type is preserved.
        * Any object with a ``.detach()`` method (raw ``torch.Tensor`` included).
        * Anything else is returned as-is.

    This covers both the xlstm_large state format ``dict[int, tuple[...]]``
    and the classic stack state format ``dict[str, dict[str, tuple[...]]]``.

    Usage::

        state = detach_states(state)   # after loss.backward(), before next chunk

    Args:
        state: Nested state produced by any xlstm-plus forward pass.

    Returns:
        A new structure of the same shape with every tensor detached.
    """
    if isinstance(state, dict):
        return {k: detach_states(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        detached = (detach_states(v) for v in state)
        return type(state)(detached)
    if hasattr(state, "detach"):
        return state.detach()
    return state


# ---------------------------------------------------------------------------
# zero_rows
# ---------------------------------------------------------------------------

def zero_rows(
    state: Union[Dict, List, Tuple, torch.Tensor, Any],
    mask: torch.Tensor,
    batch_dim: Optional[int] = None,
) -> None:
    """In-place zero selected batch rows across a nested state structure.

    ``mask`` must be a 1-D boolean tensor of shape ``(B,)`` where ``True``
    indicates rows that should be zeroed (e.g. sequences that ended).

    **Only call on detached states.**  Any tensor that still requires grad will
    raise a ``RuntimeError``.

    State-format rules (when ``batch_dim`` is not given)
    ----------------------------------------------------
    The function inspects each individual tensor's shape:

    * If ``tensor.shape[0] == B``  → batch is dim 0  (standard case).
    * If ``tensor.ndim in {4, 5}`` and ``tensor.shape[1] == B``
      → batch is dim 1  (multi-direction states like ``(D, B, …)``).
    * Both matching simultaneously is ambiguous → ``ValueError``; use
      ``batch_dim`` explicitly in that case.

    This logic handles all state formats produced by xlstm-plus:

    * ``dict[int, tuple[Tensor C, Tensor n, Tensor m]]``  (xlstm_large)
    * ``dict[str, dict[str, tuple[Tensor, ...]]]``        (classic stack)

    Args:
        state:     Nested state returned by any xlstm-plus forward pass.
        mask:      Boolean tensor of shape ``(B,)``.
        batch_dim: Explicit batch dimension index.  Auto-detected when ``None``.

    Raises:
        RuntimeError: If any tensor still requires grad.
        ValueError:   If the batch dimension cannot be determined unambiguously.
    """
    if isinstance(state, dict):
        for v in state.values():
            zero_rows(v, mask, batch_dim=batch_dim)
        return
    if isinstance(state, (list, tuple)):
        for v in state:
            zero_rows(v, mask, batch_dim=batch_dim)
        return
    if not isinstance(state, torch.Tensor):
        return

    t = state
    if t.requires_grad:
        raise RuntimeError(
            "zero_rows: encountered a tensor that requires grad. "
            "Call detach_states() or pass return_detached_states=True before zero_rows()."
        )

    B = mask.shape[0]
    if batch_dim is not None:
        idx = [slice(None)] * t.dim()
        idx[batch_dim] = mask
        t[tuple(idx)] = 0
        return

    dim0_match = t.shape[0] == B
    dim1_match = t.dim() in (4, 5) and t.shape[1] == B
    if dim0_match and dim1_match:
        raise ValueError(
            f"zero_rows: ambiguous batch dimension for tensor of shape "
            f"{list(t.shape)} (both dim 0 and dim 1 match mask length {B}). "
            "Pass batch_dim explicitly."
        )
    if dim1_match:
        t[:, mask] = 0
    elif dim0_match:
        t[mask] = 0
    else:
        raise ValueError(
            f"zero_rows: tensor shape {list(t.shape)} does not match "
            f"mask length {B} on dim 0 or dim 1."
        )
