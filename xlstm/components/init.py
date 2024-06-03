# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
import math

import torch


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py."""
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param
