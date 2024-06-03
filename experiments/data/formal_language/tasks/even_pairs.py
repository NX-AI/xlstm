# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Pöppel, Andreas Auer
from typing import Optional

import numpy as np


def even_pairs(
    batch_size: int = 1,
    vocab_size: int = 3,
    min_sequence_length: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    context_length: int = 20,
    pad_idx: int = 0,
    seed: int = 42,
    **kwargs
):
    max_sequence_length = context_length if max_sequence_length is None else max_sequence_length
    min_sequence_length = max_sequence_length if min_sequence_length is None else min_sequence_length
    rng = np.random.default_rng(seed)
    res = np.zeros([batch_size, context_length], dtype=np.int32)
    res[:, :-1] = rng.integers(vocab_size-1, size=[batch_size, context_length - 1]) + 1
    sizes = rng.integers(min_sequence_length, max_sequence_length + 1, size=[batch_size])
    prediction_mask = np.zeros_like(res)

    diffs = res[:, 1:-1] - res[:, :-2]
    cumdiffs = np.cumsum(diffs, axis=1)

    for batch_idx in range(batch_size):
        res[batch_idx, sizes[batch_idx] : ] = 0
        prediction_mask[batch_idx, sizes[batch_idx] - 1] = 1
        
        res[batch_idx, sizes[batch_idx] - 1] = np.abs(cumdiffs[batch_idx, sizes[batch_idx] - 3]) + 1

    return res, prediction_mask