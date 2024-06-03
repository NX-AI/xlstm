# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Pöppel, Andreas Auer
from typing import Optional

import numpy as np


def cycle_navigation(
    *,
    batch_size: int = 1,
    vocab_size: int = 10, # equivalent to  1 ([PAD]) + 1+2*max_step_size (e.g. 0,+/-1) + #cycle positions, 6 is minimum
    max_step_size: int = 1,
    min_sequence_length: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    context_length: int = 20,
    pad_idx: int = 0,
    seed: int = 42,
    **kwargs,
):
    rng = np.random.default_rng(seed)

    max_step_size = min(vocab_size // 2, max_step_size)

    max_sequence_length = context_length if max_sequence_length is None else max_sequence_length
    min_sequence_length = max_sequence_length if min_sequence_length is None else min_sequence_length
    
    res = np.zeros([batch_size, context_length], dtype=np.int32)
    res = rng.integers(2*max_step_size + 1, size=[batch_size, context_length]) + 1
    sizes = rng.integers(min_sequence_length, max_sequence_length + 1, size=[batch_size])

    cycle_size = vocab_size - 2 - 2*max_step_size
    # print("CYCLE SIZE: ", cycle_size)

    prediction_mask = np.zeros_like(res)
    for batch_idx in range(batch_size):
        res[batch_idx, sizes[batch_idx]-1:] = 0

        steps = np.where(
            res[batch_idx, :sizes[batch_idx]-1] <= max_step_size + 1,
            res[batch_idx, :sizes[batch_idx]-1] - 1,
            max_step_size + 1 - res[batch_idx, :sizes[batch_idx]-1])
        # print("STEP", steps)
        step_sum = np.sum(steps)
        # print("SUM", step_sum)
        res[batch_idx, sizes[batch_idx]-1] = step_sum % (cycle_size) + 2*max_step_size + 2
        prediction_mask[batch_idx, sizes[batch_idx]-1]  = 1
    
    return res, prediction_mask


cycle_navigation_dict = {
    0: "PD",
    1: " 0"
}