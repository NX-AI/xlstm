# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Pöppel, Andreas Auer

import numpy as np


# same as a half-add operation (but expanded to larger numbers eventually)
def parity(
    *,
    batch_size: int = 1,
    vocab_size: int = 3,
    min_sequence_length: int | None = None,
    max_sequence_length: int | None = None,
    context_length: int = 20,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    max_sequence_length = context_length if max_sequence_length is None else max_sequence_length
    min_sequence_length = max_sequence_length if min_sequence_length is None else min_sequence_length

    res = np.zeros([batch_size, context_length], dtype=np.int32)
    res[:, :-1] = rng.integers(1, vocab_size, size=[batch_size, context_length - 1])
    sizes = rng.integers(min_sequence_length, max_sequence_length + 1, size=[batch_size])
    prediction_mask = np.zeros_like(res)
    for batch_idx in range(batch_size):
        res[batch_idx, sizes[batch_idx] - 1 :] = np.zeros_like(res[batch_idx, sizes[batch_idx] - 1 :])
        par = (np.sum(res[batch_idx, : sizes[batch_idx] - 1]) - sizes[batch_idx] + 1) % (vocab_size - 1) + 1
        res[batch_idx, sizes[batch_idx] - 1] = par
        prediction_mask[batch_idx, sizes[batch_idx] - 1 : sizes[batch_idx]] = 1

    return res, prediction_mask
