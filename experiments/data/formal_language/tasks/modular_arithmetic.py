# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbinian Pöppel, Andreas Auer
from typing import Optional

import numpy as np


def modular_arithmetic(
    *,
    batch_size: int = 1,
    vocab_size: int = 10, # equivalent to  1 ([PAD]) + 4 (+, -, *, =) + #numbers  - smallest sensible is 7
    min_sequence_length: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
    context_length: int = 20,
    seed: int = 42,
    **kwargs,
):
    rng = np.random.default_rng(seed)

    max_sequence_length = context_length if max_sequence_length is None else max_sequence_length
    min_sequence_length = max_sequence_length if min_sequence_length is None else min_sequence_length
    
    res = np.zeros([batch_size, context_length], dtype=np.int32)
    res = rng.integers(vocab_size - 5, size=[batch_size, context_length]) + 5
    res[:, 1::2] = rng.integers(3, size=[batch_size, context_length//2]) + 1
    min_seq = min((min_sequence_length+1)//2, (max_sequence_length + 1)//2)
    # print("MSEQ", min_seq, (max_sequence_length + 2)//2)
    sizes = 2*rng.integers(min_seq, (max_sequence_length + 1)//2+1, size=[batch_size])-1

    prediction_mask = np.zeros_like(res)
    max_num = vocab_size - 5
    # print("MAX_NUM", max_num)
    for batch_idx in range(batch_size):
        # print("SIZE", sizes[batch_idx])
        res[batch_idx, sizes[batch_idx]-2] = 4
        res[batch_idx, sizes[batch_idx]:] = 0
        total_val = 0
        tmp_val = 1
        prev_val = res[batch_idx, 0] - 5
        prev_op_sign = 1
        for n, num in enumerate(res[batch_idx, 1:sizes[batch_idx]-2]):
            # at an operator
            if n % 2 == 0:
                assert num > 0 and num <= 4
                # '*'
                if num != 3:
                    total_val += prev_op_sign * prev_val
                    # print("NEW TOTAL", total_val, prev_op_sign, prev_val)
                    prev_val = 1
                    prev_op_sign = -1 if num == 2 else 1
            else:
                prev_val *= (num - 5)
        total_val += prev_op_sign * prev_val
        res[batch_idx, sizes[batch_idx]-1] = total_val % (max_num) + 5
        # print("TOTAL", total_val)
        
        prediction_mask[batch_idx, sizes[batch_idx]-1]  = 1
    
    return res, prediction_mask


modular_arithmetic_dict = {
    0: "[PAD]",
    1: "+",
    2: "-",
    3: "*",
    4: "="
}
