# Sparse Ternary Covariance (STC) Reconnaissance Report

## mLSTM Matrix Memory Update Implementation
The core matrix memory update logic for mLSTM is implemented in the recurrent step backend.

- **File Path:** `xlstm/xlstm/blocks/mlstm/backends.py`
- **Function:** `recurrent_step_stabilized_simple`
- **Tensor Shapes:**
    - `q, k, v`: `(B, NH, DH, 1)` (after squeezing/unsqueezing in the function)
    - `c_state`: `(B, NH, DH, DH)`
    - `n_state`: `(B, NH, DH, 1)`
    - `m_state`: `(B, NH, 1, 1)`
- **Update Path (Covariance):**
    ```python
    c_state_new = fg_act * c_state + ig_act * (k_scaled @ v.transpose(-1, -2))
    ```
    This is the dense outer-product update $C_t = \lambda C_{t-1} + v_t k_t^T$.

## Existing Benchmarks / Profiling
- **Experiments:** `experiments/main.py` (Parity task, Multi-Query Associative Recall).
- **Tests:** `tests/test_chunkwise_vs_recurrent.py` and `tests/template_chunkwise_vs_recurrent.py` compare different backends for numerical parity.
- **Profiling:** No dedicated profiling suite found, but `tests/template_chunkwise_vs_recurrent.py` can be adapted for latency measurements.

## Extension Infrastructure
- **Triton Kernels:** Present in `xlstm/xlstm_large` (requires GPU).
- **CUDA Kernels:** Mentioned for sLSTM in `xlstm/xlstm/blocks/slstm`.
- **Backend Selection:** `mLSTMCell` and `mLSTMLayer` allow selecting different backends via config/attributes.

## Benchmark Entry Points
- `experiments/main.py` for high-level task performance.
- `tests/template_chunkwise_vs_recurrent.py` for low-level kernel comparison.
