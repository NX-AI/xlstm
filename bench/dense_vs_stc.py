import torch
import time
import numpy as np
from xlstm.blocks.mlstm.cell import mLSTMCell, mLSTMCellConfig

def benchmark_cell(mode="dense", B=1, NH=4, DH=32):
    config = mLSTMCellConfig(
        embedding_dim=NH*DH,
        num_heads=NH,
        context_length=256,
        memory_backend="stc_sparse" if mode == "stc_sparse" else "dense",
        gate_mode="ternary" if mode == "stc_sparse" else "sigmoid" # testing both if stc_sparse
    )
    cell = mLSTMCell(config)
    cell.eval()
    
    q = torch.randn(B, 1, NH*DH)
    k = torch.randn(B, 1, NH*DH)
    v = torch.randn(B, 1, NH*DH)
    
    # Warmup
    for _ in range(10):
        _ = cell.step(q, k, v)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    iters = 100
    for _ in range(iters):
        _ = cell.step(q, k, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    latency = (end - start) / iters * 1000 # ms
    tokens_per_sec = 1000 / latency
    
    # Calculate sparsity if stc_sparse
    sparsity = 0
    if mode == "stc_sparse":
        # We can't easily get it out without modifying the code or using hooks,
        # but we can estimate it based on the threshold.
        # Q(x) = 0 if |x| <= 0.1 * EMA(|x|)
        # For Gaussian, P(|x| < 0.1 * E|x|) is small.
        # But we can measure it by manually quantizing.
        k_scaled = k.view(B, 1, NH, DH) / (DH**0.5)
        k_q = cell.k_quantizer(k_scaled)
        v_q = cell.v_quantizer(v.view(B, 1, NH, DH))
        sparsity = ( (k_q == 0).float().mean() + (v_q == 0).float().mean() ) / 2
        
    return latency, tokens_per_sec, sparsity

if __name__ == "__main__":
    print(f"{'Mode':<15} | {'Latency (ms)':<15} | {'Tokens/sec':<15} | {'Sparsity':<15}")
    print("-" * 65)
    
    l_dense, t_dense, _ = benchmark_cell(mode="dense")
    print(f"{'Dense (Baseline)':<15} | {l_dense:<15.4f} | {t_dense:<15.2f} | {'0.00':<15}")
    
    l_stc, t_stc, s_stc = benchmark_cell(mode="stc_sparse")
    print(f"{'STC Sparse':<15} | {l_stc:<15.4f} | {t_stc:<15.2f} | {s_stc:<15.4f}")
