import torch
import numpy as np

def estimate_flops(B=1, NH=4, DH=32, update_sparsity=0.9, gate_sparsity=0.5):
    # Dense update: C_new = fg * C_prev + ig * (k @ v^T)
    # Total: 4 * DH * DH per batch/head
    dense_flops = B * NH * 4 * DH * DH
    
    # STC Sparse update + Ternary Gate:
    # If gate_sparsity is s_g, then with probability s_g (gate == 0), we do 0 ops.
    # Otherwise, we do:
    # 1. Scale C_prev: DH * DH
    # 2. Update (outer product + scale + add): 3 * (1 - update_sparsity)^2 * DH * DH
    
    # Prob(gate != 0) = (1 - s_g)
    sparse_ops = B * NH * (1 - gate_sparsity) * (DH * DH + 3 * (1 - update_sparsity)**2 * DH * DH)
    
    # Minimum ops for quantization (always performed)
    quant_ops = B * NH * 3 * DH # 3 because of k, v, and gate
    sparse_ops += quant_ops
    
    savings = (dense_flops - sparse_ops) / dense_flops * 100
    speedup_potential = dense_flops / sparse_ops
    
    return dense_flops, sparse_ops, savings, speedup_potential

if __name__ == "__main__":
    print(f"{'Update Spar':<12} | {'Gate Spar':<10} | {'Dense FLOPs':<12} | {'Sparse FLOPs':<12} | {'Savings %':<10} | {'Speedup':<10}")
    print("-" * 80)
    for s_u in [0.9, 0.95]:
        for s_g in [0.5, 0.7, 0.9, 0.95, 0.99]:
            d, sp, sav, speedup = estimate_flops(update_sparsity=s_u, gate_sparsity=s_g)
            print(f"{s_u:<12.2f} | {s_g:<10.2f} | {d:<12d} | {int(sp):<12d} | {sav:<10.2f} | {speedup:<10.2f}x")
