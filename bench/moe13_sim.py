import torch
import time

def simulate_moe13_stc(num_experts=13, update_sparsity=0.9, gate_sparsity=0.9, B=4, NH=8, DH=64):
    # In MoE-13, we have 13 experts. 
    # Usually only a few are active per token, but the memory updates for ALL 
    # could be optimized if we use sparse updates.
    
    # Dense update cost for all experts
    dense_flops_per_token = num_experts * B * NH * 4 * DH * DH
    
    # Sparse STC cost
    # Prob(gate != 0) = 1 - gate_sparsity
    sparse_flops_per_token = num_experts * B * NH * (
        (1 - gate_sparsity) * (DH * DH + 3 * (1 - update_sparsity)**2 * DH * DH) + 3 * DH
    )
    
    speedup = dense_flops_per_token / sparse_flops_per_token
    
    return dense_flops_per_token, sparse_flops_per_token, speedup

if __name__ == "__main__":
    d, s, speedup = simulate_moe13_stc()
    print(f"MoE-13 Simulation Results:")
    print(f"Experts: 13")
    print(f"Update Sparsity: 90%")
    print(f"Gate Sparsity: 90%")
    print(f"Dense GFLOPs (approx): {d / 1e9:.6f}")
    print(f"Sparse GFLOPs (approx): {s / 1e9:.6f}")
    print(f"Theoretical Speedup: {speedup:.2f}x")
