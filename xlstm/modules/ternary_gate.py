# Copyright (c) NXAI GmbH and its affiliates 2024
import torch
from torch import nn
from .ternary_quantizer import TernaryQuantizer

class TernaryGate(nn.Module):
    def __init__(self, ema_alpha: float = 0.9, threshold_factor: float = 0.1):
        super().__init__()
        self.quantizer = TernaryQuantizer(ema_alpha=ema_alpha, threshold_factor=threshold_factor)

    def forward(self, memory: torch.Tensor, update: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """
        Ternary Gate Logic:
        - 0  -> exact preserve / hard lock (return memory)
        - -1 -> active inhibition / inversion (return -memory)
        - +1 -> additive update (return memory + update)
        
        Args:
            memory: Current memory state (C_t-1)
            update: Memory update (v_t k_t^T)
            gate_input: Pre-activation gate input
        """
        gate = self.quantizer(gate_input) # (B, NH, 1, 1) or similar
        
        # gate is ternary: {-1, 0, 1}
        # We can express the logic as:
        # out = (gate == 0) * memory + (gate == -1) * (-memory) + (gate == 1) * (memory + update)
        
        # Alternative implementation using masks for efficiency:
        preserve_mask = (gate == 0).to(memory.dtype)
        inhibit_mask = (gate == -1).to(memory.dtype)
        update_mask = (gate == 1).to(memory.dtype)
        
        return preserve_mask * memory + inhibit_mask * (-memory) + update_mask * (memory + update)
