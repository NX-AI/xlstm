# Copyright (c) NXAI GmbH and its affiliates 2024
import torch
from torch import nn

class TernaryQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau):
        # Q(x) = 1 if x > tau, 0 if |x| <= tau, -1 if x < -tau
        return torch.where(x > tau, 1.0, torch.where(x < -tau, -1.0, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output.clone(), None

class TernaryQuantizer(nn.Module):
    def __init__(self, ema_alpha: float = 0.9, threshold_factor: float = 0.1):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.threshold_factor = threshold_factor
        self.register_buffer("ema_magnitude", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = x.abs().mean()
        if self.training:
            self.ema_magnitude.copy_(
                self.ema_alpha * self.ema_magnitude + (1 - self.ema_alpha) * magnitude
            )
        
        tau = self.threshold_factor * self.ema_magnitude
        return TernaryQuantSTE.apply(x, tau)
