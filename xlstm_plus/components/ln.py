# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(
        self,
        ndim: int = -1,
        weight: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, normalized_shape=(self.ndim,), weight=self.weight_proxy, bias=self.bias, eps=self.eps
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = input.shape

        gn_in_1 = input.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out
