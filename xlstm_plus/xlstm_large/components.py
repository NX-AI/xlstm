import torch
from torch import nn
from abc import abstractmethod, ABC

def soft_cap(values: torch.Tensor, cap_value: float | torch.Tensor | None) -> torch.Tensor:
    """
    Soft caps a tensor to a value.

    Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
    and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
    https://arxiv.org/abs/2408.00118

    Args:
        values: The tensor to cap.
        cap_value: The value to cap the values to. If None, no cap is applied.

    Returns:
        The capped values.
    """
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)

class NormLayer(nn.Module, ABC):
    """Base class for normalization layers.
    This class contains optional learnable weight and bias parameters.
    
    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.   
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions

        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.weight = None

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.bias = None

    def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RMSNorm(NormLayer):
    """Root mean square normalization layer implementation similar
    to https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html. 

    It normalizes the input tensor by the root mean square of the last dimension.

    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.
    """

    def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        # apply rms norm over the last dimension, i.e. D dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        x = self._rms_normalize(x)
        x = self._apply_weight_bias(x)
        return x

class MultiHeadRMSNorm(RMSNorm):
    """Multi-head version of the RMSNorm layer.
    It normalizes the last dimension of the input tensor by the root mean square.

    The input is assumed to have the shape (B, S, NH, DH), where:
    B: batch size
    S: sequence length
    NH: number of heads
    DH: head dimension
    
    The normalization is applied over the last dimension (DH) of the input tensor.
    Weights and biases are applied after the normalization.

    Args:
        num_heads: The number of heads.
        head_dim: The head dimension.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions
    
    Returns:
        The normalized tensor with the shape (B, S, NH * DH).    
    """
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__(
            num_features=num_heads * head_dim,
            eps=eps,
            use_weight=use_weight,
            use_bias=use_bias,
            force_float32_reductions=force_float32_reductions,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        x: torch.Tensor, # (B, S, NH, DH)
    ) -> torch.Tensor: # (B, S, NH * DH)
        B, S, NH, DH = x.shape
        assert NH == self.num_heads, f"Expected {self.num_heads} heads, got {NH}, input shape: {x.shape}"
        assert DH == self.head_dim, f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}"

        x = self._rms_normalize(x)
        x = x.reshape(B, S, -1)
        x = self._apply_weight_bias(x)
        return x

class LayerNorm(NormLayer):
    """Layer normalization layer implementation similar to 
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html.
    
    The layer normalization is applied over the last dimension of the input tensor.

    Args:
        num_features: The number of features in the input tensor.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions.

    Returns:
        The normalized tensor.    
    """

    def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        # apply layer norm over the last dimension, i.e. D dimension
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return y.to(in_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ..., S,..., D)
        x = self._layer_normalize(x)
        x = self._apply_weight_bias(x)
        return x

class MultiHeadLayerNorm(LayerNorm):
    """Multi-head version of the LayerNorm layer.
    
    It normalizes the last dimension of the input tensor.

    The input is assumed to have the shape (B, S, NH, DH), where:
    B: batch size
    S: sequence length
    NH: number of heads
    DH: head dimension

    The normalization is applied over the last dimension (DH) of the input tensor.

    Args:
        num_heads: The number of heads.
        head_dim: The head dimension.
        eps: A small value to avoid division by zero.
        use_weight: Whether to use a learnable weight.
        use_bias: Whether to use a learnable bias.
        force_float32_reductions: Whether to force float32 reductions

    Returns:
        The normalized tensor with the shape (B, S, NH * DH).
    """
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__(
            num_features=num_heads * head_dim,
            eps=eps,
            use_weight=use_weight,
            use_bias=use_bias,
            force_float32_reductions=force_float32_reductions,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        x: torch.Tensor, # (B, S, NH, DH)
    ) -> torch.Tensor: # (B, S, NH * DH)
        B, S, NH, DH = x.shape
        assert NH == self.num_heads, f"Expected {self.num_heads} heads, got {NH}, input shape: {x.shape}"
        assert DH == self.head_dim, f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}"

        x = self._layer_normalize(x)
        x = x.reshape(B, S, -1)
        x = self._apply_weight_bias(x)
        return x
