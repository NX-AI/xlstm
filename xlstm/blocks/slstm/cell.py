# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

import torch

from torch.autograd.function import once_differentiable
from .src.cuda_init import load
from .src.vanilla import (
    slstm_forward,
    slstm_forward_step,
    slstm_pointwise_function_registry,
)
from ...components.util import conditional_decorator, round_to_multiple, ParameterProxy
from ...components.init import bias_linspace_init_

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union, Sequence
import logging
import torch
import torch.nn as nn
from math import sqrt

LOGGER = logging.getLogger(__name__)

DTYPE_DICT = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
DTYPES = Literal["bfloat16", "float16", "float32"]

# this is needed to locate the cuda source file path
curdir = Path(os.path.split(os.path.os.path.abspath(__file__))[0])

# maps the rnn function to the following values
#   states = 2 or 4

rnn_function_registry = {
    # standard variants, all connect
    "lstm": {
        "states": 2,
    },
    "slstm": {
        "states": 4,
    },
}

_python_dtype_to_cuda_dtype = {
    "float32": "float",
    "float": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
}


@dataclass
class sLSTMCellConfig:
    hidden_size: int = -1
    num_heads: int = (
        4  # this must divide the hidden size, is not yet supported by all versions in this directory
    )
    num_states: int = 4  # this is for the sLSTM, a standard LSTM  has 2
    backend: Literal["vanilla", "cuda"] = "cuda"
    # the type of function a cell computes
    function: str = "slstm"
    bias_init: Literal["powerlaw_blockdependent", "small_init", "standard"] = (
        "powerlaw_blockdependent"
    )
    recurrent_weight_init: Literal["zeros", "standard"] = "zeros"

    _block_idx: int = (
        0  # index in the block sequence for a residual stacked model, needed for forget gate init
    )
    _num_blocks: int = (
        1  # how many blocks there are in the residually stacked model, needed for forget gate init
    )

    num_gates: int = 4
    # this option cuts of the gradient for recurrent connection, i.e. no exploding gradient if False
    gradient_recurrent_cut: bool = False
    # this option clips the gradient values for recurrent connections at dy
    gradient_recurrent_clipval: Optional[float] = None
    # this option clips the y value
    forward_clipval: Optional[float] = None

    # this can be ignored internally, but may be used to optimize kernels
    batch_size: int = 8
    # input shape from external, definitions see output_shape
    input_shape: Literal["BSGNH", "SBGNH"] = "BSGNH"
    # internal input shape, may be redefined by backend
    internal_input_shape: Literal["SBNGH", "SBGNH", "SBNHG"] = "SBNGH"
    # B = batch, S sequence dim, N num heads,
    # H head dim or hidden dim (= fused dim [num_heads, head_dim]) without num_heads
    output_shape: Literal[
        "BNSH",
        "SBH",
        "BSH",
        "SBNH",
    ] = "BNSH"

    # additional compiler constants
    constants: dict = field(default_factory=dict)
    # dtypes
    dtype: DTYPES = "bfloat16"
    dtype_b: Optional[DTYPES] = "float32"  # biases
    dtype_r: Optional[DTYPES] = None  # recurrent matrix
    dtype_w: Optional[DTYPES] = None  # inputs / w matrix
    dtype_g: Optional[DTYPES] = None  # gates
    dtype_s: Optional[DTYPES] = None  # states
    dtype_a: Optional[DTYPES] = None  # internal accumulation

    # mixed precision
    enable_automatic_mixed_precision: bool = True
    initial_val: Union[float, Sequence[float]] = 0.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def input_dim(self):
        return 4 * self.hidden_size

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype]

    @property
    def torch_dtype_b(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_b]

    @property
    def torch_dtype_r(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_r]

    @property
    def torch_dtype_w(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_w]

    @property
    def torch_dtype_s(self) -> torch.dtype:
        return DTYPE_DICT[self.dtype_s]

    def __post_init__(self):
        if self.num_heads <= 0:
            self.num_heads = 1
        if self.dtype_b is None:
            self.dtype_b = self.dtype
        if self.dtype_a is None:
            self.dtype_a = self.dtype_b
        if self.dtype_r is None:
            self.dtype_r = self.dtype
        if self.dtype_w is None:
            self.dtype_w = self.dtype
        if self.dtype_s is None:
            self.dtype_s = self.dtype_w
        if self.dtype_g is None:
            self.dtype_g = self.dtype_r

        assert (
            self.function in rnn_function_registry
        ), f"RNN function {self.function} not in registry"
        self.num_states = rnn_function_registry[self.function]["states"]
        if "initial_val" in rnn_function_registry[self.function]:
            self.initial_val = rnn_function_registry[self.function]["initial_val"]

    @property
    def defines(self):
        return (
            [
                f"-DSLSTM_HIDDEN_SIZE={self.hidden_size}",
                f"-DSLSTM_BATCH_SIZE={self.batch_size}",
                f"-DSLSTM_NUM_HEADS={self.num_heads}",
                f"-DSLSTM_NUM_STATES={self.num_states}",
                f"-DSLSTM_DTYPE_B={_python_dtype_to_cuda_dtype[self.dtype_b]}",
                f"-DSLSTM_DTYPE_R={_python_dtype_to_cuda_dtype[self.dtype_r]}",
                f"-DSLSTM_DTYPE_W={_python_dtype_to_cuda_dtype[self.dtype_w]}",
                f"-DSLSTM_DTYPE_G={_python_dtype_to_cuda_dtype[self.dtype_g]}",
                f"-DSLSTM_DTYPE_S={_python_dtype_to_cuda_dtype[self.dtype_s]}",
                f"-DSLSTM_DTYPE_A={_python_dtype_to_cuda_dtype[self.dtype_a]}",
                f"-DSLSTM_NUM_GATES={4}",
                f"-DSLSTM_SIMPLE_AGG={'true'}",
            ]
            + (
                [
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID=true",
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL_VALID=false",
                    f"-DSLSTM_GRADIENT_RECURRENT_CLIPVAL=0.0",
                ]
            )
            + (
                [
                    f"-DSLSTM_FORWARD_CLIPVAL_VALID=true",
                    f"-DSLSTM_FORWARD_CLIPVAL={self.gradient_recurrent_clipval}",
                ]
                if self.gradient_recurrent_clipval is not None
                else [
                    f"-DSLSTM_FORWARD_CLIPVAL_VALID=false",
                    f"-DSLSTM_FORWARD_CLIPVAL=0.0",
                ]
            )
        )


class sLSTMCellBase(nn.Module):
    config_class = sLSTMCellConfig

    def __init__(self, config: sLSTMCellConfig):
        super().__init__()
        self.config = config
        LOGGER.debug("Init module")

        head_dim = self.config.hidden_size // self.config.num_heads
        dtype_r = (
            self.config.torch_dtype_r
            if not self.config.enable_automatic_mixed_precision
            else None
        )
        dtype_b = (
            self.config.torch_dtype_b
            if not self.config.enable_automatic_mixed_precision
            else None
        )

        self._recurrent_kernel_ = nn.Parameter(
            torch.empty(
                self.config.num_heads,
                head_dim,
                self.config.num_gates,
                head_dim,
                dtype=dtype_r,
            )
        )
        self.recurrent_kernel = ParameterProxy(
            self,
            "_recurrent_kernel",
            self._recurrent_kernel_int2ext,
            self._recurrent_kernel_ext2int,
        )
        self._recurrent_kernel_ = nn.Parameter(
            self._recurrent_kernel_ext2int(self._recurrent_kernel_.data)
        )

        self._bias_ = nn.Parameter(
            torch.empty(
                self.config.num_heads, self.config.num_gates, head_dim, dtype=dtype_b
            )
        )
        self.bias = ParameterProxy(
            self, "_bias", self._bias_int2ext, self._bias_ext2int
        )
        self._bias_ = nn.Parameter(self._bias_ext2int(self._bias_.data))

        self.reset_parameters()

        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"Hidden Size {self.config.hidden_size} must be divisible by head num {self.config.num_heads}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(function={self.config.function}, "
            f"hidden_size={self.config.hidden_size}, num_heads={self.config.num_heads})"
        )

    @property
    def _recurrent_kernel(self):
        return self._recurrent_kernel_

    @property
    def _bias(self):
        return self._bias_

    def _recurrent_kernel_ext2int(
        self, recurrent_kernel_ext: torch.Tensor
    ) -> torch.Tensor:
        return recurrent_kernel_ext

    def _bias_ext2int(self, bias_ext: torch.Tensor) -> torch.Tensor:
        return bias_ext

    def _recurrent_kernel_int2ext(
        self, recurrent_kernel_int: torch.Tensor
    ) -> torch.Tensor:
        return recurrent_kernel_int

    def _bias_int2ext(self, bias_int: torch.Tensor) -> torch.Tensor:
        return bias_int

    def parameters_to_dtype(self):
        pars = [name for name, _ in self.named_parameters()]
        for name in pars:
            par = getattr(self, name)
            if "recurrent" in name:
                setattr(
                    self,
                    name,
                    torch.nn.Parameter(
                        par.to(dtype=self.config.dtype_r),
                        requires_grad=par.requires_grad,
                    ),
                )
            if "bias" in name:
                setattr(
                    self,
                    name,
                    torch.nn.Parameter(
                        par.to(dtype=self.config.dtype_b),
                        requires_grad=par.requires_grad,
                    ),
                )

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_heads

    def _permute_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        # TODO Adapt this
        # >>> BaseRNN(BaseRNNConfig(10, 10, num_heads=2, input_shape='SBG'))._permute_input(torch.zeros((5, 2, 10))).shape
        # torch.Size([5, 2, 10])
        # >>> BaseRNN(BaseRNNConfig(10, 10, num_heads=2, input_shape='BSG'))._permute_input(torch.zeros((5, 2, 10))).shape
        # torch.Size([2, 5, 10])
        """
        if self.config.input_shape == "SBGNH":
            y = x.view(
                x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1
            )
        elif self.config.input_shape == "BSGNH":
            y = x.view(
                x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1
            ).permute(1, 0, 2, 3, 4)
        else:
            raise ValueError("Bad input_shape value")
        if self.config.internal_input_shape == "SBGNH":
            return y.view(y.shape[0], y.shape[1], -1)
        elif self.config.internal_input_shape == "SBNGH":
            return y.permute(0, 1, 3, 2, 4).reshape(y.shape[0], y.shape[1], -1)
        elif self.config.internal_input_shape == "SBNHG":
            return y.permute(0, 1, 3, 4, 2).reshape(y.shape[0], y.shape[1], -1)
        else:
            raise ValueError("Bad internal_input_shape value")

    def _permute_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        >>> BaseRNN(BaseRNNConfig(10, 16, num_heads=2, output_shape='SBH'))._permute_output(torch.zeros((5, 2, 16))).shape
        torch.Size([5, 2, 16])
        >>> BaseRNN(BaseRNNConfig(10, 16, num_heads=2, output_shape='BNSH'))._permute_output(torch.zeros((5, 3, 16))).shape
        torch.Size([3, 2, 5, 8])
        >>> BaseRNN(BaseRNNConfig(10, 16, num_heads=2, output_shape='SBNH'))._permute_output(torch.zeros((5, 3, 16))).shape
        torch.Size([5, 3, 2, 8])
        """
        if self.config.output_shape == "SBH":
            return x
        elif self.config.output_shape == "BSH":
            return x.permute(1, 0, 2)
        elif self.config.output_shape == "BNSH":
            return x.view(
                (x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim)
            ).permute(1, 2, 0, 3)
        elif self.config.output_shape == "SBNH":
            return x.view(
                (x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim)
            )

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        for h in range(self.config.num_heads):
            for i, gate in enumerate(["i", "f", "z", "o"]):
                if self.config.recurrent_weight_init == "zeros":
                    self.recurrent_kernel[h, :, i, :] = nn.init.zeros_(
                        self.recurrent_kernel[h, :, i, :]
                    )
                elif self.config.recurrent_weight_init == "standard":
                    self.recurrent_kernel[h, :, i, :] = nn.init.uniform_(
                        self.recurrent_kernel[h, :, i, :],
                        -1.0 / sqrt(self.config.hidden_size),
                        1.0 / sqrt(self.config.hidden_size),
                    )
        for h in range(self.config.num_heads):
            for i, gate in enumerate(["i", "f", "z", "o"]):
                if self.config.bias_init == "powerlaw_blockdependent":
                    if gate == "f":
                        kwargs = (
                            dict(
                                right_x=5.0,
                                range_x_neg_dir=12.0,
                                spread_lower=0.3,
                                spread_upper=3.0,
                            ),
                        )
                        ratio_0_to_1 = (
                            self.config._block_idx / (self.config._num_blocks - 1)
                            if self.config._num_blocks > 1
                            else 0.0
                        )
                        init_values = -(
                            -5.0
                            + 12.0
                            * (
                                torch.arange(self.config.head_dim)
                                / (self.config.head_dim - 1)
                            )
                            ** (0.3 + 1.3 * ratio_0_to_1)
                        )
                        with torch.no_grad():
                            self.bias[h, i, :] = init_values
                    else:
                        self.bias[h, i] = nn.init.zeros_(self.bias[h, i])
                elif self.config.bias_init == "small_init":
                    if gate == "f":
                        self.bias[h, i] = bias_linspace_init_(
                            self.bias[h, i], start=3.0, end=6.0
                        )
                    else:
                        self.bias[h, i] = nn.init.zeros_(self.bias[h, i])
                elif self.config.bias_init == "zeros":
                    self.bias[h, i] = nn.init.zeros_(self.bias[h, i])
                elif self.config.bias_init == "standard":
                    self.bias[h, i] = nn.init.uniform_(
                        self.bias[h, i],
                        -1 / sqrt(self.config.hidden_size),
                        1 / sqrt(self.config.hidden_size),
                    )

    def _check_input(self, input: torch.Tensor) -> None:
        assert self.config.hidden_size * self.config.num_gates == input.size(
            -1
        ), f"Input size mismatch: Expected input size {self.config.hidden_size * self.config.num_gates}, but got {input.size(-1)}."

    def _zero_state(self, input: torch.Tensor) -> torch.Tensor:
        """Returns a zeros state matching dtype and batch size of `input`.

        Arguments:
          input: Tensor, to specify the device and dtype of the returned tensors.

        Returns:
          zero_state: a nested structure of zero Tensors.
        """
        batch_dim = input.shape[1]
        state = torch.zeros(
            (self.config.num_states, batch_dim, self.config.hidden_size),
            dtype=input.dtype,
            device=input.device,
        )
        return state

    def _get_state(
        self, input: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if state is None:
            state = self._zero_state(input)
        else:
            assert state.shape == (
                self.config.num_states,
                input.shape[1],
                self.config.hidden_size,
            )
        return state

    def _get_final_state(self, all_states: torch.Tensor) -> torch.Tensor:
        """
        All states has the structure
        [STATES, SEQUENCE, BATCH, HIDDEN]
        """
        return all_states[:, -1]

    def _is_cuda(self) -> bool:
        is_cuda = [tensor.is_cuda for tensor in list(self.parameters())]
        if any(is_cuda) and not all(is_cuda):
            raise ValueError(
                "RNN tensors should all be CUDA tensors or none should be CUDA tensors"
            )
        return any(is_cuda)

    def step(
        self, input: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_input(input)
        input = self._permute_input(input)
        states = self._get_state(input, state)
        all_states = self._impl_step(self.training, input, states)
        output = self._permute_output(all_states[0])
        return output, state

    def forward(self, input, state=None, lengths=None):
        self._check_input(input)
        input = self._permute_input(input)
        states = self._get_state(input, state)
        all_states = self._impl(self.training, input, states)
        state = self._get_final_state(all_states)
        output = self._permute_output(all_states[0][1:])
        if torch.is_autocast_enabled():
            return output, state
        else:
            return output.to(input.dtype), state.to(input.dtype)


class sLSTMCellCUDA(object):
    mod = {}

    @classmethod
    def instance(cls, config: sLSTMCellConfig):
        if repr(config) not in cls.mod:
            cls.mod[repr(config)] = load(
                name=config.function,
                sources=[
                    str(curdir / "src" / "cuda" / "slstm.cc"),
                    str(curdir / "src" / "cuda" / "slstm_forward.cu"),
                    str(curdir / "src" / "cuda" / "slstm_backward.cu"),
                    str(curdir / "src" / "cuda" / "slstm_backward_cut.cu"),
                    str(curdir / "src" / "cuda" / f"{config.function}_pointwise.cu"),
                    str(curdir / "src" / "util" / "blas.cu"),
                    str(curdir / "src" / "util" / "cuda_error.cu"),
                ],
                extra_cflags=[
                    f"-D{const}={constval}"
                    for const, constval in config.constants.items()
                ]
                + config.defines,
            )
        return cls.mod[repr(config)]


def sLSTMCellFuncGenerator(training, config: sLSTMCellConfig):
    slstm_cuda = sLSTMCellCUDA.instance(config=config)
    slstm_mod = slstm_cuda.sLSTMFunc(
        training, config.batch_size, config.hidden_size, config.num_heads
    )

    class sLSTMCellFunction(torch.autograd.Function):
        @staticmethod
        @conditional_decorator(
            config.enable_automatic_mixed_precision, torch.cuda.amp.custom_fwd
        )
        def forward(ctx, training, *inputs):
            dtypes = (
                inputs[0].dtype,
                inputs[1].dtype,
                inputs[2].dtype,
                inputs[3].dtype,
            )
            if config.enable_automatic_mixed_precision:
                inputs = (
                    inputs[0].to(dtype=config.torch_dtype_w),
                    inputs[1].to(dtype=config.torch_dtype_s),
                    inputs[2].to(dtype=config.torch_dtype_r),
                    inputs[3].to(dtype=config.torch_dtype_b),
                )
            states, cache_g_r, cache_g_i = slstm_mod.forward(training, *inputs)

            ctx.save_for_backward(*inputs[2:], states, cache_g_r, cache_g_i)
            ctx.training = training
            return states

        @staticmethod
        @once_differentiable
        @conditional_decorator(
            config.enable_automatic_mixed_precision, torch.cuda.amp.custom_bwd
        )
        def backward(ctx, grad_s):
            if not ctx.training:
                raise RuntimeError(
                    "sLSTMCell backward can only be called in training mode"
                )
            saved = [*ctx.saved_tensors]
            saved[0] = saved[0].permute(0, 2, 1).contiguous()  # transpose R
            if config.gradient_recurrent_cut:
                grads = slstm_mod.backward_cut(*saved, grad_s.contiguous())
            else:
                grads = slstm_mod.backward(*saved, grad_s.contiguous())
            with torch.no_grad():
                S, B, H = grads[0].shape
            return (None, *grads)

    return sLSTMCellFunction


class sLSTMCell_vanilla(sLSTMCellBase):
    config_class = sLSTMCellConfig

    def __init__(self, config: sLSTMCellConfig, skip_backend_init=False):
        super().__init__(config)
        # load pointwise function
        self.pointwise = slstm_pointwise_function_registry[self.config.function]

        self.config.internal_input_shape = "SBGNH"

    def _recurrent_kernel_ext2int(
        self, recurrent_kernel_ext: torch.Tensor
    ) -> torch.Tensor:
        return (
            recurrent_kernel_ext.reshape(
                self.config.num_heads,
                self.config.head_dim,
                self.config.num_gates,
                self.config.head_dim,
            )
            .permute(0, 2, 3, 1)
            .reshape(
                self.config.num_heads,
                self.config.num_gates * self.config.head_dim,
                self.config.head_dim,
            )
        )

    def _recurrent_kernel_int2ext(
        self, recurrent_kernel_int: torch.Tensor
    ) -> torch.Tensor:
        """
        >>> (); mod = sLSTMCell_vanilla(sLSTMCellConfig(hidden_size=64, num_heads=2), skip_backend_init=True); () # doctest:+ELLIPSIS
        (...)
        >>> torch.allclose(mod._recurrent_kernel_ext2int(mod._recurrent_kernel_int2ext(mod._recurrent_kernel)), mod._recurrent_kernel)
        True
        """
        return recurrent_kernel_int.reshape(
            self.config.num_heads,
            self.config.num_gates,
            self.config.head_dim,
            self.config.head_dim,
        ).permute(0, 3, 1, 2)

    def _bias_ext2int(self, bias_ext: torch.Tensor) -> torch.Tensor:
        return (
            bias_ext.reshape(
                self.config.num_heads, self.config.num_gates, self.config.head_dim
            )
            .permute(1, 0, 2)
            .reshape(-1)
        )

    def _bias_int2ext(self, bias_int: torch.Tensor) -> torch.Tensor:
        """
        >>> (); mod = sLSTMCell_vanilla(sLSTMCellConfig(hidden_size=64, num_heads=2), skip_backend_init=True); () # doctest:+ELLIPSIS
        (...)
        >>> torch.allclose(mod._bias_ext2int(mod._bias_int2ext(mod._bias)), mod._bias)
        True
        """

        return bias_int.reshape(
            self.config.num_gates, self.config.num_heads, self.config.head_dim
        ).permute(1, 0, 2)

    def _impl(
        self, training: bool, input: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return slstm_forward(
            input,
            state,
            self._recurrent_kernel,
            self._bias,
            self.pointwise,
            constants=self.config.constants,
        )[0]

    def _impl_step(
        self, training: bool, input: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return slstm_forward_step(
            input,
            state,
            self._recurrent_kernel,
            self._bias,
            self.pointwise,
            constants=self.config.constants,
        )[0]


class sLSTMCell_cuda(sLSTMCellBase):
    config_class = sLSTMCellConfig

    def __init__(self, config: sLSTMCellConfig, skip_backend_init: bool = False):
        """
        skip device init is intended for converting models between hardware configurations / backends
        i.e. to store a model first and later convert it to a different backend form
        """
        super().__init__(config)
        self.internal_input_shape = "SBNGH"
        if not skip_backend_init:
            self.func = sLSTMCellFuncGenerator(self.training, config)

    def _recurrent_kernel_ext2int(
        self, recurrent_kernel_ext: torch.Tensor
    ) -> torch.Tensor:
        return recurrent_kernel_ext.reshape(
            self.config.num_heads,
            self.config.head_dim,
            self.config.num_gates,
            self.config.head_dim,
        ).reshape(
            self.config.num_heads,
            self.config.head_dim,
            self.config.num_gates * self.config.head_dim,
        )

    def _recurrent_kernel_int2ext(
        self, recurrent_kernel_int: torch.tensor
    ) -> torch.Tensor:
        """
        >>> (); mod = sLSTMCell_cuda(
        ...     sLSTMCellConfig(hidden_size=64, num_heads=2), skip_backend_init=True); () # doctest:+ELLIPSIS
        (...)
        >>> torch.allclose(mod._recurrent_kernel_ext2int(mod._recurrent_kernel_int2ext(mod._recurrent_kernel)), mod._recurrent_kernel)
        True
        """

        return recurrent_kernel_int.reshape(
            self.config.num_heads,
            self.config.head_dim,
            self.config.num_gates,
            self.config.head_dim,
        )

    def _bias_ext2int(self, bias_ext: torch.Tensor) -> torch.Tensor:
        return (
            bias_ext.reshape(
                self.config.num_heads, self.config.num_gates, self.config.head_dim
            )
            .permute(0, 1, 2)
            .reshape(-1)
        )

    def _bias_int2ext(self, bias_int: torch.Tensor) -> torch.Tensor:
        """
        >>> (); mod = sLSTMCell_cuda(
        ...     sLSTMCellConfig(hidden_size=64, num_heads=2), skip_backend_init=True); () # doctest:+ELLIPSIS
        (...)
        >>> torch.allclose(mod._bias_ext2int(mod._bias_int2ext(mod._bias)), mod._bias)
        True
        """

        return bias_int.reshape(
            self.config.num_heads, self.config.num_gates, self.config.head_dim
        )

    def _impl_step(
        self,
        training: bool,
        input: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.func.apply(
            training,
            input.contiguous(),
            state.contiguous(),
            self._recurrent_kernel.contiguous(),
            self._bias.contiguous(),
        )

    def _impl(
        self,
        training: bool,
        input: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.func.apply(
            training,
            input.contiguous(),
            state.contiguous(),
            self._recurrent_kernel.contiguous(),
            self._bias.contiguous(),
        )


class sLSTMCell(sLSTMCellBase):
    config_class = sLSTMCellConfig

    def __new__(cls, config: sLSTMCellConfig, skip_backend_init: bool = False):
        if config.backend == "cuda":
            return sLSTMCell_cuda(config, skip_backend_init=skip_backend_init)
        elif config.backend == "vanilla":
            return sLSTMCell_vanilla(config)
        else:
            raise RuntimeError(
                f'sLSTMCell unknown backend {config.backend}, choose from ["cuda", "vanilla"]'
            )
