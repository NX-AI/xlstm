from dataclasses import dataclass, field

try:
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
        ChunkwiseKernelType,
        SequenceKernelType,
        StepKernelType,
        DtypeType,
        BackendModeType,
    )
except ImportError:
    raise ImportError("Please install mlstm_kernels package to use mLSTM block.")


import torch
from torch import nn
from typing import Literal
from .components import MultiHeadLayerNorm, RMSNorm, soft_cap
from .utils import round_up_to_next_multiple_of
from .generate import generate_tokens, get_sampling_fn

mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]

WeightModeType = Literal["single", "fused"]


@dataclass
class xLSTMLargeConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    num_blocks: int
    """Number of blocks."""
    vocab_size: int
    """Vocabulary size."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""
    add_out_norm: bool = True
    """Whether to add a normalization layer after the block stack."""

    # mlstm layer
    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""

    # mlstm backend
    chunkwise_kernel: ChunkwiseKernelType = "chunkwise--triton_limit_chunk"
    """Kernel to use for chunkwise parallel processing of the sequence.
    Also supports fully parallel (i.e. quadratic) backends for comparison.
    E.g. 'parallel--native_autograd'.
    """
    sequence_kernel: SequenceKernelType = "native_sequence__triton"
    """The sequence kernel to use for processing sequneces step-by-step.
    Used only for parts of the prefill sequence in inference mode.
    """
    step_kernel: StepKernelType = "triton"
    """The step kernel to use for processing a single step.
    Used for generation in inference mode.
    """
    mode: BackendModeType = "train"
    """The mode of operation for the backend. Determines how the `forward` method behaves.
    Available modes are 'train', 'train_with_padding', 'inference'.
    'inference' works with arbitrary sequence lengths, and does not support training. 
    It calls a sequence of different kernels to process the sequence.
    'train_with_padding' pads the input to multiples of `chunk_size`.
    """
    chunk_size: int = 64
    """The chunk size of the chunkwise kernel.
    If `mode` is 'train_with_padding', the inputs are padded to multiples of this size.
    """
    return_last_states: bool = False
    """Whether to return the last states of the sequence in training mode.
    Inference mode always returns the last states.
    """
    autocast_kernel_dtype: DtypeType = "bfloat16"
    """The dtype to use for autocast behavior in the kernel.
    If autocast is enabled all inputs are cast to this dtype before the kernel is called.
    """
    eps: float = 1e-6
    """Epsilon value for numerical stability in the kernel."""
    inference_state_dtype: DtypeType = "float32"
    """The dtype to use for the state tensors in inference mode."""
    # feedforward
    ffn_proj_factor: float = 2.6667
    """The factor to determine the dimension of the intermediate projection in the feedforward layer."""
    ffn_round_up_to_multiple_of: int = 64
    """Round the intermediate projection dimension to the next multiple of this value."""
    
    # capping
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""
    output_logit_soft_cap: float = 30.0
    """Soft cap value for the output logits."""

    weight_mode: WeightModeType = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and gates.
    'fused' is benefitial in inference settings.
    """


class xLSTMLarge(nn.Module):
    config_class = xLSTMLargeConfig

    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.backbone = xLSTMLargeBlockStack(config)

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim, out_features=config.vocab_size, bias=False
        )

    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, mLSTMStateType]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, S].
            state: State dictionary of the model. 
                   If None, the state is initialized, the model starts from an empty initial state.
        
        Returns:
            logits: Logits tensor of shape [B, S, V].
            Tuple of logits and state: State dictionary of the model, if `return_last_states` is True.
        """

        assert x.ndim == 2, f"Input must have shape [B, S], got {x.shape}"
        B, S = x.shape

        x = self.embedding(x)

        x, state = self.backbone(x, state)

        logits = self.lm_head(x)
        logits_capped = soft_cap(logits, self.config.output_logit_soft_cap)
        if self.config.return_last_states:
            return logits_capped, state
        else:
            return logits_capped

    def generate(
        self,
        prefill_tokens: torch.Tensor,
        max_length: int,
        sampling_type: str = "greedy",
        state: mLSTMStateType | None = None,
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        """Generate tokens from the model.

        Args:
            prefill_tokens: Tensor of shape [B, S] with the prefill tokens.
            max_length: Maximum length of the generated sequence.
            sampling_type: Sampling type to use, e.g. 'greedy'.
            state: State dictionary of the model. 
                   If None, the state is initialized, the model starts from an empty initial state.
        
        Returns:
            tokens: Generated tokens tensor of shape [B, S].
            state: State dictionary of the model after the last generation step.
        """
        sampling_fn = get_sampling_fn(sampling_type)
        tokens, state = generate_tokens(
            llm_forward=self.forward,
            prefill_tokens=prefill_tokens,
            max_length=max_length,
            token_sample_fn=sampling_fn,
            state=state,
            device=str(self.embedding.weight.device),
        )
        return tokens, state


class xLSTMLargeBlockStack(nn.Module):
    config_class = xLSTMLargeConfig

    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList(
            [mLSTMBlock(config) for _ in range(config.num_blocks)]
        )

        if self.config.add_out_norm:
            self.out_norm = RMSNorm(
                num_features=config.embedding_dim,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
        else:
            self.out_norm = nn.Identity()

    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        if state is None:
            state = {i: None for i in range(len(self.blocks))}

        for i, block in enumerate(self.blocks):
            block_state = state[i]
            x, block_state_new = block(x, block_state)

            if block_state is None:
                state[i] = block_state_new
            else:
                # layer state is a tuple of three tensors: c, n, m
                # we update the state in place in order to avoid creating new tensors
                for state_idx in range(len(block_state)):
                    state[i][state_idx].copy_(block_state_new[state_idx])

        x = self.out_norm(x)

        return x, state


class FeedForward(nn.Module):
    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config

        self.up_proj_dim = round_up_to_next_multiple_of(
            config.embedding_dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )

        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
            self.proj_up = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(
                in_features=config.embedding_dim,
                out_features=2 * self.up_proj_dim,
                bias=self.config.use_bias,
            )

        self.proj_down = nn.Linear(
            in_features=self.up_proj_dim,
            out_features=config.embedding_dim,
            bias=self.config.use_bias,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y


@dataclass
class mLSTMLayerConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    num_heads: int
    """Number of heads."""
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in the normalization layers."""
    norm_reduction_force_float32: bool = True
    """Whether to force float32 reductions in the normalization layers."""

    qk_dim_factor: float = 0.5
    """The factor to determine the dimension of the query and key tensors."""
    v_dim_factor: float = 1.0
    """The factor to determine the dimension of the value tensor."""
    gate_soft_cap: float = 15.0
    """Soft cap value for the gates."""

    mlstm_backend: mLSTMBackendConfig = field(default_factory=mLSTMBackendConfig)
    """Configuration of the mLSTM backend."""

    weight_mode: WeightModeType = "single"
    """The weight mode to use for the mLSTM layer.
    Mode 'single' uses separate weights for the query, key, value, and gates.
    Mode 'fused' uses a single weight matrix for the query, key, value, and output gates.
    """


class mLSTMLayer(nn.Module):
    def __init__(self, config: mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        if self.config.weight_mode == "single":
            self.q = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.k = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.v = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )

            self.ogate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )
            self.igate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
            self.fgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
        elif self.config.weight_mode == "fused":
            self.qkv_opreact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.qk_dim + 2 * self.v_dim,
                bias=self.config.use_bias,
            )
            self.ifgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.config.num_heads,
                bias=True,
            )

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)

        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.v_dim // self.config.num_heads,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.v_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(
        self, x: torch.Tensor, state: mLSTMLayerStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape
        if self.config.weight_mode == "single":
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            o_preact = self.ogate_preact(x)
            i_preact = soft_cap(
                self.igate_preact(x), cap_value=self.config.gate_soft_cap
            )
            f_preact = soft_cap(
                self.fgate_preact(x), cap_value=self.config.gate_soft_cap
            )

        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(
                self.ifgate_preact(x), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state

        h, state = self.mlstm_backend(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
        )
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )
        assert (
            h.shape == expected_h_shape
        ), f"Got {h.shape}, expected {expected_h_shape}"

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state


class mLSTMBlock(nn.Module):
    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config
        self.norm_mlstm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.mlstm_layer = mLSTMLayer(
            mLSTMLayerConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                norm_reduction_force_float32=config.norm_reduction_force_float32,
                qk_dim_factor=config.qk_dim_factor,
                v_dim_factor=config.v_dim_factor,
                gate_soft_cap=config.gate_soft_cap,
                weight_mode=config.weight_mode,
                mlstm_backend=mLSTMBackendConfig(
                    chunkwise_kernel=config.chunkwise_kernel,
                    sequence_kernel=config.sequence_kernel,
                    step_kernel=config.step_kernel,
                    mode=config.mode,
                    chunk_size=config.chunk_size,
                    return_last_states=config.return_last_states,
                    autocast_kernel_dtype=config.autocast_kernel_dtype,
                    eps=config.eps,
                    inference_state_dtype=config.inference_state_dtype,
                ),
            )
        )
        self.norm_ffn = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.ffn = FeedForward(config)

    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm

        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn

        return x, state
