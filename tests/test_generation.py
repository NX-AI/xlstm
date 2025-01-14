import torch
import pytest

from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from xlstm.xlstm_large.generate import generate_tokens


def template_test_generation(
    batch_size: int,
    prefill_length: int,
    max_length: int,
    chunkwise_kernel_name: str = "chunkwise--native_autograd",
    sequence_kernel_name: str = "native_sequence__native",
    step_kernel_name: str = "native",
    seed: int = 0,
    vocab_size: int = 512,
    num_blocks: int = 2,
    embedding_dim: int = 256,
    num_heads: int = 4,
    use_torch_compile_model: bool = False,
    use_torch_compile_generate: bool = False,
    device: str = "cuda",
):
    """A generic test template that tests text generation.

    Args:
        batch_size: The batch size.
        prefill_length: The length of the prefill.
        max_length: The maximum length of the generated text.
        chunkwise_kernel_name: The chunkwise kernel to use.
        sequence_kernel_name: The sequence kernel to use.
        step_kernel_name: The step kernel to use.
        seed: The random seed.
        vocab_size: The vocabulary size.
        num_blocks: The number of blocks.
        embedding_dim: The embedding dimension.
        num_heads: The number of heads.
        use_torch_compile_model: Whether to compile the model.
        use_torch_compile_generate: Whether to compile the generation function.
        device: The device to use.
    """
    device_str = device
    device = torch.device(device)
    mlstm_config = xLSTMLargeConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        mode="inference",
        chunkwise_kernel=chunkwise_kernel_name,
        sequence_kernel=sequence_kernel_name,
        step_kernel=step_kernel_name,
        return_last_states=True,
    )
    mlstm = xLSTMLarge(mlstm_config)
    if use_torch_compile_model:
        mlstm = torch.compile(mlstm)

    torch.manual_seed(seed)
    mlstm = mlstm.to(device=device)

    if prefill_length > 0:
        prefill_tokens = torch.randint(0, vocab_size, (batch_size, prefill_length)).to(
            device=device
        )
    else:
        prefill_tokens = None
        batch_size = 1

    def llm_forward(tokens, state):
        return mlstm(tokens, state)

    generate_tokens_fn = generate_tokens
    if use_torch_compile_generate:
        generate_tokens_fn = torch.compile(generate_tokens)

    generated_tokens, state = generate_tokens_fn(
        llm_forward=llm_forward,
        prefill_tokens=prefill_tokens,
        max_length=max_length,
        device=device_str,
    )

    assert generated_tokens.shape == (
        batch_size,
        max_length,
    ), f"generated_tokens has wrong shape, got {generated_tokens.shape}"


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("prefill_length", [0])
@pytest.mark.parametrize("max_length", [32, 1, 128])
def test_generate_no_prefill(batch_size: int, prefill_length: int, max_length: int):
    template_test_generation(
        batch_size=batch_size,
        prefill_length=prefill_length,
        max_length=max_length,
    )


@pytest.mark.parametrize(
    "batch_size, prefill_length, max_length",
    [
        [2, 128, 128],
        [2, 150, 128],
    ],
)
@pytest.mark.parametrize(
    "chunkwise_kernel, sequence_kernel, step_kernel",
    [
        [
            "chunkwise--triton_limit_chunk",
            "native_sequence__triton",
            "triton",
        ],
        [
            "chunkwise--triton_xl_chunk",
            "native_sequence__triton",
            "triton",
        ],
    ],
)
def test_generate_prefill(
    batch_size: int,
    prefill_length: int,
    max_length: int,
    chunkwise_kernel: str,
    sequence_kernel: str,
    step_kernel: str,
):
    """Tests text generation with prefilling."""
    template_test_generation(
        chunkwise_kernel_name=chunkwise_kernel,
        sequence_kernel_name=sequence_kernel,
        step_kernel_name=step_kernel,
        batch_size=batch_size,
        prefill_length=prefill_length,
        max_length=max_length,
    )


@pytest.mark.parametrize(
    "batch_size, prefill_length, max_length",
    [
        [2, 128, 128],
    ],
)
@pytest.mark.parametrize(
    "chunkwise_kernel, sequence_kernel, step_kernel",
    [
        [
            "chunkwise--triton_limit_chunk",
            "native_sequence__triton",
            "triton",
        ],
        [
            "chunkwise--triton_xl_chunk",
            "native_sequence__triton",
            "triton",
        ],
    ],
)
@pytest.mark.parametrize(
    "compile_model, compile_generate", [[False, False], [True, False]]
)
def test_generate_compile(
    batch_size: int,
    prefill_length: int,
    max_length: int,
    chunkwise_kernel: str,
    sequence_kernel: str,
    step_kernel: str,
    compile_model: bool,
    compile_generate: bool,
):
    """Tests text generation with compilation."""
    template_test_generation(
        chunkwise_kernel_name=chunkwise_kernel,
        sequence_kernel_name=sequence_kernel,
        step_kernel_name=step_kernel,
        batch_size=batch_size,
        prefill_length=prefill_length,
        max_length=max_length,
        use_torch_compile_model=compile_model,
        use_torch_compile_generate=compile_generate,
    )
    torch.compiler.reset()

@pytest.mark.parametrize(
    "batch_size, prefill_length, max_length",
    [
        [2, 128, 128],
    ],
)
def test_model_generate(
    batch_size: int,
    prefill_length: int,
    max_length: int,
    chunkwise_kernel_name: str = "chunkwise--native_autograd",
    sequence_kernel_name: str = "native_sequence__native",
    step_kernel_name: str = "native",
    seed: int = 0,
    vocab_size: int = 512,
    num_blocks: int = 2,
    embedding_dim: int = 256,
    num_heads: int = 4,
    device: str = "cuda",
):
    """Tests calling the generation from the model."""
    torch.manual_seed(seed)
    device = torch.device(device)
    mlstm_config = xLSTMLargeConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        mode="inference",
        chunkwise_kernel=chunkwise_kernel_name,
        sequence_kernel=sequence_kernel_name,
        step_kernel=step_kernel_name,
        return_last_states=True,
    )
    mlstm = xLSTMLarge(mlstm_config).to(device=device)

    if prefill_length > 0:
        prefill_tokens = torch.randint(0, vocab_size, (batch_size, prefill_length)).to(
            device=device
        )
    else:
        prefill_tokens = None
        batch_size = 1

    generated_tokens, state = mlstm.generate(
        prefill_tokens=prefill_tokens,
        max_length=max_length,
    )

    assert generated_tokens.shape == (
        batch_size,
        max_length,
    ), f"generated_tokens has wrong shape, got {generated_tokens.shape}"
