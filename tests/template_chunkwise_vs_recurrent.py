from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
import torch
import numpy as np


def template_test_recurrent_vs_chunkwise(
    chunkwise_kernel_name: str = "chunkwise--native_autograd",
    sequence_kernel_name: str = "native_sequence__native",
    step_kernel_name: str = "native",
    seed: int = 0,
    input_shape: tuple[int, int] = (2, 256),
    vocab_size: int = 512,
    num_blocks: int = 2,
    embedding_dim: int = 256,
    num_heads: int = 4,
):
    """Tests wether the recurrent and chunkwise versions of the model produce the same output.
    """
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

    mlstm = mlstm.to("cuda")
    torch.manual_seed(seed)

    input = torch.randint(0, vocab_size, input_shape).to("cuda")

    out, state = mlstm(input)

    assert out.shape == input_shape + (
        vocab_size,
    ), f"out has wrong shape, got {out.shape}"

    assert (
        len(state) == num_blocks
    ), f"state has wrong length, got {len(state)}, expected {num_blocks}"
    assert (
        len(state[0]) == 3
    ), f"state[0] has wrong length, got {len(state[0])}, expected 3"

    step_out, step_state = mlstm(input[:, 0:1])
    assert step_out.shape == (
        input_shape[0],
        1,
        vocab_size,
    ), f"step_out has wrong shape, got {step_out.shape}"
    assert (
        len(step_state) == num_blocks
    ), f"step_state has wrong length, got {len(step_state)}, expected {num_blocks}"
    assert (
        len(step_state[0]) == 3
    ), f"step_state[0] has wrong length, got {len(step_state[0])}, expected 3"

    out_steps = []
    state = None
    for i in range(input_shape[1]):
        out_s, state = mlstm(input[:, i : i + 1], state)
        out_steps.append(out_s)

    out_steps = torch.cat(out_steps, dim=1)

    assert (
        out_steps.shape == input_shape + (vocab_size,)
    ), f"out_steps has wrong shape, got {out_steps.shape}, expected {input_shape + (vocab_size,)}"

    out_np = out.cpu().detach().numpy()
    out_steps_np = out_steps.cpu().detach().numpy()

    np.testing.assert_allclose(out_steps_np, out_np, atol=4e-3, rtol=5e-2, err_msg="out_steps and out do not match")
