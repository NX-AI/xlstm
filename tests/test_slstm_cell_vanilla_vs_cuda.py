import torch
import pytest
from xlstm.blocks.slstm.cell import sLSTMCellConfig, sLSTMCell


def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_slstm_cell(backend, dtype="float32"):
    set_seed(42)

    config = sLSTMCellConfig(
        hidden_size=64,
        num_heads=4,
        num_states=4,
        backend=backend,
        dtype=dtype,
    )

    return sLSTMCell(config)


@pytest.mark.parametrize("with_in_state", [True, False])
def test_slstm_vanilla_vs_cuda(with_in_state):
    device_cuda = 'cuda'
    cell_vanilla = get_slstm_cell('vanilla')
    cell_cuda = get_slstm_cell('cuda').to(device_cuda)

    set_seed(42)
    current_input = torch.randn((1, 1, 256))
    state = torch.randn((4, 1, 64)) if with_in_state else None

    output_vanilla, state_vanilla = cell_vanilla.forward(current_input, state)
    output_cuda, state_cuda = cell_cuda.forward(current_input.to(device_cuda), state.to(device_cuda) if state is not None else state)
    
    torch.testing.assert_close(output_vanilla, output_cuda.cpu())
    torch.testing.assert_close(state_vanilla, state_cuda.cpu())


def test_slstm_vanilla_vs_cuda_fp16():
    device_cuda = 'cuda'
    cell_vanilla = get_slstm_cell('vanilla')
    cell_cuda = get_slstm_cell('cuda', dtype="float16").to(device_cuda)

    set_seed(42)
    current_input = torch.randn((1, 1, 256))
    state = torch.randn((4, 1, 64))

    output_vanilla, state_vanilla = cell_vanilla.forward(current_input, state)
    output_cuda, state_cuda = cell_cuda.forward(current_input.to(device_cuda), state.to(device_cuda))

    torch.testing.assert_close(output_vanilla, output_cuda.cpu(), rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(state_vanilla, state_cuda.cpu(), rtol=1e-3, atol=1e-5)
