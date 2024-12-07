import pytest
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from xlstm.xlstm_large.utils import convert_single_weights_to_fused_weights
import copy
import torch
import numpy as np

@pytest.mark.parametrize("use_bias", [True, False])
def test_single_weights_to_fused_weights(use_bias: bool, device="cpu"):
    """Tests weight conversion of the model from single to fused mode."""
    mlstm_config = xLSTMLargeConfig(
        embedding_dim=512,
        num_heads=4,
        num_blocks=6,
        vocab_size=2048,
        return_last_states=True,
        mode="inference",
        chunkwise_kernel="chunkwise--native_custbw",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        weight_mode="single",
        use_bias=use_bias,
    )

    mlstm_single = xLSTMLarge(mlstm_config)

    mlstm_fused_config = copy.deepcopy(mlstm_config)
    mlstm_fused_config.weight_mode = "fused"

    mlstm_fused = xLSTMLarge(mlstm_fused_config)

    # copy weights from single to fused model
    new_state_dict = convert_single_weights_to_fused_weights(mlstm_single.state_dict())

    mlstm_fused.load_state_dict(new_state_dict)

    device = torch.device(device)

    mlstm_single.to(device)
    mlstm_fused.to(device)

    inputs = torch.randint(0, mlstm_config.vocab_size, (3, 256)).to(device)

    single_output = mlstm_single(inputs)
    fused_output = mlstm_fused(inputs)

    single_out = single_output[0].cpu().detach().numpy()
    fused_out = fused_output[0].cpu().detach().numpy()

    assert single_out.shape == fused_out.shape
    assert not (single_out == 0.0).all()

    np.testing.assert_allclose(
        actual=fused_out, desired=single_out, rtol=1e-4, atol=1e-4
    )
