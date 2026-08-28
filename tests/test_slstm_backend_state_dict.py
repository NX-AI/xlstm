"""Regression tests for sLSTM cross-backend checkpoint loading (issue #127).

The sLSTM cell stores its recurrent kernel and bias in a backend-specific
internal layout. ``state_dict`` used to serialize those internal parameters
directly, so a checkpoint saved with one backend could not be loaded into a
cell using another backend. These tests run on CPU via ``skip_backend_init``.
"""

import pytest
import torch

from xlstm.blocks.slstm.cell import (
    sLSTMCell_cuda,
    sLSTMCell_vanilla,
    sLSTMCellConfig,
)


def _external_weights(cell):
    """Return the backend-agnostic (external) recurrent kernel and bias."""
    return (
        cell._recurrent_kernel_int2ext(cell._recurrent_kernel_),
        cell._bias_int2ext(cell._bias_),
    )


def _randomize(cell):
    with torch.no_grad():
        cell._recurrent_kernel_.copy_(torch.randn_like(cell._recurrent_kernel_))
        cell._bias_.copy_(torch.randn_like(cell._bias_))


@pytest.mark.parametrize(
    "src_cls,dst_cls",
    [
        (sLSTMCell_cuda, sLSTMCell_vanilla),
        (sLSTMCell_vanilla, sLSTMCell_cuda),
    ],
)
def test_state_dict_loads_across_backends(src_cls, dst_cls):
    """A checkpoint saved with one backend loads into the other without error
    and preserves the (canonical) weights."""
    config = sLSTMCellConfig(hidden_size=16, num_heads=4)
    src = src_cls(config, skip_backend_init=True)
    _randomize(src)
    dst = dst_cls(config, skip_backend_init=True)

    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    assert missing == [] and unexpected == []

    src_kernel, src_bias = _external_weights(src)
    dst_kernel, dst_bias = _external_weights(dst)
    assert torch.allclose(dst_kernel, src_kernel)
    assert torch.allclose(dst_bias, src_bias)


@pytest.mark.parametrize("cls", [sLSTMCell_vanilla, sLSTMCell_cuda])
def test_state_dict_same_backend_round_trip(cls):
    """Saving and reloading within the same backend keeps the internal weights
    bit-for-bit identical."""
    config = sLSTMCellConfig(hidden_size=16, num_heads=4)
    src = cls(config, skip_backend_init=True)
    _randomize(src)
    dst = cls(config, skip_backend_init=True)

    dst.load_state_dict(src.state_dict())
    assert torch.allclose(dst._recurrent_kernel_, src._recurrent_kernel_)
    assert torch.allclose(dst._bias_, src._bias_)


@pytest.mark.parametrize("cls", [sLSTMCell_vanilla, sLSTMCell_cuda])
def test_state_dict_tensors_are_contiguous(cls):
    """Exported tensors must be contiguous so serializers that require it (e.g.
    ``safetensors.torch.save_file``) accept the state dict."""
    config = sLSTMCellConfig(hidden_size=16, num_heads=4)
    cell = cls(config, skip_backend_init=True)

    state_dict = cell.state_dict()
    assert state_dict["_recurrent_kernel_"].is_contiguous()
    assert state_dict["_bias_"].is_contiguous()


@pytest.mark.parametrize("cls", [sLSTMCell_vanilla, sLSTMCell_cuda])
def test_state_dict_loads_legacy_internal_checkpoint(cls):
    """A legacy checkpoint stored in the internal layout still loads into a cell
    of its original backend (backward compatibility)."""
    config = sLSTMCellConfig(hidden_size=16, num_heads=4)
    src = cls(config, skip_backend_init=True)
    _randomize(src)
    legacy = {
        "_recurrent_kernel_": src._recurrent_kernel_.detach().clone(),
        "_bias_": src._bias_.detach().clone(),
    }

    dst = cls(config, skip_backend_init=True)
    dst.load_state_dict(legacy)
    assert torch.allclose(dst._recurrent_kernel_, src._recurrent_kernel_)
    assert torch.allclose(dst._bias_, src._bias_)
