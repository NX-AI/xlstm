"""TorchScript compatibility regression test.

Ensures `torch.jit.script(xLSTMBlockStack(...))` returns without raising.
Before the `**kwargs` removal from `xLSTMBlockStack.forward`, this raised
because TorchScript requires explicit (non-variadic) signatures on all
forward methods.

If this test starts failing, a regression has reintroduced a variadic
signature into `xLSTMBlockStack.forward`. See the PR that introduced
this file for the rationale.
"""

import pytest
import torch

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm import mLSTMBlockConfig, mLSTMLayerConfig


def make_small_stack():
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(proj_factor=2.0)),
        num_blocks=2,
        embedding_dim=64,
        context_length=128,
    )
    return xLSTMBlockStack(cfg)


def test_xlstm_block_stack_is_scriptable():
    model = make_small_stack()
    # Should not raise. If it does, the **kwargs leak is back.
    scripted = torch.jit.script(model)
    assert scripted is not None


def test_xlstm_block_stack_scripted_forward_matches():
    model = make_small_stack()
    model.train(False)
    scripted = torch.jit.script(model)

    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        ref = model(x)
        out = scripted(x)
    # Within fp32 epsilon.
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_xlstm_block_stack_save_load_roundtrip(tmp_path):
    model = make_small_stack()
    scripted = torch.jit.script(model)
    path = tmp_path / "stack.pt"
    torch.jit.save(scripted, str(path))
    reloaded = torch.jit.load(str(path))
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        a = scripted(x)
        b = reloaded(x)
    assert torch.allclose(a, b, atol=1e-6)
