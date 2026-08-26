"""Regression test suite for xlstm-plus features (boundaries, auto-fallback, TBPTT, checkpointing)."""

import os
import sys
import gc
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

import xlstm_plus
from xlstm_plus import (
    detach_states,
    zero_rows,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlock,
    mLSTMBlockConfig,
    sLSTMBlock,
    sLSTMBlockConfig,
    mLSTMLayer,
    mLSTMLayerConfig,
    sLSTMLayer,
    sLSTMLayerConfig,
)
from xlstm_plus.xlstm_large import (
    xLSTMLarge,
    xLSTMLargeConfig,
    xLSTMLargeBlockStack,
    mLSTMLayer as mLSTMLayerLarge,
    mLSTMLayerConfig as mLSTMLayerConfigLarge,
)


class TestExports(unittest.TestCase):
    def test_top_level_exports(self):
        self.assertTrue(callable(xlstm_plus.detach_states))
        self.assertTrue(callable(xlstm_plus.zero_rows))

    def test_xlstm_large_exports(self):
        from xlstm_plus.xlstm_large import detach_states as ds, zero_rows as zr
        self.assertIs(ds, xlstm_plus.detach_states)
        self.assertIs(zr, xlstm_plus.zero_rows)


class TestStateUtils(unittest.TestCase):
    def test_detach_states_xlstm_large_dict(self):
        B, H, Dh = 2, 2, 8
        C = torch.randn(B, H, Dh, Dh, requires_grad=True)
        n = torch.randn(B, H, Dh, requires_grad=True)
        m = torch.randn(B, H, 1, requires_grad=True)
        state = {0: (C, n, m), 1: (C * 2, n * 2, m * 2)}

        detached = detach_states(state)
        self.assertFalse(detached[0][0].requires_grad)
        self.assertFalse(detached[0][1].requires_grad)
        self.assertFalse(detached[0][2].requires_grad)
        self.assertFalse(detached[1][0].requires_grad)
        self.assertTrue(C.requires_grad)

    def test_detach_states_classic_stack_dict(self):
        B, H = 2, 16
        c = torch.randn(B, H, requires_grad=True)
        state = {"block_0": {"conv_state": None, "slstm_state": (c, c, c, c)}}
        detached = detach_states(state)
        self.assertFalse(detached["block_0"]["slstm_state"][0].requires_grad)

    def test_zero_rows_xlstm_large(self):
        B, H, Dh = 4, 2, 8
        C = torch.ones(B, H, Dh, Dh)
        n = torch.ones(B, H, Dh)
        m = torch.ones(B, H, 1)
        state = {0: (C, n, m)}

        mask = torch.tensor([True, False, True, False])
        zero_rows(state, mask)

        self.assertEqual(C[0].abs().sum().item(), 0.0)
        self.assertGreater(C[1].abs().sum().item(), 0.0)
        self.assertEqual(C[2].abs().sum().item(), 0.0)
        self.assertGreater(C[3].abs().sum().item(), 0.0)

        self.assertEqual(n[0].abs().sum().item(), 0.0)
        self.assertGreater(n[1].abs().sum().item(), 0.0)
        self.assertEqual(m[0].abs().sum().item(), 0.0)
        self.assertGreater(m[1].abs().sum().item(), 0.0)

    def test_zero_rows_guards_attached_grad(self):
        C = torch.ones(2, 2, 4, 4, requires_grad=True)
        state = {0: (C, None, None)}
        with self.assertRaises(RuntimeError):
            zero_rows(state, torch.tensor([True, False]))


class TestXLSTMLargeFeatures(unittest.TestCase):
    def setUp(self):
        gc.collect()

    def test_packed_boundaries_isolation_auto_mode(self):
        """Document 2 outputs must be completely unaffected by Document 1 changes."""
        config = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=2,
            num_blocks=2,
            vocab_size=50,
            chunk_size=16,
            mode="auto",
            return_last_states=True,
        )
        model = xLSTMLarge(config)

        B, S = 2, 32
        x1 = torch.randint(0, 50, (B, S))
        x2 = x1.clone()
        x2[:, :16] = torch.randint(0, 50, (B, 16))

        boundaries = torch.zeros(B, S, dtype=torch.bool)
        boundaries[:, 16] = True

        with torch.no_grad():
            out1, _ = model(x1, boundaries=boundaries)
            out2, _ = model(x2, boundaries=boundaries)

        diff_doc2 = (out1[:, 16:] - out2[:, 16:]).abs().max().item()
        self.assertLess(diff_doc2, 1e-4)

    def test_auto_fallback_unaligned_sequence_lengths(self):
        """Mode='auto' handles non-multiple chunk sizes without error."""
        config = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=2,
            num_blocks=1,
            vocab_size=50,
            chunk_size=16,
            mode="auto",
            return_last_states=True,
        )
        model = xLSTMLarge(config)

        for seq_len in [1, 7, 13, 25, 33, 49]:
            x = torch.randint(0, 50, (2, seq_len))
            logits, state = model(x)
            self.assertEqual(logits.shape, (2, seq_len, 50))
            self.assertIn(0, state)
            loss = logits.sum()
            loss.backward()

    def test_return_detached_states_kwarg(self):
        """return_detached_states=True returns state tensors without grad requirement."""
        config = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=2,
            num_blocks=1,
            vocab_size=50,
            chunk_size=16,
            mode="auto",
            return_last_states=True,
        )
        model = xLSTMLarge(config)

        x = torch.randint(0, 50, (2, 16))
        logits, state = model(x, return_detached_states=True)
        C, n, m = state[0]
        self.assertFalse(C.requires_grad)
        self.assertFalse(n.requires_grad)
        self.assertFalse(m.requires_grad)

    def test_tbptt_continuous_training_loop(self):
        """Multi-step TBPTT with return_detached_states=True trains smoothly."""
        config = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=2,
            num_blocks=2,
            vocab_size=50,
            chunk_size=16,
            mode="auto",
            return_last_states=True,
        )
        model = xLSTMLarge(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        state = None
        for step in range(4):
            optimizer.zero_grad()
            x = torch.randint(0, 50, (2, 16))
            targets = torch.randint(0, 50, (2, 16))
            logits, state = model(x, state=state, return_detached_states=True)
            loss = F.cross_entropy(logits.view(-1, 50), targets.view(-1))
            loss.backward()
            optimizer.step()

    def test_activation_checkpointing_forward_backward(self):
        """use_checkpoint=True computes identical gradients to standard forward."""
        torch.manual_seed(42)
        cfg_no_ckpt = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=2,
            num_blocks=1,
            vocab_size=50,
            chunk_size=16,
            mode="auto",
            use_checkpoint=False,
        )
        model_no_ckpt = xLSTMLarge(cfg_no_ckpt)

        torch.manual_seed(42)
        cfg_ckpt = xLSTMLargeConfig(
            embedding_dim=32,
            num_heads=2,
            num_blocks=1,
            vocab_size=50,
            chunk_size=16,
            mode="auto",
            use_checkpoint=True,
        )
        model_ckpt = xLSTMLarge(cfg_ckpt)

        x = torch.randint(0, 50, (2, 16))
        loss1 = model_no_ckpt(x).sum()
        loss1.backward()

        loss2 = model_ckpt(x).sum()
        loss2.backward()

        p1 = next(model_no_ckpt.parameters())
        p2 = next(model_ckpt.parameters())
        self.assertTrue(torch.allclose(p1.grad, p2.grad, atol=1e-5))


class TestClassicStackFeatures(unittest.TestCase):
    def setUp(self):
        gc.collect()

    def test_classic_mlstm_cell_boundaries_and_states(self):
        cell_cfg = mLSTMLayerConfig(
            embedding_dim=32,
            num_heads=2,
            context_length=64,
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            proj_factor=1.0,
        )
        layer = mLSTMLayer(cell_cfg)

        B, S = 2, 32
        x = torch.randn(B, S, 32)
        boundaries = torch.zeros(B, S, dtype=torch.bool)
        boundaries[:, 16] = True

        out, state = layer(x, boundaries=boundaries, return_last_state=True, return_detached_states=True)
        self.assertEqual(out.shape, (B, S, 32))
        self.assertIsNotNone(state)
        C, n, m = state
        self.assertFalse(C.requires_grad)

    def test_classic_slstm_layer_boundaries_and_states(self):
        slstm_cfg = sLSTMLayerConfig(
            embedding_dim=32,
            num_heads=2,
            conv1d_kernel_size=0,
            backend="vanilla",
        )
        layer = sLSTMLayer(slstm_cfg)

        B, S = 2, 32
        x = torch.randn(B, S, 32)
        boundaries = torch.zeros(B, S, dtype=torch.bool)
        boundaries[:, 16] = True

        out, state = layer(x, boundaries=boundaries, return_last_state=True, return_detached_states=True)
        self.assertEqual(out.shape, (B, S, 32))
        self.assertIn("slstm_state", state)

    def test_hybrid_block_stack_tbptt(self):
        mlstm_cfg = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                embedding_dim=32,
                num_heads=2,
                context_length=64,
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                proj_factor=1.0,
            )
        )
        slstm_cfg = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                embedding_dim=32,
                num_heads=2,
                conv1d_kernel_size=0,
                backend="vanilla",
            )
        )
        stack_cfg = xLSTMBlockStackConfig(
            embedding_dim=32,
            num_blocks=2,
            context_length=64,
            mlstm_block=mlstm_cfg,
            slstm_block=slstm_cfg,
            slstm_at=[1],
        )
        stack = xLSTMBlockStack(stack_cfg)

        B, S = 2, 32
        state = None
        for step in range(3):
            x = torch.randn(B, S, 32, requires_grad=True)
            out, state = stack(x, state=state, return_last_states=True, return_detached_states=True)
            loss = out.sum()
            loss.backward()
            self.assertIn("block_0", state)
            self.assertIn("block_1", state)


if __name__ == "__main__":
    unittest.main()
