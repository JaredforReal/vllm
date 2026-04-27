# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for pack_seq_triton / unpack_seq_triton correctness and
Triton JIT recompilation behaviour.

The fix ensures that runtime-varying kernel parameters (N, B, Lmax)
are not declared as ``tl.constexpr`` so that the Triton JIT does not
recompile the kernel on every unique value.
"""

import pytest
import torch

from vllm.v1.attention.ops.common import (
    _pack_seq_kernel,
    _unpack_seq_triton_kernel,
    pack_seq_triton,
    unpack_seq_triton,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """pack → unpack and return the result."""
    packed = pack_seq_triton(x, lengths)
    return unpack_seq_triton(packed, lengths)


# ---------------------------------------------------------------------------
# Correctness: basic roundtrip
# ---------------------------------------------------------------------------


class TestPackUnpackCorrectness:
    @pytest.fixture(autouse=True)
    def _set_seed(self):
        torch.manual_seed(42)

    @pytest.mark.parametrize(
        "lengths, D",
        [
            # Single sequence
            ([10], 32),
            # Multiple equal-length sequences
            ([5, 5, 5], 64),
            # Variable lengths
            ([3, 7, 1, 4], 128),
            # Long sequence
            ([256], 576),
            # MLA-like dimensions
            ([8, 16, 4], 576),
            # Many short sequences
            ([1, 1, 1, 1, 1, 1, 1, 1], 64),
            # Single token sequences
            ([1], 576),
        ],
        ids=lambda p: str(p),
    )
    def test_roundtrip_2d(self, lengths, D):
        """Data survives pack → unpack intact for various shapes."""
        N = sum(lengths)
        x = torch.randn(N, D, device="cuda")
        lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        out = _roundtrip(x, lengths_t)

        assert out.shape == x.shape
        assert torch.allclose(x, out, atol=1e-6)

    @pytest.mark.parametrize(
        "lengths, shape",
        [
            ([3, 5], (8, 4, 4)),  # 3-D input
            ([2, 3, 1], (6, 8, 2)),  # 3-D input
        ],
        ids=lambda p: str(p),
    )
    def test_roundtrip_3d(self, lengths, shape):
        """Multi-dimensional inputs are correctly handled."""
        x = torch.randn(*shape, device="cuda")
        lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        out = _roundtrip(x, lengths_t)

        assert out.shape == x.shape
        assert torch.allclose(x, out, atol=1e-6)

    def test_padding_values(self):
        """Padded positions should equal the pad_value (-inf)."""
        lengths = [2, 5]
        D = 32
        x = torch.randn(sum(lengths), D, device="cuda")
        lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)

        packed = pack_seq_triton(x, lengths_t)
        # First batch: Lmax=5, but only 2 valid; positions 2..4 should be -inf
        assert torch.all(packed[0, 2:] == float("-inf"))
        # Second batch: all 5 valid, no padding
        assert torch.allclose(packed[1], x[2:7], atol=1e-6)

    def test_different_dtypes(self):
        """Works with float16 and bfloat16."""
        lengths = [4, 6, 2]
        D = 64
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(sum(lengths), D, device="cuda", dtype=dtype)
            lengths_t = torch.tensor(lengths, device="cuda", dtype=torch.int32)
            out = _roundtrip(x, lengths_t)
            assert out.dtype == dtype
            assert torch.allclose(x, out, atol=1e-2)


# ---------------------------------------------------------------------------
# JIT recompilation: the core fix
# ---------------------------------------------------------------------------


class TestNoRecompilation:
    def test_kernel_cache_does_not_grow_with_varying_lmax(self):
        """Calling the kernels with many different (N, B, Lmax) triples
        should NOT cause unbounded Triton JIT recompilation.

        Before the fix, every unique ``Lmax`` value triggered a full
        kernel recompilation because ``Lmax`` was declared as
        ``tl.constexpr``.  After the fix, the kernel is compiled once
        and reused for all values.
        """
        D = 576  # typical MLA dimension, constexpr in the kernel

        # Snapshot the kernel caches before the test loop.
        pack_cache_before = len(_pack_seq_kernel.cache)
        unpack_cache_before = len(_unpack_seq_triton_kernel.cache)

        # Run many steps with different batch sizes and sequence lengths.
        for step in range(30):
            B = torch.randint(1, 10, (1,)).item()
            lengths = torch.randint(1, 128, (B,), device="cuda")
            lengths_int = lengths.int()
            N = int(lengths.sum().item())
            Lmax = int(lengths.max().item())

            x = torch.randn(N, D, device="cuda")

            packed = pack_seq_triton(x, lengths_int)
            unpacked = unpack_seq_triton(packed, lengths_int)

            # Verify correctness on every step (not just the last).
            assert torch.allclose(x, unpacked, atol=1e-6), (
                f"Data mismatch at step {step}: B={B}, Lmax={Lmax}, N={N}, D={D}"
            )

        pack_cache_after = len(_pack_seq_kernel.cache)
        unpack_cache_after = len(_unpack_seq_triton_kernel.cache)

        pack_new = pack_cache_after - pack_cache_before
        unpack_new = unpack_cache_after - unpack_cache_before

        # With the fix, each kernel should compile at most a small number
        # of times (typically 1-2, for different BLOCK size combinations),
        # NOT once per unique Lmax value.
        #
        # Before the fix: 30 new entries (one per unique Lmax).
        # After the fix:   <= 3 new entries.
        assert pack_new <= 3, (
            f"_pack_seq_kernel compiled {pack_new} times for "
            f"30 different Lmax values; expected <= 3. "
            f"do_not_specialize may not be working."
        )
        assert unpack_new <= 3, (
            f"_unpack_seq_triton_kernel compiled {unpack_new} times for "
            f"30 different Lmax values; expected <= 3. "
            f"do_not_specialize may not be working."
        )
